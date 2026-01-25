"""
Volatility Forecasts (Part 2) — XGBSTES experiments.

Builds datasets via VolDatasetSpec (uses return feature templates and
NextDaySquaredReturnTarget), fits ES/STES/XGBSTES.

This script reports TWO evaluation protocols (applied uniformly to ALL models):
  1) Fixed train/test split (to remain comparable with Part 1 tables)
  2) Walk-forward (expanding-window) CV for robustness

It also includes optional hyperparameter tuning utilities for XGBSTES.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)
load_dotenv()

import logging

# Configure logging from env (AF_LOG_LEVEL) or default to INFO
_log_level = os.environ.get("AF_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from volatility_forecast.sources.simulated_garch import SimulatedGARCHSource

from volatility_forecast.pipeline import (
    build_default_ctx,
    build_vol_dataset,
    VolDatasetSpec,
    build_wide_dataset,
)
from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    JoinPolicy,
    MissingnessPolicy,
)
from alphaforge.data.cache import FileCacheBackend
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.features.selector import select_variant_columns
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget
from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel
from volatility_forecast.model.tree_stes_model import XGBoostSTESModel

# Note: we refer to all variants of this model family as XGBSTES_* in this script.
from volatility_forecast.evaluation import metrics
from volatility_forecast.evaluation.model_evaluator import (
    evaluate_model,
    rmse,
    mae,
    medae,
)
from sklearn.preprocessing import StandardScaler


def _clip01(a, eps: float):
    eps = float(eps)
    if eps <= 0.0:
        return a
    if isinstance(a, pd.Series):
        v = np.asarray(a.values, dtype=float)
        return pd.Series(np.clip(v, eps, 1.0 - eps), index=a.index, name=a.name)
    arr = np.asarray(a, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    if np.isscalar(a):
        return float(clipped)
    return clipped


def _logit(a, eps: float):
    clipped = _clip01(a, eps)
    if isinstance(clipped, pd.Series):
        v = np.asarray(clipped.values, dtype=float)
        out = np.log(v) - np.log1p(-v)
        return pd.Series(out, index=clipped.index, name=clipped.name)
    arr = np.asarray(clipped, dtype=float)
    out = np.log(arr) - np.log1p(-arr)
    if np.isscalar(a):
        return float(out)
    return out


def _compute_alpha_star(
    idx: pd.Index,
    y_true: pd.Series,
    r: pd.Series,
    y_hat: pd.Series,
    *,
    clip_eps: float,
    denom_quantile: float,
    min_denom_floor: float,
    quantile_from_train_denom_abs: pd.Series | np.ndarray | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Compute implied optimal alpha* and denom-stability mask on a given index.

    Notes:
        - v_state_t is proxied by y_hat.shift(1) (previous forecast).
        - Threshold uses TRAIN-calibrated denom magnitudes when provided.
    """
    idx = pd.Index(idx)
    y_true_s = y_true.reindex(idx)
    r_s = r.reindex(idx)
    y_hat_s = y_hat.reindex(idx)

    v_state = y_hat_s.shift(1)
    r2 = r_s.astype(float) ** 2
    denom = r2 - v_state
    numer = y_true_s - v_state

    denom_abs = denom.abs()

    if quantile_from_train_denom_abs is None:
        q_src = np.asarray(denom_abs.values, dtype=float)
    else:
        if isinstance(quantile_from_train_denom_abs, pd.Series):
            q_src = np.asarray(quantile_from_train_denom_abs.values, dtype=float)
        else:
            q_src = np.asarray(quantile_from_train_denom_abs, dtype=float)

    q_src = q_src[np.isfinite(q_src)]
    q = (
        float(np.quantile(q_src, float(denom_quantile)))
        if q_src.size > 0
        else float("nan")
    )
    thresh = (
        max(float(min_denom_floor), float(q))
        if np.isfinite(q)
        else float(min_denom_floor)
    )

    denom_v = np.asarray(denom.values, dtype=float)
    numer_v = np.asarray(numer.values, dtype=float)
    denom_valid = np.isfinite(denom_v) & np.isfinite(numer_v) & (denom_v != 0.0)

    alpha_star_raw = np.full(len(idx), np.nan, dtype=float)
    alpha_star_raw[denom_valid] = numer_v[denom_valid] / denom_v[denom_valid]
    alpha_star_s = pd.Series(alpha_star_raw, index=idx, name="alpha_star")
    alpha_star_s = _clip01(alpha_star_s, float(clip_eps))

    mask = pd.Series(
        denom_valid & (np.asarray(denom_abs.values, dtype=float) >= thresh),
        index=idx,
        name="alpha_star_mask",
    )

    return alpha_star_s, mask, float(thresh)


def _alpha_alignment_metrics(
    alpha_pred: pd.Series, alpha_star: pd.Series, mask: pd.Series, *, logit_eps: float
) -> dict:
    """Compute alpha/alpha* alignment diagnostics on masked points."""
    a = alpha_pred.astype(float)
    s = alpha_star.astype(float)
    m = mask.astype(bool)

    idx = a.index.intersection(s.index).intersection(m.index)
    a = a.reindex(idx)
    s = s.reindex(idx)
    m = m.reindex(idx)

    denom_valid = s.notna()
    denom_n = int(denom_valid.sum())
    frac_mask = float(m[denom_valid].mean()) if denom_n > 0 else np.nan

    keep = m & np.isfinite(a) & np.isfinite(s)
    n_mask = int(keep.sum())

    if n_mask <= 0:
        return {
            "alpha_mean": np.nan,
            "alpha_std": np.nan,
            "alpha_star_mean": np.nan,
            "alpha_star_std": np.nan,
            "corr_alpha_alpha_star": np.nan,
            "rmse_logit_alpha": np.nan,
            "mae_alpha": np.nan,
            "frac_mask": frac_mask,
            "n_mask": n_mask,
        }

    a_m = a.loc[keep].to_numpy(dtype=float)
    s_m = s.loc[keep].to_numpy(dtype=float)

    alpha_mean = float(np.mean(a_m))
    alpha_std = float(np.std(a_m, ddof=0))
    alpha_star_mean = float(np.mean(s_m))
    alpha_star_std = float(np.std(s_m, ddof=0))

    corr = np.nan
    if n_mask > 1 and np.std(a_m, ddof=0) > 0 and np.std(s_m, ddof=0) > 0:
        corr = float(np.corrcoef(a_m, s_m)[0, 1])

    la = _logit(pd.Series(a_m), float(logit_eps))
    ls = _logit(pd.Series(s_m), float(logit_eps))
    diff = la.to_numpy(dtype=float) - ls.to_numpy(dtype=float)
    rmse_logit = float(np.sqrt(np.mean(diff**2)))

    mae_alpha = float(np.mean(np.abs(a_m - s_m)))

    return {
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_star_mean": alpha_star_mean,
        "alpha_star_std": alpha_star_std,
        "corr_alpha_alpha_star": corr,
        "rmse_logit_alpha": rmse_logit,
        "mae_alpha": mae_alpha,
        "frac_mask": frac_mask,
        "n_mask": n_mask,
    }


# Hyperparameter tuning imports (optional)
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from scipy.stats import randint, uniform, loguniform

try:
    import optuna
except Exception:
    optuna = None


class ModelName(Enum):
    ES = "ES"
    STES_AE = "STES_AE"
    STES_SE = "STES_SE"
    STES_EAE = "STES_EAE"
    STES_ESE = "STES_ESE"
    STES_AESE = "STES_AESE"
    STES_EAESE = "STES_EAESE"
    # Option B: unified naming for all XGBSTES variants
    XGBSTES_BASE = "XGBSTES_BASE"
    XGBSTES_BASE_MONO = "XGBSTES_BASE_MONO"
    XGBSTES_BASE_HUBER = "XGBSTES_BASE_HUBER"
    XGBSTES_BASE_RESID = "XGBSTES_BASE_RESID"
    XGBSTES_BASE_MONO_HUBER = "XGBSTES_BASE_MONO_HUBER"
    XGBSTES_BASE_MONO_RESID = "XGBSTES_BASE_MONO_RESID"
    XGBSTES_BASE_HUBER_RESID = "XGBSTES_BASE_HUBER_RESID"
    XGBSTES_BASE_MONO_HUBER_RESID = "XGBSTES_BASE_MONO_HUBER_RESID"


# --- Simulation & dataset helpers (reused from Part 1) ---
ENTITY = "SIMULATED"
SOURCE = "simulated_garch"
START = pd.Timestamp("2000-01-01", tz="UTC")
END = pd.Timestamp("2023-01-01", tz="UTC")
N_LAGS = 0  # templates use k starting at 0; N_LAGS=0 requests today's return only
IS_INDEX = 500
OS_INDEX = 2000
N_RUNS = 100  # default sweep size for Part 2 experiments (increase as desired)
VARIANTS = [
    "ES",
    "STES_AE",
    "STES_SE",
    "STES_EAE",
    "STES_ESE",
    "STES_AESE",
    "STES_EAESE",
    # Option B: XGBSTES variants
    "XGBSTES_BASE",
    "XGBSTES_BASE_MONO",
    "XGBSTES_BASE_HUBER",
    "XGBSTES_BASE_RESID",
    "XGBSTES_BASE_MONO_HUBER",
    "XGBSTES_BASE_MONO_RESID",
    "XGBSTES_BASE_HUBER_RESID",
    "XGBSTES_BASE_MONO_HUBER_RESID",
]

# SPY study settings (notebook-aligned)
SPY_TICKER = "SPY"
SPY_START = pd.Timestamp("2000-01-01", tz="UTC")
SPY_END = pd.Timestamp("2023-12-31", tz="UTC")
SPY_IS_INDEX = 200
SPY_OS_INDEX = 4000
SPY_N_INITS = 100


# Flat defaults for the upgraded XGBSTES (tree_stes_model) used throughout this script.
# These are translated into xgboost.train params via _make_xgb_stes_model(...).
DEFAULT_XGBOOST_PARAMS: dict = {
    "num_boost_round": 200,
    "max_depth": 5,
    "learning_rate": 0.5,  # mapped to XGBoost 'eta'
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,  # mapped to XGBoost 'lambda'
    "reg_alpha": 0.0,  # mapped to XGBoost 'alpha'
    "verbosity": 0,
    # XGBSTES specific knobs
    # "fit_mode": "alternating",  # removed
    "init_window": 500,
    # "block_size": None,  # removed
    # "n_alt_iters": 3,
    "n_alt_iters": 10,
    "alt_clip_eps": 1e-6,
    # New model knobs
    # "alt_objective": "reg:pseudohubererror",
    "alt_objective": "reg:squarederror",
    "residual_mode": True,
    "monotonic_constraints": {},
    "denom_quantile": 0.05,
    "min_denom_floor": 1e-12,
    # Gate-learning knobs (tree_stes_model.XGBoostSTESModel)
    "gate_target": "alpha_clip",
    "eps_gate": 1e-3,
    "alpha_prior_mu": 0.0,
    "alpha_prior_tau": 1.5,
    "sigma_eps": None,  # if None, estimated each alt iteration (robust MAD)
    "map_max_iter": 50,
    "map_tol": 1e-8,
    "map_init": "ratio",
    "map_eps_for_ratio_init": 1e-6,
    "weight_mode": "denom2",
    "weight_clip_q": 0.99,
    "gate_valid_frac": 0.10,
}


def _read_tiingo_cache_dir() -> str:
    return os.environ.get("TIINGO_CACHE_DIR", ".af_cache/tiingo")


def _read_tiingo_cache_mode() -> str:
    return os.environ.get("TIINGO_CACHE_MODE", "use")


def _make_xgb_stes_model(
    *, seed: int | None, params_flat: dict | None = None
) -> XGBoostSTESModel:
    """Adapter from this script's flat param dict to tree_stes_model.XGBoostSTESModel."""
    flat = dict(DEFAULT_XGBOOST_PARAMS)
    if params_flat:
        flat.update(params_flat)

    num_boost_round = int(flat.pop("num_boost_round", 200))
    flat.pop("fit_mode", None)
    init_window = int(flat.pop("init_window", 500))
    flat.pop("block_size", None)
    n_alt_iters = int(flat.pop("n_alt_iters", 3))
    alt_clip_eps = float(flat.pop("alt_clip_eps", 1e-6))

    # --- new knobs ---
    alt_objective = flat.pop("alt_objective", "reg:pseudohubererror")
    residual_mode = flat.pop("residual_mode", True)
    monotonic_constraints = flat.pop("monotonic_constraints", {})
    denom_quantile = flat.pop("denom_quantile", 0.05)
    min_denom_floor = flat.pop("min_denom_floor", 1e-12)
    gate_target = flat.pop("gate_target", "logit_soft_map")
    eps_gate = float(flat.pop("eps_gate", 1e-3))
    alpha_prior_mu = float(flat.pop("alpha_prior_mu", 0.0))
    alpha_prior_tau = float(flat.pop("alpha_prior_tau", 1.5))
    sigma_eps = flat.pop("sigma_eps", None)
    sigma_eps = None if sigma_eps is None else float(sigma_eps)
    map_max_iter = int(flat.pop("map_max_iter", 50))
    map_tol = float(flat.pop("map_tol", 1e-8))
    map_init = str(flat.pop("map_init", "ratio"))
    map_eps_for_ratio_init = float(flat.pop("map_eps_for_ratio_init", 1e-6))
    weight_mode = flat.pop("weight_mode", "denom2")
    weight_clip_q = float(flat.pop("weight_clip_q", 0.99))
    gate_valid_frac = float(flat.pop("gate_valid_frac", 0.10))

    xgb_params: dict = {}
    for k, v in flat.items():
        if k == "learning_rate":
            xgb_params["eta"] = v
        elif k == "reg_lambda":
            xgb_params["lambda"] = v
        elif k == "reg_alpha":
            xgb_params["alpha"] = v
        else:
            xgb_params[k] = v

    return XGBoostSTESModel(
        xgb_params=xgb_params,
        num_boost_round=num_boost_round,
        init_window=init_window,
        n_alt_iters=n_alt_iters,
        alt_clip_eps=alt_clip_eps,
        random_state=seed,
        monotonic_constraints=monotonic_constraints,
        alt_objective=alt_objective,
        residual_mode=residual_mode,
        denom_quantile=denom_quantile,
        min_denom_floor=min_denom_floor,
        gate_target=gate_target,
        eps_gate=eps_gate,
        alpha_prior_mu=alpha_prior_mu,
        alpha_prior_tau=alpha_prior_tau,
        sigma_eps=sigma_eps,
        map_max_iter=map_max_iter,
        map_tol=map_tol,
        map_init=map_init,
        map_eps_for_ratio_init=map_eps_for_ratio_init,
        weight_mode=weight_mode,
        weight_clip_q=weight_clip_q,
        gate_valid_frac=gate_valid_frac,
    )


def _infer_monotone_constraints(cols: list[str]) -> dict[str, int]:
    """Heuristic: enforce +1 monotonicity on shock-magnitude style features."""
    out: dict[str, int] = {}
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["abs", "squared", "sq", "r2", "vol", "rv"]):
            out[c] = -1
    return out


def _xgb_variant_overrides(variant: str, cols: list[str]) -> dict:
    base = {
        "alt_objective": "reg:squarederror",
        "residual_mode": False,
        "monotonic_constraints": {},
    }

    # Robust inner-loop loss
    if variant in {
        "XGBSTES_BASE_HUBER",
        "XGBSTES_BASE_MONO_HUBER",
        "XGBSTES_BASE_HUBER_RESID",
        "XGBSTES_BASE_MONO_HUBER_RESID",
    }:
        base["alt_objective"] = "reg:pseudohubererror"

    # Residual / base-margin mode
    if variant in {
        "XGBSTES_BASE_RESID",
        "XGBSTES_BASE_MONO_RESID",
        "XGBSTES_BASE_HUBER_RESID",
        "XGBSTES_BASE_MONO_HUBER_RESID",
    }:
        base["residual_mode"] = True

    # Monotone constraints
    if variant in {
        "XGBSTES_BASE_MONO",
        "XGBSTES_BASE_MONO_HUBER",
        "XGBSTES_BASE_MONO_RESID",
        "XGBSTES_BASE_MONO_HUBER_RESID",
    }:
        base["monotonic_constraints"] = _infer_monotone_constraints(cols)

    return base


def _scale_train_test(
    X: pd.DataFrame, train_slice: slice, test_slice: slice
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features on the train slice and apply to test (skip const)."""
    X_tr = X.iloc[train_slice].copy()
    X_te = X.iloc[test_slice].copy()

    cols_to_scale = [c for c in X.columns if c != "const"]
    if not cols_to_scale:
        return X_tr, X_te

    scaler = StandardScaler().fit(X_tr[cols_to_scale])
    X_tr.loc[:, cols_to_scale] = pd.DataFrame(
        scaler.transform(X_tr[cols_to_scale]),
        index=X_tr.index,
        columns=cols_to_scale,
    )
    X_te.loc[:, cols_to_scale] = pd.DataFrame(
        scaler.transform(X_te[cols_to_scale]),
        index=X_te.index,
        columns=cols_to_scale,
    )

    return X_tr, X_te


def add_simulated_source(ctx, run_seed: int) -> None:
    """Attach a SimulatedGARCHSource (with per-run random_state) to the context."""
    ctx.sources[SOURCE] = SimulatedGARCHSource(
        n_periods=2500,
        random_state=run_seed,
        mu=0.0,
        omega=0.02,
        alpha=0.11,
        beta=0.87,
        eta=4.0,
        shock_prob=0.005,
        entity_id=ENTITY,
    )


def build_wide_spec(lags: int) -> VolDatasetSpec:
    """One spec that includes raw, abs, and squared lag features + target."""
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={
            "source": SOURCE,
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1.0,
        },
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[ENTITY]),
        time=TimeSpec(start=START, end=END, calendar="XNYS", grid="B", asof=None),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers for fixed-split and walk-forward
# ---------------------------------------------------------------------------


def _make_model_for_variant(
    variant: str,
    *,
    seed: int | None = None,
    xgb_params: dict | None = None,
):
    """Factory to build a model instance consistent with Part 1 / Part 2 variants."""
    if variant == "ES":
        return ESModel(random_state=seed) if seed is not None else ESModel()

    if variant.startswith("XGBSTES_"):
        return _make_xgb_stes_model(seed=seed, params_flat=xgb_params)

    # all other STES_* variants share the same STESModel class
    return STESModel(random_state=seed) if seed is not None else STESModel()


def _fit_predict_oos(
    *,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    train_slice: slice,
    test_slice: slice,
    seed: int | None = None,
    return_alpha: bool = False,
    return_alpha_star_diag: bool = False,
    xgb_params: dict | None = None,
) -> (
    tuple[pd.Index, np.ndarray, pd.Index, np.ndarray]
    | tuple[pd.Index, np.ndarray, np.ndarray, pd.Index, np.ndarray, np.ndarray]
):
    """Fit a variant on train_slice and return predictions on test_slice.

    Returns:
        (index, y_hat) or (index, y_hat, alpha) if return_alpha is True.
        Index is the time index of the test period.

    Notes:
        - We always pass returns + start/end indices to keep alignment consistent with STESModel.
        - Column selection is variant-specific via select_variant_columns.
    """
    # For all XGBSTES variants, always use XGBSTES_BASE columns
    if variant.startswith("XGBSTES_"):
        cols = select_variant_columns(X, "XGBSTES_BASE")
        if not cols:
            cols = list(X.columns)
    else:
        cols = select_variant_columns(X, variant)
        if not cols:
            cols = ["const"]

    X_sel = X[cols]

    X_tr, X_te = _scale_train_test(X_sel, train_slice, test_slice)
    y_tr, r_tr = y.iloc[train_slice], r.iloc[train_slice]
    r_te = r.iloc[test_slice]

    # Routing for XGBSTES variants
    if variant.startswith("XGBSTES_"):
        over = _xgb_variant_overrides(variant, cols)
        model = _make_xgb_stes_model(
            seed=seed,
            params_flat=(DEFAULT_XGBOOST_PARAMS | over | (xgb_params or {})),
        )
    else:
        model = _make_model_for_variant(variant, seed=seed, xgb_params=xgb_params)

    model.fit(X_tr, y_tr, returns=r_tr, start_index=0, end_index=len(X_tr))

    # IMPORTANT: warm-start OOS by predicting on concatenated [train + test]
    # and slicing the test tail. This prevents "cold start" of the variance state.
    n_tr = len(X_tr)
    X_all = pd.concat([X_tr, X_te], axis=0)
    r_all = pd.concat([r_tr, r_te], axis=0)
    train_idx = X_tr.index
    test_idx = X_te.index

    need_alpha = bool(return_alpha or return_alpha_star_diag)
    if need_alpha and hasattr(model, "predict_with_alpha"):
        y_hat_all_raw, alpha_all_raw = model.predict_with_alpha(X_all, returns=r_all)
        y_hat_all_v = np.asarray(y_hat_all_raw, dtype=float)
        alpha_all_v = np.asarray(alpha_all_raw, dtype=float)
        y_hat_all_s = pd.Series(y_hat_all_v, index=X_all.index, name="yhat")
        alpha_all_s = pd.Series(alpha_all_v, index=X_all.index, name="alpha")
    elif need_alpha and hasattr(model, "get_alphas"):
        # tree_stes_model.XGBoostSTESModel exposes alphas via get_alphas
        y_hat_raw = model.predict(
            X_all, returns=r_all, start_index=0, end_index=len(X_all)
        )
        alpha_raw = model.get_alphas(X_all, start_index=0, end_index=len(X_all))

        if isinstance(y_hat_raw, pd.Series):
            y_hat_all_s = y_hat_raw.reindex(X_all.index).astype(float)
        else:
            y_hat_all_s = pd.Series(
                np.asarray(y_hat_raw, dtype=float), index=X_all.index, name="yhat"
            )

        if isinstance(alpha_raw, pd.Series):
            alpha_all_s = alpha_raw.reindex(X_all.index).astype(float)
        else:
            alpha_all_s = pd.Series(
                np.asarray(alpha_raw, dtype=float), index=X_all.index, name="alpha"
            )

        y_hat_all_v = y_hat_all_s.to_numpy(dtype=float)
        alpha_all_v = alpha_all_s.to_numpy(dtype=float)
    else:
        y_hat_raw = model.predict(X_all, returns=r_all)
        if isinstance(y_hat_raw, pd.Series):
            y_hat_all_s = y_hat_raw.reindex(X_all.index).astype(float)
        else:
            y_hat_all_s = pd.Series(
                np.asarray(y_hat_raw, dtype=float), index=X_all.index, name="yhat"
            )
        y_hat_all_v = y_hat_all_s.to_numpy(dtype=float)
        alpha_all_v = np.full(len(X_all), np.nan, dtype=float)
        alpha_all_s = pd.Series(alpha_all_v, index=X_all.index, name="alpha")

    y_hat_te = y_hat_all_v[n_tr:]
    alpha_te = alpha_all_v[n_tr:]
    y_hat_tr = y_hat_all_v[:n_tr]
    alpha_tr = alpha_all_v[:n_tr]

    diag = None
    if return_alpha_star_diag:
        clip_eps = float(getattr(model, "alt_clip_eps", 1e-6) or 1e-6)
        denom_quantile = float(
            getattr(model, "denom_quantile", DEFAULT_XGBOOST_PARAMS["denom_quantile"])
        )
        min_denom_floor = float(
            getattr(model, "min_denom_floor", DEFAULT_XGBOOST_PARAMS["min_denom_floor"])
        )

        all_idx = X_all.index
        y_true_all = y.reindex(all_idx)
        r_all_s = r.reindex(all_idx)

        v_state_tr = y_hat_all_s.loc[train_idx].shift(1)
        denom_abs_tr = (r.loc[train_idx].astype(float) ** 2 - v_state_tr).abs()

        alpha_star_all_s, mask_all_s, thresh = _compute_alpha_star(
            all_idx,
            y_true_all,
            r_all_s,
            y_hat_all_s,
            clip_eps=clip_eps,
            denom_quantile=denom_quantile,
            min_denom_floor=min_denom_floor,
            quantile_from_train_denom_abs=denom_abs_tr,
        )

        alpha_pred_all_s = alpha_all_s.reindex(all_idx)

        tr_stats = _alpha_alignment_metrics(
            alpha_pred_all_s.loc[train_idx],
            alpha_star_all_s.loc[train_idx],
            mask_all_s.loc[train_idx],
            logit_eps=clip_eps,
        )
        te_stats = _alpha_alignment_metrics(
            alpha_pred_all_s.loc[test_idx],
            alpha_star_all_s.loc[test_idx],
            mask_all_s.loc[test_idx],
            logit_eps=clip_eps,
        )

        diag = {
            "alpha_star_denom_thresh": float(thresh),
            "train_corr_alpha_alpha_star": tr_stats["corr_alpha_alpha_star"],
            "train_rmse_logit_alpha": tr_stats["rmse_logit_alpha"],
            "train_mae_alpha": tr_stats["mae_alpha"],
            "train_alpha_std": tr_stats["alpha_std"],
            "train_alpha_star_std": tr_stats["alpha_star_std"],
            "train_frac_mask": tr_stats["frac_mask"],
            "train_n_mask": tr_stats["n_mask"],
            "test_corr_alpha_alpha_star": te_stats["corr_alpha_alpha_star"],
            "test_rmse_logit_alpha": te_stats["rmse_logit_alpha"],
            "test_mae_alpha": te_stats["mae_alpha"],
            "test_alpha_std": te_stats["alpha_std"],
            "test_alpha_star_std": te_stats["alpha_star_std"],
            "test_frac_mask": te_stats["frac_mask"],
            "test_n_mask": te_stats["n_mask"],
        }

    te_keep = np.isfinite(y_hat_te)
    te_idx = test_idx[te_keep]
    y_hat_te = y_hat_te[te_keep]
    alpha_te = alpha_te[te_keep]

    tr_keep = np.isfinite(y_hat_tr)
    tr_idx = train_idx[tr_keep]
    y_hat_tr = y_hat_tr[tr_keep]
    alpha_tr = alpha_tr[tr_keep]

    if return_alpha:
        out = (te_idx, y_hat_te, alpha_te, tr_idx, y_hat_tr, alpha_tr)
    else:
        out = (te_idx, y_hat_te, tr_idx, y_hat_tr)

    if return_alpha_star_diag:
        return (*out, diag)
    return out


def walk_forward_splits(
    n: int,
    *,
    mode: str = "expanding",
    train_size: int = 2000,
    val_size: int = 252,
    step_size: int = 252,
    max_folds: int | None = None,
) -> list[tuple[slice, slice]]:
    """Generate walk-forward splits.

    Args:
        n: total number of observations.
        mode: "expanding" or "rolling".
        train_size: initial training length (and rolling window length for mode="rolling").
        val_size: validation (test) block length.
        step_size: how far to advance the window between folds.
        max_folds: optional cap on number of folds.

    Returns:
        List of (train_slice, val_slice).

    Timeline:
        Fold k uses:
            train = [t0 : t1)
            val   = [t1 : t2)
        where t2 = t1 + val_size
    """
    if train_size <= 0 or val_size <= 0 or step_size <= 0:
        raise ValueError("train_size, val_size, step_size must be positive")

    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'")

    splits: list[tuple[slice, slice]] = []
    t1 = train_size
    folds = 0

    while True:
        t2 = t1 + val_size
        if t2 > n:
            break

        if mode == "expanding":
            tr0 = 0
            tr1 = t1
        else:
            tr0 = max(0, t1 - train_size)
            tr1 = t1

        train_sl = slice(tr0, tr1)
        val_sl = slice(t1, t2)
        splits.append((train_sl, val_sl))

        folds += 1
        if max_folds is not None and folds >= max_folds:
            break

        t1 += step_size

    return splits


def evaluate_variants_fixed_split(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    variants: list[str],
    is_index: int,
    os_index: int,
    seeds: list[int],
    sort_by: str = "rmse_mean",
) -> pd.DataFrame:
    """Fixed split evaluation, but applied uniformly to all variants."""

    def _mean_std(values: list[float]) -> tuple[float, float]:
        vals = [float(v) for v in values if np.isfinite(v)]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals))

    rows = []
    train_sl = slice(is_index, os_index)
    test_sl = slice(os_index, len(y))

    for variant in variants:
        rmses, maes, medaes = [], [], []
        rmses_tr, maes_tr, medaes_tr = [], [], []
        corr_tr, rmse_logit_tr, mae_alpha_tr = [], [], []
        alpha_std_tr, alpha_star_std_tr, frac_mask_tr = [], [], []
        corr_te, rmse_logit_te, mae_alpha_te = [], [], []
        alpha_std_te, alpha_star_std_te, frac_mask_te = [], [], []
        # Deterministic ES still runs fine with multiple seeds; keep loop uniform.
        for seed in seeds:
            try:
                if variant.startswith("XGBSTES_"):
                    res = _fit_predict_oos(
                        variant=variant,
                        X=X,
                        y=y,
                        r=r,
                        train_slice=train_sl,
                        test_slice=test_sl,
                        seed=seed,
                        return_alpha=True,
                        return_alpha_star_diag=True,
                    )
                    (
                        idx_te,
                        y_hat,
                        _alpha_te,
                        idx_tr,
                        y_hat_tr,
                        _alpha_tr,
                        diag,
                    ) = res

                    if isinstance(diag, dict):
                        corr_tr.append(diag.get("train_corr_alpha_alpha_star", np.nan))
                        rmse_logit_tr.append(diag.get("train_rmse_logit_alpha", np.nan))
                        mae_alpha_tr.append(diag.get("train_mae_alpha", np.nan))
                        alpha_std_tr.append(diag.get("train_alpha_std", np.nan))
                        alpha_star_std_tr.append(
                            diag.get("train_alpha_star_std", np.nan)
                        )
                        frac_mask_tr.append(diag.get("train_frac_mask", np.nan))

                        corr_te.append(diag.get("test_corr_alpha_alpha_star", np.nan))
                        rmse_logit_te.append(diag.get("test_rmse_logit_alpha", np.nan))
                        mae_alpha_te.append(diag.get("test_mae_alpha", np.nan))
                        alpha_std_te.append(diag.get("test_alpha_std", np.nan))
                        alpha_star_std_te.append(
                            diag.get("test_alpha_star_std", np.nan)
                        )
                        frac_mask_te.append(diag.get("test_frac_mask", np.nan))
                else:
                    res = _fit_predict_oos(
                        variant=variant,
                        X=X,
                        y=y,
                        r=r,
                        train_slice=train_sl,
                        test_slice=test_sl,
                        seed=seed,
                    )
                    idx_te, y_hat, idx_tr, y_hat_tr = res[0], res[1], res[2], res[3]

                y_true = y.loc[idx_te].values
                y_true_tr = y.loc[idx_tr].values
                rmses.append(rmse(y_true, y_hat))
                maes.append(mae(y_true, y_hat))
                medaes.append(medae(y_true, y_hat))
                rmses_tr.append(rmse(y_true_tr, y_hat_tr))
                maes_tr.append(mae(y_true_tr, y_hat_tr))
                medaes_tr.append(medae(y_true_tr, y_hat_tr))
            except Exception as e:
                logger.exception(
                    f"Fixed-split eval failed: variant={variant}, seed={seed}: {e}"
                )

        corr_tr_mean, corr_tr_std = _mean_std(corr_tr)
        rmse_logit_tr_mean, rmse_logit_tr_std = _mean_std(rmse_logit_tr)
        mae_alpha_tr_mean, mae_alpha_tr_std = _mean_std(mae_alpha_tr)
        alpha_std_tr_mean, alpha_std_tr_std = _mean_std(alpha_std_tr)
        alpha_star_std_tr_mean, alpha_star_std_tr_std = _mean_std(alpha_star_std_tr)
        frac_mask_tr_mean, frac_mask_tr_std = _mean_std(frac_mask_tr)

        corr_te_mean, corr_te_std = _mean_std(corr_te)
        rmse_logit_te_mean, rmse_logit_te_std = _mean_std(rmse_logit_te)
        mae_alpha_te_mean, mae_alpha_te_std = _mean_std(mae_alpha_te)
        alpha_std_te_mean, alpha_std_te_std = _mean_std(alpha_std_te)
        alpha_star_std_te_mean, alpha_star_std_te_std = _mean_std(alpha_star_std_te)
        frac_mask_te_mean, frac_mask_te_std = _mean_std(frac_mask_te)

        rows.append(
            {
                "variant": variant,
                "rmse_mean": float(np.mean(rmses)) if rmses else np.nan,
                "rmse_std": float(np.std(rmses)) if rmses else np.nan,
                "mae_mean": float(np.mean(maes)) if maes else np.nan,
                "mae_std": float(np.std(maes)) if maes else np.nan,
                "medae_mean": float(np.mean(medaes)) if medaes else np.nan,
                "medae_std": float(np.std(medaes)) if medaes else np.nan,
                "rmse_train_mean": float(np.mean(rmses_tr)) if rmses_tr else np.nan,
                "rmse_train_std": float(np.std(rmses_tr)) if rmses_tr else np.nan,
                "mae_train_mean": float(np.mean(maes_tr)) if maes_tr else np.nan,
                "mae_train_std": float(np.std(maes_tr)) if maes_tr else np.nan,
                "medae_train_mean": float(np.mean(medaes_tr)) if medaes_tr else np.nan,
                "medae_train_std": float(np.std(medaes_tr)) if medaes_tr else np.nan,
                "corr_alpha_alpha_star_train_mean": corr_tr_mean,
                "corr_alpha_alpha_star_train_std": corr_tr_std,
                "rmse_logit_alpha_train_mean": rmse_logit_tr_mean,
                "rmse_logit_alpha_train_std": rmse_logit_tr_std,
                "mae_alpha_train_mean": mae_alpha_tr_mean,
                "mae_alpha_train_std": mae_alpha_tr_std,
                "alpha_std_train_mean": alpha_std_tr_mean,
                "alpha_std_train_std": alpha_std_tr_std,
                "alpha_star_std_train_mean": alpha_star_std_tr_mean,
                "alpha_star_std_train_std": alpha_star_std_tr_std,
                "frac_mask_train_mean": frac_mask_tr_mean,
                "frac_mask_train_std": frac_mask_tr_std,
                "corr_alpha_alpha_star_test_mean": corr_te_mean,
                "corr_alpha_alpha_star_test_std": corr_te_std,
                "rmse_logit_alpha_test_mean": rmse_logit_te_mean,
                "rmse_logit_alpha_test_std": rmse_logit_te_std,
                "mae_alpha_test_mean": mae_alpha_te_mean,
                "mae_alpha_test_std": mae_alpha_te_std,
                "alpha_std_test_mean": alpha_std_te_mean,
                "alpha_std_test_std": alpha_std_te_std,
                "alpha_star_std_test_mean": alpha_star_std_te_mean,
                "alpha_star_std_test_std": alpha_star_std_te_std,
                "frac_mask_test_mean": frac_mask_te_mean,
                "frac_mask_test_std": frac_mask_te_std,
                "n": int(len(rmses)),
            }
        )

    return pd.DataFrame(rows).sort_values(sort_by)


def evaluate_variants_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    variants: list[str],
    splits: list[tuple[slice, slice]],
    seeds: list[int],
) -> pd.DataFrame:
    """Walk-forward evaluation applied uniformly to all variants.

    We aggregate across folds and seeds:
        rmse_mean = mean_{fold,seed} RMSE
        rmse_std  = std_{fold,seed} RMSE

    You can later extend this to also report per-fold win rates.
    """
    rows = []

    for variant in variants:
        rmses, maes, medaes = [], [], []
        for train_sl, val_sl in splits:
            for seed in seeds:
                try:
                    res = _fit_predict_oos(
                        variant=variant,
                        X=X,
                        y=y,
                        r=r,
                        train_slice=train_sl,
                        test_slice=val_sl,
                        seed=seed,
                    )
                    idx_te, y_hat, _, _ = res[0], res[1], res[2], res[3]
                    y_true = y.loc[idx_te].values
                    rmses.append(rmse(y_true, y_hat))
                    maes.append(mae(y_true, y_hat))
                    medaes.append(medae(y_true, y_hat))
                except Exception as e:
                    logger.exception(
                        f"Walk-forward eval failed: variant={variant}, seed={seed}, train={train_sl}, val={val_sl}: {e}"
                    )

        rows.append(
            {
                "variant": variant,
                "rmse_mean": float(np.mean(rmses)) if rmses else np.nan,
                "rmse_std": float(np.std(rmses)) if rmses else np.nan,
                "mae_mean": float(np.mean(maes)) if maes else np.nan,
                "mae_std": float(np.std(maes)) if maes else np.nan,
                "medae_mean": float(np.mean(medaes)) if medaes else np.nan,
                "medae_std": float(np.std(medaes)) if medaes else np.nan,
                "n": int(len(rmses)),
                "n_folds": int(len(splits)),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_mean")


def log_feature_importance(
    model: XGBoostSTESModel, feature_names: list[str], out_dir: Path
):
    """Log feature importance from XGBoost model."""
    if not isinstance(model, XGBoostSTESModel):
        logger.warning("Feature importance is only available for XGBoostSTESModel.")
        return

    booster = getattr(model, "model_", None) or getattr(model, "model", None)
    if booster is None:
        logger.warning("XGBoost model is not fitted; no feature importance available.")
        return

    out_dir.mkdir(exist_ok=True)
    importance_types = ["weight", "gain", "cover"]
    for importance_type in importance_types:
        try:
            scores = booster.get_score(importance_type=importance_type)
            if not scores:
                continue

            df = pd.DataFrame(
                {"feature": list(scores.keys()), "score": list(scores.values())}
            ).sort_values("score", ascending=False)

            # Map f0, f1, ... back to original feature names
            df["feature"] = df["feature"].apply(lambda f: feature_names[int(f[1:])])

            path = out_dir / f"xgb_feature_importance_{importance_type}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved {importance_type} feature importance to {path}")

        except Exception as e:
            logger.exception(
                f"Could not get feature importance for type {importance_type}: {e}"
            )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dump_xgb_alpha_star_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    train_slice: slice,
    test_slice: slice,
    seed: int,
    variant: str = "XGBSTES_BASE",
    xgb_params: dict | None = None,
    out_dir: Path = Path("outputs/volatility_forecast_2"),
    prefix: str = "spy_xgb_alpha_diag",
    make_plots: bool = True,
    return_df: bool = False,
) -> Path | tuple[Path, pd.DataFrame]:
    """Dump per-date OOS diagnostics for XGBSTES alpha gate vs implied alpha*.

    If return_df=True, returns (csv_path, df_test) where df_test is the OOS slice.
    """
    if not variant.startswith("XGBSTES_"):
        raise ValueError(
            f"dump_xgb_alpha_star_diagnostics expects XGBSTES_* variant, got: {variant}"
        )

    # Mirror _fit_predict_oos feature selection + scaling
    cols = select_variant_columns(X, "XGBSTES_BASE")
    if not cols:
        cols = list(X.columns)
    X_sel = X[cols]
    X_tr, X_te = _scale_train_test(X_sel, train_slice, test_slice)

    y_tr = y.iloc[train_slice]
    r_tr = r.iloc[train_slice]
    r_te = r.iloc[test_slice]

    over = _xgb_variant_overrides(variant, cols)
    model = _make_xgb_stes_model(
        seed=seed,
        params_flat=(DEFAULT_XGBOOST_PARAMS | over | (xgb_params or {})),
    )
    model.fit(X_tr, y_tr, returns=r_tr, start_index=0, end_index=len(X_tr))

    # Warm-start: predict on concatenated [train + test]
    X_all = pd.concat([X_tr, X_te], axis=0)
    r_all_fit = pd.concat([r_tr, r_te], axis=0)
    train_idx = X_tr.index
    test_idx = X_te.index
    all_idx = X_all.index

    if hasattr(model, "predict_with_alpha"):
        y_hat_all_raw, alpha_all_raw = model.predict_with_alpha(
            X_all, returns=r_all_fit
        )
        y_hat_all_s = pd.Series(
            np.asarray(y_hat_all_raw, dtype=float), index=all_idx, name="yhat"
        )
        alpha_all_s = pd.Series(
            np.asarray(alpha_all_raw, dtype=float), index=all_idx, name="alpha_pred"
        )
    elif hasattr(model, "get_alphas"):
        y_hat_raw = model.predict(
            X_all, returns=r_all_fit, start_index=0, end_index=len(X_all)
        )
        alpha_raw = model.get_alphas(X_all, start_index=0, end_index=len(X_all))
        if isinstance(y_hat_raw, pd.Series):
            y_hat_all_s = y_hat_raw.reindex(all_idx).astype(float)
        else:
            y_hat_all_s = pd.Series(
                np.asarray(y_hat_raw, dtype=float), index=all_idx, name="yhat"
            )
        if isinstance(alpha_raw, pd.Series):
            alpha_all_s = alpha_raw.reindex(all_idx).astype(float)
            alpha_all_s.name = "alpha_pred"
        else:
            alpha_all_s = pd.Series(
                np.asarray(alpha_raw, dtype=float), index=all_idx, name="alpha_pred"
            )
    else:
        y_hat_raw = model.predict(X_all, returns=r_all_fit)
        if isinstance(y_hat_raw, pd.Series):
            y_hat_all_s = y_hat_raw.reindex(all_idx).astype(float)
        else:
            y_hat_all_s = pd.Series(
                np.asarray(y_hat_raw, dtype=float), index=all_idx, name="yhat"
            )
        alpha_all_s = pd.Series(
            np.full(len(all_idx), np.nan, dtype=float),
            index=all_idx,
            name="alpha_pred",
        )

    # Alignment quantities (row-aligned to forecast date t)
    y_true_all = y.reindex(all_idx).astype(float)
    r_all = r.reindex(all_idx).astype(float)
    r2_all = r_all**2
    v_state_all = y_hat_all_s.shift(1)
    denom_all = r2_all - v_state_all
    numer_all = y_true_all - v_state_all
    abs_denom_all = denom_all.abs()

    clip_eps = float(getattr(model, "alt_clip_eps", 1e-6) or 1e-6)
    eps_gate = float(getattr(model, "eps_gate", 1e-3) or 1e-3)
    gate_target = str(
        getattr(model, "gate_target", "logit_soft_map") or "logit_soft_map"
    )
    weight_mode = str(getattr(model, "weight_mode", "denom2") or "denom2")
    weight_clip_q = float(getattr(model, "weight_clip_q", 0.99) or 0.99)
    gate_valid_frac = float(getattr(model, "gate_valid_frac", 0.10) or 0.10)
    denom_quantile = float(getattr(model, "denom_quantile", 0.05))
    min_denom_floor = float(getattr(model, "min_denom_floor", 1e-12))
    alpha_prior_mu = float(getattr(model, "alpha_prior_mu", 0.0) or 0.0)
    alpha_prior_tau = float(getattr(model, "alpha_prior_tau", 1.5) or 1.5)
    sigma_eps_used = getattr(model, "sigma_eps_", None)
    sigma_eps_used = float(sigma_eps_used) if sigma_eps_used is not None else np.nan
    map_max_iter = int(getattr(model, "map_max_iter", 50) or 50)
    map_tol = float(getattr(model, "map_tol", 1e-8) or 1e-8)
    map_init = str(getattr(model, "map_init", "ratio") or "ratio")
    map_eps_for_ratio_init = float(
        getattr(model, "map_eps_for_ratio_init", 1e-6) or 1e-6
    )

    v_state_tr = y_hat_all_s.loc[train_idx].shift(1)
    denom_abs_tr = (r.loc[train_idx].astype(float) ** 2 - v_state_tr).abs()

    # Denom stability threshold is calibrated on TRAIN only and reused on test.
    q_src = denom_abs_tr.to_numpy(dtype=float)
    q_src = q_src[np.isfinite(q_src)]
    q = (
        float(np.quantile(q_src, float(denom_quantile)))
        if q_src.size > 0
        else float("nan")
    )
    thresh = (
        max(float(min_denom_floor), float(q))
        if np.isfinite(q)
        else float(min_denom_floor)
    )

    denom_v = denom_all.to_numpy(dtype=float)
    numer_v = numer_all.to_numpy(dtype=float)
    denom_valid = np.isfinite(denom_v) & np.isfinite(numer_v) & (denom_v != 0.0)
    mask_all_v = denom_valid & (np.abs(denom_v) >= float(thresh))
    mask_all_s = pd.Series(mask_all_v, index=all_idx, name="mask_denom_ok")

    alpha_raw_v = np.full(len(all_idx), np.nan, dtype=float)
    alpha_raw_v[denom_valid] = numer_v[denom_valid] / denom_v[denom_valid]
    alpha_clip_v = np.clip(alpha_raw_v, 0.0, 1.0)
    alpha_star_all_s = pd.Series(alpha_clip_v, index=all_idx, name="alpha_star")

    # If the model did not surface sigma_eps_, estimate from TRAIN (masked) residuals.
    if (not np.isfinite(sigma_eps_used)) or sigma_eps_used <= 0.0:
        denom_tr = (r.loc[train_idx].astype(float) ** 2 - v_state_tr).astype(float)
        numer_tr = (y.loc[train_idx].astype(float) - v_state_tr).astype(float)
        alpha_pred_tr = alpha_all_s.loc[train_idx].astype(float)
        mask_tr = (
            np.isfinite(denom_tr.values)
            & np.isfinite(numer_tr.values)
            & (denom_tr.values != 0.0)
            & (np.abs(denom_tr.values) >= float(thresh))
            & np.isfinite(alpha_pred_tr.values)
        )
        resid_tr = (numer_tr.values - alpha_pred_tr.values * denom_tr.values)[mask_tr]
        resid_tr = resid_tr[np.isfinite(resid_tr)]
        if resid_tr.size > 0:
            med = float(np.median(resid_tr))
            mad = float(np.median(np.abs(resid_tr - med)))
            sigma_hat = 1.4826 * mad
        else:
            sigma_hat = float("nan")
        numer_abs = np.abs(numer_tr.values[mask_tr])
        numer_abs = numer_abs[np.isfinite(numer_abs)]
        floor_sigma = (
            1e-8 * float(np.median(numer_abs)) + 1e-12 if numer_abs.size > 0 else 1e-12
        )
        if (not np.isfinite(sigma_hat)) or sigma_hat <= 0.0:
            sigma_hat = floor_sigma
        sigma_eps_used = float(max(sigma_hat, floor_sigma))

    # Soft MAP alpha (computed only on denom-stable rows; NaN elsewhere)
    z_soft_v = np.full(len(all_idx), np.nan, dtype=float)
    alpha_soft_v = np.full(len(all_idx), np.nan, dtype=float)
    if int(mask_all_v.sum()) > 0:
        z_hat, a_hat = XGBoostSTESModel._soft_alpha_map(
            numer=numer_v[mask_all_v],
            denom=denom_v[mask_all_v],
            mu=alpha_prior_mu,
            tau=alpha_prior_tau,
            sigma_eps=float(sigma_eps_used),
            max_iter=map_max_iter,
            tol=map_tol,
            init=map_init,
            eps_ratio_init=map_eps_for_ratio_init,
        )
        z_soft_v[mask_all_v] = z_hat
        alpha_soft_v[mask_all_v] = a_hat
    alpha_soft_all_s = pd.Series(alpha_soft_v, index=all_idx, name="alpha_soft")
    z_soft_all_s = pd.Series(z_soft_v, index=all_idx, name="z_soft")

    df_all = pd.DataFrame(
        {
            "y": y_true_all,
            "r": r_all,
            "r2": r2_all,
            "yhat": y_hat_all_s,
            "v_state": v_state_all,
            "denom": denom_all,
            "numer": numer_all,
            "alpha_pred": alpha_all_s,
            "alpha_star": alpha_star_all_s,
            "mask_denom_ok": mask_all_s.astype(bool),
            "abs_denom": abs_denom_all,
        },
        index=all_idx,
    )

    df_all["alpha_soft"] = alpha_soft_all_s
    df_all["z_soft"] = z_soft_all_s
    df_all["logit_alpha_pred"] = _logit(df_all["alpha_pred"], clip_eps)
    df_all["logit_alpha_star"] = _logit(df_all["alpha_star"], clip_eps)
    df_all["logit_alpha_soft"] = _logit(df_all["alpha_soft"], clip_eps)

    # Primary mismatch is versus alpha_soft (target for gate_target="*_soft_map")
    df_all["alpha_err"] = df_all["alpha_pred"] - df_all["alpha_soft"]
    df_all["abs_alpha_err"] = df_all["alpha_err"].abs()
    df_all["alpha_err_clip"] = df_all["alpha_pred"] - df_all["alpha_star"]
    df_all["abs_alpha_err_clip"] = df_all["alpha_err_clip"].abs()
    df_all["loss"] = (df_all["y"] - df_all["yhat"]) ** 2
    df_all["thresh_abs_denom"] = float(thresh)
    df_all["D_denom_small"] = df_all["abs_denom"] < float(thresh)
    df_all["variant"] = str(variant)
    df_all["seed"] = int(seed)
    df_all["gate_target"] = gate_target
    df_all["eps_gate"] = float(eps_gate)
    df_all["alpha_prior_mu"] = float(alpha_prior_mu)
    df_all["alpha_prior_tau"] = float(alpha_prior_tau)
    df_all["sigma_eps"] = float(sigma_eps_used)
    df_all["map_max_iter"] = int(map_max_iter)
    df_all["map_tol"] = float(map_tol)
    df_all["map_init"] = str(map_init)
    df_all["map_eps_for_ratio_init"] = float(map_eps_for_ratio_init)
    df_all["weight_mode"] = weight_mode
    df_all["weight_clip_q"] = float(weight_clip_q)
    df_all["gate_valid_frac"] = float(gate_valid_frac)

    # Gate-learning space diagnostics: compare inferred z_pred vs z_soft (MAP)
    logit_eps = float(max(1e-12, clip_eps))
    df_all["z_pred"] = _logit(df_all["alpha_pred"], logit_eps)
    df_all["z_star"] = df_all["z_soft"]
    df_all["z_err"] = df_all["z_pred"] - df_all["z_soft"]
    df_all["abs_z_err"] = df_all["z_err"].abs()

    df_test = df_all.loc[test_idx]

    _ensure_dir(out_dir)
    out_path = out_dir / f"{prefix}__{variant}__seed{seed}.csv"
    df_test.to_csv(out_path, index=True)

    if make_plots:
        try:
            # 1) alpha_pred vs alpha_soft over time (also show alpha_clip)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(
                df_test.index,
                df_test["alpha_pred"].values,
                lw=1.0,
                label="alpha_pred",
            )
            ax.plot(
                df_test.index,
                df_test["alpha_soft"].values,
                lw=1.0,
                label="alpha_soft",
            )
            ax.plot(
                df_test.index,
                df_test["alpha_star"].values,
                lw=1.0,
                label="alpha_clip",
                alpha=0.8,
            )
            ax.set_title(f"{variant} seed={seed}: alpha_pred vs alpha_soft (OOS)")
            ax.set_xlabel("date")
            ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{prefix}__{variant}__seed{seed}__alpha_timeseries.png",
                dpi=150,
            )
            plt.close(fig)

            # 2) scatter alpha_pred vs alpha_soft on masked points
            sub = df_test[
                df_test["mask_denom_ok"]
                & np.isfinite(df_test["alpha_pred"].values)
                & np.isfinite(df_test["alpha_soft"].values)
            ]
            if len(sub) > 0:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(
                    sub["alpha_soft"].values,
                    sub["alpha_pred"].values,
                    s=8,
                    alpha=0.5,
                )
                ax.plot([0, 1], [0, 1], lw=1.0, alpha=0.8, color="black")
                ax.set_title(
                    f"{variant} seed={seed}: alpha_pred vs alpha_soft (masked)"
                )
                ax.set_xlabel("alpha_soft")
                ax.set_ylabel("alpha_pred")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(
                    out_dir / f"{prefix}__{variant}__seed{seed}__alpha_scatter.png",
                    dpi=150,
                )
                plt.close(fig)

            # 3) histogram of |denom| with threshold
            abs_den = df_test["abs_denom"].to_numpy(dtype=float)
            abs_den = abs_den[np.isfinite(abs_den)]
            if abs_den.size > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(abs_den, bins=60, alpha=0.9)
                ax.axvline(
                    float(thresh), lw=2.0, alpha=0.9, color="red", label="thresh"
                )
                ax.set_title(f"{variant} seed={seed}: |denom| histogram (OOS)")
                ax.set_xlabel("|r^2 - v_state|")
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(
                    out_dir / f"{prefix}__{variant}__seed{seed}__abs_denom_hist.png",
                    dpi=150,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Alpha-star diagnostic plots failed: {e}")

    if return_df:
        # Attach train-segment stats for report generation (not persisted to CSV).
        df_train = df_all.loc[train_idx]

        m_train = df_train.get("mask_denom_ok", False).astype(bool) & np.isfinite(
            df_train.get("alpha_soft", np.nan).astype(float).values
        )
        alpha_soft_mean_train = (
            float(df_train.loc[m_train, "alpha_soft"].mean())
            if int(m_train.sum()) > 0
            else np.nan
        )
        alpha_soft_std_train = (
            float(df_train.loc[m_train, "alpha_soft"].std(ddof=0))
            if int(m_train.sum()) > 0
            else np.nan
        )

        def _boundary_stats(df_seg: pd.DataFrame) -> dict:
            mask_s = (
                df_seg["mask_denom_ok"].astype(bool)
                if "mask_denom_ok" in df_seg.columns
                else pd.Series(False, index=df_seg.index)
            )
            alpha_pred_s = (
                df_seg["alpha_pred"].astype(float)
                if "alpha_pred" in df_seg.columns
                else pd.Series(np.nan, index=df_seg.index)
            )
            alpha_star_s = (
                df_seg["alpha_star"].astype(float)
                if "alpha_star" in df_seg.columns
                else pd.Series(np.nan, index=df_seg.index)
            )
            alpha_soft_s = (
                df_seg["alpha_soft"].astype(float)
                if "alpha_soft" in df_seg.columns
                else pd.Series(np.nan, index=df_seg.index)
            )

            m = (
                mask_s
                & np.isfinite(alpha_pred_s.values)
                & np.isfinite(alpha_star_s.values)
                & np.isfinite(alpha_soft_s.values)
            )
            n_seg = int(len(df_seg))
            n_m = int(m.sum())
            frac_m = float(mask_s.mean()) if n_seg > 0 else np.nan
            if n_m <= 0:
                return {
                    "n": n_seg,
                    "n_mask": n_m,
                    "frac_mask": frac_m,
                    "pct_alpha_star_clip_le_0_01": np.nan,
                    "pct_alpha_star_clip_ge_0_99": np.nan,
                    "pct_alpha_soft_le_0_01": np.nan,
                    "pct_alpha_soft_ge_0_99": np.nan,
                    "pct_alpha_pred_le_0_01": np.nan,
                    "pct_alpha_pred_ge_0_99": np.nan,
                }
            a_star = alpha_star_s.loc[m].to_numpy(dtype=float)
            a_soft = alpha_soft_s.loc[m].to_numpy(dtype=float)
            a_pred = alpha_pred_s.loc[m].to_numpy(dtype=float)
            return {
                "n": n_seg,
                "n_mask": n_m,
                "frac_mask": frac_m,
                "pct_alpha_star_clip_le_0_01": float(np.mean(a_star <= 0.01)),
                "pct_alpha_star_clip_ge_0_99": float(np.mean(a_star >= 0.99)),
                "pct_alpha_soft_le_0_01": float(np.mean(a_soft <= 0.01)),
                "pct_alpha_soft_ge_0_99": float(np.mean(a_soft >= 0.99)),
                "pct_alpha_pred_le_0_01": float(np.mean(a_pred <= 0.01)),
                "pct_alpha_pred_ge_0_99": float(np.mean(a_pred >= 0.99)),
            }

        df_test.attrs["train_boundary_stats"] = _boundary_stats(df_train)
        df_test.attrs["test_boundary_stats"] = _boundary_stats(df_test)
        df_test.attrs["gate_params"] = {
            "gate_target": gate_target,
            "eps_gate": float(eps_gate),
            "alpha_prior_mu": float(alpha_prior_mu),
            "alpha_prior_tau": float(alpha_prior_tau),
            "sigma_eps": float(sigma_eps_used),
            "map_max_iter": int(map_max_iter),
            "map_tol": float(map_tol),
            "map_init": str(map_init),
            "map_eps_for_ratio_init": float(map_eps_for_ratio_init),
            "weight_mode": weight_mode,
            "weight_clip_q": float(weight_clip_q),
            "gate_valid_frac": float(gate_valid_frac),
            "alt_clip_eps": float(clip_eps),
            "denom_quantile": float(denom_quantile),
            "min_denom_floor": float(min_denom_floor),
            "thresh_abs_denom": float(thresh),
            "alpha_soft_mean_train_masked": float(alpha_soft_mean_train),
            "alpha_soft_std_train_masked": float(alpha_soft_std_train),
        }
        return out_path, df_test
    return out_path


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
    if x.size < 2 or y.size < 2:
        return np.nan
    if float(np.std(x, ddof=0)) <= 0.0 or float(np.std(y, ddof=0)) <= 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation via rank transform + Pearson (no scipy dependency)."""
    if x.size < 2 or y.size < 2:
        return np.nan
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or y.size < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _safe_corr(xr, yr)


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() <= 0:
        return {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    err = y_true[m] - y_pred[m]
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "n": int(m.sum()),
    }


def _alpha_star_report_scalars(df_test: pd.DataFrame) -> dict:
    """Compute core scalar diagnostics used by the alpha* report.

    This is intentionally lightweight so it can be used for multi-seed summaries.
    """
    df = df_test.copy()
    mask = df.get("mask_denom_ok", False)
    if isinstance(mask, (pd.Series, np.ndarray, list)):
        mask = pd.Series(mask, index=df.index).astype(bool)
    else:
        mask = pd.Series(False, index=df.index)

    req = [
        "y",
        "yhat",
        "v_state",
        "denom",
        "alpha_pred",
        "alpha_soft",
        "alpha_star",  # hard-clipped ratio (alpha_clip)
        "abs_denom",
    ]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    n_test = int(len(df))
    frac_mask = float(mask.mean()) if n_test > 0 else np.nan

    finite_mask = (
        mask
        & np.isfinite(df["y"].values)
        & np.isfinite(df["yhat"].values)
        & np.isfinite(df["v_state"].values)
        & np.isfinite(df["denom"].values)
        & np.isfinite(df["alpha_pred"].values)
        & np.isfinite(df["alpha_soft"].values)
    )
    n_mask = int(finite_mask.sum())

    alpha_pred = df.loc[finite_mask, "alpha_pred"].to_numpy(dtype=float)
    alpha_soft = df.loc[finite_mask, "alpha_soft"].to_numpy(dtype=float)

    corr_pred_soft = _safe_corr(alpha_pred, alpha_soft)
    spearman_pred_soft = _spearman_corr(alpha_pred, alpha_soft)
    mae_pred_soft = (
        float(np.mean(np.abs(alpha_pred - alpha_soft))) if n_mask > 0 else np.nan
    )

    finite_mask_clip = finite_mask & np.isfinite(df["alpha_star"].values)
    alpha_clip = df.loc[finite_mask_clip, "alpha_star"].to_numpy(dtype=float)
    alpha_pred_clip = df.loc[finite_mask_clip, "alpha_pred"].to_numpy(dtype=float)
    corr_pred_clip = _safe_corr(alpha_pred_clip, alpha_clip)
    mae_pred_clip = (
        float(np.mean(np.abs(alpha_pred_clip - alpha_clip)))
        if int(finite_mask_clip.sum()) > 0
        else np.nan
    )

    # z-space alignment (preferred when gate_target is a logit target)
    z_pred = None
    z_soft = None
    if "z_pred" in df.columns:
        z_pred = df.loc[finite_mask, "z_pred"].to_numpy(dtype=float)
    else:
        z_pred = _logit(pd.Series(alpha_pred), 1e-12).to_numpy(dtype=float)

    if "z_soft" in df.columns:
        z_soft = df.loc[finite_mask, "z_soft"].to_numpy(dtype=float)
    elif "z_star" in df.columns:
        z_soft = df.loc[finite_mask, "z_star"].to_numpy(dtype=float)
    else:
        z_soft = _logit(pd.Series(alpha_soft), 1e-12).to_numpy(dtype=float)

    rmse_z = float(np.sqrt(np.mean((z_pred - z_soft) ** 2))) if n_mask > 0 else np.nan
    corr_z = _safe_corr(z_pred, z_soft) if n_mask > 0 else np.nan

    pct_alpha_star_le_0_01 = (
        float(np.mean(alpha_clip <= 0.01)) if alpha_clip.size > 0 else np.nan
    )
    pct_alpha_star_ge_0_99 = (
        float(np.mean(alpha_clip >= 0.99)) if alpha_clip.size > 0 else np.nan
    )
    pct_alpha_soft_le_0_01 = (
        float(np.mean(alpha_soft <= 0.01)) if n_mask > 0 else np.nan
    )
    pct_alpha_soft_ge_0_99 = (
        float(np.mean(alpha_soft >= 0.99)) if n_mask > 0 else np.nan
    )
    pct_alpha_pred_le_0_01 = (
        float(np.mean(alpha_pred <= 0.01)) if n_mask > 0 else np.nan
    )
    pct_alpha_pred_ge_0_99 = (
        float(np.mean(alpha_pred >= 0.99)) if n_mask > 0 else np.nan
    )

    # Gate reconstructions / oracle
    v_state = df["v_state"].to_numpy(dtype=float)
    denom = df["denom"].to_numpy(dtype=float)
    y_true = df["y"].to_numpy(dtype=float)
    yhat_model = df["yhat"].to_numpy(dtype=float)
    alpha_pred_all = df["alpha_pred"].to_numpy(dtype=float)
    alpha_soft_all = df["alpha_soft"].to_numpy(dtype=float)
    alpha_star_all = df["alpha_star"].to_numpy(dtype=float)

    yhat_gate_pred = v_state + alpha_pred_all * denom
    alpha_bar = float(np.mean(alpha_soft)) if n_mask > 0 else np.nan
    yhat_gate_const = (
        v_state + alpha_bar * denom
        if np.isfinite(alpha_bar)
        else np.full_like(yhat_gate_pred, np.nan)
    )
    yhat_gate_soft = np.full_like(yhat_gate_pred, np.nan)
    yhat_gate_soft[finite_mask.values] = (
        v_state[finite_mask.values]
        + alpha_soft_all[finite_mask.values] * denom[finite_mask.values]
    )
    yhat_gate_clip = np.full_like(yhat_gate_pred, np.nan)
    yhat_gate_clip[finite_mask_clip.values] = (
        v_state[finite_mask_clip.values]
        + alpha_star_all[finite_mask_clip.values] * denom[finite_mask_clip.values]
    )

    m_all = _forecast_metrics(y_true, yhat_model)
    r_all = _forecast_metrics(y_true, yhat_gate_pred)
    c_all = _forecast_metrics(y_true, yhat_gate_const)

    m_mask = (
        _forecast_metrics(y_true[finite_mask.values], yhat_model[finite_mask.values])
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    r_mask = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_pred[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    c_mask = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_const[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    o_soft = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_soft[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    o_clip = (
        _forecast_metrics(
            y_true[finite_mask_clip.values], yhat_gate_clip[finite_mask_clip.values]
        )
        if int(finite_mask_clip.sum()) > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )

    oracle_gain_soft = (
        float(r_mask["rmse"] - o_soft["rmse"])
        if np.isfinite(r_mask["rmse"]) and np.isfinite(o_soft["rmse"])
        else np.nan
    )
    oracle_gain_clip = (
        float(r_mask["rmse"] - o_clip["rmse"])
        if np.isfinite(r_mask["rmse"]) and np.isfinite(o_clip["rmse"])
        else np.nan
    )
    const_gain = (
        float(r_mask["rmse"] - c_mask["rmse"])
        if np.isfinite(r_mask["rmse"]) and np.isfinite(c_mask["rmse"])
        else np.nan
    )

    recon_diff = (
        yhat_model[finite_mask.values] - yhat_gate_pred[finite_mask.values]
        if n_mask > 0
        else np.array([], dtype=float)
    )
    recon_max_abs = float(np.max(np.abs(recon_diff))) if recon_diff.size > 0 else np.nan
    recon_corr = (
        _safe_corr(yhat_model[finite_mask.values], yhat_gate_pred[finite_mask.values])
        if n_mask > 0
        else np.nan
    )

    # Forecast vs alpha mismatch relationship
    var_err = (y_true - yhat_model).astype(float)
    abs_err = np.abs(var_err)
    alpha_err = (alpha_pred_all - alpha_soft_all).astype(float)
    abs_alpha_err = np.abs(alpha_err)
    err_corr = (
        _safe_corr(abs_err[finite_mask.values], abs_alpha_err[finite_mask.values])
        if n_mask > 0
        else np.nan
    )

    return {
        "n_test": n_test,
        "n_mask": n_mask,
        "frac_mask": frac_mask,
        "corr_alpha_pred_alpha_soft": corr_pred_soft,
        "spearman_alpha_pred_alpha_soft": spearman_pred_soft,
        "corr_alpha_pred_alpha_clip": corr_pred_clip,
        "mae_alpha_soft": mae_pred_soft,
        "mae_alpha_clip": mae_pred_clip,
        "corr_z_pred_z_soft": corr_z,
        "rmse_z_pred_z_soft": rmse_z,
        "pct_alpha_star_clip_le_0_01": pct_alpha_star_le_0_01,
        "pct_alpha_star_clip_ge_0_99": pct_alpha_star_ge_0_99,
        "pct_alpha_soft_le_0_01": pct_alpha_soft_le_0_01,
        "pct_alpha_soft_ge_0_99": pct_alpha_soft_ge_0_99,
        "pct_alpha_pred_le_0_01": pct_alpha_pred_le_0_01,
        "pct_alpha_pred_ge_0_99": pct_alpha_pred_ge_0_99,
        "rmse_model": m_all["rmse"],
        "rmse_recon_pred": r_all["rmse"],
        "rmse_oracle_soft_masked": o_soft["rmse"],
        "rmse_oracle_clip_masked": o_clip["rmse"],
        "rmse_const_alpha_masked": c_mask["rmse"],
        "oracle_gain_soft_masked": oracle_gain_soft,
        "oracle_gain_clip_masked": oracle_gain_clip,
        "const_gain_masked": const_gain,
        "recon_max_abs_diff": recon_max_abs,
        "recon_corr": recon_corr,
        "corr_abs_err_abs_alpha_err_soft": err_corr,
    }


def _write_alpha_star_report_legacy(
    df_test: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    out_dir: Path,
    prefix: str,
) -> tuple[Path, list[Path]]:
    """
    Given per-date test diagnostics df_test (already aligned), write:
      - Markdown summary table(s)
      - A small set of charts
    Returns: (md_path, list_of_chart_paths)
    """
    _ensure_dir(out_dir)

    gate_params = dict(getattr(df_test, "attrs", {}).get("gate_params", {}) or {})
    train_boundary = dict(
        getattr(df_test, "attrs", {}).get("train_boundary_stats", {}) or {}
    )

    df = df_test.copy()
    if "mask_denom_ok" not in df.columns:
        df["mask_denom_ok"] = False
    df["mask_denom_ok"] = df["mask_denom_ok"].astype(bool)

    gate_target = str(
        gate_params.get(
            "gate_target",
            (
                str(df["gate_target"].iloc[0])
                if "gate_target" in df.columns and len(df) > 0
                else "logit_soft_map"
            ),
        )
    )
    weight_mode = str(
        gate_params.get(
            "weight_mode",
            (
                str(df["weight_mode"].iloc[0])
                if "weight_mode" in df.columns and len(df) > 0
                else "denom2"
            ),
        )
    )

    required = [
        "y",
        "r",
        "r2",
        "yhat",
        "v_state",
        "denom",
        "numer",
        "alpha_pred",
        "alpha_star",
        "alpha_soft",
        "abs_denom",
        "logit_alpha_pred",
        "logit_alpha_star",
        "logit_alpha_soft",
        "z_pred",
        "z_star",
        "z_soft",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    if "z_pred" not in df_test.columns or "z_star" not in df_test.columns:
        # Backward-compat when df_test comes from disk (older CSV without z columns).
        a_pred = df["alpha_pred"].to_numpy(dtype=float)
        a_star = np.clip(df["alpha_star"].to_numpy(dtype=float), 0.0, 1.0)

        a_star_soft = eps_gate + (1.0 - 2.0 * eps_gate) * a_star
        df["z_pred"] = _logit(pd.Series(a_pred, index=df.index), eps_gate)
        df["z_star"] = _logit(pd.Series(a_star_soft, index=df.index), eps_gate)

    mask = df["mask_denom_ok"].astype(bool)
    finite_mask = (
        mask
        & np.isfinite(df["y"].values)
        & np.isfinite(df["yhat"].values)
        & np.isfinite(df["alpha_pred"].values)
        & np.isfinite(df["alpha_star"].values)
        & np.isfinite(df["abs_denom"].values)
    )

    n_test = int(len(df))
    n_mask = int(finite_mask.sum())
    frac_mask = float(mask.mean()) if n_test > 0 else np.nan

    alpha_pred_m = df.loc[finite_mask, "alpha_pred"].to_numpy(dtype=float)
    alpha_star_m = df.loc[finite_mask, "alpha_star"].to_numpy(dtype=float)

    alpha_pred_mean = float(np.mean(alpha_pred_m)) if n_mask > 0 else np.nan
    alpha_pred_std = float(np.std(alpha_pred_m, ddof=0)) if n_mask > 0 else np.nan
    alpha_star_mean = float(np.mean(alpha_star_m)) if n_mask > 0 else np.nan
    alpha_star_std = float(np.std(alpha_star_m, ddof=0)) if n_mask > 0 else np.nan
    corr_alpha = _safe_corr(alpha_pred_m, alpha_star_m) if n_mask > 0 else np.nan
    spearman_alpha = (
        _spearman_corr(alpha_pred_m, alpha_star_m) if n_mask > 0 else np.nan
    )
    mae_alpha = (
        float(np.mean(np.abs(alpha_pred_m - alpha_star_m))) if n_mask > 0 else np.nan
    )

    logit_pred_m = df.loc[finite_mask, "logit_alpha_pred"].to_numpy(dtype=float)
    logit_star_m = df.loc[finite_mask, "logit_alpha_star"].to_numpy(dtype=float)
    rmse_logit = (
        float(np.sqrt(np.mean((logit_pred_m - logit_star_m) ** 2)))
        if n_mask > 0
        else np.nan
    )

    z_pred_m = df.loc[finite_mask, "z_pred"].to_numpy(dtype=float)
    z_star_m = df.loc[finite_mask, "z_star"].to_numpy(dtype=float)
    rmse_z = (
        float(np.sqrt(np.mean((z_pred_m - z_star_m) ** 2))) if n_mask > 0 else np.nan
    )

    pct_alpha_star_le_0_01 = (
        float(np.mean(alpha_star_m <= 0.01)) if n_mask > 0 else np.nan
    )
    pct_alpha_star_ge_0_99 = (
        float(np.mean(alpha_star_m >= 0.99)) if n_mask > 0 else np.nan
    )
    pct_alpha_pred_le_0_01 = (
        float(np.mean(alpha_pred_m <= 0.01)) if n_mask > 0 else np.nan
    )
    pct_alpha_pred_ge_0_99 = (
        float(np.mean(alpha_pred_m >= 0.99)) if n_mask > 0 else np.nan
    )

    abs_denom_m = df.loc[finite_mask, "abs_denom"].to_numpy(dtype=float)
    abs_denom_p50 = float(np.quantile(abs_denom_m, 0.50)) if n_mask > 0 else np.nan
    abs_denom_p10 = float(np.quantile(abs_denom_m, 0.10)) if n_mask > 0 else np.nan
    abs_denom_p01 = float(np.quantile(abs_denom_m, 0.01)) if n_mask > 0 else np.nan

    thresh = (
        float(df["thresh_abs_denom"].iloc[0])
        if "thresh_abs_denom" in df.columns and len(df) > 0
        else np.nan
    )

    # A1: alignment summary table
    a1 = pd.DataFrame(
        {
            "metric": [
                "gate_target",
                "eps_gate",
                "weight_mode",
                "n_test",
                "n_mask",
                "frac_mask",
                "alpha_pred_mean",
                "alpha_pred_std",
                "alpha_star_mean",
                "alpha_star_std",
                "corr(alpha_pred, alpha_star)",
                "spearman(alpha_pred, alpha_star)",
                "mae(alpha_pred-alpha_star)",
                "rmse(logit(alpha_pred)-logit(alpha_star))",
                "rmse(z_pred-z_star) [logit, eps_gate]",
                "pct_alpha_star<=0.01 (masked)",
                "pct_alpha_star>=0.99 (masked)",
                "pct_alpha_pred<=0.01 (masked)",
                "pct_alpha_pred>=0.99 (masked)",
                "abs_denom_p50",
                "abs_denom_p10",
                "abs_denom_p01",
                "thresh_abs_denom",
            ],
            "value": [
                str(gate_target),
                f"{eps_gate:.4g}",
                str(weight_mode),
                str(n_test),
                str(n_mask),
                f"{frac_mask:.3f}" if np.isfinite(frac_mask) else "NA",
                f"{alpha_pred_mean:.4f}" if np.isfinite(alpha_pred_mean) else "NA",
                f"{alpha_pred_std:.4f}" if np.isfinite(alpha_pred_std) else "NA",
                f"{alpha_star_mean:.4f}" if np.isfinite(alpha_star_mean) else "NA",
                f"{alpha_star_std:.4f}" if np.isfinite(alpha_star_std) else "NA",
                f"{corr_alpha:.3f}" if np.isfinite(corr_alpha) else "NA",
                f"{spearman_alpha:.3f}" if np.isfinite(spearman_alpha) else "NA",
                f"{mae_alpha:.4f}" if np.isfinite(mae_alpha) else "NA",
                f"{rmse_logit:.3f}" if np.isfinite(rmse_logit) else "NA",
                f"{rmse_z:.3f}" if np.isfinite(rmse_z) else "NA",
                (
                    f"{100.0 * pct_alpha_star_le_0_01:.1f}%"
                    if np.isfinite(pct_alpha_star_le_0_01)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_alpha_star_ge_0_99:.1f}%"
                    if np.isfinite(pct_alpha_star_ge_0_99)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_alpha_pred_le_0_01:.1f}%"
                    if np.isfinite(pct_alpha_pred_le_0_01)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_alpha_pred_ge_0_99:.1f}%"
                    if np.isfinite(pct_alpha_pred_ge_0_99)
                    else "NA"
                ),
                _format_sci(abs_denom_p50),
                _format_sci(abs_denom_p10),
                _format_sci(abs_denom_p01),
                _format_sci(thresh),
            ],
        }
    )

    boundary_rows = []
    if train_boundary:
        boundary_rows.append(
            {
                "segment": "train",
                "n": str(train_boundary.get("n", "NA")),
                "n_mask": str(train_boundary.get("n_mask", "NA")),
                "frac_mask": (
                    f"{float(train_boundary.get('frac_mask')):.3f}"
                    if np.isfinite(train_boundary.get("frac_mask", np.nan))
                    else "NA"
                ),
                "pct_alpha_star<=0.01": (
                    f"{100.0 * float(train_boundary.get('pct_alpha_star_le_0_01')):.1f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_star_le_0_01", np.nan))
                    else "NA"
                ),
                "pct_alpha_star>=0.99": (
                    f"{100.0 * float(train_boundary.get('pct_alpha_star_ge_0_99')):.1f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_star_ge_0_99", np.nan))
                    else "NA"
                ),
                "pct_alpha_pred<=0.01": (
                    f"{100.0 * float(train_boundary.get('pct_alpha_pred_le_0_01')):.1f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_pred_le_0_01", np.nan))
                    else "NA"
                ),
                "pct_alpha_pred>=0.99": (
                    f"{100.0 * float(train_boundary.get('pct_alpha_pred_ge_0_99')):.1f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_pred_ge_0_99", np.nan))
                    else "NA"
                ),
            }
        )
    boundary_rows.append(
        {
            "segment": "test",
            "n": str(n_test),
            "n_mask": str(n_mask),
            "frac_mask": f"{frac_mask:.3f}" if np.isfinite(frac_mask) else "NA",
            "pct_alpha_star<=0.01": (
                f"{100.0 * pct_alpha_star_le_0_01:.1f}%"
                if np.isfinite(pct_alpha_star_le_0_01)
                else "NA"
            ),
            "pct_alpha_star>=0.99": (
                f"{100.0 * pct_alpha_star_ge_0_99:.1f}%"
                if np.isfinite(pct_alpha_star_ge_0_99)
                else "NA"
            ),
            "pct_alpha_pred<=0.01": (
                f"{100.0 * pct_alpha_pred_le_0_01:.1f}%"
                if np.isfinite(pct_alpha_pred_le_0_01)
                else "NA"
            ),
            "pct_alpha_pred>=0.99": (
                f"{100.0 * pct_alpha_pred_ge_0_99:.1f}%"
                if np.isfinite(pct_alpha_pred_ge_0_99)
                else "NA"
            ),
        }
    )
    boundary_tbl = pd.DataFrame(boundary_rows)

    calib_tbl = None
    if n_mask > 0:
        tmp = df.loc[finite_mask, ["alpha_star", "alpha_pred"]].copy()
        try:
            tmp["bin"] = pd.qcut(tmp["alpha_star"], q=10, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)
            rows = []
            for b, g in grp:
                rows.append(
                    {
                        "alpha_star_bin": str(b),
                        "count": int(len(g)),
                        "alpha_star_mean": float(g["alpha_star"].mean()),
                        "alpha_pred_mean": float(g["alpha_pred"].mean()),
                    }
                )
            calib_tbl = pd.DataFrame(rows)
        except Exception:
            calib_tbl = None

    # A2: forecast error vs alpha mismatch
    var_err = (df["y"] - df["yhat"]).astype(float)
    abs_err = var_err.abs()
    alpha_err = (df["alpha_pred"] - df["alpha_star"]).astype(float)
    abs_alpha_err = alpha_err.abs()

    corr_abs_err_abs_alpha = (
        _safe_corr(
            abs_err.loc[finite_mask].to_numpy(dtype=float),
            abs_alpha_err.loc[finite_mask].to_numpy(dtype=float),
        )
        if n_mask > 0
        else np.nan
    )
    corr_signed_err_signed_alpha = (
        _safe_corr(
            var_err.loc[finite_mask].to_numpy(dtype=float),
            alpha_err.loc[finite_mask].to_numpy(dtype=float),
        )
        if n_mask > 0
        else np.nan
    )
    corr_abs_err_abs_alpha2 = corr_abs_err_abs_alpha

    a2_corr = pd.DataFrame(
        {
            "metric": [
                "corr(|y-yhat|, |alpha_pred-alpha_star|)",
                "corr(y-yhat, alpha_pred-alpha_star)",
                "corr(|y-yhat|, abs(alpha_pred-alpha_star))",
            ],
            "value": [
                (
                    f"{corr_abs_err_abs_alpha:.3f}"
                    if np.isfinite(corr_abs_err_abs_alpha)
                    else "NA"
                ),
                (
                    f"{corr_signed_err_signed_alpha:.3f}"
                    if np.isfinite(corr_signed_err_signed_alpha)
                    else "NA"
                ),
                (
                    f"{corr_abs_err_abs_alpha2:.3f}"
                    if np.isfinite(corr_abs_err_abs_alpha2)
                    else "NA"
                ),
            ],
        }
    )

    bins_tbl = None
    if n_mask > 0:
        tmp = df.loc[finite_mask, ["y", "yhat", "abs_denom"]].copy()
        tmp["abs_err"] = abs_err.loc[finite_mask].values
        tmp["abs_alpha_err"] = abs_alpha_err.loc[finite_mask].values
        try:
            tmp["bin"] = pd.qcut(tmp["abs_alpha_err"], q=5, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)
            rows = []
            for b, g in grp:
                y_b = g["y"].to_numpy(dtype=float)
                yhat_b = g["yhat"].to_numpy(dtype=float)
                m = _forecast_metrics(y_b, yhat_b)
                rows.append(
                    {
                        "bin": str(b),
                        "count": int(len(g)),
                        "rmse_yhat": _format_sci(m["rmse"]),
                        "mean_abs_err": _format_sci(float(g["abs_err"].mean())),
                        "mean_abs_alpha_err": _format_sci(
                            float(g["abs_alpha_err"].mean())
                        ),
                        "mean_abs_denom": _format_sci(float(g["abs_denom"].mean())),
                    }
                )
            bins_tbl = pd.DataFrame(rows)
        except Exception:
            bins_tbl = None

    # B: oracle vs learned gate
    denom = df["denom"].to_numpy(dtype=float)
    v_state = df["v_state"].to_numpy(dtype=float)
    y_true = df["y"].to_numpy(dtype=float)
    yhat_model = df["yhat"].to_numpy(dtype=float)
    alpha_pred_all = df["alpha_pred"].to_numpy(dtype=float)
    alpha_star_all = df["alpha_star"].to_numpy(dtype=float)

    yhat_gate_pred = v_state + alpha_pred_all * denom
    alpha_bar = float(np.mean(alpha_pred_m)) if n_mask > 0 else np.nan
    yhat_gate_const = (
        v_state + alpha_bar * denom
        if np.isfinite(alpha_bar)
        else np.full_like(yhat_gate_pred, np.nan)
    )
    yhat_gate_star = np.full_like(yhat_gate_pred, np.nan)
    if n_mask > 0:
        fm = finite_mask.values
        yhat_gate_star[fm] = v_state[fm] + alpha_star_all[fm] * denom[fm]

    m_all = _forecast_metrics(y_true, yhat_model)
    r_all = _forecast_metrics(y_true, yhat_gate_pred)
    c_all = _forecast_metrics(y_true, yhat_gate_const)
    o_mask = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_star[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    r_mask = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_pred[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    c_mask = (
        _forecast_metrics(
            y_true[finite_mask.values], yhat_gate_const[finite_mask.values]
        )
        if n_mask > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )

    oracle_gain = (
        float(r_mask["rmse"] - o_mask["rmse"])
        if np.isfinite(r_mask["rmse"]) and np.isfinite(o_mask["rmse"])
        else np.nan
    )
    const_gain = (
        float(r_mask["rmse"] - c_mask["rmse"])
        if np.isfinite(r_mask["rmse"]) and np.isfinite(c_mask["rmse"])
        else np.nan
    )

    oracle_gap = oracle_gain
    denom_gap = (
        float(c_mask["rmse"] - o_mask["rmse"])
        if np.isfinite(c_mask["rmse"]) and np.isfinite(o_mask["rmse"])
        else np.nan
    )
    gap_ratio = (
        float(oracle_gap / (denom_gap + 1e-12))
        if np.isfinite(oracle_gap) and np.isfinite(denom_gap)
        else np.nan
    )

    b_tbl = pd.DataFrame(
        [
            {
                "forecast": "model_yhat",
                "rmse": _format_sci(m_all["rmse"]),
                "mae": _format_sci(m_all["mae"]),
                "medae": _format_sci(m_all["medae"]),
                "n": str(m_all["n"]),
            },
            {
                "forecast": "recon_gate_pred",
                "rmse": _format_sci(r_all["rmse"]),
                "mae": _format_sci(r_all["mae"]),
                "medae": _format_sci(r_all["medae"]),
                "n": str(r_all["n"]),
            },
            {
                "forecast": "const_alpha (masked alpha_bar)",
                "rmse": _format_sci(c_mask["rmse"]),
                "mae": _format_sci(c_mask["mae"]),
                "medae": _format_sci(c_mask["medae"]),
                "n": str(c_mask["n"]),
            },
            {
                "forecast": "oracle_alpha_star (masked)",
                "rmse": _format_sci(o_mask["rmse"]),
                "mae": _format_sci(o_mask["mae"]),
                "medae": _format_sci(o_mask["medae"]),
                "n": str(o_mask["n"]),
            },
            {
                "forecast": "gains (masked RMSE)",
                "rmse": (
                    f"oracle_gain={_format_sci(oracle_gain)}, const_gain={_format_sci(const_gain)}"
                ),
                "mae": "",
                "medae": "",
                "n": "",
            },
        ]
    )

    gap_tbl = pd.DataFrame(
        {
            "metric": ["oracle_gap (masked RMSE)", "gap_ratio"],
            "value": [
                _format_sci(oracle_gap),
                f"{gap_ratio:.3f}" if np.isfinite(gap_ratio) else "NA",
            ],
        }
    )

    # B2: reconstruction mismatch
    recon_max_abs = np.nan
    recon_corr = np.nan
    if n_mask > 0:
        diff = yhat_model[finite_mask.values] - yhat_gate_pred[finite_mask.values]
        recon_max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else np.nan
        recon_corr = _safe_corr(
            yhat_model[finite_mask.values], yhat_gate_pred[finite_mask.values]
        )
    recon_flag = bool(np.isfinite(recon_max_abs) and recon_max_abs > 1e-8)

    # Charts
    chart_paths: list[Path] = []
    try:
        # 1) alpha time series (masked)
        fig, ax = plt.subplots(figsize=(12, 4))
        a_pred = df["alpha_pred"].where(df["mask_denom_ok"])
        a_star = df["alpha_star"].where(df["mask_denom_ok"])
        ax.plot(df.index, a_pred.values, lw=1.0, label="alpha_pred (masked)")
        ax.plot(df.index, a_star.values, lw=1.0, label="alpha_star (masked)")
        ax.set_title(f"{variant} seed={seed}: alpha_pred vs alpha_star (masked)")
        ax.set_xlabel("date")
        ax.set_ylabel("alpha")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        p = out_dir / f"{prefix}__{variant}__seed{seed}__alpha_timeseries_masked.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        chart_paths.append(p)

        # 2) alpha scatter (masked)
        if n_mask > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(alpha_star_m, alpha_pred_m, s=8, alpha=0.5)
            ax.plot([0, 1], [0, 1], lw=1.0, alpha=0.8, color="black")
            ax.set_title(
                f"alpha scatter (masked)\n"
                f"corr={corr_alpha:.3f}  spearman={spearman_alpha:.3f}  rmse_z={rmse_z:.3f}"
            )
            ax.set_xlabel("alpha_star")
            ax.set_ylabel("alpha_pred")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            p = out_dir / f"{prefix}__{variant}__seed{seed}__alpha_scatter_masked.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths.append(p)

        # 2b) histogram of alpha_star and alpha_pred (masked)
        if n_mask > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                alpha_star_m,
                bins=60,
                alpha=0.6,
                label="alpha_star",
                range=(0.0, 1.0),
            )
            ax.hist(
                alpha_pred_m,
                bins=60,
                alpha=0.6,
                label="alpha_pred",
                range=(0.0, 1.0),
            )
            ax.axvline(0.01, lw=2.0, alpha=0.9, color="black")
            ax.axvline(0.99, lw=2.0, alpha=0.9, color="black")
            ax.set_title(
                "alpha distributions (masked)\n"
                f"star<=0.01: {100.0 * pct_alpha_star_le_0_01:.1f}%  "
                f"star>=0.99: {100.0 * pct_alpha_star_ge_0_99:.1f}%  "
                f"pred<=0.01: {100.0 * pct_alpha_pred_le_0_01:.1f}%  "
                f"pred>=0.99: {100.0 * pct_alpha_pred_ge_0_99:.1f}%"
            )
            ax.set_xlabel("alpha")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            p = out_dir / f"{prefix}__{variant}__seed{seed}__alpha_hist_masked.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths.append(p)

        # 2c) z_pred vs z_star scatter (masked)
        if n_mask > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(z_star_m, z_pred_m, s=8, alpha=0.5)
            lo = float(np.nanmin(np.r_[z_star_m, z_pred_m]))
            hi = float(np.nanmax(np.r_[z_star_m, z_pred_m]))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [lo, hi], lw=1.0, alpha=0.8, color="black")
            ax.set_title(
                f"z scatter (masked)  rmse_z={rmse_z:.3f}  eps_gate={eps_gate:.4g}"
            )
            ax.set_xlabel("z_star")
            ax.set_ylabel("z_pred")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            p = out_dir / f"{prefix}__{variant}__seed{seed}__z_scatter_masked.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths.append(p)

        # 3) abs error vs abs alpha mismatch (masked)
        if n_mask > 0:
            x = abs_alpha_err.loc[finite_mask].to_numpy(dtype=float)
            yv = abs_err.loc[finite_mask].to_numpy(dtype=float)
            corr_xy = _safe_corr(x, yv)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x, yv, s=8, alpha=0.5)
            ax.set_title(f"|err| vs |alpha_err| (masked)\n corr={corr_xy:.3f}")
            ax.set_xlabel("|alpha_pred - alpha_star|")
            ax.set_ylabel("|y - yhat|")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            p = (
                out_dir
                / f"{prefix}__{variant}__seed{seed}__abs_err_vs_alpha_mismatch.png"
            )
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths.append(p)

        # 3b) calibration plot: mean(alpha_pred) vs mean(alpha_star) by alpha_star deciles
        if calib_tbl is not None and not calib_tbl.empty:
            fig, ax = plt.subplots(figsize=(5, 5))
            xs = calib_tbl["alpha_star_mean"].to_numpy(dtype=float)
            ys = calib_tbl["alpha_pred_mean"].to_numpy(dtype=float)
            ax.plot(xs, ys, marker="o", lw=1.5)
            ax.plot([0, 1], [0, 1], lw=1.0, alpha=0.8, color="black")
            ax.set_title("Calibration: mean(alpha_pred) vs mean(alpha_star)")
            ax.set_xlabel("mean(alpha_star) per decile")
            ax.set_ylabel("mean(alpha_pred) per decile")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            p = out_dir / f"{prefix}__{variant}__seed{seed}__calibration.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths.append(p)

        # 4) oracle comparison (time series + squared-error diff)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax0, ax1 = axes
        ax0.plot(df.index, df["y"].values, lw=1.0, label="y (target)")
        ax0.plot(df.index, yhat_gate_pred, lw=1.0, label="yhat_gate_pred")
        if n_mask > 0:
            ax0.plot(df.index, yhat_gate_star, lw=1.0, label="yhat_gate_star (masked)")
        ax0.set_title(f"{variant} seed={seed}: oracle gate comparison (OOS)")
        ax0.set_ylabel("variance proxy")
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc="best")

        if n_mask > 0:
            se_pred = (df["y"].values - yhat_gate_pred) ** 2
            se_star = (df["y"].values - yhat_gate_star) ** 2
            diff = se_pred - se_star
            diff[~finite_mask.values] = np.nan
            ax1.plot(df.index, diff, lw=1.0)
            ax1.axhline(0.0, lw=1.0, alpha=0.8)
            ax1.set_title("(y - yhat_pred)^2 - (y - yhat_star)^2  (masked)")
        else:
            ax1.text(
                0.05,
                0.5,
                "No masked points; oracle comparison unavailable.",
                transform=ax1.transAxes,
            )
            ax1.set_axis_off()
        ax1.set_xlabel("date")
        ax1.grid(True, alpha=0.25)
        fig.tight_layout()
        p = out_dir / f"{prefix}__{variant}__seed{seed}__oracle_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        chart_paths.append(p)

        # 5) denom stability histogram
        abs_den = df["abs_denom"].to_numpy(dtype=float)
        abs_den = abs_den[np.isfinite(abs_den)]
        fig, ax = plt.subplots(figsize=(6, 4))
        if abs_den.size > 0:
            ax.hist(abs_den, bins=60, alpha=0.9)
            if np.isfinite(thresh):
                ax.axvline(
                    float(thresh), lw=2.0, alpha=0.9, color="red", label="thresh"
                )
            ax.set_title(f"|denom| hist (OOS)  frac_mask={frac_mask:.3f}")
            ax.set_xlabel("|r^2 - v_state|")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
        else:
            ax.text(0.05, 0.5, "No finite denom values.", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        p = out_dir / f"{prefix}__{variant}__seed{seed}__abs_denom_hist_report.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        chart_paths.append(p)
    except Exception as e:
        logger.warning(f"Alpha* report charts failed: {e}")

    # Write Markdown report
    md_path = out_dir / f"{prefix}__{variant}__seed{seed}__report.md"

    calib_tbl_md = None
    if calib_tbl is not None and not calib_tbl.empty:
        calib_tbl_md = pd.DataFrame(
            {
                "alpha_star_bin": calib_tbl["alpha_star_bin"],
                "count": calib_tbl["count"],
                "alpha_star_mean": calib_tbl["alpha_star_mean"].map(
                    lambda v: f"{v:.4f}" if np.isfinite(v) else "NA"
                ),
                "alpha_pred_mean": calib_tbl["alpha_pred_mean"].map(
                    lambda v: f"{v:.4f}" if np.isfinite(v) else "NA"
                ),
            }
        )

    header = [
        "<!-- generated by examples/volatility_forecast_2.py -->",
        f"# Gate-learning diagnostic report — {variant} (seed={seed})",
        "",
        f"- rows (test): {n_test}",
        f"- masked rows: {n_mask}",
        (
            f"- frac_mask: {frac_mask:.3f}"
            if np.isfinite(frac_mask)
            else "- frac_mask: NA"
        ),
        f"- denom threshold (abs): {_format_sci(thresh)}",
        f"- gate_target: {gate_target}",
        f"- eps_gate: {eps_gate:.4g}",
        f"- weight_mode: {weight_mode}",
        "",
        "## A1) Alpha alignment / denom stability (masked)",
        _df_to_markdown_table(a1),
        "### Boundary mass (masked; train+test if available)",
        _df_to_markdown_table(boundary_tbl),
        "### Calibration (alpha_star deciles; masked test)",
        (
            _df_to_markdown_table(calib_tbl_md)
            if calib_tbl_md is not None
            else "_Calibration unavailable._"
        ),
        "## A2) Forecast quality vs alpha mismatch (masked)",
        _df_to_markdown_table(a2_corr),
    ]
    if bins_tbl is not None and not bins_tbl.empty:
        header += [
            "### Binned by |alpha_pred-alpha_star| quantiles",
            _df_to_markdown_table(bins_tbl),
        ]
    else:
        header += [
            "### Binned by |alpha_pred-alpha_star| quantiles",
            "_Binning unavailable (insufficient unique values)._",
        ]

    header += [
        "## B) Counterfactual gates (oracle vs learned)",
        _df_to_markdown_table(b_tbl),
        "### Oracle gap shrink",
        _df_to_markdown_table(gap_tbl),
        "",
        "### B2) Reconstruction sanity check",
        _df_to_markdown_table(
            pd.DataFrame(
                {
                    "metric": [
                        "max_abs(yhat - yhat_gate_pred) (masked)",
                        "corr(yhat, yhat_gate_pred) (masked)",
                    ],
                    "value": [
                        _format_sci(recon_max_abs),
                        f"{recon_corr:.6f}" if np.isfinite(recon_corr) else "NA",
                    ],
                }
            )
        ),
    ]
    if recon_flag:
        header += [
            "",
            "> NOTE: Reconstruction mismatch is non-trivial; the assumed update form may differ from model mechanics. Oracle comparisons should be interpreted as a proxy.",
        ]

    if chart_paths:
        header += [
            "",
            "## C) Charts",
        ]
        for p in chart_paths:
            header.append(f"- `{p.name}`")
    header.append("")

    md_path.write_text("\n".join(header), encoding="utf-8")
    return md_path, chart_paths


def write_alpha_star_report(
    df_test: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    out_dir: Path,
    prefix: str,
) -> tuple[Path, list[Path]]:
    """Generate an alpha diagnostic report (Markdown + charts) for XGBSTES.

    This report is designed to answer:
      - Is the gate-learning objective the bottleneck?
      - Is the implied alpha label pathological (boundary mass / ill-conditioning)?

    Conventions:
      - `alpha_star` in df_test is the *hard-clipped* ratio label (alpha_clip).
      - `alpha_soft` / `z_soft` are the MAP posterior targets (interior; no hard clip).
      - All alignment metrics are computed on denom-stable rows only (`mask_denom_ok`).
    """
    _ensure_dir(out_dir)

    gate_params = dict(getattr(df_test, "attrs", {}).get("gate_params", {}) or {})
    train_boundary = dict(
        getattr(df_test, "attrs", {}).get("train_boundary_stats", {}) or {}
    )
    test_boundary = dict(
        getattr(df_test, "attrs", {}).get("test_boundary_stats", {}) or {}
    )

    df = df_test.copy()
    if "mask_denom_ok" not in df.columns:
        df["mask_denom_ok"] = False
    df["mask_denom_ok"] = df["mask_denom_ok"].astype(bool)

    required = [
        "y",
        "yhat",
        "v_state",
        "denom",
        "abs_denom",
        "alpha_pred",
        "alpha_star",  # alpha_clip
        "alpha_soft",
        "z_soft",
        "thresh_abs_denom",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    def _get_scalar(key: str, default: float) -> float:
        if key in gate_params and gate_params[key] is not None:
            try:
                return float(gate_params[key])
            except Exception:
                return float(default)
        if key in df.columns and len(df) > 0:
            try:
                return float(df[key].iloc[0])
            except Exception:
                return float(default)
        return float(default)

    gate_target = str(
        gate_params.get(
            "gate_target",
            (
                str(df["gate_target"].iloc[0])
                if "gate_target" in df.columns and len(df) > 0
                else "logit_soft_map"
            ),
        )
    )
    weight_mode = str(
        gate_params.get(
            "weight_mode",
            (
                str(df["weight_mode"].iloc[0])
                if "weight_mode" in df.columns and len(df) > 0
                else "denom2"
            ),
        )
    )

    eps_gate = _get_scalar("eps_gate", 1e-3)
    alpha_prior_mu = _get_scalar("alpha_prior_mu", 0.0)
    alpha_prior_tau = _get_scalar("alpha_prior_tau", 1.5)
    sigma_eps = _get_scalar("sigma_eps", np.nan)
    denom_quantile = _get_scalar("denom_quantile", 0.05)
    min_denom_floor = _get_scalar("min_denom_floor", 1e-12)
    thresh = _get_scalar("thresh_abs_denom", np.nan)
    alpha_soft_mean_train = _get_scalar("alpha_soft_mean_train_masked", np.nan)

    mask = df["mask_denom_ok"].astype(bool)
    finite_soft = (
        mask
        & np.isfinite(df["y"].values)
        & np.isfinite(df["yhat"].values)
        & np.isfinite(df["v_state"].values)
        & np.isfinite(df["denom"].values)
        & np.isfinite(df["abs_denom"].values)
        & np.isfinite(df["alpha_pred"].values)
        & np.isfinite(df["alpha_soft"].values)
    )
    finite_clip = (
        mask
        & np.isfinite(df["y"].values)
        & np.isfinite(df["yhat"].values)
        & np.isfinite(df["v_state"].values)
        & np.isfinite(df["denom"].values)
        & np.isfinite(df["abs_denom"].values)
        & np.isfinite(df["alpha_pred"].values)
        & np.isfinite(df["alpha_star"].values)
    )

    n_test = int(len(df))
    frac_mask = float(mask.mean()) if n_test > 0 else np.nan
    n_mask_soft = int(finite_soft.sum())
    n_mask_clip = int(finite_clip.sum())

    a_pred_soft = df.loc[finite_soft, "alpha_pred"].to_numpy(dtype=float)
    a_soft = df.loc[finite_soft, "alpha_soft"].to_numpy(dtype=float)
    a_clip_on_soft = df.loc[finite_soft, "alpha_star"].to_numpy(dtype=float)

    a_pred_clip = df.loc[finite_clip, "alpha_pred"].to_numpy(dtype=float)
    a_clip = df.loc[finite_clip, "alpha_star"].to_numpy(dtype=float)

    alpha_pred_mean = float(np.mean(a_pred_soft)) if n_mask_soft > 0 else np.nan
    alpha_pred_std = float(np.std(a_pred_soft, ddof=0)) if n_mask_soft > 0 else np.nan
    alpha_soft_mean = float(np.mean(a_soft)) if n_mask_soft > 0 else np.nan
    alpha_soft_std = float(np.std(a_soft, ddof=0)) if n_mask_soft > 0 else np.nan
    alpha_clip_mean = float(np.mean(a_clip_on_soft)) if n_mask_soft > 0 else np.nan
    alpha_clip_std = (
        float(np.std(a_clip_on_soft, ddof=0)) if n_mask_soft > 0 else np.nan
    )

    corr_pred_soft = _safe_corr(a_pred_soft, a_soft) if n_mask_soft > 0 else np.nan
    spearman_pred_soft = (
        _spearman_corr(a_pred_soft, a_soft) if n_mask_soft > 0 else np.nan
    )
    corr_pred_clip = _safe_corr(a_pred_clip, a_clip) if n_mask_clip > 0 else np.nan

    mae_pred_soft = (
        float(np.mean(np.abs(a_pred_soft - a_soft))) if n_mask_soft > 0 else np.nan
    )
    mae_pred_clip = (
        float(np.mean(np.abs(a_pred_clip - a_clip))) if n_mask_clip > 0 else np.nan
    )

    # z-space alignment (eps_gate only for numerical safety in logit)
    z_pred = _logit(a_pred_soft, eps_gate)
    if "z_soft" in df.columns:
        z_soft = df.loc[finite_soft, "z_soft"].to_numpy(dtype=float)
    else:
        z_soft = _logit(a_soft, eps_gate)
    z_pred = np.asarray(z_pred, dtype=float)
    z_soft = np.asarray(z_soft, dtype=float)

    rmse_z_pred_z_soft = (
        float(np.sqrt(np.mean((z_pred - z_soft) ** 2))) if n_mask_soft > 0 else np.nan
    )
    corr_z_pred_z_soft = _safe_corr(z_pred, z_soft) if n_mask_soft > 0 else np.nan

    a_clip_scaled = eps_gate + (1.0 - 2.0 * eps_gate) * np.clip(
        a_clip_on_soft, 0.0, 1.0
    )
    z_clip = np.asarray(_logit(a_clip_scaled, 1e-12), dtype=float)
    rmse_z_pred_z_clip = (
        float(np.sqrt(np.mean((z_pred - z_clip) ** 2))) if n_mask_soft > 0 else np.nan
    )

    # Boundary mass (masked test)
    def _pct(x: np.ndarray, which: str) -> float:
        if x.size <= 0:
            return np.nan
        if which == "le_0_01":
            return float(np.mean(x <= 0.01))
        if which == "ge_0_99":
            return float(np.mean(x >= 0.99))
        raise ValueError(which)

    pct_clip_le_0_01 = _pct(a_clip_on_soft, "le_0_01")
    pct_clip_ge_0_99 = _pct(a_clip_on_soft, "ge_0_99")
    pct_soft_le_0_01 = _pct(a_soft, "le_0_01")
    pct_soft_ge_0_99 = _pct(a_soft, "ge_0_99")
    pct_pred_le_0_01 = _pct(a_pred_soft, "le_0_01")
    pct_pred_ge_0_99 = _pct(a_pred_soft, "ge_0_99")

    abs_denom_soft = df.loc[finite_soft, "abs_denom"].to_numpy(dtype=float)
    abs_denom_p50 = (
        float(np.quantile(abs_denom_soft, 0.50)) if n_mask_soft > 0 else np.nan
    )
    abs_denom_p10 = (
        float(np.quantile(abs_denom_soft, 0.10)) if n_mask_soft > 0 else np.nan
    )
    abs_denom_p01 = (
        float(np.quantile(abs_denom_soft, 0.01)) if n_mask_soft > 0 else np.nan
    )

    if not np.isfinite(alpha_soft_mean_train):
        alpha_soft_mean_train = alpha_soft_mean

    # A1) Alpha alignment / denom stability table
    a1 = pd.DataFrame(
        {
            "metric": [
                "gate_target",
                "weight_mode",
                "prior_mu (z)",
                "prior_tau (z)",
                "sigma_eps (used)",
                "denom_quantile",
                "min_denom_floor",
                "thresh_abs_denom",
                "n_test",
                "n_mask (soft)",
                "frac_mask",
                "alpha_pred_mean",
                "alpha_pred_std",
                "alpha_soft_mean",
                "alpha_soft_std",
                "alpha_clip_mean",
                "alpha_clip_std",
                "corr(alpha_pred, alpha_soft)",
                "spearman(alpha_pred, alpha_soft)",
                "corr(alpha_pred, alpha_clip)",
                "mae(alpha_pred-alpha_soft)",
                "mae(alpha_pred-alpha_clip)",
                "rmse(z_pred-z_soft)",
                "corr(z_pred, z_soft)",
                "rmse(z_pred-z_clip)",
                "abs_denom_p50",
                "abs_denom_p10",
                "abs_denom_p01",
                "pct_alpha_clip<=0.01 (masked test)",
                "pct_alpha_clip>=0.99 (masked test)",
                "pct_alpha_soft<=0.01 (masked test)",
                "pct_alpha_soft>=0.99 (masked test)",
                "pct_alpha_pred<=0.01 (masked test)",
                "pct_alpha_pred>=0.99 (masked test)",
            ],
            "value": [
                str(gate_target),
                str(weight_mode),
                f"{alpha_prior_mu:.3g}" if np.isfinite(alpha_prior_mu) else "NA",
                f"{alpha_prior_tau:.3g}" if np.isfinite(alpha_prior_tau) else "NA",
                _format_sci(sigma_eps),
                f"{denom_quantile:.3f}" if np.isfinite(denom_quantile) else "NA",
                _format_sci(min_denom_floor),
                _format_sci(thresh),
                str(n_test),
                str(n_mask_soft),
                f"{frac_mask:.4f}" if np.isfinite(frac_mask) else "NA",
                f"{alpha_pred_mean:.4f}" if np.isfinite(alpha_pred_mean) else "NA",
                f"{alpha_pred_std:.4f}" if np.isfinite(alpha_pred_std) else "NA",
                f"{alpha_soft_mean:.4f}" if np.isfinite(alpha_soft_mean) else "NA",
                f"{alpha_soft_std:.4f}" if np.isfinite(alpha_soft_std) else "NA",
                f"{alpha_clip_mean:.4f}" if np.isfinite(alpha_clip_mean) else "NA",
                f"{alpha_clip_std:.4f}" if np.isfinite(alpha_clip_std) else "NA",
                f"{corr_pred_soft:.4f}" if np.isfinite(corr_pred_soft) else "NA",
                (
                    f"{spearman_pred_soft:.4f}"
                    if np.isfinite(spearman_pred_soft)
                    else "NA"
                ),
                f"{corr_pred_clip:.4f}" if np.isfinite(corr_pred_clip) else "NA",
                f"{mae_pred_soft:.4f}" if np.isfinite(mae_pred_soft) else "NA",
                f"{mae_pred_clip:.4f}" if np.isfinite(mae_pred_clip) else "NA",
                (
                    f"{rmse_z_pred_z_soft:.4f}"
                    if np.isfinite(rmse_z_pred_z_soft)
                    else "NA"
                ),
                (
                    f"{corr_z_pred_z_soft:.4f}"
                    if np.isfinite(corr_z_pred_z_soft)
                    else "NA"
                ),
                (
                    f"{rmse_z_pred_z_clip:.4f}"
                    if np.isfinite(rmse_z_pred_z_clip)
                    else "NA"
                ),
                _format_sci(abs_denom_p50),
                _format_sci(abs_denom_p10),
                _format_sci(abs_denom_p01),
                (
                    f"{100.0 * pct_clip_le_0_01:.2f}%"
                    if np.isfinite(pct_clip_le_0_01)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_clip_ge_0_99:.2f}%"
                    if np.isfinite(pct_clip_ge_0_99)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_soft_le_0_01:.2f}%"
                    if np.isfinite(pct_soft_le_0_01)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_soft_ge_0_99:.2f}%"
                    if np.isfinite(pct_soft_ge_0_99)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_pred_le_0_01:.2f}%"
                    if np.isfinite(pct_pred_le_0_01)
                    else "NA"
                ),
                (
                    f"{100.0 * pct_pred_ge_0_99:.2f}%"
                    if np.isfinite(pct_pred_ge_0_99)
                    else "NA"
                ),
            ],
        }
    )

    boundary_tbl = pd.DataFrame(
        [
            {
                "segment": "train (masked)",
                "n": str(train_boundary.get("n", "NA")),
                "n_mask": str(train_boundary.get("n_mask", "NA")),
                "frac_mask": (
                    f"{train_boundary.get('frac_mask', np.nan):.4f}"
                    if np.isfinite(train_boundary.get("frac_mask", np.nan))
                    else "NA"
                ),
                "clip<=0.01": (
                    f"{100.0 * train_boundary.get('pct_alpha_star_clip_le_0_01', np.nan):.2f}%"
                    if np.isfinite(
                        train_boundary.get("pct_alpha_star_clip_le_0_01", np.nan)
                    )
                    else "NA"
                ),
                "clip>=0.99": (
                    f"{100.0 * train_boundary.get('pct_alpha_star_clip_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(
                        train_boundary.get("pct_alpha_star_clip_ge_0_99", np.nan)
                    )
                    else "NA"
                ),
                "soft<=0.01": (
                    f"{100.0 * train_boundary.get('pct_alpha_soft_le_0_01', np.nan):.2f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_soft_le_0_01", np.nan))
                    else "NA"
                ),
                "soft>=0.99": (
                    f"{100.0 * train_boundary.get('pct_alpha_soft_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_soft_ge_0_99", np.nan))
                    else "NA"
                ),
                "pred<=0.01": (
                    f"{100.0 * train_boundary.get('pct_alpha_pred_le_0_01', np.nan):.2f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_pred_le_0_01", np.nan))
                    else "NA"
                ),
                "pred>=0.99": (
                    f"{100.0 * train_boundary.get('pct_alpha_pred_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(train_boundary.get("pct_alpha_pred_ge_0_99", np.nan))
                    else "NA"
                ),
            },
            {
                "segment": "test (masked)",
                "n": str(test_boundary.get("n", "NA")),
                "n_mask": str(test_boundary.get("n_mask", "NA")),
                "frac_mask": (
                    f"{test_boundary.get('frac_mask', np.nan):.4f}"
                    if np.isfinite(test_boundary.get("frac_mask", np.nan))
                    else "NA"
                ),
                "clip<=0.01": (
                    f"{100.0 * test_boundary.get('pct_alpha_star_clip_le_0_01', np.nan):.2f}%"
                    if np.isfinite(
                        test_boundary.get("pct_alpha_star_clip_le_0_01", np.nan)
                    )
                    else "NA"
                ),
                "clip>=0.99": (
                    f"{100.0 * test_boundary.get('pct_alpha_star_clip_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(
                        test_boundary.get("pct_alpha_star_clip_ge_0_99", np.nan)
                    )
                    else "NA"
                ),
                "soft<=0.01": (
                    f"{100.0 * test_boundary.get('pct_alpha_soft_le_0_01', np.nan):.2f}%"
                    if np.isfinite(test_boundary.get("pct_alpha_soft_le_0_01", np.nan))
                    else "NA"
                ),
                "soft>=0.99": (
                    f"{100.0 * test_boundary.get('pct_alpha_soft_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(test_boundary.get("pct_alpha_soft_ge_0_99", np.nan))
                    else "NA"
                ),
                "pred<=0.01": (
                    f"{100.0 * test_boundary.get('pct_alpha_pred_le_0_01', np.nan):.2f}%"
                    if np.isfinite(test_boundary.get("pct_alpha_pred_le_0_01", np.nan))
                    else "NA"
                ),
                "pred>=0.99": (
                    f"{100.0 * test_boundary.get('pct_alpha_pred_ge_0_99', np.nan):.2f}%"
                    if np.isfinite(test_boundary.get("pct_alpha_pred_ge_0_99", np.nan))
                    else "NA"
                ),
            },
        ]
    )

    # Calibration: alpha_soft deciles → mean(alpha_pred)
    calib_tbl = None
    if n_mask_soft > 0:
        try:
            tmp = df.loc[finite_soft, ["alpha_soft", "alpha_pred"]].copy()
            tmp["bin"] = pd.qcut(tmp["alpha_soft"], q=10, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)
            rows = []
            for b, g in grp:
                rows.append(
                    {
                        "alpha_soft_bin": str(b),
                        "count": int(len(g)),
                        "alpha_soft_mean": f"{float(g['alpha_soft'].mean()):.4f}",
                        "alpha_pred_mean": f"{float(g['alpha_pred'].mean()):.4f}",
                    }
                )
            calib_tbl = pd.DataFrame(rows)
        except Exception:
            calib_tbl = None

    # Forecast quality vs alpha mismatch (soft)
    var_err = (df["y"] - df["yhat"]).astype(float)
    abs_err = var_err.abs()
    alpha_err_soft_s = (df["alpha_pred"] - df["alpha_soft"]).astype(float)
    abs_alpha_err_soft_s = alpha_err_soft_s.abs()

    corr_abs_err_abs_alpha_soft = (
        _safe_corr(
            abs_err.loc[finite_soft].to_numpy(dtype=float),
            abs_alpha_err_soft_s.loc[finite_soft].to_numpy(dtype=float),
        )
        if n_mask_soft > 0
        else np.nan
    )
    corr_signed_err_alpha_soft = (
        _safe_corr(
            var_err.loc[finite_soft].to_numpy(dtype=float),
            alpha_err_soft_s.loc[finite_soft].to_numpy(dtype=float),
        )
        if n_mask_soft > 0
        else np.nan
    )
    a2_corr = pd.DataFrame(
        {
            "metric": [
                "corr(|y-yhat|, |alpha_pred-alpha_soft|)",
                "corr(y-yhat, alpha_pred-alpha_soft)",
            ],
            "value": [
                (
                    f"{corr_abs_err_abs_alpha_soft:.4f}"
                    if np.isfinite(corr_abs_err_abs_alpha_soft)
                    else "NA"
                ),
                (
                    f"{corr_signed_err_alpha_soft:.4f}"
                    if np.isfinite(corr_signed_err_alpha_soft)
                    else "NA"
                ),
            ],
        }
    )

    bins_tbl = None
    if n_mask_soft > 0:
        tmp = df.loc[finite_soft, ["y", "yhat", "abs_denom"]].copy()
        tmp["abs_err"] = abs_err.loc[finite_soft].values
        tmp["abs_alpha_err"] = abs_alpha_err_soft_s.loc[finite_soft].values
        try:
            tmp["bin"] = pd.qcut(tmp["abs_alpha_err"], q=5, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)
            rows = []
            for b, g in grp:
                y_b = g["y"].to_numpy(dtype=float)
                yhat_b = g["yhat"].to_numpy(dtype=float)
                m = _forecast_metrics(y_b, yhat_b)
                rows.append(
                    {
                        "bin": str(b),
                        "count": int(len(g)),
                        "rmse_yhat": _format_sci(m["rmse"]),
                        "mean_abs_err": _format_sci(float(g["abs_err"].mean())),
                        "mean_abs_alpha_err": _format_sci(
                            float(g["abs_alpha_err"].mean())
                        ),
                        "mean_abs_denom": _format_sci(float(g["abs_denom"].mean())),
                    }
                )
            bins_tbl = pd.DataFrame(rows)
        except Exception:
            bins_tbl = None

    # Oracle / reconstructed gate forecasts
    y_true = df["y"].to_numpy(dtype=float)
    yhat_model = df["yhat"].to_numpy(dtype=float)
    v_state = df["v_state"].to_numpy(dtype=float)
    denom = df["denom"].to_numpy(dtype=float)

    alpha_pred_all = df["alpha_pred"].to_numpy(dtype=float)
    alpha_soft_all = df["alpha_soft"].to_numpy(dtype=float)
    alpha_clip_all = df["alpha_star"].to_numpy(dtype=float)

    yhat_gate_pred = v_state + alpha_pred_all * denom
    yhat_gate_soft = v_state + alpha_soft_all * denom
    yhat_gate_clip = v_state + alpha_clip_all * denom
    yhat_gate_const = (
        v_state + float(alpha_soft_mean_train) * denom
        if np.isfinite(alpha_soft_mean_train)
        else np.full_like(yhat_gate_pred, np.nan)
    )

    m_all = _forecast_metrics(y_true, yhat_model)
    r_all = _forecast_metrics(y_true, yhat_gate_pred)
    r_mask_soft = (
        _forecast_metrics(
            y_true[finite_soft.values], yhat_gate_pred[finite_soft.values]
        )
        if n_mask_soft > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    c_mask_soft = (
        _forecast_metrics(
            y_true[finite_soft.values], yhat_gate_const[finite_soft.values]
        )
        if n_mask_soft > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    o_soft = (
        _forecast_metrics(
            y_true[finite_soft.values], yhat_gate_soft[finite_soft.values]
        )
        if n_mask_soft > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    r_mask_clip = (
        _forecast_metrics(
            y_true[finite_clip.values], yhat_gate_pred[finite_clip.values]
        )
        if n_mask_clip > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )
    o_clip = (
        _forecast_metrics(
            y_true[finite_clip.values], yhat_gate_clip[finite_clip.values]
        )
        if n_mask_clip > 0
        else {"rmse": np.nan, "mae": np.nan, "medae": np.nan, "n": 0}
    )

    oracle_gain_soft = (
        float(r_mask_soft["rmse"] - o_soft["rmse"])
        if np.isfinite(r_mask_soft["rmse"]) and np.isfinite(o_soft["rmse"])
        else np.nan
    )
    oracle_gain_clip = (
        float(r_mask_clip["rmse"] - o_clip["rmse"])
        if np.isfinite(r_mask_clip["rmse"]) and np.isfinite(o_clip["rmse"])
        else np.nan
    )
    const_gain_soft = (
        float(r_mask_soft["rmse"] - c_mask_soft["rmse"])
        if np.isfinite(r_mask_soft["rmse"]) and np.isfinite(c_mask_soft["rmse"])
        else np.nan
    )

    oracle_gap_soft = oracle_gain_soft
    denom_gap_soft = (
        float(c_mask_soft["rmse"] - o_soft["rmse"])
        if np.isfinite(c_mask_soft["rmse"]) and np.isfinite(o_soft["rmse"])
        else np.nan
    )
    gap_ratio_soft = (
        float(oracle_gap_soft / (denom_gap_soft + 1e-12))
        if np.isfinite(oracle_gap_soft) and np.isfinite(denom_gap_soft)
        else np.nan
    )

    oracle_tbl = pd.DataFrame(
        [
            {
                "forecast": "model_yhat",
                "rmse": _format_sci(m_all["rmse"]),
                "mae": _format_sci(m_all["mae"]),
                "medae": _format_sci(m_all["medae"]),
                "n": str(m_all["n"]),
            },
            {
                "forecast": "recon_gate_pred (all)",
                "rmse": _format_sci(r_all["rmse"]),
                "mae": _format_sci(r_all["mae"]),
                "medae": _format_sci(r_all["medae"]),
                "n": str(r_all["n"]),
            },
            {
                "forecast": f"const_alpha (alpha_bar={alpha_soft_mean_train:.4f}) [masked soft]",
                "rmse": _format_sci(c_mask_soft["rmse"]),
                "mae": _format_sci(c_mask_soft["mae"]),
                "medae": _format_sci(c_mask_soft["medae"]),
                "n": str(c_mask_soft["n"]),
            },
            {
                "forecast": "oracle_alpha_soft (masked)",
                "rmse": _format_sci(o_soft["rmse"]),
                "mae": _format_sci(o_soft["mae"]),
                "medae": _format_sci(o_soft["medae"]),
                "n": str(o_soft["n"]),
            },
            {
                "forecast": "oracle_alpha_clip (masked)",
                "rmse": _format_sci(o_clip["rmse"]),
                "mae": _format_sci(o_clip["mae"]),
                "medae": _format_sci(o_clip["medae"]),
                "n": str(o_clip["n"]),
            },
            {
                "forecast": "gains (masked RMSE)",
                "rmse": (
                    f"oracle_gain_soft={_format_sci(oracle_gain_soft)}, "
                    f"oracle_gain_clip={_format_sci(oracle_gain_clip)}, "
                    f"const_gain_soft={_format_sci(const_gain_soft)}"
                ),
                "mae": "",
                "medae": "",
                "n": "",
            },
        ]
    )

    gap_tbl = pd.DataFrame(
        {
            "metric": ["oracle_gap_soft (masked RMSE)", "gap_ratio_soft"],
            "value": [
                _format_sci(oracle_gap_soft),
                f"{gap_ratio_soft:.3f}" if np.isfinite(gap_ratio_soft) else "NA",
            ],
        }
    )

    # Reconstruction mismatch (sanity): does yhat ≈ v_state + alpha_pred*denom?
    recon_max_abs = np.nan
    recon_corr = np.nan
    if n_mask_soft > 0:
        diff = yhat_model[finite_soft.values] - yhat_gate_pred[finite_soft.values]
        recon_max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else np.nan
        recon_corr = _safe_corr(
            yhat_model[finite_soft.values], yhat_gate_pred[finite_soft.values]
        )
    recon_flag = bool(np.isfinite(recon_max_abs) and recon_max_abs > 1e-8)

    # Charts
    chart_paths: list[Path] = []

    def _save(fig: plt.Figure, name: str) -> None:
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        chart_paths.append(p)

    try:
        # 1) alpha time series overlay (masked)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            df.index, df["alpha_pred"].where(mask).values, lw=1.0, label="alpha_pred"
        )
        ax.plot(
            df.index, df["alpha_soft"].where(mask).values, lw=1.0, label="alpha_soft"
        )
        ax.plot(
            df.index,
            df["alpha_star"].where(mask).values,
            lw=1.0,
            alpha=0.8,
            label="alpha_clip",
        )
        ax.set_title(f"{variant} seed={seed}: alpha paths (masked)")
        ax.set_xlabel("date")
        ax.set_ylabel("alpha")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        _save(fig, f"{prefix}__{variant}__seed{seed}__alpha_timeseries_masked.png")

        # 2) alpha_pred vs alpha_soft scatter (masked)
        if n_mask_soft > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(a_soft, a_pred_soft, s=8, alpha=0.5)
            ax.plot([0, 1], [0, 1], lw=1.0, alpha=0.8, color="black")
            ax.set_title(
                f"alpha_pred vs alpha_soft (masked)\n"
                f"corr={corr_pred_soft:.3f}  spearman={spearman_pred_soft:.3f}  rmse_z={rmse_z_pred_z_soft:.3f}"
            )
            ax.set_xlabel("alpha_soft")
            ax.set_ylabel("alpha_pred")
            ax.grid(True, alpha=0.25)
            _save(
                fig, f"{prefix}__{variant}__seed{seed}__alpha_scatter_soft_masked.png"
            )

        # 3) z_pred vs z_soft scatter (masked)
        if n_mask_soft > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(z_soft, z_pred, s=8, alpha=0.5)
            lo = float(np.nanmin(np.r_[z_soft, z_pred]))
            hi = float(np.nanmax(np.r_[z_soft, z_pred]))
            ax.plot([lo, hi], [lo, hi], lw=1.0, alpha=0.8, color="black")
            ax.set_title(
                f"z_pred vs z_soft (masked)\n"
                f"corr={corr_z_pred_z_soft:.3f}  rmse={rmse_z_pred_z_soft:.3f}"
            )
            ax.set_xlabel("z_soft (MAP)")
            ax.set_ylabel("z_pred = logit(alpha_pred)")
            ax.grid(True, alpha=0.25)
            _save(fig, f"{prefix}__{variant}__seed{seed}__z_scatter_soft_masked.png")

        # 4) alpha histogram overlay (masked)
        if n_mask_soft > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                a_clip_on_soft,
                bins=60,
                alpha=0.55,
                label="alpha_clip",
                range=(0.0, 1.0),
            )
            ax.hist(a_soft, bins=60, alpha=0.55, label="alpha_soft", range=(0.0, 1.0))
            ax.hist(
                a_pred_soft, bins=60, alpha=0.55, label="alpha_pred", range=(0.0, 1.0)
            )
            ax.axvline(0.01, lw=2.0, alpha=0.9, color="black")
            ax.axvline(0.99, lw=2.0, alpha=0.9, color="black")
            ax.set_title("alpha distributions (masked test)")
            ax.set_xlabel("alpha")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            _save(fig, f"{prefix}__{variant}__seed{seed}__alpha_hist_masked.png")

        # 5) denom histogram
        abs_den = df["abs_denom"].to_numpy(dtype=float)
        abs_den = abs_den[np.isfinite(abs_den)]
        if abs_den.size > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(abs_den, bins=60, alpha=0.9)
            if np.isfinite(thresh):
                ax.axvline(
                    float(thresh), lw=2.0, alpha=0.9, color="red", label="thresh"
                )
            ax.set_title(
                f"|denom| histogram (test)  frac_mask={frac_mask:.3f}"
                if np.isfinite(frac_mask)
                else "|denom| histogram (test)"
            )
            ax.set_xlabel("|r^2 - v_state|")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            if np.isfinite(thresh):
                ax.legend(loc="best")
            _save(fig, f"{prefix}__{variant}__seed{seed}__abs_denom_hist.png")

        # 6) abs forecast error vs abs alpha mismatch (masked)
        if n_mask_soft > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = abs_alpha_err_soft_s.loc[finite_soft].to_numpy(dtype=float)
            y = abs_err.loc[finite_soft].to_numpy(dtype=float)
            ax.scatter(x, y, s=8, alpha=0.5)
            ax.set_title(
                f"|y-yhat| vs |alpha_pred-alpha_soft| (masked)\n"
                f"corr={corr_abs_err_abs_alpha_soft:.3f}"
                if np.isfinite(corr_abs_err_abs_alpha_soft)
                else "|y-yhat| vs |alpha_pred-alpha_soft| (masked)"
            )
            ax.set_xlabel("|alpha_pred - alpha_soft|")
            ax.set_ylabel("|y - yhat|")
            ax.grid(True, alpha=0.25)
            _save(
                fig,
                f"{prefix}__{variant}__seed{seed}__abs_err_vs_abs_alpha_err_soft.png",
            )

        # 7) oracle comparison panel (masked soft)
        if n_mask_soft > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1, ax2 = axes

            y_gate_pred = pd.Series(yhat_gate_pred, index=df.index).where(finite_soft)
            y_gate_oracle = pd.Series(yhat_gate_soft, index=df.index).where(finite_soft)

            ax1.plot(df.index, df["y"].values, lw=1.0, label="y (target)")
            ax1.plot(
                df.index, y_gate_pred.values, lw=1.0, label="yhat_gate_pred (masked)"
            )
            ax1.plot(
                df.index,
                y_gate_oracle.values,
                lw=1.0,
                label="yhat_oracle_soft (masked)",
            )
            ax1.set_title("Oracle alpha_soft vs learned gate reconstruction (test)")
            ax1.set_ylabel("variance")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best")

            dloss = (df["y"].values - yhat_gate_pred) ** 2 - (
                df["y"].values - yhat_gate_soft
            ) ** 2
            ax2.plot(
                df.index,
                pd.Series(dloss, index=df.index).where(finite_soft).values,
                lw=1.0,
            )
            ax2.axhline(0.0, lw=1.0, alpha=0.7)
            ax2.set_title("Loss diff: (recon_pred)^2 - (oracle_soft)^2 (masked)")
            ax2.set_ylabel("Δ loss")
            ax2.set_xlabel("date")
            ax2.grid(True, alpha=0.25)
            _save(fig, f"{prefix}__{variant}__seed{seed}__oracle_soft_comparison.png")
    except Exception as e:
        logger.warning(f"Alpha diagnostics report plots failed: {e}")

    md_path = out_dir / f"{prefix}__{variant}__seed{seed}__report.md"
    lines: list[str] = [
        "<!-- generated by examples/volatility_forecast_2.py -->",
        "",
        f"# Alpha diagnostic report — {variant} (seed={seed})",
        "",
        "## A) Alpha alignment / denom stability (test / OOS)",
        "",
        _df_to_markdown_table(a1),
        "",
        "### Boundary mass (train vs test; masked)",
        "",
        _df_to_markdown_table(boundary_tbl),
    ]
    if calib_tbl is not None:
        lines += [
            "",
            "### Calibration: alpha_soft deciles → mean(alpha_pred) (masked test)",
            "",
            _df_to_markdown_table(calib_tbl),
        ]

    lines += [
        "",
        "## B) Forecast quality vs alpha mismatch (masked test)",
        "",
        _df_to_markdown_table(a2_corr),
    ]
    if bins_tbl is not None:
        lines += [
            "",
            "### Binned by |alpha_pred-alpha_soft| (quantiles)",
            "",
            _df_to_markdown_table(bins_tbl),
        ]

    lines += [
        "",
        "## C) Oracle / reconstructed gate forecasts (test)",
        "",
        _df_to_markdown_table(oracle_tbl),
        "",
        _df_to_markdown_table(gap_tbl),
        "",
        _df_to_markdown_table(
            pd.DataFrame(
                {
                    "metric": [
                        "max_abs(yhat - yhat_gate_pred) (masked soft)",
                        "corr(yhat, yhat_gate_pred) (masked soft)",
                    ],
                    "value": [
                        _format_sci(recon_max_abs),
                        f"{recon_corr:.6f}" if np.isfinite(recon_corr) else "NA",
                    ],
                }
            )
        ),
    ]
    if recon_flag:
        lines += [
            "",
            "> NOTE: Reconstruction mismatch is non-trivial; the assumed update form may differ from model mechanics. Oracle comparisons should be interpreted as a proxy.",
        ]

    if chart_paths:
        lines += ["", "## D) Charts", ""]
        lines += [f"- `{p.name}`" for p in chart_paths]
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, chart_paths


def _event_window_mean(
    series: pd.Series, event_idx: pd.Index, window: int
) -> pd.Series:
    """Mean path around events, indexed by [-window, +window]."""
    s = series.dropna()
    locs = s.index.get_indexer(event_idx)
    locs = locs[locs >= 0]
    mat = []
    for k in locs:
        a = k - window
        b = k + window + 1
        if a < 0 or b > len(s):
            continue
        mat.append(s.iloc[a:b].values)
    rel = np.arange(-window, window + 1)
    if not mat:
        return pd.Series([np.nan] * len(rel), index=rel)
    mat = np.vstack(mat)
    return pd.Series(mat.mean(axis=0), index=rel)


def _plot_forecast_panel_stes_vs_xgb(
    df: pd.DataFrame, out_dir: Path, fname: str
) -> None:
    """Plot target, forecasts, and STES-vs-XGB loss differential."""
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1, ax2 = axes

    ax1.plot(df.index, df["y"].values, lw=1.0, label="y (target)")
    ax1.plot(df.index, df["yhat_stes"].values, lw=1.0, label="STES_EAESE forecast")
    ax1.plot(df.index, df["yhat_xgb"].values, lw=1.0, label="XGBSTES_BASE forecast")
    ax1.set_title("SPY: target and one-step forecasts (test / OOS sample)")
    ax1.set_ylabel("variance (proxy)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(df.index, df["D"].values, lw=1.0)
    ax2.axhline(0.0, lw=1.0, alpha=0.7)
    ax2.set_title(
        r"Loss differential $D_t=(y-\hat y^{STES})^2-(y-\hat y^{XGB})^2$  (positive $\Rightarrow$ XGB better)"
    )
    ax2.set_ylabel(r"$D_t$ (squared-error diff)")
    ax2.set_xlabel("date")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_event_paths(
    paths: dict[str, pd.Series],
    title: str,
    out_dir: Path,
    fname: str,
    *,
    xlabel: str = "event time k (days relative to forecast date t; k=0 is the D_t date)",
    ylabel: str | None = None,
    note: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, s in paths.items():
        ax.plot(s.index, s.values, lw=2.0, label=name)

    ax.axvline(0.0, lw=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if note is not None:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_bar_series(
    series: pd.Series,
    title: str,
    out_dir: Path,
    fname: str,
    top_k: int = 15,
    *,
    xlabel: str | None = None,
    note: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    s = series.copy().dropna()
    if s.empty:
        return

    if len(s) > top_k:
        s = s.reindex(s.abs().sort_values(ascending=False).index[:top_k])
    s = s.sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(s) + 1.5)))
    ax.barh(s.index.astype(str), s.values)

    ax.axvline(0.0, lw=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("feature")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)

    if note is not None:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_gate_panel_stes_vs_xgb(
    *,
    alpha_t: pd.Series,
    r: pd.Series,
    D_t: pd.Series | None,
    out_dir: Path,
    prefix: str,
    title_prefix: str,
    q_bins: int = 20,
) -> Path:
    """2x2 panel: alpha, alpha hist, alpha vs |r|, and D_t vs |r|."""
    _ensure_dir(out_dir)

    # Here, alpha_t and r_t are contemporaneous with the forecast made at time t (about t+1).
    alpha_now = alpha_t
    r_now = r

    idx = alpha_now.index.intersection(r_now.index)
    if D_t is not None:
        idx = idx.intersection(D_t.index)

    alpha_now = alpha_now.reindex(idx)
    r_now = r_now.reindex(idx)
    if D_t is not None:
        D_t = D_t.reindex(idx)

    df = pd.DataFrame({"alpha": alpha_now, "r": r_now}, index=idx)
    df["abs_r"] = df["r"].abs()
    if D_t is not None:
        df["D_t"] = D_t
    df = df.dropna(subset=["alpha", "r", "abs_r"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.plot(df.index, df["alpha"].values, lw=1.0)
    ax00.set_title(f"{title_prefix}: $\\alpha_t$ time series")
    ax00.set_ylabel("$\\alpha_t$")
    ax00.set_xlabel("date")
    ax00.grid(True, alpha=0.25)

    ax01.hist(df["alpha"].values, bins=50, alpha=0.9)
    ax01.set_title(f"{title_prefix}: distribution of $\\alpha_t$")
    ax01.set_xlabel("$\\alpha_t$")
    ax01.set_ylabel("count")
    ax01.grid(True, alpha=0.25)

    try:
        tmp = df[["alpha", "abs_r"]].copy()
        tmp["bin"] = pd.qcut(tmp["abs_r"], q=q_bins, duplicates="drop")
        grp = tmp.groupby("bin", observed=True)["alpha"].mean()
        ax10.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
        ax10.set_title(f"{title_prefix}: binned mean $\\alpha_t$ vs $|r_t|$ quantiles")
        ax10.set_xlabel("quantile bin of $|r_t|$ (low → high)")
        ax10.set_ylabel("mean $\\alpha_t$")
        ax10.grid(True, alpha=0.25)
    except Exception as e:
        ax10.text(
            0.05,
            0.5,
            f"binned plot failed:\n{e}",
            transform=ax10.transAxes,
            fontsize=10,
        )
        ax10.set_axis_off()

    if D_t is not None and "D_t" in df.columns:
        try:
            tmp = df[["D_t", "abs_r"]].dropna().copy()
            tmp["bin"] = pd.qcut(tmp["abs_r"], q=q_bins, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)["D_t"].mean()
            ax11.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
            ax11.axhline(0.0, lw=1.0)
            ax11.set_title(
                "Binned mean $D_t$ vs $|r_t|$ quantiles\n($D_t>0$ means XGB better than STES)"
            )
            ax11.set_xlabel("quantile bin of $|r_t|$ (low → high)")
            ax11.set_ylabel("mean $D_t$")
            ax11.grid(True, alpha=0.25)
        except Exception as e:
            ax11.text(
                0.05,
                0.5,
                f"D_t binned plot failed:\n{e}",
                transform=ax11.transAxes,
                fontsize=10,
            )
            ax11.set_axis_off()
    else:
        ax11.text(
            0.05,
            0.55,
            "No $D_t$ provided.\n\nPass D_t = (y-ŷ_STES)^2 - (y-ŷ_XGB)^2\n(aligned on the same date index)\n\nto show when XGB helps.",
            transform=ax11.transAxes,
            fontsize=10,
        )
        ax11.set_axis_off()

    fig.tight_layout()
    out_path = out_dir / f"{prefix}_gate_helpfulness_2x2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze_spy_stes_vs_xgb_stes(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    stes_variant: str = "STES_EAESE",
    xgb_params: dict,
    seeds: list[int],
    q: float = 0.10,
    window: int = 10,
    out_dir: Path,
):
    """
    Mirrors Part 1 style:
    - Fit STES_EAESE and XGBSTES_BASE on same IS
    - Pick best seed for XGB (by OOS RMSE) or average over seeds (you decide)
    - Build per-date D_t = loss_STES - loss_XGB (positive => XGB better)
    - Define win/lose events by top/bottom q quantiles of D_t
    - Produce event-study plots and summary tables
    """
    logger.info(f"Starting STES vs XGBSTES analysis (baseline: {stes_variant})")
    out_dir.mkdir(exist_ok=True)

    # Use a fixed train/test split for this analysis
    train_sl = slice(SPY_IS_INDEX, SPY_OS_INDEX)
    test_sl = slice(SPY_OS_INDEX, len(y))

    # --- 1. Get STES baseline predictions ---
    (
        idx_stes_te,
        y_hat_stes_te,
        alpha_stes_te,
        _,
        _,
        _,
    ) = _fit_predict_oos(
        variant=stes_variant,
        X=X,
        y=y,
        r=r,
        train_slice=train_sl,
        test_slice=test_sl,
        seed=seeds[0],  # STES is deterministic, one seed is enough
        return_alpha=True,
    )

    # --- 2. Get XGBSTES predictions (best seed or average) ---
    xgb_preds = []
    xgb_alphas = []
    for seed in seeds:
        idx_xgb_te, y_hat_xgb_seed_te, alpha_xgb_seed_te, _, _, _ = _fit_predict_oos(
            variant="XGBSTES_BASE",
            X=X,
            y=y,
            r=r,
            train_slice=train_sl,
            test_slice=test_sl,
            seed=seed,
            return_alpha=True,
            xgb_params=xgb_params,
        )
        xgb_preds.append(pd.Series(y_hat_xgb_seed_te, index=idx_xgb_te))
        xgb_alphas.append(pd.Series(alpha_xgb_seed_te, index=idx_xgb_te))

    # Averaging predictions over seeds
    y_hat_xgb_s = pd.concat(xgb_preds, axis=1).mean(axis=1)
    alpha_xgb_s = pd.concat(xgb_alphas, axis=1).mean(axis=1)

    idx_te = idx_stes_te.intersection(y_hat_xgb_s.index)
    y_true = y.loc[idx_te].values
    y_hat_stes_te = pd.Series(y_hat_stes_te, index=idx_stes_te).loc[idx_te].to_numpy()
    alpha_stes_te = pd.Series(alpha_stes_te, index=idx_stes_te).loc[idx_te].to_numpy()
    y_hat_xgb = y_hat_xgb_s.loc[idx_te].to_numpy()
    alpha_xgb = alpha_xgb_s.loc[idx_te].to_numpy()

    # --- 3. Compute per-date loss differential ---
    loss_stes = (y_true - y_hat_stes_te) ** 2
    loss_xgb = (y_true - y_hat_xgb) ** 2
    D = loss_stes - loss_xgb  # Positive => XGB is better

    # --- 4. Build timeseries DataFrame for analysis ---
    df = pd.DataFrame(
        {
            "y": y_true,
            "yhat_stes": y_hat_stes_te,
            "yhat_xgb": y_hat_xgb,
            "loss_stes": loss_stes,
            "loss_xgb": loss_xgb,
            "D": D,
            "r": r.loc[idx_te],
            "abs_r": np.abs(r.loc[idx_te]),
            "r2": r.loc[idx_te] ** 2,
            "r2_lag": (r.loc[idx_te] ** 2).shift(1),
            "alpha_stes_lag": pd.Series(alpha_stes_te, index=idx_te).shift(1),
            "alpha_xgb_lag": pd.Series(alpha_xgb, index=idx_te).shift(1),
        },
        index=idx_te,
    )
    df.to_csv(out_dir / "spy_stes_vs_xgb_timeseries.csv")
    logger.info(
        f"Saved timeseries analysis to {out_dir / 'spy_stes_vs_xgb_timeseries.csv'}"
    )

    df["alpha_stes"] = pd.Series(alpha_stes_te, index=idx_te)
    df["alpha_xgb"] = pd.Series(alpha_xgb, index=idx_te)
    df["delta_alpha"] = df["alpha_xgb"] - df["alpha_stes"]

    # State proxy: the variance level v_t is the *previous* forecast made at t-1 about t.
    df["v_stes_state"] = df["yhat_stes"].shift(1)
    df["u"] = df["r2"] - df["v_stes_state"]

    df = df.dropna()

    # --- 5. Event definitions (wins/losses by D) ---
    hi = df["D"].quantile(1.0 - q)
    lo = df["D"].quantile(q)
    win_idx = df.index[df["D"] >= hi]  # XGB wins
    lose_idx = df.index[df["D"] <= lo]  # XGB loses

    event_def = (
        f"Event definition (OOS): WIN = top {int(q*100)}% of D_t (XGB better), "
        f"LOSE = bottom {int(q*100)}% of D_t (XGB worse)."
    )

    def _summ(mask):
        sub = df.loc[mask]
        return {
            "n": int(len(sub)),
            "mean_D": float(sub["D"].mean()),
            "mean_loss_stes": float(sub["loss_stes"].mean()),
            "mean_loss_xgb": float(sub["loss_xgb"].mean()),
            "mean_alpha_stes": float(sub["alpha_stes"].mean()),
            "mean_alpha_xgb": float(sub["alpha_xgb"].mean()),
            "mean_delta_alpha": float(sub["delta_alpha"].mean()),
            "frac_delta_alpha_pos": float((sub["delta_alpha"] > 0).mean()),
            "mean_u": float(sub["u"].mean()),
            "frac_u_pos": float((sub["u"] > 0).mean()),
            "mean_abs_r": float(sub["abs_r"].mean()),
        }

    win_stats = _summ(df.index.isin(win_idx))
    lose_stats = _summ(df.index.isin(lose_idx))
    pd.DataFrame([win_stats, lose_stats], index=["win", "lose"]).to_csv(
        out_dir / "spy_stes_vs_xgb_event_summary.csv"
    )

    # --- 6. Plots (reuse Part 1 chart style) ---
    _plot_forecast_panel_stes_vs_xgb(
        df[["y", "yhat_stes", "yhat_xgb", "D"]],
        out_dir,
        "spy_stes_vs_xgb_forecasts_and_D.png",
    )

    paths_alpha = {
        "alpha_XGB (wins)": _event_window_mean(df["alpha_xgb"], win_idx, window),
        "alpha_XGB (losses)": _event_window_mean(df["alpha_xgb"], lose_idx, window),
        "alpha_STES (wins)": _event_window_mean(df["alpha_stes"], win_idx, window),
        "alpha_STES (losses)": _event_window_mean(df["alpha_stes"], lose_idx, window),
    }
    _plot_event_paths(
        paths_alpha,
        r"Event study: gate $\alpha_t$ around WIN vs LOSE dates",
        out_dir,
        "spy_event_alpha_stes_vs_xgb.png",
        ylabel=r"mean $\alpha_{t+k}$",
        note=event_def,
    )

    paths_absr = {
        "|r| (wins)": _event_window_mean(df["abs_r"], win_idx, window),
        "|r| (losses)": _event_window_mean(df["abs_r"], lose_idx, window),
    }
    _plot_event_paths(
        paths_absr,
        r"Event study: $|r_t|$ around WIN vs LOSE dates",
        out_dir,
        "spy_event_absr_stes_vs_xgb.png",
        ylabel=r"mean $|r_{t+k}|$",
        note=event_def,
    )

    paths_u = {
        "u_t (wins)": _event_window_mean(df["u"], win_idx, window),
        "u_t (losses)": _event_window_mean(df["u"], lose_idx, window),
    }
    _plot_event_paths(
        paths_u,
        r"Event study: innovation proxy $u_t=r_t^2-\hat v_t^{STES}$ around WIN vs LOSE dates",
        out_dir,
        "spy_event_u_stes_vs_xgb.png",
        ylabel=r"mean $u_{t+k}$",
        note=event_def,
    )

    # Mechanism view: delta alpha vs innovation proxy
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(
        df["u"].values,
        df["delta_alpha"].values,
        s=6,
        alpha=0.15,
        label="all",
        color="#999999",
    )
    ax.scatter(
        df.loc[win_idx, "u"].values,
        df.loc[win_idx, "delta_alpha"].values,
        s=10,
        alpha=0.35,
        label="wins",
        color="#2ca02c",
    )
    ax.scatter(
        df.loc[lose_idx, "u"].values,
        df.loc[lose_idx, "delta_alpha"].values,
        s=10,
        alpha=0.35,
        label="losses",
        color="#d62728",
    )
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    ax.set_title(
        r"Mechanism view: $\Delta\alpha_t=\alpha^{XGB}_t-\alpha^{STES}_t$ vs $u_t$"
        + f"\n(WIN/LOSE are top/bottom {int(q*100)}% of D_t in the OOS sample)"
    )
    ax.set_xlabel("$u_t=r_t^2-\\hat v_t^{STES}$")
    ax.set_ylabel("$\\Delta\\alpha_t=\\alpha^{XGB}_t-\\alpha^{STES}_t$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "spy_delta_alpha_vs_u_stes_vs_xgb.png", dpi=150)
    plt.close(fig)

    # Gate panels
    _plot_gate_panel_stes_vs_xgb(
        alpha_t=df["alpha_stes"],
        r=df["r"],
        D_t=df["D"],
        out_dir=out_dir,
        prefix="spy_oos_stes",
        title_prefix="SPY STES_EAESE",
    )
    _plot_gate_panel_stes_vs_xgb(
        alpha_t=df["alpha_xgb"],
        r=df["r"],
        D_t=df["D"],
        out_dir=out_dir,
        prefix="spy_oos_xgb",
        title_prefix="SPY XGBSTES_BASE",
    )

    # Optional: STES gate-score contribution diagnostics (linear gate only)
    try:
        cols_stes = select_variant_columns(X, stes_variant) or ["const"]
        X_sel = X[cols_stes]
        model_stes = STESModel(random_state=seeds[0])
        model_stes.fit(
            X_sel.iloc[train_sl],
            y.iloc[train_sl],
            returns=r.iloc[train_sl],
            start_index=0,
            end_index=len(X_sel.iloc[train_sl]),
        )
        beta = np.asarray(model_stes.params, dtype=float).reshape(-1)
        X_aligned = X_sel.loc[df.index]
        contrib = X_aligned.mul(beta, axis=1)
        contrib_win = contrib.loc[win_idx].mean().sort_values(ascending=False)
        contrib_lose = contrib.loc[lose_idx].mean().sort_values(ascending=False)
        contrib_diff = (contrib_win - contrib_lose).sort_values(ascending=False)

        pd.DataFrame(
            {
                "win_mean": contrib_win,
                "lose_mean": contrib_lose,
                "win_minus_lose": contrib_win - contrib_lose,
            }
        ).to_csv(out_dir / "spy_stes_gate_contrib_summary__xgb_win_minus_lose.csv")

        _plot_bar_series(
            contrib_diff,
            "STES gate-score contribution differences: XGB WIN minus LOSE",
            out_dir,
            "spy_stes_gate_contrib_xgb_win_minus_lose.png",
            top_k=15,
            xlabel=r"mean score contrib diff  E[c_{j,t}|WIN] - E[c_{j,t}|LOSE]",
            note=event_def,
        )
    except Exception as e:
        logger.exception(f"STES contribution diagnostics failed: {e}")

    logger.info(f"Saved STES vs XGBSTES charts to {out_dir.resolve()}")


def _mean_oos_pred_alpha(
    *,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    train_slice: slice,
    test_slice: slice,
    seeds: list[int],
    xgb_params: dict,
) -> tuple[pd.Series, pd.Series]:
    """Return mean OOS prediction and alpha series across seeds (same index)."""
    use_seeds = seeds if variant.startswith("XGBSTES_") else [seeds[0]]
    preds: list[pd.Series] = []
    alphas: list[pd.Series] = []
    for seed in use_seeds:
        idx_te, y_hat_te, alpha_te, _, _, _ = _fit_predict_oos(
            variant=variant,
            X=X,
            y=y,
            r=r,
            train_slice=train_slice,
            test_slice=test_slice,
            seed=seed,
            return_alpha=True,
            xgb_params=xgb_params,
        )
        preds.append(pd.Series(y_hat_te, index=idx_te))
        alphas.append(pd.Series(alpha_te, index=idx_te))

    y_hat_mean = pd.concat(preds, axis=1).mean(axis=1)
    alpha_mean = pd.concat(alphas, axis=1).mean(axis=1)
    idx = y_hat_mean.index.intersection(alpha_mean.index)
    return y_hat_mean.loc[idx], alpha_mean.loc[idx]


def _trimmed_rmse(y_true: np.ndarray, y_pred: np.ndarray, *, trim_frac: float) -> float:
    if trim_frac <= 0.0:
        return float(rmse(y_true, y_pred))
    if trim_frac >= 1.0:
        return np.nan
    se = np.asarray((y_true - y_pred) ** 2, dtype=float)
    se = se[np.isfinite(se)]
    if se.size == 0:
        return np.nan
    k = int(np.floor((1.0 - trim_frac) * se.size))
    k = max(1, min(k, se.size))
    se_sorted = np.sort(se)
    return float(np.sqrt(np.mean(se_sorted[:k])))


def generate_spy_blog_tables_4_to_6(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    seeds: list[int],
    out_dir: Path,
    xgb_params: dict,
    trim_frac: float = 0.01,
) -> None:
    """Generate the last 3 tables in the Part 2 blog post (Tables 4–6)."""
    _ensure_dir(out_dir)
    train_sl = slice(SPY_IS_INDEX, SPY_OS_INDEX)
    test_sl = slice(SPY_OS_INDEX, len(y))

    variants = ["STES_EAESE", "XGBSTES_BASE", "XGBSTES_BASE_MONO_HUBER"]
    preds: dict[str, pd.Series] = {}
    alphas: dict[str, pd.Series] = {}
    for variant in variants:
        y_hat_s, alpha_s = _mean_oos_pred_alpha(
            variant=variant,
            X=X,
            y=y,
            r=r,
            train_slice=train_sl,
            test_slice=test_sl,
            seeds=seeds,
            xgb_params=xgb_params,
        )
        preds[variant] = y_hat_s
        alphas[variant] = alpha_s

    idx_common = None
    for s in preds.values():
        idx_common = s.index if idx_common is None else idx_common.intersection(s.index)
    if idx_common is None or len(idx_common) == 0:
        logger.warning("No overlapping OOS index for Table 4–6 diagnostics; skipping.")
        return

    y_true_s = y.loc[idx_common]
    r_s = r.loc[idx_common]

    # -------------------------
    # Table 4: error diagnostics
    # -------------------------
    rows4 = []
    for variant in variants:
        y_pred_s = preds[variant].loc[idx_common]
        err = (y_true_s.values - y_pred_s.values).astype(float)
        abs_err = np.abs(err)
        q50, q80, q90, q95, q99 = np.quantile(abs_err, [0.50, 0.80, 0.90, 0.95, 0.99])
        rows4.append(
            {
                "variant": variant,
                "p50": float(q50),
                "p80": float(q80),
                "p90": float(q90),
                "p95": float(q95),
                "p99": float(q99),
                "max_abs_err": float(np.max(abs_err)),
                "trimmed_rmse_1pct": _trimmed_rmse(
                    y_true_s.values, y_pred_s.values, trim_frac=trim_frac
                ),
            }
        )
    tbl4 = pd.DataFrame(rows4)
    tbl4.to_csv(out_dir / "spy_table4_error_quantiles.csv", index=False)

    blog4 = pd.DataFrame(
        {
            "Model": tbl4["variant"],
            "p50": tbl4["p50"].map(_format_sci),
            "p80": tbl4["p80"].map(_format_sci),
            "p90": tbl4["p90"].map(_format_sci),
            "p95": tbl4["p95"].map(_format_sci),
            "p99": tbl4["p99"].map(_format_sci),
            "max": tbl4["max_abs_err"].map(_format_sci),
            "Trimmed RMSE (1%)": tbl4["trimmed_rmse_1pct"].map(_format_sci),
        }
    )
    (out_dir / "spy_table4_error_quantiles_blog.md").write_text(
        "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
        + _df_to_markdown_table(blog4),
        encoding="utf-8",
    )

    # -------------------------
    # Table 5: regime metrics
    # -------------------------
    # Regime proxy: realized variance (y) quantiles (low/med/high)
    y_vals = y_true_s.values.astype(float)
    q1, q2 = np.quantile(y_vals, [1.0 / 3.0, 2.0 / 3.0])
    low = y_vals <= q1
    mid = (y_vals > q1) & (y_vals <= q2)
    high = y_vals > q2

    rows5 = []
    for variant in variants:
        y_pred = preds[variant].loc[idx_common].values.astype(float)
        rows5.append(
            {
                "variant": variant,
                "low_rmse": float(rmse(y_vals[low], y_pred[low])),
                "mid_rmse": float(rmse(y_vals[mid], y_pred[mid])),
                "high_rmse": float(rmse(y_vals[high], y_pred[high])),
                "low_medae": float(medae(y_vals[low], y_pred[low])),
                "high_medae": float(medae(y_vals[high], y_pred[high])),
            }
        )
    tbl5 = pd.DataFrame(rows5)
    tbl5.to_csv(out_dir / "spy_table5_regime_metrics.csv", index=False)

    blog5 = pd.DataFrame(
        {
            "Model": tbl5["variant"],
            "Low vol RMSE": tbl5["low_rmse"].map(_format_sci),
            "Mid vol RMSE": tbl5["mid_rmse"].map(_format_sci),
            "High vol RMSE": tbl5["high_rmse"].map(_format_sci),
            "Low vol MedAE": tbl5["low_medae"].map(_format_sci),
            "High vol MedAE": tbl5["high_medae"].map(_format_sci),
        }
    )
    (out_dir / "spy_table5_regime_metrics_blog.md").write_text(
        "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
        + _df_to_markdown_table(blog5),
        encoding="utf-8",
    )

    # -------------------------
    # Table 6: gate summary stats
    # -------------------------
    rows6 = []
    for variant in variants:
        alpha_s = alphas[variant].loc[idx_common].astype(float)
        abs_r_s = np.abs(r_s.loc[idx_common].astype(float))
        mask = np.isfinite(alpha_s.values) & np.isfinite(abs_r_s.values)
        a = alpha_s.values[mask]
        ar = abs_r_s.values[mask]
        rows6.append(
            {
                "variant": variant,
                "alpha_mean": float(np.mean(a)) if a.size else np.nan,
                "alpha_std": float(np.std(a)) if a.size else np.nan,
                "pct_alpha_lt_0_1": float(np.mean(a < 0.1)) if a.size else np.nan,
                "pct_alpha_gt_0_9": float(np.mean(a > 0.9)) if a.size else np.nan,
                "corr_alpha_abs_r": (
                    float(np.corrcoef(a, ar)[0, 1])
                    if a.size >= 2 and np.std(ar) > 0 and np.std(a) > 0
                    else np.nan
                ),
            }
        )
    tbl6 = pd.DataFrame(rows6)
    tbl6.to_csv(out_dir / "spy_table6_gate_summary.csv", index=False)

    def _fmt_pct(x: float) -> str:
        if x is None or not np.isfinite(x):
            return "NA"
        return f"{100.0 * x:.1f}%"

    blog6 = pd.DataFrame(
        {
            "Model": tbl6["variant"],
            "alpha mean": tbl6["alpha_mean"].map(
                lambda v: f"{v:.3f}" if np.isfinite(v) else "NA"
            ),
            "alpha std": tbl6["alpha_std"].map(
                lambda v: f"{v:.3f}" if np.isfinite(v) else "NA"
            ),
            "pct alpha < 0.1": tbl6["pct_alpha_lt_0_1"].map(_fmt_pct),
            "pct alpha > 0.9": tbl6["pct_alpha_gt_0_9"].map(_fmt_pct),
        }
    )
    (out_dir / "spy_table6_gate_summary_blog.md").write_text(
        "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
        + _df_to_markdown_table(blog6),
        encoding="utf-8",
    )

    logger.info(f"Wrote blog Tables 4–6 to {out_dir.resolve()}")


def build_spy_spec(lags: int) -> VolDatasetSpec:
    """Wide spec for SPY using Tiingo source (raw/abs/squared + next-day target)."""
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": lags,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": lags,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": lags,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={
            "source": "tiingo",
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1.0,
        },
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[SPY_TICKER]),
        time=TimeSpec(
            start=SPY_START, end=SPY_END, calendar="XNYS", grid="B", asof=None
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def build_spy_dataset(ctx, n_lags: int = 5):
    spec = build_spy_spec(n_lags)
    X, y, returns, catalog = build_vol_dataset(ctx, spec, persist=False)

    X1 = X.xs("SPY", level="entity_id").sort_index().copy()
    if "const" not in X1.columns:
        X1["const"] = 1.0
    y1 = y.xs("SPY", level="entity_id").sort_index()
    r1 = returns.xs("SPY", level="entity_id").sort_index()

    # build date array of len(y)+1 so evaluate_model can attach date[1:] -> y
    date = pd.DatetimeIndex([X1.index[0] - pd.Timedelta(days=1)] + list(X1.index))

    # Use the realized wide feature columns (FeatureRequest has no `.name`).
    feature_names = list(X1.columns)
    return X1, y1, r1, date, feature_names


def evaluate_models(
    X: pd.DataFrame, y: pd.Series, os_index: int, returns: pd.Series | None = None
):
    models = {
        "ES": ESModel(),
        "STES": STESModel(),
    }
    models["XGBSTES_BASE"] = _make_xgb_stes_model(
        seed=0,
        params_flat={"max_depth": 4, "learning_rate": 0.1, "num_boost_round": 50},
    )
    results = {}
    for name, model in models.items():
        logger.info(f"Fitting {name}...")
        try:
            # prefer passing returns and start/end indices when supported by the model
            X_is, X_os = _scale_train_test(
                X, slice(0, os_index), slice(os_index, len(X))
            )
            y_is = y.iloc[:os_index]
            r_is = returns.iloc[:os_index] if returns is not None else None
            try:
                model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
            except TypeError:
                # fallback for models that don't accept returns/start_index
                model.fit(X_is, y_is)
            r_os = returns.iloc[os_index:] if returns is not None else None
            try:
                y_pred = model.predict(X_os, returns=r_os)
            except TypeError:
                y_pred = model.predict(X_os)

            y_true = np.asarray(y.iloc[os_index:].values, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            keep = np.isfinite(y_pred_arr)
            rmse = np.sqrt(np.mean((y_true[keep] - y_pred_arr[keep]) ** 2))
            qlike = metrics.qlike(y_true[keep], y_pred_arr[keep])
            results[name] = {"rmse": rmse, "qlike": qlike}
            logger.info(f" OK — RMSE={rmse:.6f}, QLIKE={qlike:.6f}")
        except Exception as e:
            logger.warning(f" FAIL ({e})")
            results[name] = None
    return results


def random_cv_tune_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    n_iter: int = 80,
    n_splits: int = 3,
    cv_seeds: list[int] | None = None,
):
    """Randomized search using walk-forward splits and _fit_predict_oos.

    Args:
        X: Wide feature matrix
        y: Target series
        r: Returns series
        n_iter: Number of parameter samples to try.
        n_splits: Number of walk-forward folds for cross-validation.
        cv_seeds: Seeds to average over for each fold (e.g., [0, 1, 2]).
    """
    if cv_seeds is None:
        cv_seeds = [0]

    logger.info(
        f"Starting RandomizedSearchCV tuning (n_iter={n_iter}, n_splits={n_splits}, seeds={cv_seeds})"
    )

    param_distributions = {
        "num_boost_round": [1, 5, 10, 20],
        "max_depth": randint(1, 5),
        "learning_rate": loguniform(0.01, 10),
        "reg_lambda": loguniform(0.1, 10.0),
        "colsample_bytree": uniform(0.1, 0.6),
        "colsample_bylevel": uniform(0.1, 0.6),
        "colsample_bynode": uniform(0.1, 0.6),
    }

    param_list = list(
        ParameterSampler(param_distributions, n_iter=n_iter, random_state=0)
    )

    # Use walk-forward splits consistent with main evaluation
    splits = walk_forward_splits(
        n=len(y),
        mode="expanding",
        train_size=2000,
        val_size=252,
        step_size=252,
        max_folds=n_splits,
    )

    best_score_rmse = float("inf")
    best_params = None
    best_score_mae = float("inf")
    best_score_medae = float("inf")

    for params in param_list:
        fold_rmses, fold_maes, fold_medaes = [], [], []
        for train_sl, val_sl in splits:
            seed_rmses, seed_maes, seed_medaes = [], [], []
            for seed in cv_seeds:
                try:
                    res = _fit_predict_oos(
                        variant="XGBSTES_BASE",
                        X=X,
                        y=y,
                        r=r,
                        train_slice=train_sl,
                        test_slice=val_sl,
                        seed=seed,
                        xgb_params=params,
                    )
                    idx_te, y_hat, _, _ = res[0], res[1], res[2], res[3]
                    y_true = y.loc[idx_te].values

                    seed_rmses.append(rmse(y_true, y_hat))
                    seed_maes.append(mae(y_true, y_hat))
                    seed_medaes.append(medae(y_true, y_hat))

                except Exception as e:
                    logger.exception(
                        f"CV tuning failed for params {params} and seed {seed}: {e}"
                    )
                    seed_rmses.append(np.inf)
                    seed_maes.append(np.inf)
                    seed_medaes.append(np.inf)

            # Average metrics across seeds for this fold
            fold_rmses.append(np.mean(seed_rmses))
            fold_maes.append(np.mean(seed_maes))
            fold_medaes.append(np.mean(seed_medaes))

        # Objective is the mean RMSE across all folds
        mean_rmse = float(np.mean(fold_rmses))
        mean_mae = float(np.mean(fold_maes))
        mean_medae = float(np.mean(fold_medaes))

        if mean_rmse < best_score_rmse:
            best_score_rmse = mean_rmse
            best_params = params
            best_score_mae = mean_mae
            best_score_medae = mean_medae
            logger.info(
                f"New best params: {params} -> RMSE={mean_rmse:.6f}, MAE={mean_mae:.6f}, MedAE={mean_medae:.6f}"
            )

    logger.info(f"RandomizedSearchCV best RMSE={best_score_rmse:.6f}")
    return best_params, best_score_rmse, best_score_mae, best_score_medae


def xgb_stes_optuna_objective(trial, X, y, r, n_splits=3, cv_seeds=None):
    """Optuna objective (returns mean RMSE to be minimized)."""
    if optuna is None:
        raise RuntimeError("optuna not available")

    if cv_seeds is None:
        cv_seeds = [0]

    param = {
        "verbosity": 0,
        "num_boost_round": trial.suggest_int("num_boost_round", 1, 20),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.6),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.6),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.6),
    }

    splits = walk_forward_splits(
        n=len(y),
        mode="expanding",
        train_size=2000,
        val_size=252,
        step_size=252,
        max_folds=n_splits,
    )

    fold_rmses = []
    for train_sl, val_sl in splits:
        seed_rmses = []
        for seed in cv_seeds:
            try:
                res = _fit_predict_oos(
                    variant="XGBSTES_BASE",
                    X=X,
                    y=y,
                    r=r,
                    train_slice=train_sl,
                    test_slice=val_sl,
                    seed=seed,
                    xgb_params=param,
                )
                idx_te, y_hat, _, _ = res[0], res[1], res[2], res[3]
                y_true = y.loc[idx_te].values
                seed_rmses.append(rmse(y_true, y_hat))
            except Exception as e:
                logger.debug(
                    f"Optuna trial failed for params {param} and seed {seed}: {e}"
                )
                seed_rmses.append(np.inf)
        fold_rmses.append(np.mean(seed_rmses))

    return float(np.mean(fold_rmses))


def fit_and_score(variant: str, X: pd.DataFrame, y: pd.Series, r: pd.Series) -> float:
    """Fit model on in-sample, compute OOS RMSE (mirrors Part1)."""
    if len(y) <= OS_INDEX:
        raise ValueError(f"Insufficient rows for slicing: len(y)={len(y)}")

    cols = select_variant_columns(X, variant)
    if not cols:
        cols = ["const"]

    X_sel = X[cols]
    X_is, X_os = _scale_train_test(
        X_sel, slice(IS_INDEX, OS_INDEX), slice(OS_INDEX, len(X_sel))
    )
    y_is, y_os = y.iloc[IS_INDEX:OS_INDEX], y.iloc[OS_INDEX:]
    r_is, r_os = r.iloc[IS_INDEX:OS_INDEX], r.iloc[OS_INDEX:]

    if variant == "ES":
        model = ESModel()
    elif variant == "XGBSTES_BASE":
        model = _make_xgb_stes_model(seed=0, params_flat=DEFAULT_XGBOOST_PARAMS)
    else:
        model = STESModel()

    # Pass returns and index bounds where supported
    model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
    # predict on OOS slice using provided returns
    y_hat = model.predict(X_os, returns=r_os)
    y_hat_arr = np.asarray(y_hat, dtype=float)
    y_os_arr = np.asarray(y_os.values, dtype=float)
    keep = np.isfinite(y_hat_arr)
    return float(np.sqrt(np.mean((y_os_arr[keep] - y_hat_arr[keep]) ** 2)))


def fit_and_score_metrics(
    variant: str, X: pd.DataFrame, y: pd.Series, r: pd.Series
) -> dict[str, float]:
    """Fit model on in-sample, compute OOS RMSE/MAE/MedAE (simulated experiment)."""
    if len(y) <= OS_INDEX:
        raise ValueError(f"Insufficient rows for slicing: len(y)={len(y)}")

    cols = select_variant_columns(X, variant)
    if not cols:
        cols = ["const"]

    X_sel = X[cols]
    X_is, X_os = _scale_train_test(
        X_sel, slice(IS_INDEX, OS_INDEX), slice(OS_INDEX, len(X_sel))
    )
    y_is, y_os = y.iloc[IS_INDEX:OS_INDEX], y.iloc[OS_INDEX:]
    r_is, r_os = r.iloc[IS_INDEX:OS_INDEX], r.iloc[OS_INDEX:]

    if variant == "ES":
        model = ESModel()
    elif variant == "XGBSTES_BASE":
        model = _make_xgb_stes_model(seed=0, params_flat=DEFAULT_XGBOOST_PARAMS)
    else:
        model = STESModel()

    model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
    y_hat = model.predict(X_os, returns=r_os)

    y_hat_arr = np.asarray(y_hat, dtype=float)
    y_os_arr = np.asarray(y_os.values, dtype=float)
    keep = np.isfinite(y_hat_arr)
    err = y_os_arr[keep] - y_hat_arr[keep]

    rmse_val = float(np.sqrt(np.mean(err**2)))
    mae_val = float(np.mean(np.abs(err)))
    medae_val = float(np.median(np.abs(err)))

    return {"rmse": rmse_val, "mae": mae_val, "medae": medae_val}


def _format_sci(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.2e}"


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table formatter (avoids requiring tabulate)."""
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isfinite(v):
                vals.append(str(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _make_blog_table(df: pd.DataFrame, *, title_col: str = "variant") -> pd.DataFrame:
    """Reduce eval tables to blog-friendly columns and scientific formatting."""
    out = pd.DataFrame(
        {
            "Model": df[title_col],
            "RMSE": df["rmse_mean"].map(_format_sci),
            "MAE": df["mae_mean"].map(_format_sci),
            "MedAE": df["medae_mean"].map(_format_sci),
        }
    )
    return out


def run_simulated_experiment(ctx, n_runs: int = N_RUNS) -> pd.DataFrame:
    logger.info("Starting simulated experiment")
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=n_runs)

    rmse_acc = {v: [] for v in VARIANTS}
    mae_acc = {v: [] for v in VARIANTS}
    medae_acc = {v: [] for v in VARIANTS}

    for i, seed in enumerate(seeds, start=1):
        if i % 5 == 0:
            logger.info(f"  Run {i}/{n_runs}")

        add_simulated_source(ctx, int(seed))
        spec = build_wide_spec(N_LAGS)
        try:
            X_wide, y, r, _ = build_wide_dataset(ctx, spec, entity_id=ENTITY)
        except Exception as e:
            logger.warning(f"Skipping run {i}: {e}")
            continue

        # Defensive: some pipelines may return a single column as a Series.
        if isinstance(X_wide, pd.Series):
            X_wide = X_wide.to_frame()

        for variant in VARIANTS:
            try:
                m = fit_and_score_metrics(variant, X_wide, y, r)
                rmse_acc[variant].append(m["rmse"])
                mae_acc[variant].append(m["mae"])
                medae_acc[variant].append(m["medae"])
            except Exception as e:
                # log failure details for debugging
                logger.exception(f"Variant {variant} failed on run {i}: {e}")
                continue

    logger.info("Simulated experiment complete")

    rows = []
    for variant in VARIANTS:
        rmses = rmse_acc[variant]
        maes = mae_acc[variant]
        medaes = medae_acc[variant]
        if rmses:
            rows.append(
                {
                    "variant": variant,
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "medae_mean": float(np.mean(medaes)),
                    "medae_std": float(np.std(medaes)),
                    "n": int(len(rmses)),
                }
            )
            logger.info(
                f"{variant:12s}: "
                f"RMSE={np.mean(rmses):.6f} ± {np.std(rmses):.6f} | "
                f"MAE={np.mean(maes):.6f} ± {np.std(maes):.6f} | "
                f"MedAE={np.mean(medaes):.6f} ± {np.std(medaes):.6f} "
                f"({len(rmses)}/{n_runs})"
            )
        else:
            rows.append(
                {
                    "variant": variant,
                    "rmse_mean": np.nan,
                    "rmse_std": np.nan,
                    "mae_mean": np.nan,
                    "mae_std": np.nan,
                    "medae_mean": np.nan,
                    "medae_std": np.nan,
                    "n": 0,
                }
            )
            logger.info(f"{variant:12s}: N/A (0/{n_runs})")

    return pd.DataFrame(rows).sort_values("rmse_mean")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Part 2 experiments and optional tuning for XGBSTES"
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Number of simulated runs to execute (default: N_RUNS in script)",
    )

    parser.add_argument(
        "--skip-spy",
        action="store_true",
        help="Skip SPY/Tiingo study (useful for quick debugging)",
    )
    parser.add_argument(
        "--spy-n-inits",
        type=int,
        default=SPY_N_INITS,
        help="Number of random initializations/seeds for SPY fixed-split table",
    )
    parser.add_argument(
        "--cv-seeds",
        type=int,
        default=10,
        help="Number of seeds to average over in walk-forward CV",
    )
    parser.add_argument(
        "--cv-max-folds",
        type=int,
        default=10,
        help="Max number of walk-forward folds (caps runtime)",
    )
    parser.add_argument(
        "--tune-sklearn",
        action="store_true",
        help="Run sklearn RandomizedSearchCV tuning for XGBSTES on SPY",
    )
    parser.add_argument(
        "--tune-sklearn-iter",
        type=int,
        default=80,
        help="Number of parameter samples for sklearn randomized search",
    )
    parser.add_argument(
        "--tune-optuna",
        action="store_true",
        help="Run Optuna tuning for XGBSTES on SPY",
    )
    parser.add_argument(
        "--tune-optuna-trials", type=int, default=100, help="Number of Optuna trials"
    )

    args = parser.parse_args()

    api_key = os.environ.get("TIINGO_API")

    cache_root = Path(_read_tiingo_cache_dir())
    file_cache = FileCacheBackend(cache_root)
    ctx = build_default_ctx(
        tiingo_api_key=api_key,
        tiingo_cache_backends=[file_cache],
        tiingo_cache_mode=_read_tiingo_cache_mode(),
    )

    out_dir = Path("outputs/volatility_forecast_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run simulated experiment (no Tiingo key required)
    sim_tbl = run_simulated_experiment(ctx, n_runs=args.n_runs)
    sim_tbl.to_csv(out_dir / "simulated_metrics.csv", index=False)
    sim_blog_tbl = _make_blog_table(sim_tbl)
    (out_dir / "simulated_metrics_blog.md").write_text(
        "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
        + _df_to_markdown_table(sim_blog_tbl),
        encoding="utf-8",
    )

    if args.skip_spy:
        logger.info("--skip-spy set; skipping SPY study.")
        return

    if not api_key:
        logger.warning(
            "TIINGO_API not set; set in .env to run Part 2 with real data. Skipping SPY study."
        )
        return

    # Build SPY wide dataset and run randomized-initialization averaging (match Part 1)
    try:
        X, y, returns, date, feature_names = build_spy_dataset(ctx, n_lags=N_LAGS)
    except Exception as e:
        logger.warning(
            "SPY via Tiingo failed. Ensure TIINGO_API is set in .env and valid.\n"
            f"Reason: {e}"
        )
        X = y = returns = date = feature_names = None

    # ------------------------------------------------------------------
    # Unified evaluation protocol across ALL models (for apples-to-apples)
    # ------------------------------------------------------------------
    if X is not None and len(y) > SPY_OS_INDEX:
        # Keep the same variants list you already use for tables
        variants = list(VARIANTS)

        # Use a small seed set by default for CV to keep runtime reasonable.
        # You can increase later.
        fixed_seeds = list(range(args.spy_n_inits))
        cv_seeds = list(range(min(args.cv_seeds, args.spy_n_inits)))

        logger.info("\n" + "=" * 80)
        logger.info("SPY (Fixed Split) — unified across ES/STES/XGBSTES")
        logger.info("=" * 80)
        fixed_tbl = evaluate_variants_fixed_split(
            X,
            y,
            returns,
            variants=variants,
            is_index=SPY_IS_INDEX,
            os_index=SPY_OS_INDEX,
            seeds=fixed_seeds,
        )
        logger.info("\n" + fixed_tbl.to_string(index=False))

        fixed_tbl.to_csv(out_dir / "spy_fixed_split_metrics.csv", index=False)

        # Per-date diagnostic dump + alpha* report (single seed)
        try:
            diag_seed = int(fixed_seeds[0]) if fixed_seeds else 0
            diag_out_dir = out_dir / "alpha_star_diag"
            diag_path, df_test = dump_xgb_alpha_star_diagnostics(
                X,
                y,
                returns,
                train_slice=slice(SPY_IS_INDEX, SPY_OS_INDEX),
                test_slice=slice(SPY_OS_INDEX, len(y)),
                seed=diag_seed,
                variant="XGBSTES_BASE",
                xgb_params={},
                out_dir=diag_out_dir,
                prefix="spy_xgb_alpha_diag",
                make_plots=False,
                return_df=True,
            )
            md_path, chart_paths = write_alpha_star_report(
                df_test,
                variant="XGBSTES_BASE",
                seed=diag_seed,
                out_dir=diag_out_dir,
                prefix="spy_xgb_alpha_diag",
            )
            logger.info(f"Wrote alpha* diagnostics CSV to {diag_path}")
            logger.info(f"Wrote alpha* report to {md_path}")
            if chart_paths:
                logger.info(f"Wrote {len(chart_paths)} alpha* charts to {diag_out_dir}")

            # Multi-seed scalar summary (first K seeds)
            k = max(1, min(10, len(fixed_seeds)))
            rows = []
            base = dict(_alpha_star_report_scalars(df_test))
            base["seed"] = int(diag_seed)
            rows.append(base)
            for s in fixed_seeds[1:k]:
                _, df_s = dump_xgb_alpha_star_diagnostics(
                    X,
                    y,
                    returns,
                    train_slice=slice(SPY_IS_INDEX, SPY_OS_INDEX),
                    test_slice=slice(SPY_OS_INDEX, len(y)),
                    seed=int(s),
                    variant="XGBSTES_BASE",
                    xgb_params={},
                    out_dir=diag_out_dir,
                    prefix="spy_xgb_alpha_diag",
                    make_plots=False,
                    return_df=True,
                )
                row = dict(_alpha_star_report_scalars(df_s))
                row["seed"] = int(s)
                rows.append(row)

            df_ms = pd.DataFrame(rows)

            # Blog-friendly per-seed summary table (always written)
            try:
                keep_cols = [
                    "seed",
                    "corr_alpha_pred_alpha_soft",
                    "spearman_alpha_pred_alpha_soft",
                    "rmse_model",
                    "rmse_recon_pred",
                    "rmse_oracle_soft_masked",
                    "oracle_gain_soft_masked",
                    "rmse_const_alpha_masked",
                    "rmse_z_pred_z_soft",
                    "pct_alpha_soft_ge_0_99",
                    "pct_alpha_pred_ge_0_99",
                    "frac_mask",
                    "n_mask",
                ]
                keep_cols = [c for c in keep_cols if c in df_ms.columns]
                df_sum = df_ms[keep_cols].copy()

                def _fmt_sum(col: str, v: float) -> str:
                    if v is None or not np.isfinite(v):
                        return "NA"
                    if col in {"n_test", "n_mask"}:
                        return str(int(round(v)))
                    if col.startswith("pct_"):
                        return f"{100.0 * v:.2f}%"
                    if "corr" in col or "spearman" in col or "frac" in col:
                        return f"{v:.4f}"
                    return _format_sci(float(v))

                for c in df_sum.columns:
                    if c == "seed":
                        df_sum[c] = (
                            pd.to_numeric(df_sum[c], errors="coerce")
                            .fillna(-1)
                            .astype(int)
                        )
                        continue
                    vv = pd.to_numeric(df_sum[c], errors="coerce").to_numpy(dtype=float)
                    df_sum[c] = [_fmt_sum(c, x) for x in vv]

                summary_path = out_dir / "spy_xgb_alpha_diag_summary__XGBSTES_BASE.md"
                summary_path.write_text(
                    "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
                    + _df_to_markdown_table(df_sum),
                    encoding="utf-8",
                )
                logger.info(f"Wrote alpha diag summary to {summary_path}")
            except Exception as e:
                logger.warning(f"Could not write alpha diag summary md: {e}")

            if len(df_ms) >= 2:
                metrics = [c for c in df_ms.columns if c != "seed"]
                out_rows = []
                for c in metrics:
                    v = pd.to_numeric(df_ms[c], errors="coerce").to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    out_rows.append(
                        {
                            "metric": c,
                            "mean": float(np.mean(v)) if v.size else np.nan,
                            "std": float(np.std(v)) if v.size else np.nan,
                        }
                    )
                ms_tbl = pd.DataFrame(out_rows)

                def _fmt_ms(metric: str, v: float) -> str:
                    if v is None or not np.isfinite(v):
                        return "NA"
                    m = str(metric).lower()
                    if m.startswith("n_"):
                        return f"{v:.1f}"
                    if "pct" in m:
                        return f"{100.0 * v:.2f}%"
                    if "corr" in m or "spearman" in m or "frac" in m:
                        return f"{v:.4f}"
                    if "alpha" in m and ("mae" in m or "std" in m):
                        return f"{v:.4f}"
                    return _format_sci(float(v))

                ms_tbl_fmt = pd.DataFrame(
                    {
                        "metric": ms_tbl["metric"].astype(str),
                        "mean": [
                            _fmt_ms(m, v)
                            for m, v in zip(ms_tbl["metric"], ms_tbl["mean"])
                        ],
                        "std": [
                            _fmt_ms(m, v)
                            for m, v in zip(ms_tbl["metric"], ms_tbl["std"])
                        ],
                    }
                )

                extra = [
                    "## E) Multi-seed summary (first K seeds)",
                    f"K={k}",
                    _df_to_markdown_table(ms_tbl_fmt),
                    "",
                ]
                md_path.write_text(
                    md_path.read_text(encoding="utf-8") + "\n" + "\n".join(extra),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.exception(f"Alpha* diagnostic dump failed: {e}")

        fixed_blog_tbl = _make_blog_table(fixed_tbl)
        (out_dir / "spy_fixed_split_metrics_blog.md").write_text(
            "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
            + _df_to_markdown_table(fixed_blog_tbl),
            encoding="utf-8",
        )

        logger.info("\n" + "=" * 80)
        logger.info("SPY (Walk-Forward CV) — unified across ES/STES/XGBSTES")
        logger.info("=" * 80)

        # Default walk-forward settings (reasonable runtime):
        # - expanding window
        # - start with ~8 years (2000 trading days)
        # - validate 1 year (252)
        # - refit yearly (252)
        # - cap folds to keep runtime bounded
        splits = walk_forward_splits(
            n=len(y),
            mode="expanding",
            train_size=2000,
            val_size=252,
            step_size=252,
            max_folds=args.cv_max_folds,
        )
        cv_tbl = evaluate_variants_walk_forward(
            X,
            y,
            returns,
            variants=variants,
            splits=splits,
            seeds=cv_seeds,
        )
        logger.info("\n" + cv_tbl.to_string(index=False))

        cv_tbl.to_csv(out_dir / "spy_walk_forward_metrics.csv", index=False)
        cv_blog_tbl = _make_blog_table(cv_tbl)
        (out_dir / "spy_walk_forward_metrics_blog.md").write_text(
            "<!-- generated by examples/volatility_forecast_2.py -->\n\n"
            + _df_to_markdown_table(cv_blog_tbl),
            encoding="utf-8",
        )

        # --- Head-to-head analysis: STES vs. XGBSTES ---
        # Using default params for now; you can plug in tuned params later.
        analysis_out_dir = out_dir / "spy_stes_vs_xgb"
        analyze_spy_stes_vs_xgb_stes(
            X,
            y,
            returns,
            stes_variant="STES_EAESE",
            xgb_params={},
            seeds=cv_seeds,
            out_dir=analysis_out_dir,
        )
        generate_spy_blog_tables_4_to_6(
            X,
            y,
            returns,
            seeds=fixed_seeds,
            out_dir=out_dir,
            xgb_params={},
        )

    else:
        logger.info("SPY unified evaluation skipped (insufficient data).")

    # Optional hyperparameter tuning for XGBSTES (sklearn RandomizedSearchCV + Optuna)
    if args.tune_sklearn or args.tune_optuna:
        if X is None:
            logger.warning("SPY dataset unavailable; skipping tuning")
        else:
            if args.tune_sklearn:
                (
                    rcv_best_params,
                    rcv_best_score_rmse,
                    rcv_best_score_mae,
                    rcv_best_score_medae,
                ) = random_cv_tune_xgboost_model(
                    X, y, returns, n_iter=args.tune_sklearn_iter
                )
                logger.info(
                    f"Sklearn tuning best params: {rcv_best_params} | "
                    f"RMSE: {rcv_best_score_rmse:.6f}, MAE: {rcv_best_score_mae:.6f}, MedAE: {rcv_best_score_medae:.6f}"
                )
                tuned_params = DEFAULT_XGBOOST_PARAMS | (rcv_best_params or {})
                try:
                    tuned_model = _make_xgb_stes_model(seed=0, params_flat=tuned_params)
                    # Fit the model on the full training data before logging importance
                    tuned_model.fit(
                        X.iloc[:SPY_OS_INDEX],
                        y.iloc[:SPY_OS_INDEX],
                        returns=returns.iloc[:SPY_OS_INDEX],
                    )
                    log_feature_importance(
                        tuned_model, feature_names, Path("outputs/spy_importance")
                    )
                    # User can add evaluation of the tuned model here if desired
                except Exception as e:
                    logger.exception(f"Failed to process tuned sklearn model: {e}")

            if args.tune_optuna:
                if optuna is None:
                    logger.warning("optuna not available; skipping optuna tuning")
                else:
                    sampler = optuna.samplers.TPESampler(seed=0)
                    study = optuna.create_study(direction="minimize", sampler=sampler)
                    study.optimize(
                        lambda trial: xgb_stes_optuna_objective(trial, X, y, returns),
                        n_trials=args.tune_optuna_trials,
                    )
                    trial = study.best_trial
                    tuned_params = DEFAULT_XGBOOST_PARAMS | trial.params
                    logger.info(f"Optuna best params: {tuned_params}")
                    try:
                        tuned_model = _make_xgb_stes_model(
                            seed=0, params_flat=tuned_params
                        )
                        # Fit the model on the full training data before logging importance
                        tuned_model.fit(
                            X.iloc[:SPY_OS_INDEX],
                            y.iloc[:SPY_OS_INDEX],
                            returns=returns.iloc[:SPY_OS_INDEX],
                        )
                        log_feature_importance(
                            tuned_model, feature_names, Path("outputs/spy_importance")
                        )
                        # User can add evaluation of the tuned model here if desired
                    except Exception as e:
                        logger.exception(f"Failed to process tuned optuna model: {e}")

    # NOTE: The detailed SPY seed-averaged fixed-split table is now produced by
    # evaluate_variants_fixed_split(...) above (applied uniformly to all variants).
    # The walk-forward CV table is produced by evaluate_variants_walk_forward(...).


if __name__ == "__main__":
    main()
