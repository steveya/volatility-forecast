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
    "max_depth": 3,
    "learning_rate": 0.05,  # mapped to XGBoost 'eta'
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
    "n_alt_iters": 3,
    "alt_clip_eps": 1e-6,
    # New model knobs
    "alt_objective": "reg:pseudohubererror",
    "residual_mode": True,
    "monotonic_constraints": {},
    "denom_quantile": 0.05,
    "min_denom_floor": 1e-12,
}


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
    )


def _infer_monotone_constraints(cols: list[str]) -> dict[str, int]:
    """Heuristic: enforce +1 monotonicity on shock-magnitude style features."""
    out: dict[str, int] = {}
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["abs", "squared", "sq", "r2", "vol", "rv"]):
            out[c] = 1
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
    xgb_params: dict | None = None,
) -> tuple[pd.Index, np.ndarray] | tuple[pd.Index, np.ndarray, np.ndarray]:
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

    X_tr, y_tr, r_tr = (
        X_sel.iloc[train_slice],
        y.iloc[train_slice],
        r.iloc[train_slice],
    )
    X_te, r_te = X_sel.iloc[test_slice], r.iloc[test_slice]

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

    idx = X_te.index

    if return_alpha and hasattr(model, "predict_with_alpha"):
        y_hat, alpha = model.predict_with_alpha(X_te, returns=r_te)
        y_hat_arr = np.asarray(y_hat, dtype=float)
        alpha_arr = np.asarray(alpha, dtype=float)
    elif return_alpha and hasattr(model, "get_alphas"):
        # tree_stes_model.XGBoostSTESModel exposes alphas via get_alphas
        y_hat_s = model.predict(X_te, returns=r_te, start_index=0, end_index=len(X_te))
        alpha_s = model.get_alphas(X_te, start_index=0, end_index=len(X_te))
        y_hat_arr = np.asarray(y_hat_s, dtype=float)
        alpha_arr = np.asarray(alpha_s, dtype=float)
    else:
        y_hat_s = model.predict(X_te, returns=r_te)
        y_hat_arr = np.asarray(y_hat_s, dtype=float)
        alpha_arr = np.full_like(y_hat_arr, np.nan, dtype=float)

    keep = np.isfinite(y_hat_arr)
    idx = idx[keep]
    y_hat_arr = y_hat_arr[keep]
    alpha_arr = alpha_arr[keep]

    if return_alpha:
        return idx, y_hat_arr, alpha_arr
    return idx, y_hat_arr


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
    rows = []
    train_sl = slice(is_index, os_index)
    test_sl = slice(os_index, len(y))

    for variant in variants:
        rmses, maes, medaes = [], [], []
        # Deterministic ES still runs fine with multiple seeds; keep loop uniform.
        for seed in seeds:
            try:
                res = _fit_predict_oos(
                    variant=variant,
                    X=X,
                    y=y,
                    r=r,
                    train_slice=train_sl,
                    test_slice=test_sl,
                    seed=seed,
                )
                idx_te, y_hat = res[0], res[1]
                y_true = y.loc[idx_te].values
                rmses.append(rmse(y_true, y_hat))
                maes.append(mae(y_true, y_hat))
                medaes.append(medae(y_true, y_hat))
            except Exception as e:
                logger.exception(
                    f"Fixed-split eval failed: variant={variant}, seed={seed}: {e}"
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
                    idx_te, y_hat = res[0], res[1]
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
    idx_stes, y_hat_stes, alpha_stes = _fit_predict_oos(
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
        idx_xgb, y_hat_xgb_seed, alpha_xgb_seed = _fit_predict_oos(
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
        xgb_preds.append(pd.Series(y_hat_xgb_seed, index=idx_xgb))
        xgb_alphas.append(pd.Series(alpha_xgb_seed, index=idx_xgb))

    # Averaging predictions over seeds
    y_hat_xgb_s = pd.concat(xgb_preds, axis=1).mean(axis=1)
    alpha_xgb_s = pd.concat(xgb_alphas, axis=1).mean(axis=1)

    idx_te = idx_stes.intersection(y_hat_xgb_s.index)
    y_true = y.loc[idx_te].values
    y_hat_stes = pd.Series(y_hat_stes, index=idx_stes).loc[idx_te].to_numpy()
    alpha_stes = pd.Series(alpha_stes, index=idx_stes).loc[idx_te].to_numpy()
    y_hat_xgb = y_hat_xgb_s.loc[idx_te].to_numpy()
    alpha_xgb = alpha_xgb_s.loc[idx_te].to_numpy()

    # --- 3. Compute per-date loss differential ---
    loss_stes = (y_true - y_hat_stes) ** 2
    loss_xgb = (y_true - y_hat_xgb) ** 2
    D = loss_stes - loss_xgb  # Positive => XGB is better

    # --- 4. Build timeseries DataFrame for analysis ---
    df = pd.DataFrame(
        {
            "y": y_true,
            "yhat_stes": y_hat_stes,
            "yhat_xgb": y_hat_xgb,
            "loss_stes": loss_stes,
            "loss_xgb": loss_xgb,
            "D": D,
            "r": r.loc[idx_te],
            "abs_r": np.abs(r.loc[idx_te]),
            "r2": r.loc[idx_te] ** 2,
            "r2_lag": (r.loc[idx_te] ** 2).shift(1),
            "alpha_stes_lag": pd.Series(alpha_stes, index=idx_te).shift(1),
            "alpha_xgb_lag": pd.Series(alpha_xgb, index=idx_te).shift(1),
        },
        index=idx_te,
    )
    df.to_csv(out_dir / "spy_stes_vs_xgb_timeseries.csv")
    logger.info(
        f"Saved timeseries analysis to {out_dir / 'spy_stes_vs_xgb_timeseries.csv'}"
    )

    df["alpha_stes"] = pd.Series(alpha_stes, index=idx_te)
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
        idx, y_hat, alpha = _fit_predict_oos(
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
        preds.append(pd.Series(y_hat, index=idx))
        alphas.append(pd.Series(alpha, index=idx))

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


# Retain the original build_spy_dataset (unchanged)
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
            X_is, y_is = X.iloc[:os_index], y.iloc[:os_index]
            r_is = returns.iloc[:os_index] if returns is not None else None
            try:
                model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
            except TypeError:
                # fallback for models that don't accept returns/start_index
                model.fit(X_is, y_is)

            X_os = X.iloc[os_index:]
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
                    idx_te, y_hat = res[0], res[1]
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
                idx_te, y_hat = res[0], res[1]
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
    X_is, y_is = X_sel.iloc[IS_INDEX:OS_INDEX], y.iloc[IS_INDEX:OS_INDEX]
    X_os, y_os = X_sel.iloc[OS_INDEX:], y.iloc[OS_INDEX:]
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
    X_is, y_is = X_sel.iloc[IS_INDEX:OS_INDEX], y.iloc[IS_INDEX:OS_INDEX]
    X_os, y_os = X_sel.iloc[OS_INDEX:], y.iloc[OS_INDEX:]
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

    ctx = build_default_ctx(tiingo_api_key=api_key)

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
