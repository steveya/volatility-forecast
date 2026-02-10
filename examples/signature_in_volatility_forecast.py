"""signature_in_volatility_forecast.py

SPY volatility-forecast experiments with optional signature features.

This script is meant to be a *drop-in companion* to examples/volatility_forecast_2.py.
The baseline models (ES / STES_* / XGBSTES_*) should reproduce the exact same
numbers as volatility_forecast_2.py when signature features are disabled.

Why baseline drift happens
-------------------------
Signature features require an additional lookback window. If you build a dataset
with signatures and use a global "drop_if_any_nan" policy, the first (window-1)
rows will be dropped, which shifts the integer-based split indices (IS/OS) and
changes results even for models that *ignore* signature columns.

Fix implemented here
--------------------
1) Build a *baseline* SPY dataset with the exact same feature set as Part 2
   (lag returns / abs returns / squared returns; N_LAGS default 0).

2) Build a SPY dataset that *adds* signature features.

3) Align the signature dataset back to the baseline index via reindexing.
   - Rows that exist in baseline but were dropped in the signature build (due to
     signature warmup NaNs) are reintroduced.
   - Signature columns for those early rows are filled with 0.0.

This preserves:
- identical y / returns series
- identical calendar index and therefore identical IS/OS split dates
- identical baseline model behavior (since baseline variants select only the
  non-signature columns)

Design note on burn-in vs signature window
------------------------------------------
Using an STES-style burn-in (e.g., init_window=500) is compatible with signature
lookbacks shorter than that. However, for the *training split* used in Part 2
(SPY_IS_INDEX=200), you should also keep signature lookback <= SPY_IS_INDEX
(otherwise the training features at the split start will still be NaN before
alignment/fill).

This file removes the simulated GARCH section entirely; it focuses on SPY only.

"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv

# Repo-local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)
load_dotenv()

import logging

_log_level = os.environ.get("AF_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    JoinPolicy,
    MissingnessPolicy,
)
from alphaforge.data.cache import FileCacheBackend
from volatility_forecast.pipeline import (
    build_default_ctx,
    build_vol_dataset,
    VolDatasetSpec,
)
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.features.signature_features import SignatureFeaturesTemplate
from volatility_forecast.features.selector import select_variant_columns
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget

from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel
from volatility_forecast.model.tree_stes_model import XGBoostSTESModel

from volatility_forecast.evaluation.model_evaluator import rmse, mae, medae


# ---------------------------------------------------------------------------
# SPY settings (kept consistent with examples/volatility_forecast_2.py)
# ---------------------------------------------------------------------------

SPY_TICKER = "SPY"
SPY_START = pd.Timestamp("2000-01-01", tz="UTC")
SPY_END = pd.Timestamp("2023-12-31", tz="UTC")
SPY_BDAYS = pd.bdate_range(SPY_START, SPY_END, tz="UTC")
SPY_START_BDAY = SPY_BDAYS.min()
SPY_END_BDAY = SPY_BDAYS.max()

# Same as Part 2 script
SPY_IS_INDEX = 200
SPY_OS_INDEX = 4000

# Feature lags for the base return templates. N_LAGS=0 => today's return only.
N_LAGS = 0

# This matches the initialization window used by the upgraded XGBSTES.
# (We keep it explicit here so signature lookbacks can be chosen to be <= this.)
INIT_WINDOW = 500


# ---------------------------------------------------------------------------
# XGBSTES defaults (aligned with examples/volatility_forecast_2.py)
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "booster": "gblinear",
    "updater": "coord_descent",
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "verbosity": 0,
}
XGB_ROUNDS = 200

# Scale returns by 100 (Vol Points) so variance is scaled by 10000.
# This keeps XGBoostSTES gradients well-conditioned.
SCALE_FACTOR = 100.0


def _read_tiingo_cache_dir() -> str:
    return os.environ.get("TIINGO_CACHE_DIR", ".af_cache")


def _ensure_spy_cache_coverage(cache: FileCacheBackend) -> None:
    """Ensure SPY cache covers [SPY_START, SPY_END] to avoid API calls."""
    cache_key = f"{SPY_TICKER}|adj=True"
    cached = cache.get(cache_key)
    if cached is None or cached.empty:
        raise RuntimeError(
            "Tiingo cache is empty for SPY. Populate cache first before running in cache-only mode."
        )

    dates = pd.to_datetime(cached["date"], utc=True).dt.normalize()
    cached_min = dates.min()
    cached_max = dates.max()

    # Use business-day bounds to avoid false failures on weekends/holidays.
    if SPY_BDAYS.empty:
        raise RuntimeError("No business days found between SPY_START and SPY_END.")
    target_min = SPY_START_BDAY
    target_max = SPY_END_BDAY

    if cached_min > target_min or cached_max < target_max:
        raise RuntimeError(
            "Tiingo cache does not fully cover SPY range. "
            f"Cached [{cached_min.date()}..{cached_max.date()}], "
            f"needed [{target_min.date()}..{target_max.date()}]."
        )


def make_xgbstes_grids():
    """Define hyperparameter grids for CV (aligned with volatility_forecast_2.py)."""
    grid_e2e = [
        {"min_child_weight": 5.0, "learning_rate": 0.05, "max_depth": 3},
        {"min_child_weight": 20.0, "learning_rate": 0.1, "max_depth": 3},
        {"min_child_weight": 50.0, "learning_rate": 0.05, "max_depth": 2},
    ]
    return grid_e2e


def _make_xgb_stes_model(
    *,
    seed: int | None,
    fit_method: str = "alternating",
    loss: str = "mse",
    xgb_params: dict | None = None,
    monotonic_constraints: dict[str, int] | None = None,
) -> XGBoostSTESModel:
    params = dict(XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)
    # Keep E2E defaults aligned with examples/volatility_forecast_2.py
    e2e_kwargs = {}
    if fit_method == "end_to_end":
        e2e_kwargs = {
            "e2e_grad_hess_scale": 1.0,
            "e2e_debug": True,
            "e2e_debug_print_once": True,
        }

    return XGBoostSTESModel(
        xgb_params=params,
        num_boost_round=XGB_ROUNDS,
        init_window=INIT_WINDOW,
        fit_method=fit_method,  # "alternating" | "end_to_end"
        loss=loss,  # "mse" | "pseudohuber"
        random_state=seed,
        monotonic_constraints=monotonic_constraints,
        **e2e_kwargs,
    )


def _infer_monotone_constraints(cols: list[str]) -> dict[str, int]:
    """Heuristic: enforce -1 monotonicity on shock-magnitude style features."""
    out: dict[str, int] = {}
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["abs", "squared", "sq", "r2", "vol", "rv"]):
            out[c] = -1
    return out


def _xgb_variant_overrides(variant: str, cols: list[str]) -> dict:
    """Translate legacy XGBSTES_* variant names into current constructor knobs."""
    v = variant.upper()
    fit_method = "end_to_end" if "E2E" in v or "END_TO_END" in v else "alternating"
    loss = "pseudohuber" if "HUBER" in v else "mse"
    mono = _infer_monotone_constraints(cols) if "MONO" in v else None
    return {"fit_method": fit_method, "loss": loss, "monotonic_constraints": mono}


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------


def _base_feature_requests(lags: int) -> tuple[FeatureRequest, ...]:
    return (
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


@dataclass(frozen=True)
class SignatureConfig:
    name: str
    lookback: int
    level: int
    augmentations: list[str]
    sig_tfm: str


def build_spy_spec_baseline(lags: int = N_LAGS) -> VolDatasetSpec:
    """Baseline spec identical to Part 2 (no signature features)."""
    features = _base_feature_requests(lags)

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
            start=SPY_START_BDAY,
            end=SPY_END_BDAY,
            calendar="XNYS",
            grid="B",
            asof=None,
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def build_spy_spec_with_signature(sig: SignatureConfig, lags: int = N_LAGS):
    """Spec that adds signature features on top of the baseline features."""
    base = list(_base_feature_requests(lags))

    # Signature template is assumed to compute features from SPY return channels.
    # Keep params explicit so changes are localized.
    sig_req = FeatureRequest(
        template=SignatureFeaturesTemplate(),
        params={
            "lags": int(sig.lookback),
            "sig_level": int(sig.level),
            "sig_tfm": str(sig.sig_tfm),
            # SignatureFeaturesTemplate expects a *string* tag / ordered pipeline spec.
            # Examples: "all", "none", "cumsum->basepoint->addtime->leadlag"
            "augmentation_list": (
                "none" if not sig.augmentations else "->".join(sig.augmentations)
            ),
            "source": "tiingo",
            "table": "market.ohlcv",
            "price_col": "close",
        },
    )

    features = tuple(base + [sig_req])

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
            start=SPY_START_BDAY,
            end=SPY_END_BDAY,
            calendar="XNYS",
            grid="B",
            asof=None,
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        # Keep consistent with Part 2; we handle the signature warmup alignment manually.
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def _split_entity_frame(
    df: pd.DataFrame | pd.Series, entity: str
) -> pd.DataFrame | pd.Series:
    if isinstance(df.index, pd.MultiIndex) and "entity_id" in df.index.names:
        return df.xs(entity, level="entity_id").sort_index()
    return df.sort_index()


def build_spy_dataset_baseline(ctx, *, lags: int = N_LAGS):
    """Build baseline dataset and return (X, y, r)."""
    spec = build_spy_spec_baseline(lags)
    X, y, returns, _catalog = build_vol_dataset(ctx, spec, persist=False)

    X1 = _split_entity_frame(X, SPY_TICKER).copy()
    y1 = _split_entity_frame(y, SPY_TICKER)
    r1 = _split_entity_frame(returns, SPY_TICKER)

    if isinstance(X1, pd.Series):
        X1 = X1.to_frame()

    if "const" not in X1.columns:
        X1["const"] = 1.0

    # Ensure alignment
    idx = y1.index.intersection(X1.index).intersection(r1.index)
    X1 = X1.loc[idx]
    y1 = y1.loc[idx]
    r1 = r1.loc[idx]

    return X1, y1, r1


def build_spy_dataset_with_signature(
    ctx,
    *,
    sig: SignatureConfig,
    baseline_X: pd.DataFrame,
    baseline_index: pd.Index,
    baseline_columns: list[str],
    lags: int = N_LAGS,
) -> pd.DataFrame:
    """Build a signature-feature dataset and align it to the baseline index.

    Parameters
    ----------
    baseline_X:
        The baseline feature matrix (already cleaned/dropped by the Part 2
        missingness policy). We use it to *restore* baseline columns for dates
        that were dropped from the signature dataset during signature warm-up.

    Returns
    -------
    X_aligned:
        DataFrame indexed by baseline_index and containing baseline_columns +
        signature columns. Signature columns for warmup rows are filled with 0.0.

    Notes
    -----
    We intentionally preserve the Part 2 calendar grid by:

    1) building a baseline dataset (no signatures) using the same missingness
       policy as volatility_forecast_2.py,
    2) building a signature dataset (which will typically drop early rows due to
       signature warm-up),
    3) reindexing the signature dataset to the baseline index, and
    4) copying baseline columns from baseline_X and filling missing signature
       rows with 0.0.

    This guarantees that *baseline variants* (that ignore signature columns)
    match volatility_forecast_2.py exactly, while signature variants are trained
    and evaluated on the same date ranges.
    """
    spec = build_spy_spec_with_signature(sig, lags)
    X, _y, _returns, _catalog = build_vol_dataset(ctx, spec, persist=False)

    X1 = _split_entity_frame(X, SPY_TICKER)
    if isinstance(X1, pd.Series):
        X1 = X1.to_frame()

    if "const" not in X1.columns:
        X1["const"] = 1.0

    # Align to baseline index: reintroduce rows dropped during signature warm-up.
    X_aligned = X1.reindex(baseline_index)

    # Ensure baseline columns exist in the aligned frame.
    missing_base = [c for c in baseline_columns if c not in X_aligned.columns]
    if missing_base:
        raise ValueError(
            f"Signature dataset is missing baseline columns: {missing_base}"
        )

    # Restore baseline columns from the baseline dataset (exact match to Part 2).
    # This is crucial: those rows may have been dropped entirely from X1, so
    # reindexing alone would leave NaNs in baseline columns.
    X_aligned.loc[baseline_index, baseline_columns] = baseline_X.loc[
        baseline_index, baseline_columns
    ].to_numpy()

    # Identify signature-only columns (those not in baseline_columns)
    sig_cols = [c for c in X_aligned.columns if c not in set(baseline_columns)]

    # Fill warmup NaNs in signature features only.
    if sig_cols:
        X_aligned[sig_cols] = X_aligned[sig_cols].fillna(0.0)

    # Defensive: baseline columns should now be NaN-free.
    bad = X_aligned[baseline_columns].isna().any(axis=1)
    if bool(bad.any()):
        n_bad = int(bad.sum())
        first_bad = X_aligned.index[bad][0]
        raise ValueError(
            f"Found NaNs in baseline columns after alignment (n={n_bad}, first={first_bad}). "
            "This suggests the baseline dataset itself contains missing values."
        )

    return X_aligned


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


class ModelName(str, Enum):
    ES = "ES"
    STES_EAESE = "STES_EAESE"
    XGBSTES_E2E = "XGBSTES_E2E"


def _variant_name(variant: str | Enum) -> str:
    return variant.value if isinstance(variant, Enum) else str(variant)


def _is_signature_col(col: str) -> bool:
    return col.startswith("sig.")


def _filter_signature_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if not _is_signature_col(c)]


def _scale_train_test(
    X: pd.DataFrame, train_slice: slice, test_slice: slice
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize non-signature features on train slice and apply to test.

    Excludes:
    - 'const' column (if present)
    - signature features (columns starting with 'sig.')

    Signature features are assumed to be already standardized.
    """
    X_tr = X.iloc[train_slice].copy()
    X_te = X.iloc[test_slice].copy()

    # Identify columns to scale: exclude 'const' and signature features
    cols_to_scale = [c for c in X.columns if c != "const" and not _is_signature_col(c)]

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


def _make_model(
    variant: str,
    *,
    seed: int | None = None,
    xgb_params: dict | None = None,
    cols: list[str] | None = None,
):
    variant_name = _variant_name(variant)
    if variant_name == ModelName.ES.value:
        return ESModel(random_state=seed) if seed is not None else ESModel()
    if variant_name.startswith("XGBSTES_"):
        over = _xgb_variant_overrides(variant_name, cols or [])
        return _make_xgb_stes_model(seed=seed, xgb_params=xgb_params, **over)
    # STES variants
    return STESModel(random_state=seed) if seed is not None else STESModel()


def _select_columns(
    X: pd.DataFrame, variant: str, *, extra_cols: list[str] | None = None
) -> list[str]:
    """Select base columns for a variant, optionally adding extra columns (e.g. signatures)."""
    variant_name = _variant_name(variant)
    if variant_name.startswith("XGBSTES_"):
        base_cols = select_variant_columns(X, "XGBSTES_BASE")
        if not base_cols:
            base_cols = list(X.columns)
    else:
        base_cols = select_variant_columns(X, variant_name)
        if not base_cols:
            base_cols = ["const"]

    base_cols = _filter_signature_cols(base_cols)

    if extra_cols:
        # Keep deterministic ordering: base then extras not already included
        seen = set(base_cols)
        base_cols = base_cols + [c for c in extra_cols if c not in seen]

    return base_cols


def _fit_predict_oos(
    *,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    train_slice: slice,
    test_slice: slice,
    seed: int | None = None,
    xgb_params: dict | None = None,
    extra_cols: list[str] | None = None,
    return_alpha: bool = False,
) -> tuple[pd.Index, np.ndarray] | tuple[pd.Index, np.ndarray, pd.Series]:
    variant_name = _variant_name(variant)
    if variant_name.startswith("XGBSTES_"):
        # Preserve column order to match volatility_forecast_2.py behavior.
        # Baseline XGBSTES ignores signature columns; signature variants keep all columns.
        if extra_cols:
            cols = list(X.columns)
        else:
            cols = [c for c in X.columns if not _is_signature_col(c)]

        X_sel = X[cols]
        X_tr, X_te = _scale_train_test(X_sel, train_slice, test_slice)
        y_tr, r_tr = y.iloc[train_slice], r.iloc[train_slice]
        r_te = r.iloc[test_slice]

        over = _xgb_variant_overrides(variant_name, cols)
        # Align with volatility_forecast_2.py: do not set random_state for XGBSTES
        model = _make_xgb_stes_model(seed=None, xgb_params=xgb_params, **over)
    else:
        cols = _select_columns(X, variant_name, extra_cols=extra_cols)
        X_sel = X[cols]

        X_tr, X_te = _scale_train_test(X_sel, train_slice, test_slice)
        y_tr, r_tr = y.iloc[train_slice], r.iloc[train_slice]
        r_te = r.iloc[test_slice]

        model = _make_model(variant_name, seed=seed, xgb_params=xgb_params, cols=cols)

    # Keep alignment consistent with Part 2: always pass returns and start/end indices.
    if variant_name.startswith("XGBSTES_"):
        r_tr_scaled = r_tr * SCALE_FACTOR
        y_tr_scaled = y_tr * (SCALE_FACTOR**2)
        grid_e2e = make_xgbstes_grids()
        model.fit(
            X_tr,
            y_tr_scaled,
            returns=r_tr_scaled,
            perform_cv=True,
            cv_grid=grid_e2e,
            cv_splits=3,
            start_index=0,
            end_index=len(X_tr),
        )
    else:
        model.fit(X_tr, y_tr, returns=r_tr, start_index=0, end_index=len(X_tr))

    # IMPORTANT: warm-start OOS recursion by predicting on concatenated
    # [train + test] and slicing the test tail. This avoids "cold start"
    # of the variance state at the beginning of the OOS block.
    n_tr = len(X_tr)
    X_all = pd.concat([X_tr, X_te], axis=0)
    r_all = pd.concat([r_tr, r_te], axis=0)
    if variant_name.startswith("XGBSTES_"):
        r_all_scaled = r_all * SCALE_FACTOR
        y_hat_all_scaled = model.predict(X_all, returns=r_all_scaled)
        y_hat_all = np.asarray(y_hat_all_scaled, dtype=float) / (SCALE_FACTOR**2)
        alpha_all = model.get_alphas(X_all)
    else:
        y_hat_all = model.predict(X_all, returns=r_all)
        if isinstance(model, STESModel):
            _yhat_vals, alpha_vals = model.predict_with_alpha(X_all, returns=r_all)
            alpha_all = pd.Series(alpha_vals, index=X_all.index, name="alpha")
        else:
            alpha_val = getattr(model, "alpha_", np.nan)
            alpha_all = pd.Series(alpha_val, index=X_all.index, name="alpha")

    y_hat_arr = np.asarray(y_hat_all, dtype=float)[n_tr:]
    keep = np.isfinite(y_hat_arr)
    if return_alpha:
        return X_te.index[keep], y_hat_arr[keep], alpha_all
    return X_te.index[keep], y_hat_arr[keep]


def evaluate_fixed_split(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    variants: list[str],
    seeds: list[int],
    is_index: int = SPY_IS_INDEX,
    os_index: int = SPY_OS_INDEX,
    extra_cols_by_variant: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    train_sl = slice(is_index, os_index)
    test_sl = slice(os_index, len(y))

    rows = []
    for variant in variants:
        rmses, maes, medaes = [], [], []
        extra_cols = (extra_cols_by_variant or {}).get(variant)

        for seed in seeds:
            try:
                idx_te, y_hat = _fit_predict_oos(
                    variant=variant,
                    X=X,
                    y=y,
                    r=r,
                    train_slice=train_sl,
                    test_slice=test_sl,
                    seed=seed,
                    extra_cols=extra_cols,
                )
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

    return pd.DataFrame(rows).sort_values("rmse_mean")


def _format_sci(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.2e}"


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _make_blog_table(df: pd.DataFrame) -> pd.DataFrame:
    data = {
        "Model": df["variant"],
        "RMSE": df["rmse_mean"].map(_format_sci),
        "MAE": df["mae_mean"].map(_format_sci),
        "MedAE": df["medae_mean"].map(_format_sci),
    }
    if "signature" in df.columns:
        data = {"Signature": df["signature"], **data}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def main():
    import argparse

    DEFAULT_AUGS = ["cumsum", "basepoint", "addtime", "leadlag"]

    def normalize_augmentations(base: list[str], extras: list[str] | None) -> list[str]:
        out = list(base)
        if extras:
            out.extend(extras)
        seen = set()
        out2 = []
        for a in out:
            if a and a not in seen:
                out2.append(a)
                seen.add(a)
        return out2

    def _parse_augmentations(values: list[str] | None) -> list[str]:
        if not values:
            return []
        if (
            len(values) == 1
            and isinstance(values[0], str)
            and values[0].lower() == "none"
        ):
            return []
        out = []
        for v in values:
            if not v:
                continue
            s = v.replace("->", ",").replace("+", ",")
            out.extend([a.strip() for a in s.split(",") if a.strip()])
        return out

    def _parse_variants(value: str) -> list[str]:
        return [v.strip() for v in value.split(",") if v.strip()]

    parser = argparse.ArgumentParser(
        description="SPY Part-2-style evaluation with optional signature features (baseline-aligned)."
    )
    parser.add_argument(
        "--spy-n-inits",
        type=int,
        default=100,
        help="Number of random seeds to average over",
    )
    parser.add_argument(
        "--sig",
        action="append",
        default=[],
        help=(
            "Signature config encoded as name:lookback:level[:augmentations], e.g. "
            "L10_L2:10:2:cumsum->basepoint->addtime->leadlag or L10_L2:10:2:none. "
            "If augmentations are omitted, defaults + --augmentations are used. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--base-variants",
        type=str,
        default="ES,STES_EAESE,XGBSTES_E2E",
        help="Comma-separated baseline variants for the sanity-check table.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/signature_volatility_forecast",
        help="Output directory",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=[],
        help=(
            "Extra signature augmentations to append to the default pipeline when --sig does not "
            "explicitly provide augmentations. Use 'none' for no extras."
        ),
    )
    parser.add_argument(
        "--sig-tfm",
        type=str,
        default="signature",
        choices=["signature", "logsignature"],
        help="Signature transform to use for all --sig configs.",
    )
    args = parser.parse_args()

    # Add default comparison signature configs only when none are provided.
    default_sig_specs = [
        "L10_L2:10:2:none",
        "L20_L2:20:2:none",
        "L40_L2:40:2:none",
        "L20_L3:20:3:none",
        "L40_L3:40:3:none",
    ]
    if args.sig is None:
        args.sig = []
    if not args.sig:
        args.sig.extend(default_sig_specs)

    extra_augs_global = _parse_augmentations(args.augmentations)
    logger.info(
        "Signature augmentations default=%s extras=%s",
        DEFAULT_AUGS,
        extra_augs_global,
    )

    # Cache-only mode: avoid hitting Tiingo API.
    cache = FileCacheBackend(Path(_read_tiingo_cache_dir()))
    _ensure_spy_cache_coverage(cache)
    ctx = build_default_ctx(
        tiingo_api_key=None,
        tiingo_cache_backends=[cache],
        tiingo_cache_mode="use",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Baseline dataset (authoritative index and split dates)
    # ------------------------------------------------------------------
    X_base, y_base, r_base = build_spy_dataset_baseline(ctx, lags=N_LAGS)

    if len(y_base) <= SPY_OS_INDEX:
        raise ValueError(
            f"Insufficient rows for SPY_OS_INDEX={SPY_OS_INDEX}: len(y)={len(y_base)}"
        )

    baseline_index = y_base.index
    baseline_columns = list(X_base.columns)

    logger.info(f"Baseline dataset rows: {len(y_base)}")
    logger.info(f"Baseline IS date: {baseline_index[SPY_IS_INDEX]}")
    logger.info(f"Baseline OS date: {baseline_index[SPY_OS_INDEX]}")

    # ------------------------------------------------------------------
    # 2) Evaluate baseline variants on baseline dataset (sanity check)
    # ------------------------------------------------------------------
    seeds = list(range(int(args.spy_n_inits)))
    baseline_variants = _parse_variants(args.base_variants)

    logger.info(
        "Evaluating baselines (should match examples/volatility_forecast_2.py) ..."
    )
    base_tbl = evaluate_fixed_split(
        X_base, y_base, r_base, variants=baseline_variants, seeds=seeds
    )
    base_tbl.to_csv(out_dir / "spy_fixed_split_baseline.csv", index=False)
    (out_dir / "spy_fixed_split_baseline_blog.md").write_text(
        "<!-- generated by signature_in_volatility_forecast.py -->\n\n"
        + _df_to_markdown_table(_make_blog_table(base_tbl)),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 3) Signature configs
    # ------------------------------------------------------------------
    sig_cfgs: list[SignatureConfig] = []
    for s in args.sig:
        try:
            parts = s.split(":")
            if len(parts) not in {3, 4}:
                raise ValueError
            name, lookback, level = parts[:3]
            sig_override = _parse_augmentations([parts[3]]) if len(parts) == 4 else None
            if sig_override is None:
                sig_augs = normalize_augmentations(DEFAULT_AUGS, extra_augs_global)
            else:
                sig_augs = sig_override
            sig_cfgs.append(
                SignatureConfig(
                    name=name,
                    lookback=int(lookback),
                    level=int(level),
                    augmentations=sig_augs,
                    sig_tfm=args.sig_tfm,
                )
            )
        except Exception:
            raise ValueError(
                f"Bad --sig format: {s!r}. Expected name:lookback:level[:augmentations]."
            )

    if not sig_cfgs:
        logger.info("No --sig provided; stopping after baseline sanity check.")
        return

    # ------------------------------------------------------------------
    # 4) For each signature config: build aligned X, run baselines + signature variants
    # ------------------------------------------------------------------
    all_sig_tables = []
    for sig in sig_cfgs:
        if sig.lookback > SPY_IS_INDEX:
            logger.warning(
                f"Signature lookback {sig.lookback} > SPY_IS_INDEX {SPY_IS_INDEX}. "
                "The earliest training rows at the split may not have meaningful signature information."
            )
        if sig.lookback > INIT_WINDOW:
            logger.warning(
                f"Signature lookback {sig.lookback} > INIT_WINDOW {INIT_WINDOW}. "
                "Many early rows will have zero-filled signature features."
            )

        logger.info("-" * 80)
        logger.info(f"Building signature dataset: {sig}")

        X_sig = build_spy_dataset_with_signature(
            ctx,
            sig=sig,
            baseline_X=X_base,
            baseline_index=baseline_index,
            baseline_columns=baseline_columns,
            lags=N_LAGS,
        )

        # Signature-only columns for this config
        sig_cols = [c for c in X_sig.columns if c not in set(baseline_columns)]
        logger.info(f"Signature columns added: {len(sig_cols)}")

        # Compose variants: baselines + versions that include signature columns
        stes_sig_name = f"{_variant_name(ModelName.STES_EAESE)}_SIG_{sig.name}"
        xgb_sig_name = f"{_variant_name(ModelName.XGBSTES_E2E)}_SIG_{sig.name}"

        variants = [
            _variant_name(ModelName.ES),
            _variant_name(ModelName.STES_EAESE),
            _variant_name(ModelName.XGBSTES_E2E),
            stes_sig_name,
            xgb_sig_name,
        ]

        # Map signature variants back to underlying model names; we only change the column set.
        extra_cols_by_variant = {
            stes_sig_name: sig_cols,
            xgb_sig_name: sig_cols,
        }

        # Evaluate by reusing baseline y/r and aligned X
        logger.info(f"Evaluating signature config {sig.name} (fixed split) ...")

        def _eval_variant(v: str) -> str:
            es_name = _variant_name(ModelName.ES)
            xgb_name = _variant_name(ModelName.XGBSTES_E2E)
            stes_name = _variant_name(ModelName.STES_EAESE)
            if v.startswith(es_name):
                return es_name
            if v.startswith(xgb_name):
                return xgb_name
            return stes_name

        # Wrapper: translate synthetic names to true model variants for training.
        # (We keep the reported name as-is.)
        def evaluate_with_aliases():
            rows = []
            alpha_series = {}
            train_sl = slice(SPY_IS_INDEX, SPY_OS_INDEX)
            test_sl = slice(SPY_OS_INDEX, len(y_base))

            for v in variants:
                base_v = _eval_variant(v)
                extra = extra_cols_by_variant.get(v)

                rmses, maes, medaes = [], [], []
                for i, seed in enumerate(seeds):
                    try:
                        want_alpha = i == 0
                        res = _fit_predict_oos(
                            variant=base_v,
                            X=X_sig,
                            y=y_base,
                            r=r_base,
                            train_slice=train_sl,
                            test_slice=test_sl,
                            seed=seed,
                            extra_cols=extra,
                            return_alpha=want_alpha,
                        )
                        if want_alpha:
                            idx_te, y_hat, alpha_all = res
                            alpha_series[v] = alpha_all
                        else:
                            idx_te, y_hat = res
                        y_true = y_base.loc[idx_te].values
                        rmses.append(rmse(y_true, y_hat))
                        maes.append(mae(y_true, y_hat))
                        medaes.append(medae(y_true, y_hat))
                    except Exception as e:
                        logger.exception(
                            f"Eval failed: variant={v} (base={base_v}), seed={seed}: {e}"
                        )

                rows.append(
                    {
                        "variant": v,
                        "rmse_mean": float(np.mean(rmses)) if rmses else np.nan,
                        "rmse_std": float(np.std(rmses)) if rmses else np.nan,
                        "mae_mean": float(np.mean(maes)) if maes else np.nan,
                        "mae_std": float(np.std(maes)) if maes else np.nan,
                        "medae_mean": float(np.mean(medaes)) if medaes else np.nan,
                        "medae_std": float(np.std(medaes)) if medaes else np.nan,
                        "n": int(len(rmses)),
                    }
                )

            return pd.DataFrame(rows).sort_values("rmse_mean"), alpha_series

        tbl, alpha_series = evaluate_with_aliases()
        tbl = tbl.copy()
        tbl.insert(0, "signature", sig.name)
        all_sig_tables.append(tbl)
        tbl.to_csv(out_dir / f"spy_fixed_split_{sig.name}.csv", index=False)
        (out_dir / f"spy_fixed_split_{sig.name}_blog.md").write_text(
            "<!-- generated by signature_in_volatility_forecast.py -->\n\n"
            + _df_to_markdown_table(_make_blog_table(tbl)),
            encoding="utf-8",
        )

        if alpha_series:
            df_alpha = pd.concat(alpha_series, axis=1)
            alpha_path = out_dir / f"spy_fixed_split_{sig.name}_alphas.csv"
            df_alpha.to_csv(alpha_path)
            logger.info(f"Saved alpha time series to {alpha_path}")

        logger.info("\n" + tbl.to_string(index=False))

    if all_sig_tables:
        df_all = pd.concat(all_sig_tables, axis=0, ignore_index=True)
        df_all.to_csv(out_dir / "spy_fixed_split_all_signatures.csv", index=False)
        (out_dir / "spy_fixed_split_all_signatures_blog.md").write_text(
            "<!-- generated by signature_in_volatility_forecast.py -->\n\n"
            + _df_to_markdown_table(_make_blog_table(df_all)),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
