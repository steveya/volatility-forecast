"""
Volatility Forecasts (Part 2) — Monte Carlo & Robustness Analysis.

1. Simulated Contaminated GARCH: 100 Runs (New Path + New Init each run).
2. SPY Data: 100 Runs (Fixed Data + New Init each run).

Outputs:
- Aggregated Metrics (Mean ± Std)
- Full Alpha Histories for all runs.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm

# --- Setup Paths & Logging ---
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)
load_dotenv()

logging.basicConfig(
    level=os.environ.get("AF_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- AlphaForge / Model Imports ---
from volatility_forecast.sources.simulated_garch import SimulatedGARCHAdapter
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

# Feature Templates
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget

# Models
from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel
from volatility_forecast.model.tree_stes_model import XGBoostSTESModel

# --- Configuration ---
OUT_DIR = Path("outputs/volatility_forecast_2_monte_carlo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configs
START_DATE = pd.Timestamp("2000-01-01", tz="UTC")
END_DATE = pd.Timestamp("2023-01-01", tz="UTC")
N_LAGS = 0  # Matches 'copy.py' (current features only)

# Monte Carlo Config
N_RUNS = 1  # Number of seeds/paths

# Split Configs
# Simulated: Train [500:2000], Test [2000:]
SIM_N_PERIODS = 2500
SIM_TRAIN_START = 500
SIM_SPLIT_INDEX = 2000

# SPY: Train [200:4000], Test [4000:]
SPY_END_DATE = pd.Timestamp("2023-12-31", tz="UTC")
SPY_TRAIN_START = 200
SPY_SPLIT_INDEX = 4000

# XGBoost Params
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
SCALE_FACTOR = 100.0  # Scale returns by 100 (Vol Points) so Variance is scaled by 10000
# This aligns gradients to O(1) for stable End-to-End fitting.

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def calc_metrics(y_true, y_pred, prefix):
    """Calculate RMSE, MAE, MedAE."""
    e = y_true - y_pred
    ae = np.abs(e)
    return {
        f"{prefix}_rmse": np.sqrt(np.mean(e**2)),
        f"{prefix}_mae": np.mean(ae),
        f"{prefix}_medae": np.median(ae),
    }


def scale_data(X_train, X_test):
    """Standard scale features (fit on train, apply to test)."""
    scaler = StandardScaler()
    cols_to_scale = [c for c in X_train.columns if c != "const"]

    if not cols_to_scale:
        return X_train, X_test

    X_tr_s = X_train.copy()
    X_te_s = X_test.copy()

    X_tr_s[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_te_s[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_tr_s, X_te_s


def inverse_scale_pred(y_pred_scaled):
    """Convert scaled variance predictions back to original decimal units."""
    return y_pred_scaled / (SCALE_FACTOR**2)


def make_stes_grids():
    """Define hyperparameter grids for CV."""
    grid_l2 = [
        {"l2_reg": 0.01},
        {"l2_reg": 0.1},
        {"l2_reg": 1.0},
    ]
    return grid_l2


def make_xgbstes_grids():
    """Define hyperparameter grids for CV."""
    # Grid for End-to-End (Recursive) - H is ~1.0 due to scaling, so MCW is 'days'
    grid_e2e = [
        {"min_child_weight": 5.0, "learning_rate": 0.05, "max_depth": 3},
        {"min_child_weight": 20.0, "learning_rate": 0.1, "max_depth": 3},
        {"min_child_weight": 50.0, "learning_rate": 0.05, "max_depth": 2},
    ]
    # Grid for Alternating (Regression) - Standard XGB tuning
    grid_alt = [
        {"min_child_weight": 1.0, "learning_rate": 0.05, "max_depth": 4},
        {"min_child_weight": 10.0, "learning_rate": 0.1, "max_depth": 3},
    ]
    return grid_e2e, grid_alt


def fit_single_run(
    run_id: int,
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    train_start: int,
    split_index: int,
):
    """
    Executes one run of model fitting and prediction for a specific seed.
    Returns dictionaries of metrics, alpha series, and predictions.
    """
    # 1. Split
    X_train = X.iloc[train_start:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[train_start:split_index]
    y_test = y.iloc[split_index:]
    r_train = r.iloc[train_start:split_index]
    r_test = r.iloc[split_index:]

    # 2. Feature Scaling (StandardScaler)
    X_train_s, X_test_s = scale_data(X_train, X_test)

    # 3. Target/Return Scaling (Vol Points)
    # We scale Inputs (r) and Targets (y) for training stability
    r_train_scaled = r_train * SCALE_FACTOR
    y_train_scaled = y_train * (SCALE_FACTOR**2)

    # 4. Define Models (seeded) & Grids
    grid_e2e, grid_alt = make_xgbstes_grids()
    grid_l2 = make_stes_grids()

    models = {
        "ES": ESModel(),
        "STES": STESModel(),
        "XGBSTES_Alt": XGBoostSTESModel(
            xgb_params=XGB_PARAMS,
            num_boost_round=XGB_ROUNDS,
            fit_method="alternating",
        ),
        "XGBSTES_E2E": XGBoostSTESModel(
            xgb_params=XGB_PARAMS,
            num_boost_round=XGB_ROUNDS,
            fit_method="end_to_end",
            e2e_grad_hess_scale=1.0,  # Data is scaled, so gradients are O(1) naturally
            e2e_debug=True,
            e2e_debug_print_once=True,
        ),
    }

    run_metrics = {}
    run_alphas = {}
    run_preds = {}

    for name, model in models.items():
        # Handle ES feature constraint
        if name == "ES":
            X_tr_use = X_train_s[["const"]]
            X_te_use = X_test_s[["const"]]
        else:
            X_tr_use = X_train_s
            X_te_use = X_test_s

        try:
            # --- Fit (with Scaling & Auto-CV) ---
            if "XGBSTES" in name:
                # Determine which grid to use
                grid = grid_alt if "Alt" in name else grid_e2e

                model.fit(
                    X_tr_use,
                    y_train_scaled,  # TRAIN ON SCALED TARGETS
                    returns=r_train_scaled,  # TRAIN ON SCALED RETURNS
                    perform_cv=True,
                    cv_grid=grid,
                    cv_splits=3,
                )
            elif "STES" in name:
                # Fit STES on original (unscaled) data. Features are scaled,
                # and regularization is impose on the margin of the model.
                model.fit(
                    X_tr_use,
                    y_train,
                    returns=r_train,
                    # perform_cv=True,
                    # cv_grid=grid_l2,
                    # cv_splits=3,
                )
            else:
                # Baseline models (ES/STES) might not need scaling or CV in this context
                # But for fairness, let's fit them on original data or handle internally.
                # The existing ES/STES implementations are robust to scale, but let's use raw.
                model.fit(X_tr_use, y_train, returns=r_train)

            # --- Prediction Strategy: Concat + Slice (Warm Start) ---
            # To match the old script and ensure variance state (v_t) flows
            # correctly from Train -> Test, we predict on the full concatenated series.

            X_all = pd.concat([X_tr_use, X_te_use])
            r_all = pd.concat([r_train, r_test])

            # For XGB models, we must predict using scaled returns to match training physics
            r_all_scaled = r_all * SCALE_FACTOR
            n_train = len(X_tr_use)

            # Predict on full history
            if isinstance(model, XGBoostSTESModel):
                # Predict scaled variance
                y_pred_scaled = model.predict(X_all, returns=r_all_scaled)
                # Inverse transform to original units
                y_pred_all = inverse_scale_pred(y_pred_scaled)
                alpha_all = model.get_alphas(X_all)

            elif isinstance(model, STESModel):
                y_pred_vals, alpha_vals = model.predict_with_alpha(X_all, returns=r_all)
                y_pred_all = pd.Series(y_pred_vals, index=X_all.index)
                alpha_all = pd.Series(alpha_vals, index=X_all.index)

            else:  # ES Model
                y_pred_all = model.predict(X_all, returns=r_all)
                # ES alpha is typically constant
                alpha_val = getattr(model, "alpha_", np.nan)
                alpha_all = pd.Series(alpha_val, index=X_all.index)

            # --- Slice IS / OS ---
            y_pred_is = y_pred_all.iloc[:n_train]
            y_pred_os = y_pred_all.iloc[n_train:]

            alpha_is = alpha_all.iloc[:n_train]
            alpha_os = alpha_all.iloc[n_train:]

            # Calculate Metrics
            def get_scores(yt, yp, tag):
                idx = yt.index.intersection(yp.index)
                return calc_metrics(yt.loc[idx], yp.loc[idx], tag)

            m_is = get_scores(y_train, pd.Series(y_pred_is, index=X_tr_use.index), "IS")
            m_os = get_scores(y_test, pd.Series(y_pred_os, index=X_te_use.index), "OS")

            # Store Metrics
            flat_m = {"Model": name, "Run": run_id}
            flat_m.update(m_is)
            flat_m.update(m_os)
            run_metrics[name] = flat_m

            # Store Alphas (concatenated full history)
            # Use distinct column name: Model_RunID
            full_alpha = pd.concat([alpha_is, alpha_os])
            run_alphas[f"{name}_run_{run_id}"] = full_alpha

            # Store Predictions (concatenated full history)
            run_preds[f"{name}_run_{run_id}"] = y_pred_all

        except Exception as e:
            logger.error(f"Run {run_id} Model {name} failed: {e}")
            run_metrics[name] = None
            run_alphas[f"{name}_run_{run_id}"] = None
            run_preds[f"{name}_run_{run_id}"] = None

    return run_metrics, run_alphas, run_preds


# -------------------------------------------------------------------------
# Orchestrators
# -------------------------------------------------------------------------


def process_results(all_metrics, all_alphas, all_preds, y_true, experiment_name):
    """Aggregates metrics and saves alpha history and predictions."""

    # 1. Metrics Aggregation
    flat_list = []
    for run_dict in all_metrics:
        for m_name, m_data in run_dict.items():
            if m_data:
                flat_list.append(m_data)

    df_raw = pd.DataFrame(flat_list)

    # Group by Model to get Mean/Std
    summary = df_raw.groupby("Model").agg(
        {
            "IS_rmse": ["mean", "std"],
            "IS_mae": ["mean", "std"],
            "IS_medae": ["mean", "std"],
            "OS_rmse": ["mean", "std"],
            "OS_mae": ["mean", "std"],
            "OS_medae": ["mean", "std"],
        }
    )

    # Flatten columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save Metrics
    metrics_path = OUT_DIR / f"{experiment_name}_metrics_summary.csv"
    summary.to_csv(metrics_path, index=False)

    raw_metrics_path = OUT_DIR / f"{experiment_name}_metrics_all_runs.csv"
    df_raw.to_csv(raw_metrics_path, index=False)

    # 2. Alpha Aggregation
    # Concatenate all series into one wide DataFrame
    valid_alphas = [
        s for run_alpha in all_alphas for s in run_alpha.values() if s is not None
    ]

    if valid_alphas:
        df_alphas = pd.concat(
            valid_alphas,
            axis=1,
            keys=[
                k
                for run_alpha in all_alphas
                for k in run_alpha.keys()
                if run_alpha[k] is not None
            ],
        )
        alpha_path = OUT_DIR / f"{experiment_name}_alphas_all_runs.csv"
        # Warning: This file can be large (e.g. 2500 rows * 400 cols)
        df_alphas.to_csv(alpha_path)
        logger.info(f"Saved {experiment_name} alphas to {alpha_path}")

    # 3. Predictions Aggregation
    valid_preds = [
        s for run_pred in all_preds for s in run_pred.values() if s is not None
    ]

    if valid_preds:
        df_preds = pd.concat(
            valid_preds,
            axis=1,
            keys=[
                k
                for run_pred in all_preds
                for k in run_pred.keys()
                if run_pred[k] is not None
            ],
        )
        # Add the true y values as the first column
        if y_true is not None:
            df_preds.insert(0, "y_true", y_true)
        pred_path = OUT_DIR / f"{experiment_name}_predictions_all_runs.csv"
        df_preds.to_csv(pred_path)
        logger.info(f"Saved {experiment_name} predictions to {pred_path}")

    # Print Summary
    n_runs = len(df_raw["run"].unique()) if "run" in df_raw.columns else N_RUNS
    print(
        f"\n=== Results: {experiment_name} ({n_runs} run{'s' if n_runs != 1 else ''}) ==="
    )

    # For single runs, omit std columns
    if n_runs == 1:
        print(summary[["Model", "IS_rmse_mean", "OS_rmse_mean"]].to_string(index=False))
    else:
        print(
            summary[
                ["Model", "IS_rmse_mean", "IS_rmse_std", "OS_rmse_mean", "OS_rmse_std"]
            ].to_string(index=False)
        )
    print("-" * 60)


def _build_sim_ctx(*, run_seed: int, tiingo_cache_backends):
    return build_default_ctx(
        tiingo_cache_backends=tiingo_cache_backends,
        extra_adapters=(
            SimulatedGARCHAdapter(
                source_name="simulated_garch",
                n_periods=SIM_N_PERIODS,
                random_state=run_seed,
                mu=0.0,
                omega=0.02,
                alpha=0.11,
                beta=0.87,
                eta=4.0,
                shock_prob=0.005,
                entity_id="SIM",
            ),
        ),
    )


def run_simulated_study(*, tiingo_cache_backends):
    logger.info(f"Starting Simulated Study ({N_RUNS} runs)...")

    all_metrics = []
    all_alphas = []
    all_preds = []
    y_true = None

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_RUNS)

    for seed in tqdm(seeds, desc="Simulated Runs"):
        sim_ctx = _build_sim_ctx(
            run_seed=int(seed), tiingo_cache_backends=tiingo_cache_backends
        )
        spec = build_dataset_spec(
            "simulated_garch", "SIM", START_DATE, END_DATE, N_LAGS
        )
        X, y, r, _ = build_wide_dataset(sim_ctx, spec, entity_id="SIM")

        if isinstance(X.index, pd.MultiIndex):
            X = X.droplevel(0)
            y = y.droplevel(0)
            r = r.droplevel(0)
        X, y, r = prep_data_frames(X, y, r)

        # Re-index for generic integer steps (since simulated dates might shift or not matter)
        # This ensures alpha concatenation works smoothly across runs
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        r = r.reset_index(drop=True)

        # Store y from the first run (all runs will have different data for simulated)
        if y_true is None:
            y_true = y.copy()

        m, a, p = fit_single_run(seed, X, y, r, SIM_TRAIN_START, SIM_SPLIT_INDEX)
        all_metrics.append(m)
        all_alphas.append(a)
        all_preds.append(p)

    process_results(all_metrics, all_alphas, all_preds, y_true, "Simulated_GARCH")


def run_spy_study(ctx):
    logger.info(f"Starting SPY Study (1 run)...")

    # 1. Load Data ONCE
    spec_spy = build_dataset_spec("tiingo", "SPY", START_DATE, SPY_END_DATE, N_LAGS)
    try:
        X, y, r, _ = build_vol_dataset(ctx, spec_spy, persist=False)
        X = X.xs("SPY", level="entity_id")
        y = y.xs("SPY", level="entity_id")
        r = r.xs("SPY", level="entity_id")
        X, y, r = prep_data_frames(X, y, r)
    except Exception as e:
        logger.error(f"SPY Data Load Failed: {e}")
        return

    all_metrics = []
    all_alphas = []
    all_preds = []

    # Single run with fixed seed
    seed = 42
    m, a, p = fit_single_run(seed, X, y, r, SPY_TRAIN_START, SPY_SPLIT_INDEX)
    all_metrics.append(m)
    all_alphas.append(a)
    all_preds.append(p)

    process_results(all_metrics, all_alphas, all_preds, y, "SPY_Analysis")


# -------------------------------------------------------------------------
# Dataset Builders
# -------------------------------------------------------------------------


def build_dataset_spec(
    source: str, ticker: str, start, end, lags: int
) -> VolDatasetSpec:
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": lags,
                "source": source,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": lags,
                "source": source,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": lags,
                "source": source,
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={
            "source": source,
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1.0,
        },
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[ticker]),
        time=TimeSpec(start=start, end=end, calendar="XNYS", grid="B", asof=None),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def prep_data_frames(X_raw, y_raw, r_raw):
    X = X_raw.sort_index().copy()
    if "const" not in X.columns:
        X["const"] = 1.0
    X = X.astype(float)
    y = y_raw.sort_index().astype(float)
    r = r_raw.sort_index().astype(float)
    return X, y, r


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main():
    tiingo_key = os.environ.get("TIINGO_API")
    cache = FileCacheBackend(Path(".af_cache"))
    ctx = build_default_ctx(tiingo_api_key=tiingo_key, tiingo_cache_backends=[cache])

    # 1. Simulated
    run_simulated_study(tiingo_cache_backends=[cache])

    # 2. SPY
    if tiingo_key:
        run_spy_study(ctx)
    else:
        logger.warning("TIINGO_API missing. Skipping SPY.")


if __name__ == "__main__":
    main()
