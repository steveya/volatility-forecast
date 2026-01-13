"""
Volatility Forecasts (Part 2) — XGBoost-STES experiments.

Builds datasets via VolDatasetSpec (uses return feature templates and
NextDaySquaredReturnTarget), fits ES/STES/XGBoost-STES, and prints OOS metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd

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

# Simulated data source
from volatility_forecast.sources.simulated_garch import SimulatedGARCHSource

from volatility_forecast.pipeline import (
    build_default_ctx,
    build_vol_dataset,
    VolDatasetSpec,
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
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget
from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel

from volatility_forecast.model.xgboost_stes_model import (
    XGBoostSTESModel,
    DEFAULT_XGBOOST_PARAMS,
)

from volatility_forecast.evaluation import metrics
from enum import Enum
from functools import partial

from volatility_forecast.evaluation.model_evaluator import (
    evaluate_model,
    root_mean_squared_error,
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
    XGBOOST_STES = "XGBoost_STES"


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
    "XGBoost-STES",
]

# SPY study settings (notebook-aligned)
SPY_TICKER = "SPY"
SPY_START = pd.Timestamp("2000-01-01", tz="UTC")
SPY_END = pd.Timestamp("2023-12-31", tz="UTC")
SPY_IS_INDEX = 200
SPY_OS_INDEX = 4000
SPY_N_INITS = 100


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


def prepare_wide(ctx: "DataContext", spec: VolDatasetSpec, *, entity_id: str = ENTITY):
    """Build wide dataset once; return X, y, returns for a single entity and a column catalog."""
    X, y, returns, catalog = build_vol_dataset(ctx, spec, persist=False)

    X1 = X.xs(entity_id, level="entity_id").sort_index().copy()
    y1 = y.xs(entity_id, level="entity_id").sort_index()
    r1 = returns.xs(entity_id, level="entity_id").sort_index()

    if "const" not in X1.columns:
        X1["const"] = 1.0

    # ensure strict alignment
    idx = X1.index.intersection(y1.index).intersection(r1.index)
    X1, y1, r1 = X1.loc[idx], y1.loc[idx], r1.loc[idx]

    return X1, y1, r1, catalog


def detect_group_from_name(col: str) -> str:
    """Heuristic group detection from column name."""
    lc = col.lower()
    if "abs" in lc:
        return "abs"
    if "squared" in lc or "sq" in lc:
        return "sq"
    if "log" in lc or "ret" in lc:
        return "raw"
    return "raw"  # default


def select_columns_for_variant(X: pd.DataFrame, variant: str) -> list[str]:
    """
    Choose columns by group without recomputing:
    - raw = log-return lags
    - abs  = abs(log-return) lags
    - sq   = squared(log-return) lags
    Always include 'const' if present.
    """
    groups_needed = {
        "ES": set(),
        "STES_AE": {"abs"},
        "STES_SE": {"sq"},
        "STES_EAE": {"raw", "abs"},
        "STES_ESE": {"raw", "sq"},
        "STES_AESE": {"abs", "sq"},
        "STES_EAESE": {"raw", "abs", "sq"},
        "XGBoost-STES": {"raw", "abs", "sq"},
    }[variant]

    cols = []
    for c in X.columns:
        if c == "const":
            continue
        g = detect_group_from_name(c)
        if g in groups_needed:
            cols.append(c)

    if variant == "ES":
        return ["const"] if "const" in X.columns else []

    if "const" in X.columns:
        cols = ["const"] + cols
    return cols


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

    return X1, y1, r1, date


def evaluate_models(
    X: pd.DataFrame, y: pd.Series, os_index: int, returns: pd.Series | None = None
):
    models = {
        "ES": ESModel(),
        "STES": STESModel(),
    }
    if XGBoostSTESModel is not None:
        models["XGBoost-STES"] = XGBoostSTESModel(
            params={"max_depth": 4, "learning_rate": 0.1, "num_boost_round": 50}
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

            y_true = y.iloc[os_index:].values
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            qlike = metrics.qlike(y_true, y_pred)
            results[name] = {"rmse": rmse, "qlike": qlike}
            logger.info(f" OK — RMSE={rmse:.6f}, QLIKE={qlike:.6f}")
        except Exception as e:
            logger.warning(f" FAIL ({e})")
            results[name] = None
    return results


def random_cv_tune_xgboost_model(data_provider, n_iter: int = 80, n_splits: int = 3):
    """Randomized search using TimeSeriesSplit and evaluate_model.

    data_provider: callable returning (X, y, returns, date)
    """
    logger.info(
        f"Starting RandomizedSearchCV tuning (n_iter={n_iter}, n_splits={n_splits})"
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

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score = float("inf")
    best_params = None

    indices = np.ones((4000, 1))
    for params in param_list:
        scores = []
        for train_index, val_index in tscv.split(indices):
            try:
                model = XGBoostSTESModel(**params)
            except Exception as e:
                logger.exception(
                    f"Failed to instantiate XGBoostSTESModel with params {params}: {e}"
                )
                scores.append(np.inf)
                continue

            res, _, _, _ = evaluate_model(
                data_provider,
                model,
                root_mean_squared_error,
                0,
                val_index[0],
                val_index[-1],
            )
            scores.append(res)

        mean_rmse = float(np.mean(scores))
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    logger.info(f"RandomizedSearchCV best RMSE={best_score:.6f}")
    return best_params, best_score


def xgb_stes_optuna_objective(trial, data_provider):
    """Optuna objective (returns mean(-rmse) so that study can maximize)."""
    if optuna is None:
        raise RuntimeError("optuna not available")

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

    model = XGBoostSTESModel(**param)

    indices = np.ones((4000, 1))
    tscv = TimeSeriesSplit(n_splits=3)

    rmses = []
    for train_idx, valid_idx in tscv.split(indices):
        res, _, _, _ = evaluate_model(
            data_provider,
            model,
            root_mean_squared_error,
            0,
            valid_idx[0],
            valid_idx[-1],
        )
        rmses.append(-res)

    return float(np.mean(rmses))


def fit_and_score(variant: str, X: pd.DataFrame, y: pd.Series, r: pd.Series) -> float:
    """Fit model on in-sample, compute OOS RMSE (mirrors Part1)."""
    if len(y) <= OS_INDEX:
        raise ValueError(f"Insufficient rows for slicing: len(y)={len(y)}")

    cols = select_columns_for_variant(X, variant)
    if not cols:
        cols = ["const"]

    X_sel = X[cols]
    X_is, y_is = X_sel.iloc[IS_INDEX:OS_INDEX], y.iloc[IS_INDEX:OS_INDEX]
    X_os, y_os = X_sel.iloc[OS_INDEX:], y.iloc[OS_INDEX:]
    r_is, r_os = r.iloc[IS_INDEX:OS_INDEX], r.iloc[OS_INDEX:]

    if variant == "ES":
        model = ESModel()
    elif variant == "XGBoost-STES":
        if XGBoostSTESModel is None:
            raise RuntimeError("XGBoost-STES requested but xgboost is not available")
        model = XGBoostSTESModel(**DEFAULT_XGBOOST_PARAMS)
    else:
        model = STESModel()

    # Pass returns and index bounds where supported
    model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
    # predict on OOS slice using provided returns
    y_hat = model.predict(X_os, returns=r_os)
    return float(np.sqrt(np.mean((y_os.values - y_hat) ** 2)))


def run_simulated_experiment(ctx, n_runs: int = N_RUNS):
    logger.info("Starting simulated experiment")
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=n_runs)

    rmse_acc = {v: [] for v in VARIANTS}

    for i, seed in enumerate(seeds, start=1):
        if i % 5 == 0:
            logger.info(f"  Run {i}/{n_runs}")

        add_simulated_source(ctx, int(seed))
        spec = build_wide_spec(N_LAGS)
        try:
            X_wide, y, r, _ = prepare_wide(ctx, spec)
        except Exception as e:
            logger.warning(f"Skipping run {i}: {e}")
            continue

        for variant in VARIANTS:
            try:
                rmse = fit_and_score(variant, X_wide, y, r)
                rmse_acc[variant].append(rmse)
            except Exception as e:
                # log failure details for debugging
                logger.exception(f"Variant {variant} failed on run {i}: {e}")
                continue

    logger.info("Simulated experiment complete")
    for variant in VARIANTS:
        vals = rmse_acc[variant]
        if vals:
            logger.info(
                f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{n_runs})"
            )
        else:
            logger.info(f"{variant:12s}: N/A (0/{n_runs})")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Part 2 experiments and optional tuning for XGBoost-STES"
    )
    parser.add_argument(
        "--tune-sklearn",
        action="store_true",
        help="Run sklearn RandomizedSearchCV tuning for XGBoost-STES on SPY",
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
        help="Run Optuna tuning for XGBoost-STES on SPY",
    )
    parser.add_argument(
        "--tune-optuna-trials", type=int, default=100, help="Number of Optuna trials"
    )

    args = parser.parse_args()

    api_key = os.environ.get("TIINGO_API")

    ctx = build_default_ctx(tiingo_api_key=api_key)

    # Run simulated experiment (no Tiingo key required)
    run_simulated_experiment(ctx, n_runs=N_RUNS)

    if not api_key:
        logger.warning(
            "TIINGO_API not set; set in .env to run Part 2 with real data. Skipping SPY study."
        )
        return

    # Build SPY wide dataset and run randomized-initialization averaging (match Part 1)
    try:
        X, y, returns, date = build_spy_dataset(ctx, n_lags=N_LAGS)
    except Exception as e:
        logger.warning(
            "SPY via Tiingo failed. Ensure TIINGO_API is set in .env and valid.\n"
            f"Reason: {e}"
        )
        X = y = returns = date = None

    # Optional hyperparameter tuning for XGBoost-STES (sklearn RandomizedSearchCV + Optuna)
    if args.tune_sklearn or args.tune_optuna:
        if X is None:
            logger.warning("SPY dataset unavailable; skipping tuning")
        else:
            spy_data_provider = lambda: (X, y, returns, date)

            if args.tune_sklearn:
                if XGBoostSTESModel is None:
                    logger.warning("XGBoost not available; skipping sklearn tuning")
                else:
                    rcv_best_params, rcv_best_score = random_cv_tune_xgboost_model(
                        spy_data_provider, n_iter=args.tune_sklearn_iter
                    )
                    logger.info(
                        f"Sklearn tuning best params: {rcv_best_params} | RMSE: {rcv_best_score:.6f}"
                    )
                    tuned_params = DEFAULT_XGBOOST_PARAMS | (rcv_best_params or {})
                    try:
                        tuned_model = XGBoostSTESModel(**tuned_params)
                        os_res, is_res, _, _ = evaluate_model(
                            spy_data_provider,
                            tuned_model,
                            root_mean_squared_error,
                            10,
                            4000,
                        )
                        logger.info(
                            f"Sklearn tuned XGBoost-STES — Test RMSE={os_res:.6f}, Train RMSE={is_res:.6f}"
                        )
                    except Exception as e:
                        logger.exception(f"Failed to evaluate tuned sklearn model: {e}")

            if args.tune_optuna:
                if optuna is None:
                    logger.warning("optuna not available; skipping optuna tuning")
                elif XGBoostSTESModel is None:
                    logger.warning("XGBoost not available; skipping optuna tuning")
                else:
                    sampler = optuna.samplers.TPESampler(seed=0)
                    study = optuna.create_study(direction="maximize", sampler=sampler)
                    study.optimize(
                        lambda trial: xgb_stes_optuna_objective(
                            trial, spy_data_provider
                        ),
                        n_trials=args.tune_optuna_trials,
                    )
                    trial = study.best_trial
                    tuned_params = DEFAULT_XGBOOST_PARAMS | trial.params
                    logger.info(f"Optuna best params: {tuned_params}")
                    try:
                        tuned_model = XGBoostSTESModel(**tuned_params)
                        os_res, is_res, _, _ = evaluate_model(
                            spy_data_provider,
                            tuned_model,
                            root_mean_squared_error,
                            10,
                            4000,
                        )
                        logger.info(
                            f"Optuna tuned XGBoost-STES — Test RMSE={os_res:.6f}, Train RMSE={is_res:.6f}"
                        )
                    except Exception as e:
                        logger.exception(f"Failed to evaluate tuned optuna model: {e}")

    rmse_spy_oos = {v: [] for v in VARIANTS}
    rmse_spy_is = {v: [] for v in VARIANTS}

    if X is not None and len(y) > SPY_OS_INDEX:
        for variant in VARIANTS:
            for seed in range(SPY_N_INITS):
                try:
                    cols = select_columns_for_variant(X, variant)
                    if not cols:
                        cols = ["const"]
                    X_sel = X[cols]
                    X_is, y_is = (
                        X_sel.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                        y.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                    )
                    X_os, y_os = X_sel.iloc[SPY_OS_INDEX:], y.iloc[SPY_OS_INDEX:]
                    r_is, r_os = (
                        returns.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                        returns.iloc[SPY_OS_INDEX:],
                    )

                    if variant == "ES":
                        model = ESModel(random_state=seed)
                    elif variant == "XGBoost-STES":
                        if XGBoostSTESModel is None:
                            raise RuntimeError(
                                "XGBoost-STES requested but xgboost is not available"
                            )
                        params = dict(DEFAULT_XGBOOST_PARAMS)
                        params["random_state"] = seed
                        model = XGBoostSTESModel(**params)
                    else:
                        model = STESModel(random_state=seed)

                    model.fit(
                        X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is)
                    )

                    # In-sample RMSE
                    y_hat_is = model.predict(X_is, returns=r_is)
                    rmse_spy_is[variant].append(
                        float(np.sqrt(np.mean((y_is.values - y_hat_is) ** 2)))
                    )

                    # Out-of-sample RMSE
                    y_hat_os = model.predict(X_os, returns=r_os)
                    rmse_spy_oos[variant].append(
                        float(np.sqrt(np.mean((y_os.values - y_hat_os) ** 2)))
                    )
                except Exception as e:
                    logger.exception(
                        f"SPY variant {variant} failed on seed {seed}: {e}"
                    )
                    continue

        logger.info("\nIn-Sample RMSE on SPY (mean ± std)")
        for variant in VARIANTS:
            vals = rmse_spy_is[variant]
            if vals:
                logger.info(
                    f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{SPY_N_INITS})"
                )
            else:
                logger.info(f"{variant:12s}: N/A (0/{SPY_N_INITS})")

        logger.info("\nOut-of-Sample RMSE on SPY (mean ± std)")
        for variant in VARIANTS:
            vals = rmse_spy_oos[variant]
            if vals:
                logger.info(
                    f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{SPY_N_INITS})"
                )
            else:
                logger.info(f"{variant:12s}: N/A (0/{SPY_N_INITS})")
    else:
        logger.info("SPY study skipped (insufficient data).")


if __name__ == "__main__":
    main()
