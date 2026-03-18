"""Pure stage functions for the horserace execution DAG.

Each function implements one stage of the pipeline:
  fetch_data  →  split_data  →  fit_or_load  →  forecast_next_day  →  score_and_record
"""

from __future__ import annotations

import importlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from alphaforge.features.dataset_spec import (
    FeatureRequest,
    JoinPolicy,
    MissingnessPolicy,
    TargetRequest,
    TimeSpec,
    UniverseSpec,
)

from volatility_forecast.evaluation import metrics as eval_metrics
from volatility_forecast.features.return_features import (
    LagAbsLogReturnTemplate,
    LagLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.horserace.refit_policy import RefitPolicy
from volatility_forecast.horserace.types import DataBundle, FitResult, SplitSpec
from volatility_forecast.pipeline import (
    VolDatasetSpec,
    build_default_ctx,
    build_wide_dataset,
)
from volatility_forecast.storage import VolForecastStore
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "horserace.yaml"


def load_config(path: Optional[str] = None) -> dict:
    """Load and return the horserace configuration."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _import_class(dotted: str) -> type:
    module_path, class_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_model_registry(config: dict) -> Dict[str, Any]:
    """Instantiate each model from config. Returns {name: model_instance}."""
    registry: Dict[str, Any] = {}
    for name, spec in config["models"].items():
        try:
            cls = _import_class(spec["class"])
            params = spec.get("params", {})
            registry[name] = cls(**params)
            logger.info("Registered model %s (%s)", name, spec["class"])
        except Exception:
            logger.warning("Failed to import model %s, skipping", name, exc_info=True)
    return registry


# ---------------------------------------------------------------------------
# Stage 1: Data fetch
# ---------------------------------------------------------------------------

_TIINGO_PARAMS = {
    "source": "tiingo",
    "table": "market.ohlcv",
    "price_col": "close",
}


def _build_dataset_spec(config: dict, ticker: str, end_date: date) -> VolDatasetSpec:
    """Build a VolDatasetSpec for a single ticker up to *end_date*."""
    data_cfg = config["data"]
    lags = data_cfg.get("feature_lags", 5)
    scale = data_cfg.get("target_scale", 1.0)

    features = tuple(
        FeatureRequest(
            template=tmpl(),
            params={"lags": lags, **_TIINGO_PARAMS},
        )
        for tmpl in [
            LagLogReturnTemplate,
            LagAbsLogReturnTemplate,
            LagSquaredLogReturnTemplate,
        ]
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={"scale": scale, **_TIINGO_PARAMS},
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[ticker]),
        time=TimeSpec(
            start=pd.Timestamp(data_cfg["history_start"], tz="UTC"),
            end=pd.Timestamp(str(end_date), tz="UTC"),
            calendar="XNYS",
            grid="B",
            asof=None,
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def fetch_data(config: dict, ticker: str, as_of_date: date) -> DataBundle:
    """Stage 1: Fetch aligned (X, y, returns) for *ticker* through *as_of_date*."""
    ctx = build_default_ctx()
    spec = _build_dataset_spec(config, ticker, as_of_date)
    X, y, returns, _catalog = build_wide_dataset(ctx, spec, entity_id=ticker)
    dates = pd.DatetimeIndex(X.index)
    return DataBundle(ticker=ticker, X=X, y=y, returns=returns, dates=dates)


# ---------------------------------------------------------------------------
# Stage 2: Data split
# ---------------------------------------------------------------------------


def split_data(bundle: DataBundle, config: dict) -> SplitSpec:
    """Stage 2: Compute train/OOS split indices.

    Uses an expanding training window: train_end = len(dates) (all available data).
    ``oos_start`` marks where the out-of-sample scoring region begins.
    """
    burn_in = config["training"].get("burn_in", 100)
    oos_start_str = config["training"].get("oos_start")

    if oos_start_str:
        oos_ts = pd.Timestamp(oos_start_str, tz="UTC")
        oos_idx = int(bundle.dates.searchsorted(oos_ts))
    else:
        # No OOS cutoff configured — treat everything as training, no scoring region
        oos_idx = len(bundle.dates)

    return SplitSpec(
        burn_in=min(burn_in, len(bundle.dates)),
        train_end=len(bundle.dates),
        oos_start=oos_idx,
        oos_end=len(bundle.dates),
    )


# ---------------------------------------------------------------------------
# Stage 3: Fit or load
# ---------------------------------------------------------------------------


def _fit_model(
    model: Any,
    api: str,
    X: Any,
    y: Any,
    returns: Any,
    burn_in: int,
    train_end: int,
) -> Any:
    """Fit a model using the appropriate API convention.

    For sklearn-style models, X/y/returns may be DataFrames/Series (models may
    need column names, e.g. ESModel does X[["const"]]).
    For pgarch models, numpy arrays are expected.
    """
    if api == "pgarch":
        Xn = np.asarray(X, dtype=float)
        yn = np.asarray(y, dtype=float)
        model.fit(yn[:train_end], Xn[:train_end])
    else:
        try:
            model.fit(X, y, returns=returns, start_index=burn_in, end_index=train_end)
        except TypeError:
            try:
                model.fit(X, y, returns, burn_in, train_end)
            except TypeError:
                try:
                    model.fit(X, y, returns=returns)
                except TypeError:
                    model.fit(X, y)
    return model


def _predict_model(
    model: Any,
    api: str,
    X: Any,
    y: Any,
    returns: Any,
) -> np.ndarray:
    """Generate predictions using the appropriate API convention."""
    if api == "pgarch":
        Xn = np.asarray(X, dtype=float)
        yn = np.asarray(y, dtype=float)
        return model.predict_variance(yn, Xn)
    else:
        try:
            return model.predict(X, returns=returns)
        except TypeError:
            return model.predict(X)


def fit_or_load(
    model_name: str,
    model_instance: Any,
    api: str,
    bundle: DataBundle,
    split: SplitSpec,
    config: dict,
    store: VolForecastStore,
    policy: RefitPolicy,
) -> FitResult:
    """Stage 3: Decide whether to retrain, then fit or load the model."""
    should_train = policy.should_refit(store, model_name, bundle)

    if should_train:
        logger.info("Training %s ...", model_name)
        burn_in = split.burn_in
        train_end = split.train_end

        # For sklearn-style models, pass DataFrames (they may need column names).
        # For pgarch, _fit_model handles numpy conversion internally.
        X_fit = bundle.X
        y_fit = bundle.y
        r_fit = bundle.returns

        fitted = _fit_model(model_instance, api, X_fit, y_fit, r_fit, burn_in, train_end)

        # Compute in-sample metric
        yhat = _predict_model(fitted, api, X_fit, y_fit, r_fit)
        yn_flat = bundle.y.values.astype(float)
        try:
            metric_val = float(
                eval_metrics.qlike(yn_flat[burn_in:train_end], yhat[burn_in:train_end])
            )
        except Exception:
            metric_val = None

        dates = bundle.dates
        model_id = store.register_model(
            model_type=model_name,
            model_obj=fitted,
            trained_start=pd.Timestamp(dates[0]),
            trained_end=pd.Timestamp(dates[train_end - 1]),
            metric_name="qlike_is",
            metric_value=metric_val,
        )
        logger.info("Registered %s  qlike_is=%s", model_id, metric_val)
        return FitResult(
            model_id=model_id,
            fitted_model=fitted,
            was_retrained=True,
            train_metric=metric_val,
        )
    else:
        model_id = store.latest_model_id(model_name)
        fitted = store.load_latest_model(model_name)
        logger.info("Using existing %s", model_id)
        return FitResult(
            model_id=model_id or "",
            fitted_model=fitted,
            was_retrained=False,
            train_metric=None,
        )


# ---------------------------------------------------------------------------
# Stage 4: Forecast
# ---------------------------------------------------------------------------


def next_trading_day(d: date) -> date:
    """Return the next weekday after *d* (simple weekend skip)."""
    nxt = d + timedelta(days=1)
    while pd.Timestamp(nxt).weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def forecast_next_day(
    fit_result: FitResult,
    api: str,
    bundle: DataBundle,
    store: VolForecastStore,
    ticker: str,
    as_of_date: date,
    target_date: date,
    horizon: int = 1,
) -> Optional[str]:
    """Stage 4: Generate and store a one-step-ahead forecast. Returns forecast_id."""
    if fit_result.fitted_model is None:
        logger.warning("No fitted model for forecast, skipping")
        return None

    yhat_arr = _predict_model(fit_result.fitted_model, api, bundle.X, bundle.y, bundle.returns)
    yhat_val = float(yhat_arr[-1])

    forecast_id = store.upsert_forecast(
        model_id=fit_result.model_id,
        ticker=ticker,
        asof_utc=pd.Timestamp(str(as_of_date), tz="UTC"),
        target_utc=pd.Timestamp(str(target_date), tz="UTC"),
        horizon=horizon,
        yhat=yhat_val,
    )
    logger.info(
        "Forecast %s  model=%s  yhat=%.6f  target=%s",
        forecast_id,
        fit_result.model_id,
        yhat_val,
        target_date,
    )
    return forecast_id


# ---------------------------------------------------------------------------
# Stage 5: Score and record
# ---------------------------------------------------------------------------


def record_actuals(
    store: VolForecastStore,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    ticker: str,
    horizon: int = 1,
) -> int:
    """Upsert realized volatility for all dates in *y*. Returns count."""
    count = 0
    for dt, val in zip(dates, y.values):
        if np.isfinite(val):
            store.upsert_actual(
                ticker=ticker,
                target_utc=pd.Timestamp(dt),
                horizon=horizon,
                y=float(val),
            )
            count += 1
    return count


_METRIC_FNS = {
    "qlike": eval_metrics.qlike,
    "rmse": eval_metrics.rmse,
    "mae": eval_metrics.mae,
    "hit_rate": eval_metrics.hit_rate,
    "corr": eval_metrics.corr,
}

_LOWER_IS_BETTER = {"qlike", "rmse", "mae"}


def compute_leaderboard(
    store: VolForecastStore,
    ticker: str,
    window_days: int = 252,
    metric_names: Optional[List[str]] = None,
    primary_metric: str = "qlike",
    as_of_utc: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute a leaderboard from stored forecasts vs actuals."""
    if metric_names is None:
        metric_names = ["qlike", "rmse", "mae", "hit_rate", "corr"]

    end = as_of_utc or pd.Timestamp.now("UTC")
    start = end - pd.Timedelta(days=int(window_days * 1.5))

    df = store.join_forecast_vs_actual(ticker=ticker, start_utc=start, end_utc=end)
    if df.empty:
        logger.warning("No forecast-vs-actual data for %s", ticker)
        return pd.DataFrame()

    df["model_type"] = df["model_id"].str.split(":").str[0]

    rows = []
    for model_type, grp in df.groupby("model_type"):
        grp = grp.sort_values("target_utc").tail(window_days)
        y_actual = grp["y_actual"].values
        yhat = grp["yhat"].values

        row: Dict[str, Any] = {"model": model_type, "n_obs": len(grp)}
        for mname in metric_names:
            fn = _METRIC_FNS.get(mname)
            if fn is not None:
                try:
                    row[mname] = float(fn(y_actual, yhat))
                except Exception:
                    row[mname] = float("nan")
        rows.append(row)

    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb

    ascending = primary_metric in _LOWER_IS_BETTER
    lb = lb.sort_values(primary_metric, ascending=ascending).reset_index(drop=True)
    lb.index = lb.index + 1
    lb.index.name = "rank"
    return lb
