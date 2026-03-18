"""Horserace execution DAG for volatility forecast model competition."""

from volatility_forecast.horserace.dag import run_backfill, run_dag
from volatility_forecast.horserace.refit_policy import (
    EveryNDaysPolicy,
    ExpandingWindowPolicy,
    RefitPolicy,
    build_refit_policy,
    resolve_refit_policy,
)
from volatility_forecast.horserace.stages import (
    build_model_registry,
    compute_leaderboard,
    fetch_data,
    fit_or_load,
    forecast_next_day,
    load_config,
    record_actuals,
    split_data,
)
from volatility_forecast.horserace.types import DataBundle, FitResult, SplitSpec

__all__ = [
    "run_dag",
    "run_backfill",
    "load_config",
    "build_model_registry",
    "fetch_data",
    "split_data",
    "fit_or_load",
    "forecast_next_day",
    "record_actuals",
    "compute_leaderboard",
    "RefitPolicy",
    "EveryNDaysPolicy",
    "ExpandingWindowPolicy",
    "build_refit_policy",
    "resolve_refit_policy",
    "DataBundle",
    "SplitSpec",
    "FitResult",
]
