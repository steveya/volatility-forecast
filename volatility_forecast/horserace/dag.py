"""Horserace DAG orchestrators: run_dag (single day) and run_backfill (walk-forward)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from volatility_forecast.horserace.refit_policy import resolve_refit_policy
from volatility_forecast.horserace.stages import (
    build_model_registry,
    compute_leaderboard,
    fetch_data,
    fit_or_load,
    forecast_next_day,
    next_trading_day,
    record_actuals,
    split_data,
)
from volatility_forecast.storage import VolForecastStore

logger = logging.getLogger(__name__)


def run_dag(
    config: dict,
    as_of_date: date,
    *,
    force_retrain: bool = False,
    model_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Execute one day of the horserace DAG.

    Stages
    ------
    1. Fetch data (once per ticker, shared across models)
    2. Compute split indices
    3. For each model: check refit policy → fit or load
    4. For each model: generate forecast for the next trading day
    5. Record actuals + compute leaderboard

    Parameters
    ----------
    config : dict
        Loaded horserace config.
    as_of_date : date
        The date whose close data is available. Forecasts target the next trading day.
    force_retrain : bool
        Override refit policy and retrain all models.
    model_filter : str, optional
        Run only this model name.

    Returns
    -------
    pd.DataFrame
        The leaderboard after this run.
    """
    store = VolForecastStore(root=config["store"]["root"])
    training_cfg = config["training"]
    lb_cfg = config.get("leaderboard", {})

    # Build model registry
    registry = build_model_registry(config)
    if model_filter:
        registry = {k: v for k, v in registry.items() if k == model_filter}

    for ticker in config["data"]["tickers"]:
        logger.info("=== %s  as_of=%s ===", ticker, as_of_date)

        # Stage 1: fetch data
        bundle = fetch_data(config, ticker, as_of_date)
        n = len(bundle.X)
        if n < training_cfg.get("min_history_days", 500):
            logger.warning("Only %d observations for %s, skipping", n, ticker)
            continue

        # Stage 2: split
        split = split_data(bundle, config)
        logger.info(
            "Split: burn_in=%d  train_end=%d  oos_start=%d  oos_end=%d",
            split.burn_in, split.train_end, split.oos_start, split.oos_end,
        )

        # Record actuals for all dates we have realized data
        n_actuals = record_actuals(store, bundle.y, bundle.dates, ticker)
        logger.info("Recorded %d actuals for %s", n_actuals, ticker)

        # Determine target date
        target_date = next_trading_day(as_of_date)

        # Stages 3 + 4: per model
        for model_name, model_instance in registry.items():
            api = config["models"][model_name].get("api", "sklearn")
            policy = resolve_refit_policy(config, model_name, force_retrain)

            # Stage 3: fit or load
            fit_result = fit_or_load(
                model_name, model_instance, api, bundle, split,
                config, store, policy,
            )

            # Stage 4: forecast
            forecast_next_day(
                fit_result, api, bundle, store,
                ticker, as_of_date, target_date,
            )

    # Stage 5: compute leaderboard
    leaderboard = pd.DataFrame()
    for ticker in config["data"]["tickers"]:
        lb = compute_leaderboard(
            store,
            ticker,
            window_days=lb_cfg.get("rolling_window_days", 252),
            metric_names=lb_cfg.get("metrics"),
            primary_metric=lb_cfg.get("primary_metric", "qlike"),
            as_of_utc=pd.Timestamp(str(as_of_date), tz="UTC"),
        )
        if not lb.empty:
            lb.insert(0, "ticker", ticker)
            leaderboard = pd.concat([leaderboard, lb])

    return leaderboard


def run_backfill(
    config: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Walk-forward backfill: run the DAG for each trading day in [start, end].

    Each day, models are trained on all data available up to that date
    (expanding window). The refit policy controls how often retraining occurs.
    """
    bdays = pd.bdate_range(start=str(start_date), end=str(end_date), freq="B")
    logger.info("Backfilling %d trading days: %s to %s", len(bdays), start_date, end_date)

    last_leaderboard = pd.DataFrame()
    for i, ts in enumerate(bdays):
        d = ts.date()
        logger.info("[%d/%d] Running for %s", i + 1, len(bdays), d)
        last_leaderboard = run_dag(config, d)

    return last_leaderboard
