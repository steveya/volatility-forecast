"""CLI entry point for the volatility forecast horserace."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from volatility_forecast.horserace import (
    compute_leaderboard,
    load_config,
    run_backfill,
    run_dag,
)
from volatility_forecast.storage import VolForecastStore


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_run(args):
    config = load_config(args.config)
    as_of = _parse_date(args.date) if args.date else date.today()
    lb = run_dag(config, as_of, force_retrain=args.retrain, model_filter=args.model)
    if not lb.empty:
        print("\n=== Leaderboard ===")
        print(lb.to_string())
    else:
        print("No leaderboard data yet.")


def cmd_backfill(args):
    config = load_config(args.config)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.oos_start:
        config["training"]["oos_start"] = args.oos_start
    lb = run_backfill(config, start, end)
    if not lb.empty:
        print("\n=== Final Leaderboard ===")
        print(lb.to_string())
    else:
        print("No leaderboard data after backfill.")


def cmd_leaderboard(args):
    config = load_config(args.config)
    store = VolForecastStore(root=config["store"]["root"])
    lb_cfg = config.get("leaderboard", {})
    window = args.window or lb_cfg.get("rolling_window_days", 252)

    for ticker in config["data"]["tickers"]:
        lb = compute_leaderboard(
            store,
            ticker,
            window_days=window,
            metric_names=lb_cfg.get("metrics"),
            primary_metric=lb_cfg.get("primary_metric", "qlike"),
        )
        if not lb.empty:
            print(f"\n=== {ticker} Leaderboard ({window}d window) ===")
            print(lb.to_string())
        else:
            print(f"No data for {ticker}.")


def cmd_retrain(args):
    config = load_config(args.config)
    as_of = _parse_date(args.date) if args.date else date.today()
    lb = run_dag(config, as_of, force_retrain=True, model_filter=args.model)
    if not lb.empty:
        print("\n=== Leaderboard ===")
        print(lb.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="volatility_forecast",
        description="Volatility forecast horserace CLI",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to horserace.yaml (default: config/horserace.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the daily horserace")
    p_run.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD, default: today)")
    p_run.add_argument("--retrain", action="store_true", help="Force retraining all models")
    p_run.add_argument("--model", default=None, help="Run only this model")
    p_run.set_defaults(func=cmd_run)

    # backfill
    p_bf = sub.add_parser("backfill", help="Backfill OOS forecasts over a date range")
    p_bf.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_bf.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_bf.add_argument("--oos-start", default=None, help="Override OOS start date (YYYY-MM-DD)")
    p_bf.set_defaults(func=cmd_backfill)

    # leaderboard
    p_lb = sub.add_parser("leaderboard", help="Print current leaderboard")
    p_lb.add_argument("--window", type=int, default=None, help="Rolling window in trading days")
    p_lb.set_defaults(func=cmd_leaderboard)

    # retrain
    p_rt = sub.add_parser("retrain", help="Force retrain a model")
    p_rt.add_argument("--model", required=True, help="Model name to retrain")
    p_rt.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD, default: today)")
    p_rt.set_defaults(func=cmd_retrain)

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
