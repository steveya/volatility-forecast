"""Unit tests for the horserace runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from volatility_forecast.horserace import (
    build_model_registry,
    compute_leaderboard,
    load_config,
    record_actuals,
)
from volatility_forecast.horserace.stages import _import_class
from volatility_forecast.storage import VolForecastStore


@pytest.fixture
def store(tmp_path):
    return VolForecastStore(root=str(tmp_path / "store"))


@pytest.fixture
def sample_config():
    return {
        "data": {
            "tickers": ["SPY"],
            "history_start": "2005-01-01",
            "feature_lags": 5,
            "target_scale": 1.0,
        },
        "models": {
            "ES": {
                "class": "volatility_forecast.model.es_model.ESModel",
                "params": {"loss": "mse"},
                "api": "sklearn",
            },
            "STES": {
                "class": "volatility_forecast.model.stes_model.STESModel",
                "params": {"loss": "mse"},
                "api": "sklearn",
            },
        },
        "training": {
            "recalibrate_every_n_days": 30,
            "burn_in": 100,
            "min_history_days": 500,
        },
        "store": {"root": "/tmp/test_store"},
        "leaderboard": {
            "rolling_window_days": 252,
            "metrics": ["qlike", "rmse", "mae"],
            "primary_metric": "qlike",
        },
    }


def test_import_class():
    cls = _import_class("volatility_forecast.model.es_model.ESModel")
    assert cls.__name__ == "ESModel"


def test_build_model_registry(sample_config):
    registry = build_model_registry(sample_config)
    assert "ES" in registry
    assert "STES" in registry
    assert len(registry) == 2


def test_build_model_registry_skips_bad_import(sample_config):
    sample_config["models"]["Bad"] = {
        "class": "nonexistent.module.FakeModel",
        "params": {},
    }
    registry = build_model_registry(sample_config)
    assert "Bad" not in registry
    assert "ES" in registry


def test_load_config():
    config = load_config()
    assert "data" in config
    assert "models" in config
    assert "training" in config
    assert "store" in config


def test_record_actuals_idempotent(store):
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2024-06-03", tz="UTC"), pd.Timestamp("2024-06-04", tz="UTC")]
    )
    y = pd.Series([0.5, 0.7], index=dates)

    n1 = record_actuals(store, y, dates, "SPY")
    n2 = record_actuals(store, y, dates, "SPY")
    assert n1 == 2
    assert n2 == 2

    # Verify no duplicates via actuals table
    with store._conn() as con:
        count = con.execute("SELECT COUNT(*) FROM actuals WHERE ticker = 'SPY'").fetchone()[0]
    assert count == 2


def test_compute_leaderboard_empty(store):
    lb = compute_leaderboard(store, "SPY")
    assert lb.empty


def test_compute_leaderboard_ranking(store):
    """Seed store with synthetic data and verify leaderboard ranking."""
    # Register two dummy models
    from volatility_forecast.model.es_model import ESModel

    m1 = ESModel()
    m2 = ESModel()
    mid1 = store.register_model("GoodModel", m1)
    mid2 = store.register_model("BadModel", m2)

    # Create forecasts and actuals
    rng = np.random.default_rng(42)
    n = 50
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n):
        dt = base + pd.Timedelta(days=i)
        actual = float(rng.uniform(0.5, 2.0))

        store.upsert_actual(ticker="SPY", target_utc=dt, horizon=1, y=actual)

        # GoodModel: close to actual
        store.upsert_forecast(
            model_id=mid1, ticker="SPY",
            asof_utc=dt - pd.Timedelta(days=1), target_utc=dt,
            horizon=1, yhat=actual + rng.normal(0, 0.05),
        )
        # BadModel: far from actual
        store.upsert_forecast(
            model_id=mid2, ticker="SPY",
            asof_utc=dt - pd.Timedelta(days=1), target_utc=dt,
            horizon=1, yhat=actual + rng.normal(0, 0.5),
        )

    lb = compute_leaderboard(
        store, "SPY", window_days=100, primary_metric="rmse",
        as_of_utc=pd.Timestamp("2025-03-01", tz="UTC"),
    )
    assert len(lb) == 2
    # GoodModel should rank higher (lower RMSE)
    assert lb.iloc[0]["model"] == "GoodModel"
    assert lb.iloc[0]["rmse"] < lb.iloc[1]["rmse"]
