"""Tests for horserace DAG stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from volatility_forecast.horserace.refit_policy import ExpandingWindowPolicy, EveryNDaysPolicy
from volatility_forecast.horserace.stages import (
    build_model_registry,
    fit_or_load,
    split_data,
)
from volatility_forecast.horserace.types import DataBundle, SplitSpec
from volatility_forecast.storage import VolForecastStore


@pytest.fixture
def store(tmp_path):
    return VolForecastStore(root=str(tmp_path / "store"))


def _make_bundle(n=500, ticker="SPY", start="2020-01-01"):
    """Create a synthetic DataBundle for testing."""
    rng = np.random.default_rng(42)
    dates = pd.DatetimeIndex(pd.bdate_range(start, periods=n, tz="UTC"))
    returns = pd.Series(rng.normal(0, 0.01, n), index=dates, name="logret")
    y = pd.Series(returns.values ** 2, index=dates, name="y")
    X = pd.DataFrame(
        {
            "lag_ret_1": np.roll(returns.values, 1),
            "lag_abs_ret_1": np.abs(np.roll(returns.values, 1)),
            "const": 1.0,
        },
        index=dates,
    )
    # Zero out first row to avoid look-ahead from roll
    X.iloc[0] = [0.0, 0.0, 1.0]
    return DataBundle(ticker=ticker, X=X, y=y, returns=returns, dates=dates)


def test_split_data_expanding():
    bundle = _make_bundle(n=500, start="2019-01-01")
    config = {
        "training": {
            "burn_in": 50,
            "oos_start": "2020-06-01",
        }
    }
    split = split_data(bundle, config)
    assert split.burn_in == 50
    assert split.train_end == 500  # expanding: train on all data
    assert split.oos_start > 0
    assert split.oos_start < 500
    assert split.oos_end == 500
    # oos_start should correspond to a date >= 2020-06-01
    assert bundle.dates[split.oos_start] >= pd.Timestamp("2020-06-01", tz="UTC")


def test_split_data_no_oos_start():
    bundle = _make_bundle(n=200)
    config = {"training": {"burn_in": 30}}
    split = split_data(bundle, config)
    assert split.oos_start == 200  # no OOS region
    assert split.train_end == 200


def test_split_data_burn_in_clamped():
    bundle = _make_bundle(n=50)
    config = {"training": {"burn_in": 100}}
    split = split_data(bundle, config)
    assert split.burn_in == 50  # clamped to len(dates)


def test_fit_or_load_first_run(store):
    """No model in store → should train and register."""
    bundle = _make_bundle(n=300)
    config = {
        "training": {"burn_in": 50, "oos_start": "2021-06-01"},
        "models": {"ES": {"class": "volatility_forecast.model.es_model.ESModel", "params": {"loss": "mse"}, "api": "sklearn"}},
    }
    split = split_data(bundle, config)

    from volatility_forecast.model.es_model import ESModel
    model = ESModel()
    policy = ExpandingWindowPolicy()

    result = fit_or_load("ES", model, "sklearn", bundle, split, config, store, policy)
    assert result.was_retrained is True
    assert result.model_id.startswith("ES:")
    assert result.fitted_model is not None
    assert result.train_metric is not None


def test_fit_or_load_reuses_existing(store):
    """Model in store + EveryNDays policy says no → should load."""
    bundle = _make_bundle(n=300)
    config = {
        "training": {"burn_in": 50, "oos_start": "2021-06-01"},
        "models": {"ES": {"class": "volatility_forecast.model.es_model.ESModel", "params": {"loss": "mse"}, "api": "sklearn"}},
    }
    split = split_data(bundle, config)

    from volatility_forecast.model.es_model import ESModel
    model = ESModel()

    # First: train and register
    result1 = fit_or_load("ES", model, "sklearn", bundle, split, config, store, ExpandingWindowPolicy())
    assert result1.was_retrained is True

    # Second: with EveryNDays(30) should load (just created)
    model2 = ESModel()
    result2 = fit_or_load("ES", model2, "sklearn", bundle, split, config, store, EveryNDaysPolicy(n_days=30))
    assert result2.was_retrained is False
    assert result2.model_id == result1.model_id
