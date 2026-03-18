"""Tests for the refit policy module."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from volatility_forecast.horserace.refit_policy import (
    EveryNDaysPolicy,
    ExpandingWindowPolicy,
    ForceRefitPolicy,
    build_refit_policy,
    resolve_refit_policy,
)
from volatility_forecast.horserace.types import DataBundle
from volatility_forecast.storage import VolForecastStore


@pytest.fixture
def store(tmp_path):
    return VolForecastStore(root=str(tmp_path / "store"))


@pytest.fixture
def dummy_bundle():
    dates = pd.DatetimeIndex(
        pd.bdate_range("2024-01-01", periods=100, tz="UTC")
    )
    return DataBundle(
        ticker="SPY",
        X=pd.DataFrame({"const": 1.0}, index=dates),
        y=pd.Series(0.5, index=dates),
        returns=pd.Series(0.01, index=dates),
        dates=dates,
    )


def test_every_n_days_no_model(store, dummy_bundle):
    """No model in store → should always refit."""
    policy = EveryNDaysPolicy(n_days=30)
    assert policy.should_refit(store, "ES", dummy_bundle) is True


def test_every_n_days_fresh(store, dummy_bundle):
    """Model just created → should NOT refit."""
    from volatility_forecast.model.es_model import ESModel

    store.register_model("ES", ESModel())
    policy = EveryNDaysPolicy(n_days=30)
    assert policy.should_refit(store, "ES", dummy_bundle) is False


def test_every_n_days_stale(store, dummy_bundle):
    """Model created 31+ days ago with n_days=30 → should refit."""
    from volatility_forecast.model.es_model import ESModel

    model_id = store.register_model("ES", ESModel())
    # Backdate the created_utc
    with store._conn() as con:
        con.execute(
            "UPDATE models SET created_utc = ? WHERE model_id = ?",
            [pd.Timestamp("2020-01-01", tz="UTC"), model_id],
        )
    policy = EveryNDaysPolicy(n_days=30)
    assert policy.should_refit(store, "ES", dummy_bundle) is True


def test_expanding_always_true(store, dummy_bundle):
    """ExpandingWindowPolicy always returns True."""
    from volatility_forecast.model.es_model import ESModel

    store.register_model("ES", ESModel())
    policy = ExpandingWindowPolicy()
    assert policy.should_refit(store, "ES", dummy_bundle) is True


def test_build_refit_policy_every_n_days():
    cfg = {"type": "every_n_days", "params": {"n_days": 15}}
    policy = build_refit_policy(cfg)
    assert isinstance(policy, EveryNDaysPolicy)
    assert policy.n_days == 15


def test_build_refit_policy_expanding():
    cfg = {"type": "expanding_window", "params": {}}
    policy = build_refit_policy(cfg)
    assert isinstance(policy, ExpandingWindowPolicy)


def test_build_refit_policy_unknown():
    with pytest.raises(ValueError, match="Unknown refit policy"):
        build_refit_policy({"type": "nonexistent"})


def test_resolve_force_retrain():
    config = {"models": {"ES": {}}, "training": {"refit_policy": {"type": "every_n_days", "params": {"n_days": 30}}}}
    policy = resolve_refit_policy(config, "ES", force_retrain=True)
    assert isinstance(policy, ForceRefitPolicy)


def test_resolve_per_model_override():
    config = {
        "models": {
            "STES": {"refit_policy": {"type": "expanding_window", "params": {}}},
        },
        "training": {"refit_policy": {"type": "every_n_days", "params": {"n_days": 30}}},
    }
    policy = resolve_refit_policy(config, "STES", force_retrain=False)
    assert isinstance(policy, ExpandingWindowPolicy)


def test_resolve_global_fallback():
    config = {
        "models": {"ES": {}},
        "training": {"refit_policy": {"type": "every_n_days", "params": {"n_days": 7}}},
    }
    policy = resolve_refit_policy(config, "ES", force_retrain=False)
    assert isinstance(policy, EveryNDaysPolicy)
    assert policy.n_days == 7
