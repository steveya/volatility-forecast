"""Tests for VolGRU cross-validation (run_cv / perform_cv)."""

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from volatility_forecast.model.volgru_config import VolGRUConfig
from volatility_forecast.model.volgru_model import VolGRUModel


def _make_synthetic(n_obs: int = 180, seed: int = 123):
    rng = np.random.default_rng(seed)
    t = np.arange(n_obs, dtype=float)
    vol = 0.008 * (1.0 + 0.4 * np.sin(2.0 * np.pi * t / 30.0))
    returns = (rng.normal(size=n_obs) * vol).astype(np.float64)
    prev_ret = np.roll(returns, 1)
    prev_ret[0] = 0.0

    X = pd.DataFrame(
        {
            "const": np.ones(n_obs, dtype=np.float64),
            "prev_ret": prev_ret,
            "prev_abs_ret": np.abs(prev_ret),
            "prev_sq_ret": prev_ret**2,
        }
    )

    y = np.roll(returns**2, -1)
    y[-1] = y[-2]
    return X, y.astype(np.float64), returns


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_run_cv_returns_sorted_results():
    """run_cv returns (score, avg_epochs, params) list sorted ascending by score."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="mse_r2",
        lr=1e-2,
        max_epochs=5,
        early_stopping_patience=5,
    )
    model = VolGRUModel(config=cfg, random_state=0)

    grid = [
        {"lr": 1e-2, "weight_decay_gate": 0.0},
        {"lr": 1e-3, "weight_decay_gate": 1e-3},
    ]

    results = model.run_cv(
        X.values,
        y,
        returns=returns,
        param_grid=grid,
        n_splits=2,
        start_index=0,
    )

    assert len(results) == 2
    assert results[0][0] <= results[1][0], "Results should be sorted ascending"
    for score, epochs, params in results:
        assert np.isfinite(score)
        assert score > 0
        assert isinstance(epochs, int)
        assert epochs > 0
        assert isinstance(params, dict)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_perform_cv_applies_best_config():
    """fit(perform_cv=True) applies the best config and produces a fitted model."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="mse_r2",
        lr=1e-2,
        max_epochs=5,
        early_stopping_patience=5,
    )
    model = VolGRUModel(config=cfg, random_state=0)

    grid = [
        {"lr": 1e-2, "weight_decay_gate": 0.0, "weight_decay_candidate": 0.0},
        {"lr": 5e-3, "weight_decay_gate": 1e-4, "weight_decay_candidate": 1e-4},
    ]

    model.fit(
        X,
        y,
        returns=returns,
        perform_cv=True,
        cv_grid=grid,
        cv_splits=2,
    )

    assert model.is_fitted_
    assert model.init_var_ is not None

    preds = model.predict(X, returns=returns)
    assert np.isfinite(preds).all()
    assert len(preds) == len(X)

    # Config should have been updated to one of the grid entries
    assert model.config.lr in (1e-2, 5e-3)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_perform_cv_requires_grid():
    """fit(perform_cv=True) without cv_grid raises ValueError."""
    X, y, returns = _make_synthetic()
    cfg = VolGRUConfig(backend="torch", max_epochs=3)
    model = VolGRUModel(config=cfg, random_state=0)

    with pytest.raises(ValueError, match="cv_grid"):
        model.fit(X, y, returns=returns, perform_cv=True)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_cv_with_beta_ref():
    """CV works when beta_ref is provided."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        lr=1e-2,
        max_epochs=5,
        early_stopping_patience=5,
        beta_stay_close_lambda=1e-3,
    )
    model = VolGRUModel(config=cfg, random_state=0)
    beta_ref = np.zeros(X.shape[1], dtype=np.float64)

    grid = [
        {"beta_stay_close_lambda": 0.0},
        {"beta_stay_close_lambda": 1e-2},
    ]

    model.fit(
        X,
        y,
        returns=returns,
        perform_cv=True,
        cv_grid=grid,
        cv_splits=2,
        beta_ref=beta_ref,
    )

    assert model.is_fitted_
    assert model.config.beta_stay_close_lambda in (0.0, 1e-2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_val_fraction_triggers_early_stopping():
    """val_fraction > 0 produces a validation_loss_history_ and stops earlier."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic(n_obs=300)

    # Without val_fraction (baseline)
    cfg_no_val = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        lr=1e-2,
        max_epochs=200,
        early_stopping_patience=10,
        val_fraction=0.0,
    )
    m_no_val = VolGRUModel(config=cfg_no_val, random_state=0)
    m_no_val.fit(X, y, returns=returns)
    assert m_no_val.validation_loss_history_ is None

    # With val_fraction
    cfg_val = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        lr=1e-2,
        max_epochs=200,
        early_stopping_patience=10,
        val_fraction=0.15,
    )
    m_val = VolGRUModel(config=cfg_val, random_state=0)
    m_val.fit(X, y, returns=returns)
    assert m_val.validation_loss_history_ is not None
    assert len(m_val.validation_loss_history_) == len(m_val.training_loss_history_)
    assert all(np.isfinite(v) for v in m_val.validation_loss_history_)

    # Model with val-based ES should generally stop earlier (or at the same epoch)
    # because validation loss rises before training loss plateaus.
    assert len(m_val.training_loss_history_) <= len(m_no_val.training_loss_history_)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_val_fraction_config_validation():
    """val_fraction outside [0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="val_fraction"):
        VolGRUConfig(val_fraction=1.0)
    with pytest.raises(ValueError, match="val_fraction"):
        VolGRUConfig(val_fraction=-0.1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_perform_cv_multidim():
    """fit(perform_cv=True) works with state_dim=3 under QLIKE."""
    torch.manual_seed(5)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="gru_linear",
        candidate_mode="linear_pos",
        state_dim=3,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=8,
        early_stopping_patience=8,
    )
    model = VolGRUModel(config=cfg, random_state=5)

    grid = [
        {"lr": 1e-2, "weight_decay_gate": 0.0, "weight_decay_candidate": 0.0},
        {"lr": 5e-3, "weight_decay_gate": 1e-4, "weight_decay_candidate": 1e-4},
    ]

    model.fit(
        X,
        y,
        returns=returns,
        perform_cv=True,
        cv_grid=grid,
        cv_splits=2,
    )

    assert model.is_fitted_
    assert model.init_var_ is not None

    preds, gates, cands = model.predict_with_gates(X, returns=returns)
    assert np.isfinite(preds).all()
    assert preds.shape == (len(X),)
    assert gates.shape == (len(X), 3)
    assert cands.shape == (len(X), 3)
    assert model.config.lr in (1e-2, 5e-3)
