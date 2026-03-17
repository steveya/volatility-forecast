"""Tests for VolGRU QLIKE loss support."""

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


def _make_synthetic(n_obs: int = 180, seed: int = 42):
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
def test_volgru_qlike_loss_decreases():
    """QLIKE loss should decrease during training."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=30,
        early_stopping_patience=30,
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.fit(X.iloc[:140], y[:140], returns=returns[:140])

    history = model.training_loss_history_
    assert history is not None
    assert len(history) > 1
    assert history[-1] < history[0], "QLIKE loss should decrease during training"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_qlike_predictions_positive_and_finite():
    """Predictions from QLIKE-trained model should be positive and finite."""
    torch.manual_seed(1)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=20,
        early_stopping_patience=20,
    )
    model = VolGRUModel(config=cfg, random_state=1)
    model.fit(X.iloc[:140], y[:140], returns=returns[:140])
    preds = model.predict(X.iloc[140:], returns=returns[140:])

    assert len(preds) == 40
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_cv_qlike_scoring():
    """CV fold scoring should use QLIKE when loss_mode='qlike'."""
    torch.manual_seed(2)
    X, y, returns = _make_synthetic(n_obs=120)

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=5,
        early_stopping_patience=5,
    )
    model = VolGRUModel(config=cfg, random_state=2)

    grid = [
        {"lr": 1e-2, "weight_decay_gate": 0.0},
        {"lr": 1e-3, "weight_decay_gate": 1e-3},
    ]

    results = model.run_cv(
        X.to_numpy(dtype=float),
        y,
        returns=returns,
        param_grid=grid,
        n_splits=2,
        start_index=0,
    )

    assert len(results) == 2
    # Results should be sorted ascending by score
    assert results[0][0] <= results[1][0]
    # QLIKE scores should be finite and non-negative
    for score, _, _ in results:
        assert np.isfinite(score)
        assert score >= 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_stes_equivalence_under_qlike():
    """Constrained 1D-GRU with QLIKE should still reduce to STES predictions."""
    torch.manual_seed(3)
    rng = np.random.default_rng(7)
    n_obs = 200
    returns = rng.normal(0.0, 0.01, size=n_obs).astype(np.float64)
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

    beta = np.array([0.1, -0.2, 0.3, -0.05], dtype=np.float64)

    # Under QLIKE, the forward pass is the same — only the loss changes.
    # So with fixed beta, predictions should match MSE version exactly.
    cfg_qlike = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="stes_r2",
        state_dim=1,
        loss_mode="qlike",
    )
    model_qlike = VolGRUModel(config=cfg_qlike, random_state=0)
    model_qlike.set_gate_beta(beta)
    sigma2_qlike, z_qlike, v_cand_qlike = model_qlike.predict_with_gates(
        X, returns=returns
    )

    cfg_mse = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="stes_r2",
        state_dim=1,
        loss_mode="mse_r2",
    )
    model_mse = VolGRUModel(config=cfg_mse, random_state=0)
    model_mse.set_gate_beta(beta)
    sigma2_mse, z_mse, v_cand_mse = model_mse.predict_with_gates(X, returns=returns)

    # Forward pass is loss-mode-agnostic
    assert np.allclose(sigma2_qlike, sigma2_mse, atol=1e-10)
    assert np.allclose(z_qlike, z_mse, atol=1e-10)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_qlike_multidim_loss_decreases():
    """QLIKE loss should decrease during training with state_dim=3."""
    torch.manual_seed(10)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="gru_linear",
        candidate_mode="linear_pos",
        state_dim=3,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=30,
        early_stopping_patience=30,
    )
    model = VolGRUModel(config=cfg, random_state=10)
    model.fit(X.iloc[:140], y[:140], returns=returns[:140])

    history = model.training_loss_history_
    assert history is not None
    assert len(history) > 1
    assert history[-1] < history[0], "QLIKE loss should decrease with state_dim=3"

    preds, gates, cands = model.predict_with_gates(X.iloc[140:], returns=returns[140:])
    assert preds.shape == (40,)
    assert gates.shape == (40, 3)
    assert cands.shape == (40, 3)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)
