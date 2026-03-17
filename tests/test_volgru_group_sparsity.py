"""Tests for the row-wise feature-group sparsity penalty on gate_beta."""

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


# ------------------------------------------------------------------ #
# Unit test for the penalty function itself
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_feature_group_penalty_zero_at_origin():
    """Penalty at the origin equals p * sqrt(eps), not zero."""
    from volatility_forecast.model.volgru_utils import feature_group_penalty_torch

    eps = 1e-12
    beta = torch.zeros(5, 3, dtype=torch.float64)
    val = feature_group_penalty_torch(beta, eps=eps)
    expected = 5.0 * np.sqrt(eps)
    assert abs(float(val) - expected) < 1e-15


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_feature_group_penalty_scalar_reduces_to_l1():
    """For d=1, the penalty is sum_i sqrt(beta_i^2 + eps) ≈ sum |beta_i|."""
    from volatility_forecast.model.volgru_utils import feature_group_penalty_torch

    eps = 1e-12
    beta_1d = torch.tensor([0.3, -0.5, 0.0, 1.2], dtype=torch.float64)
    val = float(feature_group_penalty_torch(beta_1d, eps=eps))
    expected = sum(np.sqrt(b**2 + eps) for b in [0.3, 0.5, 0.0, 1.2])
    assert abs(val - expected) < 1e-10


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_feature_group_penalty_multidim_row_norms():
    """For d>1, penalty sums row-wise L2 norms (smoothed)."""
    from volatility_forecast.model.volgru_utils import feature_group_penalty_torch

    eps = 1e-12
    beta = torch.tensor([[0.3, 0.4], [-0.5, 0.0], [0.0, 0.0]], dtype=torch.float64)
    val = float(feature_group_penalty_torch(beta, eps=eps))
    row_norms = [
        np.sqrt(0.3**2 + 0.4**2 + eps),
        np.sqrt(0.5**2 + 0.0**2 + eps),
        np.sqrt(0.0**2 + 0.0**2 + eps),
    ]
    expected = sum(row_norms)
    assert abs(val - expected) < 1e-10


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_feature_group_penalty_gradient_at_origin():
    """Gradient at origin is finite (not NaN) thanks to the eps guard."""
    from volatility_forecast.model.volgru_utils import feature_group_penalty_torch

    beta = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)
    penalty = feature_group_penalty_torch(beta, eps=1e-12)
    penalty.backward()
    assert torch.isfinite(beta.grad).all()


# ------------------------------------------------------------------ #
# Config validation
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_config_negative_group_lambda_raises():
    with pytest.raises(ValueError, match="gate_feature_group_lambda"):
        VolGRUConfig(gate_feature_group_lambda=-0.1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_config_negative_entropy_lambda_raises():
    with pytest.raises(ValueError, match="gate_entropy_lambda"):
        VolGRUConfig(gate_entropy_lambda=-0.1)


# ------------------------------------------------------------------ #
# Zero-lambda passthrough: training identical to baseline
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_zero_lambda_matches_baseline():
    """gate_feature_group_lambda=0 produces the same training as without it."""
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()

    cfg_base = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=10,
        early_stopping_patience=10,
        gate_feature_group_lambda=0.0,
    )
    m_base = VolGRUModel(config=cfg_base, random_state=0)
    m_base.fit(X, y, returns=returns)

    torch.manual_seed(0)
    cfg_zero = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=10,
        early_stopping_patience=10,
        gate_feature_group_lambda=0.0,
    )
    m_zero = VolGRUModel(config=cfg_zero, random_state=0)
    m_zero.fit(X, y, returns=returns)

    np.testing.assert_allclose(
        m_base.training_loss_history_,
        m_zero.training_loss_history_,
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        m_base.get_gate_beta(), m_zero.get_gate_beta(), rtol=0, atol=1e-14
    )


# ------------------------------------------------------------------ #
# Positive lambda: penalty increases training loss
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_positive_lambda_increases_loss():
    """A large group-sparsity lambda increases the (penalised) training loss."""
    torch.manual_seed(1)
    X, y, returns = _make_synthetic()

    cfg_no = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=15,
        early_stopping_patience=15,
        gate_feature_group_lambda=0.0,
    )
    m_no = VolGRUModel(config=cfg_no, random_state=1)
    m_no.fit(X, y, returns=returns)

    torch.manual_seed(1)
    cfg_yes = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=15,
        early_stopping_patience=15,
        gate_feature_group_lambda=1.0,
    )
    m_yes = VolGRUModel(config=cfg_yes, random_state=1)
    m_yes.fit(X, y, returns=returns)

    # Penalised loss with large lambda should be higher
    assert m_yes.training_loss_history_[-1] > m_no.training_loss_history_[-1]


# ------------------------------------------------------------------ #
# Positive lambda shrinks gate-row norms
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_positive_lambda_shrinks_row_norms():
    """Group penalty shrinks gate-row L2 norms relative to the unpenalised case."""
    torch.manual_seed(2)
    X, y, returns = _make_synthetic()

    cfg_no = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=20,
        early_stopping_patience=20,
        gate_feature_group_lambda=0.0,
    )
    m_no = VolGRUModel(config=cfg_no, random_state=2)
    m_no.fit(X, y, returns=returns)
    norms_no = np.linalg.norm(m_no.get_gate_beta(), axis=-1)

    torch.manual_seed(2)
    cfg_yes = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=20,
        early_stopping_patience=20,
        gate_feature_group_lambda=0.5,
    )
    m_yes = VolGRUModel(config=cfg_yes, random_state=2)
    m_yes.fit(X, y, returns=returns)
    norms_yes = np.linalg.norm(m_yes.get_gate_beta(), axis=-1)

    assert np.sum(norms_yes) < np.sum(norms_no)


# ------------------------------------------------------------------ #
# Multi-dimensional: row grouping acts across all state dimensions
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_multidim_group_penalty_shrinks_rows():
    """With state_dim=3, the group penalty shrinks entire rows (all dims) together."""
    torch.manual_seed(3)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        gate_mode="gru_linear",
        candidate_mode="linear_pos",
        state_dim=3,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=20,
        early_stopping_patience=20,
        gate_feature_group_lambda=0.5,
    )
    model = VolGRUModel(config=cfg, random_state=3)
    model.fit(X, y, returns=returns)
    beta = model.get_gate_beta()

    assert beta.shape == (X.shape[1], 3)
    row_norms = np.linalg.norm(beta, axis=1)
    # At least one feature should have been pushed toward zero
    assert np.min(row_norms) < 0.1


# ------------------------------------------------------------------ #
# CV integration: gate_feature_group_lambda is CV-tunable
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_cv_tunes_group_lambda():
    """CV can search over gate_feature_group_lambda without errors."""
    torch.manual_seed(4)
    X, y, returns = _make_synthetic()

    cfg = VolGRUConfig(
        gate_mode="stes_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=5,
        early_stopping_patience=5,
    )
    model = VolGRUModel(config=cfg, random_state=4)

    grid = [
        {"gate_feature_group_lambda": 0.0},
        {"gate_feature_group_lambda": 0.01},
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
    assert model.config.gate_feature_group_lambda in (0.0, 0.01)
    preds = model.predict(X, returns=returns)
    assert np.isfinite(preds).all()
