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


def _make_synthetic(
    n_obs: int = 180, seed: int = 123
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
def test_volgru_torch_training_sanity() -> None:
    torch.manual_seed(0)
    X, y, returns = _make_synthetic()
    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="gru_linear",
        candidate_mode="linear_pos",
        state_dim=1,
        loss_mode="mse_r2",
        lr=5e-2,
        max_epochs=12,
        early_stopping_patience=12,
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.set_gate_beta(np.zeros(X.shape[1], dtype=np.float64))
    beta_before = model.get_gate_beta().copy()

    model.fit(X, y, returns=returns, start_index=5, end_index=len(X))
    history = model.training_loss_history_
    beta_after = model.get_gate_beta()
    preds = model.predict(X, returns=returns)

    assert len(history) >= 2
    assert model.init_var_ is not None
    assert model.last_var_ is not None
    assert np.isfinite(history).all()
    assert np.isfinite(preds).all()
    assert np.isfinite(beta_after).all()
    assert np.min(history) <= history[0]
    assert (not np.allclose(beta_before, beta_after)) or (np.min(history) < history[0])


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_torch_multidim_training_sanity() -> None:
    torch.manual_seed(0)
    X, y, returns = _make_synthetic(seed=456)
    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="gru_linear",
        candidate_mode="linear_pos",
        state_dim=3,
        loss_mode="mse_r2",
        lr=5e-2,
        max_epochs=10,
        early_stopping_patience=10,
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.set_gate_beta(np.zeros(X.shape[1], dtype=np.float64))

    model.fit(X, y, returns=returns, start_index=5, end_index=len(X))
    history = model.training_loss_history_
    preds, gates, cands = model.predict_with_gates(X, returns=returns)
    beta_after = model.get_gate_beta()

    assert len(history) >= 2
    assert np.isfinite(history).all()
    assert np.isfinite(preds).all()
    assert np.isfinite(gates).all()
    assert np.isfinite(cands).all()
    assert preds.shape == (len(X),)
    assert gates.shape == (len(X), 3)
    assert cands.shape == (len(X), 3)
    assert beta_after.shape == (X.shape[1], 3)
    assert model.last_state_ is not None
    assert model.last_state_.shape == (3,)
    assert np.min(history) <= history[0]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_torch_multidim_mlp_candidate() -> None:
    """Torch backend trains state_dim=3 with mlp_pos candidate and reset gate."""
    torch.manual_seed(7)
    X, y, returns = _make_synthetic(seed=999)
    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="gru_linear",
        candidate_mode="mlp_pos",
        use_reset_gate=True,
        state_dim=3,
        loss_mode="qlike",
        lr=1e-2,
        max_epochs=15,
        early_stopping_patience=15,
    )
    model = VolGRUModel(config=cfg, random_state=7)
    model.fit(X.iloc[:140], y[:140], returns=returns[:140])

    history = model.training_loss_history_
    preds, gates, cands = model.predict_with_gates(X, returns=returns)

    assert len(history) >= 2
    assert (
        history[-1] < history[0]
    ), "QLIKE loss should decrease with mlp_pos + state_dim=3"
    assert preds.shape == (len(X),)
    assert gates.shape == (len(X), 3)
    assert cands.shape == (len(X), 3)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)
