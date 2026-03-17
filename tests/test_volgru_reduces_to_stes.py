import numpy as np
import pandas as pd
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from volatility_forecast.model.stes_model import STESModel
from volatility_forecast.model.volgru_config import VolGRUConfig
from volatility_forecast.model.volgru_model import VolGRUModel


def _make_stes_features(returns: np.ndarray) -> pd.DataFrame:
    prev_ret = np.roll(returns, 1)
    prev_ret[0] = 0.0
    return pd.DataFrame(
        {
            "const": np.ones_like(returns),
            "prev_ret": prev_ret,
            "prev_abs_ret": np.abs(prev_ret),
            "prev_sq_ret": prev_ret**2,
        }
    )


def _run_equivalence_check(backend: str) -> None:
    rng = np.random.default_rng(7)
    n_obs = 200
    returns = rng.normal(0.0, 0.01, size=n_obs).astype(np.float64)
    X = _make_stes_features(returns).astype(np.float64)
    beta = np.array([0.1, -0.2, 0.3, -0.05], dtype=np.float64)

    stes = STESModel(params=beta.copy())
    sigma2_stes, alpha_stes = stes.predict_with_alpha(X, returns=returns)

    cfg = VolGRUConfig(
        backend=backend,
        gate_mode="stes_linear",
        candidate_mode="stes_r2",
        state_dim=1,
        loss_mode="mse_r2",
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.set_gate_beta(beta)
    sigma2_volgru, z_volgru, v_cand = model.predict_with_gates(X, returns=returns)

    atol = 1e-10
    assert np.allclose(sigma2_volgru, sigma2_stes, rtol=0.0, atol=atol)
    assert np.allclose(z_volgru, alpha_stes, rtol=0.0, atol=atol)
    assert np.allclose(v_cand, returns**2, rtol=0.0, atol=atol)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_torch_reduces_to_stes_exactly() -> None:
    torch.manual_seed(0)
    _run_equivalence_check("torch")
