import dataclasses
import sys
from pathlib import Path

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
from volatility_forecast.model.volgru_utils import gate_entropy_term_torch


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


def _numpy_float32_stes_reference(
    X: pd.DataFrame, returns: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_32 = X.to_numpy(dtype=np.float32)
    r_32 = np.asarray(returns, dtype=np.float32).reshape(-1)
    beta_32 = np.asarray(beta, dtype=np.float32).reshape(-1)
    logits = np.clip(X_32 @ beta_32, np.float32(-60.0), np.float32(60.0))
    z = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))).astype(np.float32)
    cand = (r_32 * r_32).astype(np.float32)
    n = len(r_32)
    sigma2 = np.empty(n, dtype=np.float32)
    v_t = np.float32(cand[0])
    for t in range(n):
        v_t = np.float32((np.float32(1.0) - z[t]) * v_t + z[t] * cand[t])
        sigma2[t] = v_t
    return sigma2, z, cand


def _run_equivalence_check(
    backend: str,
    *,
    return_scale: float = 0.01,
    dtype: str = "float64",
    rtol: float = 0.0,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n_obs = 200
    returns = rng.normal(0.0, return_scale, size=n_obs).astype(np.float64)
    X = _make_stes_features(returns).astype(np.float64)
    beta = np.array([0.1, -0.2, 0.3, -0.05], dtype=np.float64)

    stes = STESModel(params=beta.copy())
    sigma2_stes, alpha_stes = stes.predict_with_alpha(X, returns=returns)

    cfg = VolGRUConfig(
        backend=backend,  # type: ignore[arg-type]
        gate_mode="stes_linear",
        candidate_mode="stes_r2",
        state_dim=1,
        loss_mode="mse_r2",
        dtype=dtype,  # type: ignore[arg-type]
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.set_gate_beta(beta)
    sigma2_volgru, z_volgru, v_cand = model.predict_with_gates(X, returns=returns)

    expected_np_dtype = np.float64 if dtype == "float64" else np.float32
    assert sigma2_volgru.dtype == expected_np_dtype
    assert z_volgru.dtype == expected_np_dtype
    assert v_cand.dtype == expected_np_dtype

    assert np.allclose(sigma2_volgru, sigma2_stes, rtol=rtol, atol=atol)
    assert np.allclose(z_volgru, alpha_stes, rtol=rtol, atol=atol)
    assert np.allclose(v_cand, returns**2, rtol=rtol, atol=atol)
    return sigma2_volgru, z_volgru, v_cand, sigma2_stes, alpha_stes


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("return_scale", [0.01, 1.0])
def test_volgru_torch_reduces_to_stes_exactly(return_scale: float) -> None:
    torch.manual_seed(0)
    _, _, _, sigma2_stes, alpha_stes = _run_equivalence_check(
        "torch",
        return_scale=return_scale,
        dtype="float64",
        rtol=0.0,
        atol=1e-12,
    )

    # Discriminating perturbation: 1e-9 shift in beta must fail 1e-12 parity
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, return_scale, size=200).astype(np.float64)
    X = _make_stes_features(returns).astype(np.float64)
    beta_perturbed = np.array([0.1 + 1e-9, -0.2, 0.3, -0.05], dtype=np.float64)
    model_pert = VolGRUModel(
        config=VolGRUConfig(
            backend="torch",
            gate_mode="stes_linear",
            candidate_mode="stes_r2",
            state_dim=1,
            loss_mode="mse_r2",
            dtype="float64",
        ),
        random_state=0,
    )
    model_pert.set_gate_beta(beta_perturbed)
    sigma2_pert, z_pert, _ = model_pert.predict_with_gates(X, returns=returns)
    assert not np.allclose(z_pert, alpha_stes, rtol=0.0, atol=1e-12)
    assert not np.allclose(sigma2_pert, sigma2_stes, rtol=1e-12, atol=0.0)
    if return_scale >= 1.0:
        assert not np.allclose(sigma2_pert, sigma2_stes, rtol=0.0, atol=1e-12)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("return_scale", [0.01, 1.0])
def test_volgru_torch_float32_default_precision_contract(return_scale: float) -> None:
    torch.manual_seed(0)
    assert VolGRUConfig().dtype == "float32"

    sigma2_volgru, z_volgru, v_cand, sigma2_stes, alpha_stes = _run_equivalence_check(
        "torch",
        return_scale=return_scale,
        dtype="float32",
        rtol=1e-6,
        atol=1e-6,
    )

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, return_scale, size=200).astype(np.float64)
    X = _make_stes_features(returns).astype(np.float64)
    beta = np.array([0.1, -0.2, 0.3, -0.05], dtype=np.float64)

    sigma2_ref32, z_ref32, cand_ref32 = _numpy_float32_stes_reference(X, returns, beta)
    assert np.allclose(sigma2_volgru, sigma2_ref32, rtol=0.0, atol=5e-7)
    assert np.allclose(z_volgru, z_ref32, rtol=0.0, atol=5e-7)
    assert np.allclose(v_cand, cand_ref32, rtol=0.0, atol=5e-7)

    # Discriminating perturbation: 1e-4 shift in beta must fail rtol=1e-6, atol=1e-6
    beta_perturbed = np.array([0.1 + 1e-4, -0.2, 0.3, -0.05], dtype=np.float64)
    model_pert = VolGRUModel(
        config=VolGRUConfig(
            backend="torch",
            gate_mode="stes_linear",
            candidate_mode="stes_r2",
            state_dim=1,
            loss_mode="mse_r2",
        ),
        random_state=0,
    )
    model_pert.set_gate_beta(beta_perturbed)
    _, z_pert, _ = model_pert.predict_with_gates(X, returns=returns)
    assert not np.allclose(z_pert, alpha_stes, rtol=1e-6, atol=1e-6)

    # Explicit boundary condition: VolGRU clamps v_next >= config.eps whereas unclamped STES does not
    zeros_r = np.zeros(10, dtype=np.float64)
    X_zero = _make_stes_features(zeros_r).astype(np.float64)
    stes_zero = STESModel(params=beta.copy())
    s_zero, _ = stes_zero.predict_with_alpha(X_zero, returns=zeros_r, init_var=0.0)
    m_zero = VolGRUModel(
        config=VolGRUConfig(gate_mode="stes_linear", candidate_mode="stes_r2"),
        random_state=0,
    )
    m_zero.set_gate_beta(beta)
    v_zero = m_zero.predict(X_zero, returns=zeros_r, init_var=0.0)
    assert np.all(s_zero == 0.0)
    assert np.allclose(v_zero, m_zero.config.eps, rtol=0.0, atol=1e-18)


def test_volgru_dtype_validation_and_float64_fit_roundtrip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dtype"):
        VolGRUConfig(dtype="float16")  # type: ignore[arg-type]

    if not HAS_TORCH:
        pytest.skip("torch not installed")

    rng = np.random.default_rng(11)
    n_obs = 90
    returns = rng.normal(0.0, 0.02, size=n_obs).astype(np.float64)
    X = _make_stes_features(returns).astype(np.float64)
    y = np.roll(returns**2, -1)
    y[-1] = returns[-1] ** 2

    cfg = VolGRUConfig(
        backend="torch",
        gate_mode="stes_linear",
        candidate_mode="stes_r2",
        state_dim=1,
        loss_mode="mse_r2",
        dtype="float64",
        max_epochs=5,
        early_stopping_patience=5,
    )
    model = VolGRUModel(config=cfg, random_state=0)
    model.fit(X, y, returns=returns, start_index=5, end_index=n_obs)

    beta = model.get_gate_beta()
    assert beta.dtype == np.float64
    preds = model.predict(X, returns=returns)
    s2, z, cand = model.predict_with_gates(X, returns=returns)
    assert preds.dtype == np.float64
    assert s2.dtype == np.float64 and z.dtype == np.float64 and cand.dtype == np.float64
    assert np.allclose(preds, s2, rtol=0.0, atol=0.0)

    cv_res = model.run_cv(
        X.to_numpy(dtype=float),
        y,
        returns=returns,
        param_grid=[{"lr": 1e-2}, {"lr": 5e-3}],
        n_splits=2,
        start_index=5,
    )
    assert len(cv_res) == 2

    artifact = tmp_path / "volgru_f64.joblib"
    model.save(str(artifact))
    loaded = VolGRUModel.load(str(artifact))
    s2_loaded, z_loaded, cand_loaded = loaded.predict_with_gates(X, returns=returns)
    assert s2_loaded.dtype == np.float64
    assert np.array_equal(s2_loaded, s2)
    assert np.array_equal(z_loaded, z)
    assert np.array_equal(cand_loaded, cand)

    f32_model = VolGRUModel(
        config=VolGRUConfig(gate_mode="stes_linear", candidate_mode="stes_r2"),
        random_state=0,
    )
    f32_model.set_gate_beta(beta)
    legacy_f32_path = tmp_path / "legacy_f32.joblib"
    object.__delattr__(f32_model.config, "dtype")
    object.__delattr__(f32_model.config, "gate_feature_group_lambda")
    f32_model.save(str(legacy_f32_path))
    loaded_legacy_f32 = VolGRUModel.load(str(legacy_f32_path))
    assert loaded_legacy_f32.config.dtype == "float32"
    assert loaded_legacy_f32.config.gate_feature_group_lambda == 0.0
    assert repr(loaded_legacy_f32)
    assert loaded_legacy_f32.get_params()["config"] == loaded_legacy_f32.config
    assert dataclasses.asdict(loaded_legacy_f32.config)["dtype"] == "float32"
    assert loaded_legacy_f32.predict(X, returns=returns).dtype == np.float32

    # Pre-4dcacf7 float64 artifact missing dtype slot keeps float64 instead of downcasting
    legacy_f64_model = VolGRUModel(
        config=VolGRUConfig(
            gate_mode="stes_linear", candidate_mode="stes_r2", dtype="float64"
        ),
        random_state=0,
    )
    legacy_f64_model.set_gate_beta(beta)
    legacy_f64_path = tmp_path / "legacy_f64.joblib"
    object.__delattr__(legacy_f64_model.config, "dtype")
    object.__delattr__(legacy_f64_model.config, "gate_feature_group_lambda")
    legacy_f64_model.save(str(legacy_f64_path))
    loaded_legacy_f64 = VolGRUModel.load(str(legacy_f64_path))
    assert loaded_legacy_f64.config.dtype == "float64"
    assert all(p.dtype == torch.float64 for p in loaded_legacy_f64.torch_module_.parameters())
    assert loaded_legacy_f64.predict(X, returns=returns).dtype == np.float64

    # Mutating config.dtype syncs both get_gate_beta() and predict()
    loaded_legacy_f64.config.dtype = "float32"
    assert loaded_legacy_f64.get_gate_beta().dtype == np.float32
    assert loaded_legacy_f64.predict(X, returns=returns).dtype == np.float32


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_volgru_float32_saturated_entropy_and_direct_torch_module() -> None:
    from volatility_forecast.model.volgru_torch import VolGRUCellTorch, VolGRUModelTorch

    z_sat = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float32, requires_grad=True)
    entropy_val = gate_entropy_term_torch(z_sat, eps=1e-12)
    assert torch.isfinite(entropy_val)
    entropy_val.backward()
    assert z_sat.grad is not None and torch.all(torch.isfinite(z_sat.grad))

    rng = np.random.default_rng(17)
    r_sat = rng.normal(0.0, 0.02, size=50).astype(np.float64)
    X_sat = np.ones((50, 1), dtype=np.float64) * 20.0
    sat_model = VolGRUModel(
        config=VolGRUConfig(
            dtype="float32", gate_entropy_lambda=0.01, max_epochs=5
        ),
        random_state=0,
    )
    sat_model.set_gate_beta(np.array([1.0]))
    sat_model.fit(X_sat, r_sat**2, returns=r_sat)
    assert all(np.isfinite(sat_model.training_loss_history_))

    cfg_f64 = VolGRUConfig(dtype="float64", gate_mode="gru_linear", candidate_mode="mlp_pos", state_dim=2)
    cell = VolGRUCellTorch(config=cfg_f64, n_features=4)
    mod = VolGRUModelTorch(config=cfg_f64, n_features=4)
    assert cell.gate_beta.dtype == torch.float64
    assert all(p.dtype == torch.float64 for p in mod.parameters())

    # General non-STES perform_cv in float64 preserves double precision
    X_cv = _make_stes_features(r_sat).astype(np.float64)
    gen_model = VolGRUModel(
        config=VolGRUConfig(
            gate_mode="gru_linear",
            candidate_mode="mlp_pos",
            state_dim=2,
            loss_mode="qlike",
            dtype="float64",
            val_fraction=0.2,
            max_epochs=3,
            early_stopping_patience=3,
        ),
        random_state=0,
    )
    gen_model.fit(
        X_cv,
        r_sat**2,
        returns=r_sat,
        start_index=2,
        end_index=50,
        perform_cv=True,
        cv_grid=[{"lr": 1e-2}],
        cv_splits=2,
    )
    gen_preds = gen_model.predict(X_cv, returns=r_sat)
    assert gen_preds.dtype == np.float64 and np.all(np.isfinite(gen_preds))


def test_volgru_torch_absent_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = VolGRUConfig(dtype="float64")
    assert cfg.dtype == "float64"
    monkeypatch.setitem(sys.modules, "torch", None)
    model = VolGRUModel(config=cfg)
    with pytest.raises(ImportError, match="torch is required"):
        model.set_gate_beta(np.array([0.1, 0.2]))
    with pytest.raises(ImportError, match="torch is required"):
        model.fit(np.ones((10, 2)), np.ones(10), returns=np.ones(10))
