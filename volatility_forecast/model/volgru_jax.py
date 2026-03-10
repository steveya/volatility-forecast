"""JAX backend for VolGRU volatility models."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

from .volgru_config import VolGRUConfig
from .volgru_utils import positive_transform_jax

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


VolGRUParamsJax: TypeAlias = dict[str, Any]


def _require_jax() -> None:
    if jax is None or jnp is None:  # pragma: no cover - optional dependency
        raise ImportError("jax is required for backend='jax'.")


def init_params_jax(
    config: VolGRUConfig,
    n_features: int,
    key: Any,
) -> VolGRUParamsJax:
    """Initialize VolGRU JAX parameters."""
    _require_jax()

    dtype = jnp.float64
    state_dim = int(config.state_dim)
    params: VolGRUParamsJax = {
        "gate": {
            "beta": jnp.zeros((int(n_features), state_dim), dtype=dtype),
        }
    }
    if config.gate_mode == "gru_linear":
        params["gate"]["state_weight"] = jnp.zeros((state_dim,), dtype=dtype)

    if config.use_reset_gate:
        params["reset"] = {
            "gamma": jnp.zeros((int(n_features), state_dim), dtype=dtype),
            "state_weight": jnp.zeros((state_dim,), dtype=dtype),
        }

    if config.candidate_mode == "stes_r2":
        params["candidate"] = {}
    elif config.candidate_mode == "linear_pos":
        params["candidate"] = {
            "weight": jnp.zeros((state_dim, 4 + state_dim), dtype=dtype),
            "bias": jnp.zeros((state_dim,), dtype=dtype),
        }
    elif config.candidate_mode == "mlp_pos":
        hidden = int(config.mlp_hidden_dim)
        key_w1, key_w2 = jax.random.split(key)
        params["candidate"] = {
            "weight1": 0.01 * jax.random.normal(key_w1, (hidden, 3 + state_dim), dtype=dtype),
            "bias1": jnp.zeros((hidden,), dtype=dtype),
            "weight2": 0.01 * jax.random.normal(key_w2, (state_dim, hidden), dtype=dtype),
            "bias2": jnp.zeros((state_dim,), dtype=dtype),
        }
    else:
        raise ValueError(f"Unsupported candidate_mode={config.candidate_mode!r}")
    return params


def _compute_update_gate(
    params: VolGRUParamsJax,
    config: VolGRUConfig,
    X_t: Any,
    v_t: Any,
) -> Any:
    gate = params["gate"]
    gate_logit = jnp.dot(X_t, gate["beta"])
    if config.gate_mode == "gru_linear":
        gate_logit = gate_logit + gate["state_weight"] * v_t
    return jax.nn.sigmoid(gate_logit)


def _compute_reset_gate(
    params: VolGRUParamsJax,
    config: VolGRUConfig,
    X_t: Any,
    v_t: Any,
) -> Any:
    if not config.use_reset_gate:
        return None
    reset = params["reset"]
    reset_logit = jnp.dot(X_t, reset["gamma"]) + reset["state_weight"] * v_t
    return jax.nn.sigmoid(reset_logit)


def _compute_candidate(
    params: VolGRUParamsJax,
    config: VolGRUConfig,
    r_t: Any,
    v_t: Any,
    reset_t: Any,
) -> Any:
    if config.candidate_mode == "stes_r2":
        return jnp.ones_like(v_t) * (r_t * r_t)

    state_for_cand = v_t if reset_t is None else reset_t * v_t
    abs_r = jnp.abs(r_t)
    r2 = r_t * r_t
    cand = params["candidate"]

    if config.candidate_mode == "linear_pos":
        base = jnp.stack([jnp.ones_like(r_t), r_t, abs_r, r2])
        x = jnp.concatenate([base, state_for_cand], axis=0)
        raw = (cand["weight"] @ x) + cand["bias"]
        return positive_transform_jax(raw, config.positive_transform)

    if config.candidate_mode == "mlp_pos":
        base = jnp.stack([r_t, abs_r, r2])
        x = jnp.concatenate([base, state_for_cand], axis=0)
        h = jnp.tanh((cand["weight1"] @ x) + cand["bias1"])
        raw = (cand["weight2"] @ h) + cand["bias2"]
        return positive_transform_jax(raw, config.positive_transform)

    raise ValueError(f"Unsupported candidate_mode={config.candidate_mode!r}")


def volgru_forward_sequence(
    params: VolGRUParamsJax,
    config: VolGRUConfig,
    X: np.ndarray,
    returns: np.ndarray,
    init_var: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Run VolGRU sequence recursion with :func:`jax.lax.scan`."""
    _require_jax()

    X_jnp = jnp.asarray(X, dtype=jnp.float64)
    r_jnp = jnp.asarray(returns, dtype=jnp.float64)
    if X_jnp.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_jnp.shape}.")
    if r_jnp.ndim != 1:
        raise ValueError(f"returns must be 1D. Got shape {r_jnp.shape}.")
    if X_jnp.shape[0] != r_jnp.shape[0]:
        raise ValueError(
            f"Length mismatch: len(X)={X_jnp.shape[0]} vs len(returns)={r_jnp.shape[0]}"
        )

    state_dim = int(config.state_dim)
    if init_var is None:
        v0 = jnp.full((state_dim,), r_jnp[0] * r_jnp[0], dtype=jnp.float64)
    else:
        init_arr = jnp.asarray(init_var, dtype=jnp.float64)
        if init_arr.ndim == 0:
            v0 = jnp.full((state_dim,), init_arr, dtype=jnp.float64)
        else:
            v0 = jnp.reshape(init_arr, (state_dim,))

    def step(v_t: Any, inputs: tuple[Any, Any]) -> tuple[Any, tuple[Any, Any, Any]]:
        X_t, r_t = inputs
        z_t = _compute_update_gate(params=params, config=config, X_t=X_t, v_t=v_t)
        reset_t = _compute_reset_gate(params=params, config=config, X_t=X_t, v_t=v_t)
        v_cand_t = _compute_candidate(
            params=params,
            config=config,
            r_t=r_t,
            v_t=v_t,
            reset_t=reset_t,
        )
        v_next = (1.0 - z_t) * v_t + z_t * v_cand_t
        v_next = jnp.maximum(v_next, float(config.eps))
        return v_next, (jnp.mean(v_next), z_t, v_cand_t)

    final_var, (sigma2_next, z_t, v_cand_t) = jax.lax.scan(step, v0, (X_jnp, r_jnp))
    if state_dim == 1:
        z_t = jnp.reshape(z_t, (z_t.shape[0],))
        v_cand_t = jnp.reshape(v_cand_t, (v_cand_t.shape[0],))
    return sigma2_next, z_t, v_cand_t, final_var
