"""Shared loss helpers for PGARCH-family models."""

from __future__ import annotations

import numpy as np

VALID_PGARCH_LOSSES = frozenset({"mse", "rmse", "qlike"})


def _prepare_variance_inputs(
    y: np.ndarray,
    h: np.ndarray,
    h_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_np = np.asarray(y, dtype=float)
    h_np = np.asarray(h, dtype=float)
    if y_np.ndim != 1 or h_np.ndim != 1:
        raise ValueError("y and h must be 1-D arrays.")
    if y_np.shape != h_np.shape:
        raise ValueError("y and h must have matching shapes.")
    if y_np.size < 2:
        raise ValueError(
            "At least 2 observations are required to exclude the warm start."
        )
    if not np.all(np.isfinite(y_np)) or not np.all(np.isfinite(h_np)):
        raise ValueError("y and h must contain only finite values.")
    variance_floor = float(h_min)
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("h_min must be finite and positive.")
    return y_np, np.maximum(h_np, variance_floor)


def optimization_loss_name(loss: str) -> str:
    if loss not in VALID_PGARCH_LOSSES:
        raise ValueError(f"loss must be one of {sorted(VALID_PGARCH_LOSSES)}")
    return "mse" if loss == "rmse" else loss


def score_from_variance(
    y: np.ndarray,
    h: np.ndarray,
    *,
    metric: str,
    h_min: float,
) -> float:
    if metric not in VALID_PGARCH_LOSSES:
        raise ValueError(f"metric must be one of {sorted(VALID_PGARCH_LOSSES)}")

    y_np, h_safe = _prepare_variance_inputs(y, h, h_min)
    y_eff = y_np[1:]
    h_eff = h_safe[1:]

    if metric == "mse":
        return float(np.mean((y_eff - h_eff) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_eff - h_eff) ** 2)))
    return float(np.mean(np.log(h_eff) + y_eff / h_eff))


def loss_from_variance(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
) -> float:
    return score_from_variance(
        y,
        h,
        metric=optimization_loss_name(loss),
        h_min=h_min,
    )


def loss_grad_wrt_h(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
) -> np.ndarray:
    y_np, h_safe = _prepare_variance_inputs(y, h, h_min)
    grad = np.zeros_like(h_safe)
    n_eff = float(len(y_np) - 1)
    if optimization_loss_name(loss) == "mse":
        grad[1:] = 2.0 * (h_safe[1:] - y_np[1:]) / n_eff
        return grad
    grad[1:] = (1.0 / h_safe[1:] - y_np[1:] / (h_safe[1:] ** 2)) / n_eff
    return grad


def loss_hess_wrt_h(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
) -> np.ndarray:
    y_np, h_safe = _prepare_variance_inputs(y, h, h_min)
    hess = np.zeros_like(h_safe)
    n_eff = float(len(y_np) - 1)
    if optimization_loss_name(loss) == "mse":
        hess[1:] = 2.0 / n_eff
        return hess
    hess[1:] = (-1.0 / (h_safe[1:] ** 2) + 2.0 * y_np[1:] / (h_safe[1:] ** 3)) / n_eff
    return hess


def loss_hess_weight_wrt_h(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
) -> np.ndarray:
    y_np, h_safe = _prepare_variance_inputs(y, h, h_min)
    weight = np.zeros_like(h_safe)
    n_eff = float(len(y_np) - 1)
    if optimization_loss_name(loss) == "mse":
        weight[1:] = 2.0 / n_eff
        return weight
    weight[1:] = 1.0 / (n_eff * (h_safe[1:] ** 2))
    weight[1:] = np.clip(weight[1:], 1e-12, 1e12)
    return weight
