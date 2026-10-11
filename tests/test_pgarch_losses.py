from __future__ import annotations

import numpy as np
import pytest

from volatility_forecast.model.components.losses import (
    loss_from_variance,
    loss_grad_wrt_h,
    loss_hess_weight_wrt_h,
    loss_hess_wrt_h,
    optimization_loss_name,
    score_from_variance,
)


def _finite_difference_gradient(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.zeros_like(h)
    for idx in range(h.size):
        step = np.zeros_like(h)
        step[idx] = eps
        grad[idx] = (
            loss_from_variance(y, h + step, loss=loss, h_min=h_min)
            - loss_from_variance(y, h - step, loss=loss, h_min=h_min)
        ) / (2.0 * eps)
    return grad


def _finite_difference_hessian(
    y: np.ndarray,
    h: np.ndarray,
    *,
    loss: str,
    h_min: float,
    eps: float = 1e-6,
) -> np.ndarray:
    hess = np.zeros_like(h)
    for idx in range(h.size):
        step = np.zeros_like(h)
        step[idx] = eps
        grad_plus = loss_grad_wrt_h(y, h + step, loss=loss, h_min=h_min)
        grad_minus = loss_grad_wrt_h(y, h - step, loss=loss, h_min=h_min)
        hess[idx] = (grad_plus[idx] - grad_minus[idx]) / (2.0 * eps)
    return hess


def test_manual_loss_values_match_helpers() -> None:
    y = np.array([0.04, 0.05, 0.07, 0.06], dtype=float)
    h = np.array([0.04, 0.045, 0.055, 0.065], dtype=float)

    mse_manual = np.mean((y[1:] - h[1:]) ** 2)
    qlike_manual = np.mean(np.log(h[1:]) + y[1:] / h[1:])

    assert optimization_loss_name("rmse") == "mse"
    assert np.isclose(score_from_variance(y, h, metric="mse", h_min=1e-12), mse_manual)
    assert np.isclose(
        score_from_variance(y, h, metric="rmse", h_min=1e-12), np.sqrt(mse_manual)
    )
    assert np.isclose(
        score_from_variance(y, h, metric="qlike", h_min=1e-12), qlike_manual
    )


def test_loss_gradients_and_hessians_match_finite_differences() -> None:
    y = np.array([0.04, 0.05, 0.07, 0.06, 0.055], dtype=float)
    h = np.array([0.04, 0.045, 0.055, 0.065, 0.058], dtype=float)

    for loss in ("mse", "qlike"):
        grad = loss_grad_wrt_h(y, h, loss=loss, h_min=1e-12)
        grad_fd = _finite_difference_gradient(y, h, loss=loss, h_min=1e-12)
        hess = loss_hess_wrt_h(y, h, loss=loss, h_min=1e-12)
        hess_fd = _finite_difference_hessian(y, h, loss=loss, h_min=1e-12)

        assert grad[0] == 0.0
        assert hess[0] == 0.0
        assert np.allclose(grad, grad_fd, rtol=1e-5, atol=1e-6)
        assert np.allclose(hess, hess_fd, rtol=1e-5, atol=1e-6)


def test_qlike_hessian_weights_match_expected_gauss_newton_form() -> None:
    y = np.array([0.04, 0.05, 0.07, 0.06], dtype=float)
    h = np.array([0.04, 0.045, 0.055, 0.065], dtype=float)

    weights = loss_hess_weight_wrt_h(y, h, loss="qlike", h_min=1e-12)
    expected = np.zeros_like(h)
    expected[1:] = 1.0 / ((len(y) - 1) * (h[1:] ** 2))

    assert np.allclose(weights, expected)


@pytest.mark.parametrize(
    "helper",
    [
        score_from_variance,
        loss_from_variance,
        loss_grad_wrt_h,
        loss_hess_wrt_h,
        loss_hess_weight_wrt_h,
    ],
)
@pytest.mark.parametrize(
    "y, h",
    [
        (np.ones((3, 1)), np.ones(3)),
        (np.ones(3), np.ones((3, 1))),
        (np.ones(3), np.ones(4)),
        (np.ones(0), np.ones(0)),
        (np.ones(1), np.ones(1)),
        (np.array([1.0, np.nan]), np.ones(2)),
        (np.array([1.0, np.inf]), np.ones(2)),
        (np.ones(2), np.array([1.0, np.nan])),
        (np.ones(2), np.array([1.0, np.inf])),
    ],
)
def test_loss_helpers_reject_invalid_samples(helper, y, h) -> None:
    kwargs = {"metric": "mse"} if helper is score_from_variance else {"loss": "mse"}

    with pytest.raises(ValueError):
        helper(y, h, h_min=1e-12, **kwargs)


@pytest.mark.parametrize("h_min", [0.0, -1.0, np.nan, np.inf])
def test_loss_helpers_require_positive_finite_variance_floor(h_min) -> None:
    with pytest.raises(ValueError, match="h_min must be finite and positive"):
        score_from_variance(np.ones(3), np.ones(3), metric="mse", h_min=h_min)
