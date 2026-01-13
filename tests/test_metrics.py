import numpy as np
from volatility_forecast.evaluation import metrics


def test_qlike_clipping_handles_zero():
    y = np.array([0.0, 1e-6, 0.1])
    yhat = np.array([0.0, 1e-7, 0.2])
    v = metrics.qlike(y, yhat, epsilon=1e-8)
    assert np.isfinite(v)


def test_qlike_known_value():
    y = np.array([0.1, 0.2])
    yhat = np.array([0.08, 0.25])
    # manual computation
    ratio = y / yhat
    expected = np.mean(ratio - np.log(ratio) - 1.0)
    v = metrics.qlike(y, yhat)
    assert np.allclose(v, expected)


def test_rmse_and_mae():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([0.9, 2.1, 2.9])
    assert np.isclose(metrics.mae(y, yhat), np.mean(np.abs(y - yhat)))
    assert np.isclose(metrics.rmse(y, yhat), np.sqrt(np.mean((y - yhat) ** 2)))
