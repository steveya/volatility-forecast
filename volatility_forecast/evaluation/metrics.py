"""Standard evaluation metrics for volatility forecasts.

Includes:
- RMSE
- MAE
- QLIKE (with clipping to epsilon)
- Hit rate for direction
- Correlation between realized and forecast
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Tuple


def rmse(y: Iterable[float], yhat: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.sqrt(np.mean((y - yhat) ** 2))


def mae(y: Iterable[float], yhat: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean(np.abs(y - yhat))


def qlike(y: Iterable[float], yhat: Iterable[float], epsilon: float = 1e-8) -> float:
    """QLIKE loss for variance forecasts.

    QLIKE(y, yhat) = y / yhat - log(y / yhat) - 1

    To avoid numerical issues, both y and yhat are clipped to [epsilon, +inf).
    Returns the mean QLIKE across observations.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")

    y_clipped = np.clip(y, epsilon, None)
    yhat_clipped = np.clip(yhat, epsilon, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y_clipped / yhat_clipped
        loss = ratio - np.log(ratio) - 1.0
    return np.nanmean(loss)


def hit_rate(y: Iterable[float], yhat: Iterable[float]) -> float:
    """Directional hit rate for changes in volatility.

    Compares sign of day-over-day changes of y and yhat.
    Returns fraction of days where signs match (excluding NaNs).
    """
    y = pd.Series(y).astype(float).pct_change().dropna()
    yhat = pd.Series(yhat).astype(float).pct_change().dropna()
    common_index = y.index.intersection(yhat.index)
    if len(common_index) == 0:
        return float("nan")
    y_ch = np.sign(y.loc[common_index])
    yhat_ch = np.sign(yhat.loc[common_index])
    return float((y_ch == yhat_ch).mean())


def corr(y: Iterable[float], yhat: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1])
