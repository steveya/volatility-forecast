"""Diebold-Mariano test for comparing forecast accuracy.

Implements a HAC / Newey-West variance estimator for the loss differential series.
Returns DM statistic, p-value, sample size, and mean loss differential.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from math import floor
from scipy import stats


def _autocov(x: np.ndarray, lag: int) -> float:
    n = x.size
    if lag == 0:
        return np.sum((x - x.mean()) ** 2) / n
    return np.sum((x[: n - lag] - x.mean()) * (x[lag:] - x.mean())) / n


def _newey_west_var_mean(x: np.ndarray, max_lags: int | None = None) -> float:
    n = x.size
    if max_lags is None:
        # Common automatic bandwidth (Newey-West) rule-of-thumb
        max_lags = int(floor(4 * (n / 100) ** (2.0 / 9.0)))
    gamma0 = _autocov(x, 0)
    s2 = gamma0
    for l in range(1, max_lags + 1):
        weight = 1.0 - l / float(max_lags + 1)
        s2 += 2.0 * weight * _autocov(x, l)
    # return variance of the sample mean
    return s2 / n


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    max_lags: int | None = None,
    two_sided: bool = True,
) -> Dict[str, float]:
    """Perform Diebold-Mariano test comparing two loss series.

    Parameters
    - loss_a, loss_b: sequences of loss values (must be same length)
    - h: forecast horizon (currently informational for API)
    - max_lags: bandwidth for Newey-West HAC. If None, use automatic rule.

    Returns dict with keys:
    - dm_stat: DM t-statistic
    - p_value: two-sided p-value
    - n: sample size used
    - mean_d: mean loss differential (A - B)
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss series must have the same shape")
    d = a - b
    n = d.size
    mean_d = float(np.nanmean(d))
    # HAC variance of mean
    var_mean = _newey_west_var_mean(d, max_lags)
    if var_mean <= 0 or np.isnan(var_mean):
        dm_stat = float("nan")
        p_value = float("nan")
    else:
        dm_stat = mean_d / np.sqrt(var_mean)
        # t-distribution approximation with df = n - 1
        df = max(n - 1, 1)
        if two_sided:
            p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=df))
        else:
            p_value = 1.0 - stats.t.cdf(dm_stat, df=df)
    return {"dm_stat": dm_stat, "p_value": p_value, "n": int(n), "mean_d": mean_d}
