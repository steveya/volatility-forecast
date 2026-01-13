"""Mincer-Zarnowitz regression utilities for variance forecasts.

Performs regression of y on yhat (variance scale) or log-variance if requested.
Returns parameters, t-stats, R^2 and a summary table suitable for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def mincer_zarnowitz(y, yhat, log_scale: bool = False, add_const: bool = True):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    if log_scale:
        y = np.log(np.clip(y, 1e-8, None))
        yhat = np.log(np.clip(yhat, 1e-8, None))

    X = yhat.reshape(-1, 1)
    if add_const:
        X = np.column_stack([np.ones(X.shape[0]), X])

    # OLS via numpy
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X.dot(beta)
    resid = y - y_pred
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = (resid**2).sum() / dof
    # Use pseudo-inverse to handle near-singular design matrices robustly
    xtx_inv = np.linalg.pinv(X.T.dot(X))
    se = np.sqrt(np.diag(xtx_inv) * s2)
    # guard division by zero (if se==0, set t-stat to nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se == 0, np.nan, beta / se)
    t_stats = t_stats.tolist()
    # convert back to numpy arrays as needed for downstream
    t_stats = np.asarray(t_stats)
    ssr = ((y_pred - y.mean()) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    r2 = ssr / sst if sst != 0 else float("nan")

    param_names = ["const", "beta"] if add_const else ["beta"]
    params = dict(zip(param_names, beta.tolist()))
    tstats = dict(zip(param_names, t_stats.tolist()))

    return {
        "params": params,
        "t_stats": tstats,
        "r2": float(r2),
        "n": int(n),
        "resid": resid,
        "y_pred": y_pred,
    }


def plot_mincer_zarnowitz(y, yhat, log_scale: bool = False, ax=None):
    """Create a scatter plot of y vs yhat with fitted regression line and 45-degree line.

    Returns the matplotlib Axes object (creates figure/axes if not provided).
    """
    import matplotlib.pyplot as plt

    res = mincer_zarnowitz(y, yhat, log_scale=log_scale)
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yhat, y, alpha=0.6, s=20)
    # fitted line
    yy = res["y_pred"]
    # if add_const, yy corresponds to masked y order
    ax.plot(yhat, yy, color="C1", label="MZ fit")
    # 45-degree line
    mn = min(y.min(), yhat.min())
    mx = max(y.max(), yhat.max())
    ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="45°")
    ax.set_xlabel("Forecast")
    ax.set_ylabel("Realized")
    ax.legend()
    ax.set_title(f"Mincer–Zarnowitz (R²={res['r2']:.3f})")
    return ax
