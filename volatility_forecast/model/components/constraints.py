"""Shared constrained transforms for PGARCH-family models."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    pos = z_arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    if np.isscalar(z):
        return float(out)
    return out


def softplus(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=float)
    out = np.log1p(np.exp(-np.abs(z_arr))) + np.maximum(z_arr, 0.0)
    if np.isscalar(z):
        return float(out)
    return out


def logit(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    p_arr = np.asarray(p, dtype=float)
    p_arr = np.clip(p_arr, eps, 1.0 - eps)
    out = np.log(p_arr) - np.log1p(-p_arr)
    if np.isscalar(p):
        return float(out)
    return out


def softplus_inverse(y: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    y_arr = np.asarray(y, dtype=float)
    y_arr = np.maximum(y_arr, eps)
    out = y_arr + np.log(-np.expm1(-y_arr))
    if np.isscalar(y):
        return float(out)
    return out
