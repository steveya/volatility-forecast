"""Shared helpers for VolGRU backends."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _schema_hash(cols: list[str]) -> str:
    payload = json.dumps(list(cols), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _coerce_X(X: Any) -> tuple[np.ndarray, list[str] | None]:
    """Convert features to numpy and collect column names if present."""
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        return X.to_numpy(dtype=float), list(X.columns)
    return np.asarray(X, dtype=float), None


def _set_schema(model: Any, feature_names: list[str] | None, n_features: int) -> None:
    """Store schema metadata on a fitted model."""
    model.n_features_ = int(n_features)
    names = (
        feature_names
        if feature_names is not None
        else [f"x{i}" for i in range(n_features)]
    )
    model.feature_names_ = list(names)
    model.feature_schema_hash_ = _schema_hash(model.feature_names_)


def _check_schema(
    model: Any, X_feature_names: list[str] | None, X_np: np.ndarray
) -> None:
    """Check inference-time feature schema against training schema."""
    if getattr(model, "n_features_", None) is None:
        return
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
    if X_np.shape[1] != model.n_features_:
        raise ValueError(
            f"Feature count mismatch: trained {model.n_features_}, got {X_np.shape[1]}"
        )
    if (
        X_feature_names is not None
        and getattr(model, "feature_names_", None) is not None
    ):
        if list(X_feature_names) != list(model.feature_names_):
            raise ValueError(
                "Feature schema mismatch.\n"
                f"Trained: {model.feature_names_[:8]}...\n"
                f"Got:     {list(X_feature_names)[:8]}..."
            )


def align_returns_next(returns: np.ndarray) -> np.ndarray:
    """Return r_{t+1} aligned to time t by shifting left with edge padding."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return r.copy()
    out = np.empty_like(r)
    if r.size == 1:
        out[0] = r[0]
        return out
    out[:-1] = r[1:]
    out[-1] = r[-1]
    return out


def positive_transform_numpy(x: np.ndarray, mode: str) -> np.ndarray:
    """Apply a positivity transform in numpy."""
    if mode == "softplus":
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    if mode == "exp":
        return np.exp(x)
    raise ValueError(f"Unsupported positive_transform={mode!r}")


def positive_transform_torch(x: Any, mode: str) -> Any:
    """Apply positivity transform in torch."""
    try:
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for torch backend transforms.") from exc

    if mode == "softplus":
        return F.softplus(x)
    if mode == "exp":
        return x.exp()
    raise ValueError(f"Unsupported positive_transform={mode!r}")


def nll_gaussian_torch(returns_next: Any, sigma2_next: Any, eps: float = 1e-12) -> Any:
    """Mean Gaussian negative log-likelihood under zero mean."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for torch backend losses.") from exc

    var = torch.clamp(sigma2_next, min=float(eps))
    return 0.5 * torch.mean(torch.log(var) + (returns_next**2) / var)


def gate_entropy_term_torch(z: Any, eps: float = 1e-12) -> Any:
    """Mean z*log(z) + (1-z)*log(1-z), clamped for numerical safety."""
    zc = z.clamp(min=float(eps), max=1.0 - float(eps))
    return (zc * zc.log() + (1.0 - zc) * (1.0 - zc).log()).mean()


def qlike_torch(y_true: Any, y_pred: Any, eps: float = 1e-12) -> Any:
    """Mean QLIKE loss: mean(y/pred - log(y/pred) - 1), always >= 0."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for torch backend losses.") from exc

    y_safe = torch.clamp(y_true, min=float(eps))
    pred_safe = torch.clamp(y_pred, min=float(eps))
    ratio = y_safe / pred_safe
    return torch.mean(ratio - torch.log(ratio) - 1.0)


def feature_group_penalty_torch(beta: Any, eps: float = 1e-12) -> Any:
    """Row-wise group-lasso penalty: sum of L2 norms of each feature row.

    For ``beta`` of shape ``(p, d)``, computes
    ``sum_i sqrt(sum_j beta[i,j]**2 + eps)``.
    The ``eps`` inside the sqrt ensures differentiability at the origin
    (important because ``gate_beta`` is initialised to zeros).
    For ``d == 1`` this reduces to an L1-like penalty on scalar gate weights.
    """
    import torch

    if beta.ndim == 1:
        return torch.sum(torch.sqrt(beta**2 + float(eps)))
    return torch.sum(torch.sqrt(torch.sum(beta**2, dim=1) + float(eps)))
