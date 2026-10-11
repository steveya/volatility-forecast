"""Reusable PGARCH score transforms, recursion, and score blending."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .components.channel_layout import VALID_PGARCH_CHANNELS as VALID_PGARCH_CHANNELS
from .components.constraints import (
    logit as _logit,
)
from .components.constraints import (
    sigmoid as _sigmoid,
)
from .components.constraints import (
    softplus as _softplus,
)
from .components.constraints import (
    softplus_inverse as _softplus_inverse,
)
from .components.losses import (
    loss_from_variance,
    loss_grad_wrt_h,
    loss_hess_weight_wrt_h,
    optimization_loss_name,
    score_from_variance,
)
from .components.recursion import pgarch_step, variance_path_from_components


@dataclass(frozen=True, slots=True)
class PGARCHBounds:
    mu_min: float = 1e-12
    phi_min: float = 1e-4
    phi_max: float = 1.0 - 1e-4
    g_min: float = 1e-4
    g_max: float = 1.0 - 1e-4
    h_min: float = 1e-12

    def __post_init__(self) -> None:
        if self.mu_min <= 0.0 or self.h_min <= 0.0:
            raise ValueError("mu_min and h_min must be > 0.")
        if not (0.0 < self.phi_min < self.phi_max < 1.0):
            raise ValueError("Require 0 < phi_min < phi_max < 1.")
        if not (0.0 < self.g_min < self.g_max < 1.0):
            raise ValueError("Require 0 < g_min < g_max < 1.")


@dataclass(frozen=True, slots=True)
class PGARCHRawScores:
    mu: np.ndarray
    phi: np.ndarray
    g: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=float))
        object.__setattr__(self, "phi", np.asarray(self.phi, dtype=float))
        object.__setattr__(self, "g", np.asarray(self.g, dtype=float))

    @property
    def a(self) -> np.ndarray:
        return self.mu

    @property
    def b(self) -> np.ndarray:
        return self.phi

    @property
    def c(self) -> np.ndarray:
        return self.g

    def validate_length(self, expected: int) -> None:
        for name, arr in {"a": self.a, "b": self.b, "c": self.c}.items():
            if arr.shape != (expected,):
                raise ValueError(f"{name} must have shape {(expected,)}, got {arr.shape}.")


@dataclass(frozen=True, slots=True)
class PGARCHRecursionState:
    scores: PGARCHRawScores
    mu: np.ndarray
    phi: np.ndarray
    g: np.ndarray
    mu_prime: np.ndarray
    phi_prime: np.ndarray
    g_prime: np.ndarray
    h: np.ndarray
    q: np.ndarray
    rho: np.ndarray
    local_impulse_mu: np.ndarray
    local_impulse_phi: np.ndarray
    local_impulse_g: np.ndarray
    floor_active: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "a": self.scores.a,
            "b": self.scores.b,
            "c": self.scores.c,
            "mu": self.mu,
            "phi": self.phi,
            "g": self.g,
            "mu_prime": self.mu_prime,
            "phi_prime": self.phi_prime,
            "g_prime": self.g_prime,
            "h": self.h,
            "q": self.q,
            "rho": self.rho,
            "local_impulse_mu": self.local_impulse_mu,
            "local_impulse_phi": self.local_impulse_phi,
            "local_impulse_g": self.local_impulse_g,
            "floor_active": self.floor_active,
        }


class PGARCHCore:
    """Shared PGARCH score transforms and recursion mechanics.

    This core is intentionally model-agnostic: boosted trees, mixture-of-experts,
    or future attention-style routers can all emit raw channel scores and reuse
    the same PGARCH recursion, loss geometry, and score blending helpers.
    """

    def __init__(self, *, loss: str = "qlike", bounds: PGARCHBounds | None = None) -> None:
        valid_losses = {"mse", "rmse", "qlike"}
        if loss not in valid_losses:
            raise ValueError(f"loss must be one of {sorted(valid_losses)}")
        self.loss = loss
        self.bounds = bounds or PGARCHBounds()

    @property
    def optimization_loss_name(self) -> str:
        return optimization_loss_name(self.loss)

    def link_mu(self, a: np.ndarray) -> np.ndarray:
        return self.bounds.mu_min + np.asarray(_softplus(np.asarray(a, dtype=float)), dtype=float)

    def link_phi(self, b: np.ndarray) -> np.ndarray:
        sigma = np.asarray(_sigmoid(np.asarray(b, dtype=float)), dtype=float)
        return self.bounds.phi_min + (self.bounds.phi_max - self.bounds.phi_min) * sigma

    def link_g(self, c: np.ndarray) -> np.ndarray:
        sigma = np.asarray(_sigmoid(np.asarray(c, dtype=float)), dtype=float)
        return self.bounds.g_min + (self.bounds.g_max - self.bounds.g_min) * sigma

    def link_mu_prime(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(_sigmoid(np.asarray(a, dtype=float)), dtype=float)

    def link_phi_prime(self, b: np.ndarray) -> np.ndarray:
        sigma = np.asarray(_sigmoid(np.asarray(b, dtype=float)), dtype=float)
        return (self.bounds.phi_max - self.bounds.phi_min) * sigma * (1.0 - sigma)

    def link_g_prime(self, c: np.ndarray) -> np.ndarray:
        sigma = np.asarray(_sigmoid(np.asarray(c, dtype=float)), dtype=float)
        return (self.bounds.g_max - self.bounds.g_min) * sigma * (1.0 - sigma)

    def inverse_link_mu(self, mu: np.ndarray) -> np.ndarray:
        return np.asarray(_softplus_inverse(np.asarray(mu, dtype=float) - self.bounds.mu_min), dtype=float)

    def inverse_link_phi(self, phi: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(phi, dtype=float) - self.bounds.phi_min) / (
            self.bounds.phi_max - self.bounds.phi_min
        )
        return np.asarray(_logit(scaled), dtype=float)

    def inverse_link_g(self, g: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(g, dtype=float) - self.bounds.g_min) / (
            self.bounds.g_max - self.bounds.g_min
        )
        return np.asarray(_logit(scaled), dtype=float)

    def inverse_link_scores(self, mu: np.ndarray, phi: np.ndarray, g: np.ndarray) -> PGARCHRawScores:
        return PGARCHRawScores(
            mu=self.inverse_link_mu(mu),
            phi=self.inverse_link_phi(phi),
            g=self.inverse_link_g(g),
        )

    def forward_with_scores(
        self,
        y: np.ndarray,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        h0: float,
    ) -> PGARCHRecursionState:
        y_np = np.asarray(y, dtype=float)
        scores = PGARCHRawScores(mu=a, phi=b, g=c)
        T = len(y_np)
        scores.validate_length(T)

        mu = self.link_mu(scores.a)
        phi = self.link_phi(scores.b)
        g = self.link_g(scores.c)
        mu_prime = self.link_mu_prime(scores.a)
        phi_prime = self.link_phi_prime(scores.b)
        g_prime = self.link_g_prime(scores.c)

        h = np.empty(T, dtype=float)
        h[0] = max(float(h0), self.bounds.h_min)
        q = np.zeros(T, dtype=float)
        rho = np.zeros(T, dtype=float)
        local_impulse_mu = np.zeros(T, dtype=float)
        local_impulse_phi = np.zeros(T, dtype=float)
        local_impulse_g = np.zeros(T, dtype=float)
        floor_active = np.zeros(T, dtype=bool)

        for row in range(T - 1):
            step = pgarch_step(
                mu=float(mu[row]),
                phi=float(phi[row]),
                g=float(g[row]),
                y_prev=float(y_np[row]),
                h_prev=float(h[row]),
                h_min=self.bounds.h_min,
                mu_prime=float(mu_prime[row]),
                phi_prime=float(phi_prime[row]),
                g_prime=float(g_prime[row]),
            )
            q[row] = step.q
            h[row + 1] = step.h_next
            rho[row] = step.rho
            local_impulse_mu[row] = step.local_impulse_mu
            local_impulse_phi[row] = step.local_impulse_phi
            local_impulse_g[row] = step.local_impulse_g
            floor_active[row] = step.floor_active

        q[-1] = g[-1] * y_np[-1] + (1.0 - g[-1]) * h[-1]
        return PGARCHRecursionState(
            scores=scores,
            mu=mu,
            phi=phi,
            g=g,
            mu_prime=mu_prime,
            phi_prime=phi_prime,
            g_prime=g_prime,
            h=h,
            q=q,
            rho=rho,
            local_impulse_mu=local_impulse_mu,
            local_impulse_phi=local_impulse_phi,
            local_impulse_g=local_impulse_g,
            floor_active=floor_active,
        )

    def variance_path_from_components(
        self,
        y: np.ndarray,
        *,
        mu: np.ndarray,
        phi: np.ndarray,
        g: np.ndarray,
        h0: float,
    ) -> np.ndarray:
        return variance_path_from_components(
            y,
            mu=mu,
            phi=phi,
            g=g,
            h0=h0,
            h_min=self.bounds.h_min,
        )

    def loss_from_variance(self, y: np.ndarray, h: np.ndarray) -> float:
        return loss_from_variance(y, h, loss=self.loss, h_min=self.bounds.h_min)

    def loss_grad_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return loss_grad_wrt_h(y, h, loss=self.loss, h_min=self.bounds.h_min)

    def loss_hess_weight_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return loss_hess_weight_wrt_h(y, h, loss=self.loss, h_min=self.bounds.h_min)

    def score_from_variance(self, y: np.ndarray, h: np.ndarray, metric: str) -> float:
        return score_from_variance(y, h, metric=metric, h_min=self.bounds.h_min)


def blend_raw_scores(
    experts: Sequence[PGARCHRawScores],
    weights: np.ndarray,
) -> PGARCHRawScores:
    """Blend expert raw scores with scalar or time-varying weights.

    ``weights`` may be shape ``(n_experts,)`` for global weights or
    ``(n_experts, T)`` for time-varying attention / routing weights.
    """

    if not experts:
        raise ValueError("At least one expert score triplet is required.")

    T = int(len(experts[0].a))
    for raw in experts:
        raw.validate_length(T)

    weight_np = np.asarray(weights, dtype=float)
    n_experts = len(experts)
    if weight_np.ndim == 1:
        if weight_np.shape != (n_experts,):
            raise ValueError(f"Global weights must have shape {(n_experts,)}, got {weight_np.shape}.")
        total = float(np.sum(weight_np))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Global weights must sum to a finite positive value.")
        norm = weight_np / total
        return PGARCHRawScores(
            mu=np.tensordot(norm, np.stack([raw.mu for raw in experts], axis=0), axes=(0, 0)),
            phi=np.tensordot(norm, np.stack([raw.phi for raw in experts], axis=0), axes=(0, 0)),
            g=np.tensordot(norm, np.stack([raw.g for raw in experts], axis=0), axes=(0, 0)),
        )

    if weight_np.ndim == 2:
        if weight_np.shape != (n_experts, T):
            raise ValueError(f"Time-varying weights must have shape {(n_experts, T)}, got {weight_np.shape}.")
        totals = np.sum(weight_np, axis=0)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise ValueError("Each time step needs a finite positive total expert weight.")
        norm = weight_np / totals
        return PGARCHRawScores(
            mu=np.sum(norm * np.stack([raw.mu for raw in experts], axis=0), axis=0),
            phi=np.sum(norm * np.stack([raw.phi for raw in experts], axis=0), axis=0),
            g=np.sum(norm * np.stack([raw.g for raw in experts], axis=0), axis=0),
        )

    raise ValueError(f"weights must be 1D or 2D, got shape {weight_np.shape}.")
