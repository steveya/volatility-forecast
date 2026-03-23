"""Reusable PGARCH score transforms, recursion, and score blending."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .xgb_pgarch_model import _logit, _sigmoid, _softplus, _softplus_inverse

VALID_PGARCH_CHANNELS = ("mu", "phi", "g")


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
        return "mse" if self.loss == "rmse" else self.loss

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
            q[row] = g[row] * y_np[row] + (1.0 - g[row]) * h[row]
            h_raw_next = (1.0 - phi[row]) * mu[row] + phi[row] * q[row]
            if h_raw_next <= self.bounds.h_min:
                h[row + 1] = self.bounds.h_min
                rho[row] = 0.0
                local_impulse_mu[row] = 0.0
                local_impulse_phi[row] = 0.0
                local_impulse_g[row] = 0.0
                floor_active[row] = True
                continue

            h[row + 1] = h_raw_next
            rho[row] = phi[row] * (1.0 - g[row])
            local_impulse_mu[row] = (1.0 - phi[row]) * mu_prime[row]
            local_impulse_phi[row] = (q[row] - mu[row]) * phi_prime[row]
            local_impulse_g[row] = phi[row] * (y_np[row] - h[row]) * g_prime[row]

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
        y_np = np.asarray(y, dtype=float)
        mu_np = np.asarray(mu, dtype=float)
        phi_np = np.asarray(phi, dtype=float)
        g_np = np.asarray(g, dtype=float)
        T = len(y_np)
        for name, arr in {"mu": mu_np, "phi": phi_np, "g": g_np}.items():
            if arr.shape != (T,):
                raise ValueError(f"{name} must have shape {(T,)}, got {arr.shape}.")

        h = np.empty(T, dtype=float)
        h[0] = max(float(h0), self.bounds.h_min)
        for row in range(T - 1):
            q_row = g_np[row] * y_np[row] + (1.0 - g_np[row]) * h[row]
            h[row + 1] = max((1.0 - phi_np[row]) * mu_np[row] + phi_np[row] * q_row, self.bounds.h_min)
        return h

    def loss_from_variance(self, y: np.ndarray, h: np.ndarray) -> float:
        return self.score_from_variance(y, h, metric=self.optimization_loss_name)

    def loss_grad_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.bounds.h_min)
        grad = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self.optimization_loss_name == "mse":
            grad[1:] = 2.0 * (h_safe[1:] - y_np[1:]) / n_eff
            return grad
        grad[1:] = (1.0 / h_safe[1:] - y_np[1:] / (h_safe[1:] ** 2)) / n_eff
        return grad

    def loss_hess_weight_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.bounds.h_min)
        weight = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self.optimization_loss_name == "mse":
            weight[1:] = 2.0 / n_eff
            return weight
        weight[1:] = 1.0 / (n_eff * (h_safe[1:] ** 2))
        weight[1:] = np.clip(weight[1:], 1e-12, 1e12)
        return weight

    def score_from_variance(self, y: np.ndarray, h: np.ndarray, metric: str) -> float:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.bounds.h_min)
        y_eff = y_np[1:]
        h_eff = h_safe[1:]
        if metric == "mse":
            return float(np.mean((y_eff - h_eff) ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean((y_eff - h_eff) ** 2)))
        if metric == "qlike":
            return float(np.mean(np.log(h_eff) + y_eff / h_eff))
        raise ValueError(f"metric must be one of ['mse', 'rmse', 'qlike'], got {metric!r}.")


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
