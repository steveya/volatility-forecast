"""Shared recursion helpers for PGARCH-family models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PGARCHStepResult:
    """Single PGARCH recursion step and its local sensitivities."""

    q: float
    h_raw_next: float
    h_next: float
    rho: float
    local_impulse_mu: float
    local_impulse_phi: float
    local_impulse_g: float
    floor_active: bool


def pgarch_step(
    *,
    mu: float,
    phi: float,
    g: float,
    y_prev: float,
    h_prev: float,
    h_min: float,
    mu_prime: float = 0.0,
    phi_prime: float = 0.0,
    g_prime: float = 0.0,
) -> PGARCHStepResult:
    """Advance the PGARCH recursion by one step."""

    q = float(g) * float(y_prev) + (1.0 - float(g)) * float(h_prev)
    h_raw_next = (1.0 - float(phi)) * float(mu) + float(phi) * q

    if h_raw_next <= float(h_min):
        return PGARCHStepResult(
            q=q,
            h_raw_next=float(h_raw_next),
            h_next=float(h_min),
            rho=0.0,
            local_impulse_mu=0.0,
            local_impulse_phi=0.0,
            local_impulse_g=0.0,
            floor_active=True,
        )

    return PGARCHStepResult(
        q=q,
        h_raw_next=float(h_raw_next),
        h_next=float(h_raw_next),
        rho=float(phi) * (1.0 - float(g)),
        local_impulse_mu=(1.0 - float(phi)) * float(mu_prime),
        local_impulse_phi=(q - float(mu)) * float(phi_prime),
        local_impulse_g=float(phi) * (float(y_prev) - float(h_prev)) * float(g_prime),
        floor_active=False,
    )


def variance_path_from_components(
    y: np.ndarray,
    *,
    mu: np.ndarray,
    phi: np.ndarray,
    g: np.ndarray,
    h0: float,
    h_min: float,
) -> np.ndarray:
    """Run the causal PGARCH recursion from rowwise component sequences."""

    y_np = np.asarray(y, dtype=float)
    mu_np = np.asarray(mu, dtype=float)
    phi_np = np.asarray(phi, dtype=float)
    g_np = np.asarray(g, dtype=float)
    T = len(y_np)

    for name, arr in {"mu": mu_np, "phi": phi_np, "g": g_np}.items():
        if arr.shape != (T,):
            raise ValueError(f"{name} must have shape {(T,)}, got {arr.shape}.")

    h = np.empty(T, dtype=float)
    h[0] = max(float(h0), float(h_min))

    for row in range(T - 1):
        step = pgarch_step(
            mu=float(mu_np[row]),
            phi=float(phi_np[row]),
            g=float(g_np[row]),
            y_prev=float(y_np[row]),
            h_prev=float(h[row]),
            h_min=float(h_min),
        )
        h[row + 1] = step.h_next

    return h
