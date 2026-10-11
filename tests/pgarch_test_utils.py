from __future__ import annotations

import numpy as np

from volatility_forecast.model.pgarch_linear_model import _sigmoid, _softplus


def make_synthetic_pgarch_data(
    n: int = 160,
    d: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X_aug = np.column_stack([np.ones(n, dtype=float), X])

    w_mu = np.array([-2.0, 0.25, -0.15], dtype=float)[: d + 1]
    w_phi = np.array([2.0, 0.15, 0.10], dtype=float)[: d + 1]
    w_g = np.array([-2.5, -0.20, 0.20], dtype=float)[: d + 1]

    y = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    y[0] = 0.04
    h[0] = 0.04

    for t in range(1, n):
        x = X_aug[t - 1]
        mu = 1e-12 + float(_softplus(np.dot(w_mu, x)))
        phi = 1e-4 + (1.0 - 2e-4) * float(_sigmoid(np.dot(w_phi, x)))
        g = 1e-4 + (1.0 - 2e-4) * float(_sigmoid(np.dot(w_g, x)))
        h[t] = (1.0 - phi) * mu + phi * (g * y[t - 1] + (1.0 - g) * h[t - 1])
        y[t] = max(h[t] * (0.8 + 0.4 * rng.random()), 1e-8)

    return y, X
