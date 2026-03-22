"""Full three-channel XGBoost PGARCH model."""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .xgb_pgarch_model import (
    PGARCHLinearModel,
    _ConstantPGARCHInitializer,
    _coerce_X,
    _logit,
    _sigmoid,
    _softplus,
    _softplus_inverse,
    xgb,
)

logger = logging.getLogger(__name__)


class XGBPGARCHModel:
    """Full three-channel generalized boosted PGARCH model.

    ``predict_components(X)`` returns rowwise ``mu``, ``phi``, and ``g`` values
    aligned with feature row ``X[t]``. The recursive variance path instead uses
    row ``t`` to affect the next-step forecast ``h[t+1]``. The terminal row
    therefore contributes zero gradient and zero Hessian in every channel.
    """

    def __init__(
        self,
        loss: str = "qlike",
        n_outer_rounds: int = 25,
        trees_per_channel_per_round: int = 1,
        learning_rate: float = 0.02,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        mu_min: float = 1e-12,
        phi_min: float = 1e-4,
        phi_max: float = 1.0 - 1e-4,
        g_min: float = 1e-4,
        g_max: float = 1.0 - 1e-4,
        h_min: float = 1e-12,
        init_method: str = "linear_pgarch",
        init_model: object | None = None,
        channel_update_order: tuple[str, ...] = ("mu", "phi", "g"),
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = None,
        verbosity: int = 0,
        channel_features: dict[str, list[int]] | None = None,
        booster: str = "gbtree",
    ) -> None:
        valid_losses = {"mse", "rmse", "qlike"}
        valid_init_methods = {"linear_pgarch", "intercept_only_pgarch"}
        valid_eval_metrics = {None, "mse", "rmse", "qlike"}
        valid_boosters = {"gbtree", "gblinear"}
        if booster not in valid_boosters:
            raise ValueError(f"booster must be one of {sorted(valid_boosters)}")
        valid_channels = ("mu", "phi", "g")

        if loss not in valid_losses:
            raise ValueError(f"loss must be one of {sorted(valid_losses)}")
        if init_method not in valid_init_methods:
            raise ValueError(f"init_method must be one of {sorted(valid_init_methods)}")
        if n_outer_rounds <= 0:
            raise ValueError("n_outer_rounds must be positive")
        if trees_per_channel_per_round <= 0:
            raise ValueError("trees_per_channel_per_round must be positive")
        if learning_rate <= 0.0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and > 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_child_weight <= 0.0 or not np.isfinite(min_child_weight):
            raise ValueError("min_child_weight must be finite and > 0")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if not (0.0 < colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must be in (0, 1]")
        if reg_alpha < 0.0 or reg_lambda < 0.0 or gamma < 0.0:
            raise ValueError("reg_alpha, reg_lambda, and gamma must be >= 0")
        if mu_min <= 0.0 or h_min <= 0.0:
            raise ValueError("mu_min and h_min must be > 0")
        if not (0.0 < phi_min < phi_max < 1.0):
            raise ValueError("Require 0 < phi_min < phi_max < 1")
        if not (0.0 < g_min < g_max < 1.0):
            raise ValueError("Require 0 < g_min < g_max < 1")
        if len(tuple(channel_update_order)) != len(valid_channels) or tuple(
            sorted(channel_update_order)
        ) != tuple(sorted(valid_channels)):
            raise ValueError("channel_update_order must contain each of 'mu', 'phi', and 'g' exactly once")
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive when provided")
        if eval_metric not in valid_eval_metrics:
            raise ValueError("eval_metric must be one of {None, 'mse', 'rmse', 'qlike'}")

        self.loss = loss
        self.n_outer_rounds = int(n_outer_rounds)
        self.trees_per_channel_per_round = int(trees_per_channel_per_round)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.mu_min = float(mu_min)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.h_min = float(h_min)
        self.init_method = init_method
        self.init_model = init_model
        self.channel_update_order = tuple(channel_update_order)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.verbosity = int(verbosity)
        self.channel_features = channel_features
        self.booster = booster

        self.booster_mu_: Any = None
        self.booster_phi_: Any = None
        self.booster_g_: Any = None
        self.initializer_: object | None = None
        self.init_method_: str | None = None
        self.n_features_in_: int | None = None
        self.is_fitted_: bool = False
        self.train_loss_: float | None = None
        self.best_iteration_: int | None = None
        self.feature_names_: list[str] | None = None
        self.baseline_train_: dict[str, np.ndarray] | None = None
        self.a0_: np.ndarray | None = None
        self.b0_: np.ndarray | None = None
        self.c0_: np.ndarray | None = None
        self.channel_history_: list[dict[str, Any]] = []

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBPGARCHModel":
        """Fit the full three-channel boosted PGARCH model."""
        self._require_xgboost()
        X_np_raw, cols = _coerce_X(X)
        y_np, X_np = self._validate_inputs(y, X_np_raw)

        self.n_features_in_ = int(X_np.shape[1])
        self.feature_names_ = cols
        self.is_fitted_ = False
        self.best_iteration_ = None
        self.channel_history_ = []
        self.booster_mu_ = None
        self.booster_phi_ = None
        self.booster_g_ = None

        initializer = self._fit_initializer(y_np, X)
        baseline_train = self._extract_baseline_sequences(y_np, X, initializer)
        a, b, c = self._initialize_raw_scores(baseline_train)
        self.initializer_ = initializer
        self.baseline_train_ = baseline_train
        self.a0_ = a.copy()
        self.b0_ = b.copy()
        self.c0_ = c.copy()

        h0_train = max(float(y_np[0]), self.h_min)

        y_eval: np.ndarray | None = None
        X_eval_raw: np.ndarray | None = None
        h0_eval: float | None = None
        if eval_set is not None:
            y_eval_raw, X_eval_input = eval_set
            X_eval_raw_np, eval_cols = _coerce_X(X_eval_input)
            y_eval, X_eval_raw = self._validate_inputs(y_eval_raw, X_eval_raw_np)
            self._check_schema(eval_cols, X_eval_raw)
            h0_eval = max(float(y_eval[0]), self.h_min)

        best_eval: float | None = None
        best_snapshot: dict[str, Any] | None = None
        no_improve_rounds = 0
        eval_metric = self.eval_metric or self._optimization_loss_name

        for outer_round in range(self.n_outer_rounds):
            for channel in self.channel_update_order:
                updated_scores, booster = self._fit_channel_update(channel, X_np, y_np, a, b, c, h0_train)
                if channel == "mu":
                    a = updated_scores
                    self.booster_mu_ = booster
                elif channel == "phi":
                    b = updated_scores
                    self.booster_phi_ = booster
                else:
                    c = updated_scores
                    self.booster_g_ = booster

                train_state = self._forward_recursion_with_scores(y_np, a, b, c, h0=h0_train)
                self.channel_history_.append(
                    {
                        "outer_round": outer_round,
                        "channel": channel,
                        "train_loss": self._loss_from_state(y_np, train_state["h"]),
                    }
                )

            if y_eval is None or X_eval_raw is None:
                continue

            a_eval, b_eval, c_eval = self._predict_raw_scores(X_eval_raw)
            eval_state = self._forward_recursion_with_scores(y_eval, a_eval, b_eval, c_eval, h0=float(h0_eval))
            eval_value = self._score_from_state(y_eval, eval_state["h"], metric=eval_metric)
            self.channel_history_.append(
                {
                    "outer_round": outer_round,
                    "channel": "eval",
                    "eval_metric": eval_metric,
                    "eval_loss": eval_value,
                }
            )

            if self.early_stopping_rounds is None:
                continue

            if best_eval is None or eval_value < best_eval:
                best_eval = float(eval_value)
                best_snapshot = self._snapshot_boosters()
                self.best_iteration_ = outer_round
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= self.early_stopping_rounds:
                    if best_snapshot is not None:
                        self._restore_boosters(best_snapshot)
                    break

        a_final, b_final, c_final = self._predict_raw_scores(X_np)
        state_train = self._forward_recursion_with_scores(y_np, a_final, b_final, c_final, h0=h0_train)
        self.train_loss_ = self._loss_from_state(y_np, state_train["h"])
        self.is_fitted_ = True
        return self

    def predict_variance(
        self,
        y: np.ndarray,
        X: np.ndarray,
        *,
        h0: float | None = None,
    ) -> np.ndarray:
        """Return the recursion-aware variance path for the supplied lagged target path."""
        self._check_is_fitted()
        X_np_raw, cols = _coerce_X(X)
        y_np, X_np = self._validate_inputs(y, X_np_raw)
        self._check_schema(cols, X_np)
        a, b, c = self._predict_raw_scores(X)
        start = max(float(y_np[0]) if h0 is None else float(h0), self.h_min)
        state = self._forward_recursion_with_scores(y_np, a, b, c, h0=start)
        return state["h"]

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return rowwise ``mu``, ``phi``, and ``g`` aligned to ``X[t]``."""
        self._check_is_fitted()
        X_np_raw, cols = _coerce_X(X)
        X_np = self._validate_feature_matrix(X_np_raw)
        self._check_schema(cols, X_np)
        a, b, c = self._predict_raw_scores(X)
        return {
            "mu": self._link_mu(a),
            "phi": self._link_phi(b),
            "g": self._link_g(c),
        }

    def implied_garch_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return rowwise implied ``omega``, ``alpha``, and ``beta``."""
        components = self.predict_components(X)
        mu = components["mu"]
        phi = components["phi"]
        g = components["g"]
        return {
            "omega": (1.0 - phi) * mu,
            "alpha": phi * g,
            "beta": phi * (1.0 - g),
        }

    def score(self, y: np.ndarray, X: np.ndarray, metric: str = "qlike") -> float:
        """Evaluate the recursive variance path using the requested metric."""
        valid_metrics = {"qlike", "mse", "rmse"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {sorted(valid_metrics)}")
        X_np_raw, _ = _coerce_X(X)
        y_np, _ = self._validate_inputs(y, X_np_raw)
        h = self.predict_variance(y_np, X)
        return self._score_from_state(y_np, h, metric=metric)

    def _validate_inputs(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=float)
        X_np = np.asarray(X, dtype=float)
        if y_np.ndim != 1:
            raise ValueError(f"y must be 1D. Got shape {y_np.shape}.")
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if len(y_np) != X_np.shape[0]:
            raise ValueError(
                f"Length mismatch: len(y)={len(y_np)} and X.shape[0]={X_np.shape[0]}."
            )
        if len(y_np) < 3:
            raise ValueError("At least 3 observations are required.")
        if not np.all(np.isfinite(y_np)) or not np.all(np.isfinite(X_np)):
            raise ValueError("Inputs must not contain NaN or inf.")
        return y_np.astype(float, copy=False), X_np.astype(float, copy=False)

    def _fit_initializer(self, y: np.ndarray, X: np.ndarray) -> object:
        if self.init_model is not None:
            initializer = self.init_model
            try:
                self._extract_baseline_sequences(y, X, initializer)
            except Exception:
                if not hasattr(initializer, "fit"):
                    raise ValueError(
                        "init_model must be compatible with predict_components and predict_variance."
                    )
                initializer = initializer.fit(y, X)
                self._extract_baseline_sequences(y, X, initializer)
            if isinstance(initializer, _ConstantPGARCHInitializer):
                self.init_method_ = "intercept_only_pgarch"
            elif PGARCHLinearModel is not None and isinstance(initializer, PGARCHLinearModel):
                self.init_method_ = "linear_pgarch"
            else:
                self.init_method_ = self.init_method
            return initializer

        if self.init_method == "linear_pgarch":
            if PGARCHLinearModel is None:
                warnings.warn(
                    "PGARCHLinearModel is unavailable. Falling back to intercept-only PGARCH initialization.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                initializer = self._fit_intercept_only_initializer(y)
                self.init_method_ = "intercept_only_pgarch"
                return initializer

            initializer = PGARCHLinearModel(
                loss=self.loss,
                random_state=self.random_state,
                channel_features=self.channel_features,
            )
            initializer.fit(y, X)
            self.init_method_ = "linear_pgarch"
            return initializer

        initializer = self._fit_intercept_only_initializer(y)
        self.init_method_ = "intercept_only_pgarch"
        return initializer

    def _extract_baseline_sequences(
        self,
        y: np.ndarray,
        X: np.ndarray,
        initializer: object,
    ) -> dict[str, np.ndarray]:
        if not hasattr(initializer, "predict_components") or not hasattr(
            initializer, "predict_variance"
        ):
            raise ValueError(
                "Initializer must implement predict_components(X) and predict_variance(y, X)."
            )

        components = initializer.predict_components(X)
        h = np.asarray(initializer.predict_variance(y, X), dtype=float)
        mu = np.asarray(components["mu"], dtype=float)
        phi = np.asarray(components["phi"], dtype=float)
        g = np.asarray(components["g"], dtype=float)
        T = len(np.asarray(y, dtype=float))

        for name, arr in {"mu": mu, "phi": phi, "g": g, "h": h}.items():
            if arr.shape != (T,):
                raise ValueError(f"Initializer returned {name} with shape {arr.shape}, expected {(T,)}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Initializer returned non-finite values for {name}.")

        if np.any(mu <= 0.0):
            raise ValueError("Baseline mu must be strictly positive.")
        if np.any((phi <= 0.0) | (phi >= 1.0)):
            raise ValueError("Baseline phi must lie in (0, 1).")
        if np.any((g <= 0.0) | (g >= 1.0)):
            raise ValueError("Baseline g must lie in (0, 1).")

        h = np.maximum(h, self.h_min)
        return {
            "mu": mu,
            "phi": phi,
            "g": g,
            "h": h,
            "a0": self._inverse_link_mu(mu),
            "b0": self._inverse_link_phi(phi),
            "c0": self._inverse_link_g(g),
        }

    def _initialize_raw_scores(
        self,
        baseline: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(baseline["a0"], dtype=float).copy(),
            np.asarray(baseline["b0"], dtype=float).copy(),
            np.asarray(baseline["c0"], dtype=float).copy(),
        )

    def _predict_raw_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_np_raw, cols = _coerce_X(X)
        X_np = self._validate_feature_matrix(X_np_raw)
        self._check_schema(cols, X_np)
        if self.initializer_ is None:
            raise ValueError("Initializer is not available.")

        components = self.initializer_.predict_components(X)
        a = self._inverse_link_mu(np.asarray(components["mu"], dtype=float))
        b = self._inverse_link_phi(np.asarray(components["phi"], dtype=float))
        c = self._inverse_link_g(np.asarray(components["g"], dtype=float))

        if self.booster_mu_ is not None:
            a = a + self._predict_margin(self.booster_mu_, X_np, channel="mu")
        if self.booster_phi_ is not None:
            b = b + self._predict_margin(self.booster_phi_, X_np, channel="phi")
        if self.booster_g_ is not None:
            c = c + self._predict_margin(self.booster_g_, X_np, channel="g")
        return a, b, c

    def _link_mu(self, a: np.ndarray) -> np.ndarray:
        a_np = np.asarray(a, dtype=float)
        return self.mu_min + np.asarray(_softplus(a_np), dtype=float)

    def _link_phi(self, b: np.ndarray) -> np.ndarray:
        b_np = np.asarray(b, dtype=float)
        sigma = np.asarray(_sigmoid(b_np), dtype=float)
        return self.phi_min + (self.phi_max - self.phi_min) * sigma

    def _link_g(self, c: np.ndarray) -> np.ndarray:
        c_np = np.asarray(c, dtype=float)
        sigma = np.asarray(_sigmoid(c_np), dtype=float)
        return self.g_min + (self.g_max - self.g_min) * sigma

    def _link_mu_prime(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(_sigmoid(np.asarray(a, dtype=float)), dtype=float)

    def _link_phi_prime(self, b: np.ndarray) -> np.ndarray:
        b_np = np.asarray(b, dtype=float)
        sigma = np.asarray(_sigmoid(b_np), dtype=float)
        return (self.phi_max - self.phi_min) * sigma * (1.0 - sigma)

    def _link_g_prime(self, c: np.ndarray) -> np.ndarray:
        c_np = np.asarray(c, dtype=float)
        sigma = np.asarray(_sigmoid(c_np), dtype=float)
        return (self.g_max - self.g_min) * sigma * (1.0 - sigma)

    def _inverse_link_mu(self, mu: np.ndarray) -> np.ndarray:
        mu_np = np.asarray(mu, dtype=float)
        return np.asarray(_softplus_inverse(mu_np - self.mu_min), dtype=float)

    def _inverse_link_phi(self, phi: np.ndarray) -> np.ndarray:
        phi_np = np.asarray(phi, dtype=float)
        scaled = (phi_np - self.phi_min) / (self.phi_max - self.phi_min)
        return np.asarray(_logit(scaled), dtype=float)

    def _inverse_link_g(self, g: np.ndarray) -> np.ndarray:
        g_np = np.asarray(g, dtype=float)
        scaled = (g_np - self.g_min) / (self.g_max - self.g_min)
        return np.asarray(_logit(scaled), dtype=float)

    def _forward_recursion_with_scores(
        self,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        *,
        h0: float,
    ) -> dict[str, np.ndarray]:
        y_np = np.asarray(y, dtype=float)
        a_np = np.asarray(a, dtype=float)
        b_np = np.asarray(b, dtype=float)
        c_np = np.asarray(c, dtype=float)
        T = len(y_np)

        for name, arr in {"a": a_np, "b": b_np, "c": c_np}.items():
            if arr.shape != (T,):
                raise ValueError(f"{name} must have shape {(T,)}, got {arr.shape}.")

        mu = self._link_mu(a_np)
        phi = self._link_phi(b_np)
        g = self._link_g(c_np)
        mu_prime = self._link_mu_prime(a_np)
        phi_prime = self._link_phi_prime(b_np)
        g_prime = self._link_g_prime(c_np)

        h = np.empty(T, dtype=float)
        h[0] = max(float(h0), self.h_min)
        q = np.zeros(T, dtype=float)
        rho = np.zeros(T, dtype=float)
        local_impulse_mu = np.zeros(T, dtype=float)
        local_impulse_phi = np.zeros(T, dtype=float)
        local_impulse_g = np.zeros(T, dtype=float)
        floor_active = np.zeros(T, dtype=bool)

        for row in range(T - 1):
            q[row] = g[row] * y_np[row] + (1.0 - g[row]) * h[row]
            h_raw_next = (1.0 - phi[row]) * mu[row] + phi[row] * q[row]
            if h_raw_next <= self.h_min:
                h[row + 1] = self.h_min
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
        return {
            "a": a_np,
            "b": b_np,
            "c": c_np,
            "mu": mu,
            "phi": phi,
            "g": g,
            "mu_prime": mu_prime,
            "phi_prime": phi_prime,
            "g_prime": g_prime,
            "h": h,
            "q": q,
            "rho": rho,
            "local_impulse_mu": local_impulse_mu,
            "local_impulse_phi": local_impulse_phi,
            "local_impulse_g": local_impulse_g,
            "floor_active": floor_active,
        }

    def _loss_from_state(self, y: np.ndarray, h: np.ndarray) -> float:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        y_eff = y_np[1:]
        h_eff = h_safe[1:]
        if self._optimization_loss_name == "mse":
            return float(np.mean((y_eff - h_eff) ** 2))
        return float(np.mean(np.log(h_eff) + y_eff / h_eff))

    def _loss_grad_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        grad = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self._optimization_loss_name == "mse":
            grad[1:] = 2.0 * (h_safe[1:] - y_np[1:]) / n_eff
            return grad
        grad[1:] = (1.0 / h_safe[1:] - y_np[1:] / (h_safe[1:] ** 2)) / n_eff
        return grad

    def _loss_hess_weight_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        weight = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self._optimization_loss_name == "mse":
            weight[1:] = 2.0 / n_eff
            return weight
        weight[1:] = 1.0 / (n_eff * (h_safe[1:] ** 2))
        weight[1:] = np.clip(weight[1:], 1e-12, 1e12)
        return weight

    def _backward_adjoint(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> np.ndarray:
        u = self._loss_grad_wrt_h(y, state["h"])
        lam = np.zeros_like(u)
        lam[-1] = u[-1]
        for row in range(len(u) - 2, -1, -1):
            lam[row] = u[row] + state["rho"][row] * lam[row + 1]
        return lam

    def _rowwise_grad_hess_mu(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rowwise_grad_hess_from_impulse(y, state, "local_impulse_mu")

    def _rowwise_grad_hess_phi(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rowwise_grad_hess_from_impulse(y, state, "local_impulse_phi")

    def _rowwise_grad_hess_g(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rowwise_grad_hess_from_impulse(y, state, "local_impulse_g")

    def _channel_col_indices(self, channel: str) -> np.ndarray | None:
        """Return the column indices for *channel*, or ``None`` when all
        columns are shared (no ``channel_features`` specified)."""
        if self.channel_features is None:
            return None
        return np.asarray(self.channel_features[channel], dtype=int)

    def _make_dmatrix(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        channel: str | None = None,
    ) -> xgb.DMatrix:
        self._require_xgboost()
        X_np, cols = _coerce_X(X)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}.")
        if cols is not None and self.feature_names_ is not None and list(cols) != list(self.feature_names_):
            raise ValueError("Feature schema mismatch.")

        ch_idx = self._channel_col_indices(channel) if channel is not None else None
        if ch_idx is not None:
            X_np = X_np[:, ch_idx]
            feature_names = (
                [self.feature_names_[i] for i in ch_idx]
                if self.feature_names_ is not None
                else ([cols[i] for i in ch_idx] if cols is not None else None)
            )
        else:
            feature_names = self.feature_names_ if self.feature_names_ is not None else cols

        label = None if y is None else np.asarray(y, dtype=float)
        return xgb.DMatrix(X_np, label=label, feature_names=feature_names)

    def _fit_channel_update(
        self,
        channel: str,
        X: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        h0: float,
    ) -> tuple[np.ndarray, Any]:
        if channel not in {"mu", "phi", "g"}:
            raise ValueError("channel must be one of {'mu', 'phi', 'g'}")
        if self.a0_ is None or self.b0_ is None or self.c0_ is None:
            raise ValueError("Baseline raw scores are not initialized.")

        base_map = {"mu": self.a0_, "phi": self.b0_, "g": self.c0_}
        booster_map = {
            "mu": self.booster_mu_,
            "phi": self.booster_phi_,
            "g": self.booster_g_,
        }
        grad_fn_map = {
            "mu": self._rowwise_grad_hess_mu,
            "phi": self._rowwise_grad_hess_phi,
            "g": self._rowwise_grad_hess_g,
        }

        dtrain = self._make_dmatrix(X, y, channel=channel)
        dtrain.set_base_margin(np.asarray(base_map[channel], dtype=float))

        def objective(preds: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
            del dmatrix
            preds_np = np.asarray(preds, dtype=float)
            if channel == "mu":
                state = self._forward_recursion_with_scores(y, preds_np, b, c, h0=h0)
            elif channel == "phi":
                state = self._forward_recursion_with_scores(y, a, preds_np, c, h0=h0)
            else:
                state = self._forward_recursion_with_scores(y, a, b, preds_np, h0=h0)
            return grad_fn_map[channel](y, state)

        train_kwargs: dict[str, Any] = {
            "params": self._xgb_params(),
            "dtrain": dtrain,
            "num_boost_round": self.trees_per_channel_per_round,
            "obj": objective,
            "xgb_model": booster_map[channel],
        }
        if self.verbosity > 0:
            train_kwargs["evals"] = [(dtrain, "train")]
            train_kwargs["verbose_eval"] = True

            def channel_metric(preds: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
                del dmatrix
                preds_np = np.asarray(preds, dtype=float)
                if channel == "mu":
                    state = self._forward_recursion_with_scores(y, preds_np, b, c, h0=h0)
                elif channel == "phi":
                    state = self._forward_recursion_with_scores(y, a, preds_np, c, h0=h0)
                else:
                    state = self._forward_recursion_with_scores(y, a, b, preds_np, h0=h0)
                metric_name = self.eval_metric or self._optimization_loss_name
                return metric_name, self._score_from_state(y, state["h"], metric=metric_name)

            if "custom_metric" in inspect.signature(xgb.train).parameters:
                train_kwargs["custom_metric"] = channel_metric
            else:
                train_kwargs["feval"] = channel_metric

        booster = xgb.train(**train_kwargs)
        updated_scores = np.asarray(base_map[channel], dtype=float) + self._predict_margin(booster, X, channel=channel)
        return updated_scores, booster

    @property
    def _optimization_loss_name(self) -> str:
        return "mse" if self.loss == "rmse" else self.loss

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.initializer_ is None:
            raise ValueError("Model is not fitted.")

    def _validate_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if not np.all(np.isfinite(X_np)):
            raise ValueError("X must not contain NaN or inf.")
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}.")
        return X_np.astype(float, copy=False)

    def _check_schema(self, cols: list[str] | None, X_np: np.ndarray) -> None:
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}.")
        if cols is not None and self.feature_names_ is not None and list(cols) != list(self.feature_names_):
            raise ValueError("Feature schema mismatch.")

    def _require_xgboost(self) -> None:
        if xgb is None:
            raise ImportError("xgboost is required to use XGBPGARCHModel.")

    def _fit_intercept_only_initializer(self, y: np.ndarray) -> _ConstantPGARCHInitializer:
        y_np = np.asarray(y, dtype=float)
        if PGARCHLinearModel is not None:
            zero_features = np.empty((len(y_np), 0), dtype=float)
            model = PGARCHLinearModel(
                loss=self.loss,
                standardize_features=False,
                random_state=self.random_state,
            )
            model.fit(y_np, zero_features)
            components = model.predict_components(zero_features)
            h = model.predict_variance(y_np, zero_features)
            return _ConstantPGARCHInitializer(
                mu_=float(components["mu"][0]),
                phi_=float(components["phi"][0]),
                g_=float(components["g"][0]),
                h0_=float(h[0]),
                h_min=self.h_min,
            )

        h0 = max(float(y_np[0]), self.h_min)
        theta0 = np.array(
            [
                float(_softplus_inverse(max(h0 - self.mu_min, 1e-12))),
                float(_logit((0.98 - self.phi_min) / (self.phi_max - self.phi_min))),
                float(_logit((0.05 - self.g_min) / (self.g_max - self.g_min))),
            ],
            dtype=float,
        )

        def unpack(theta: np.ndarray) -> tuple[float, float, float]:
            mu = self.mu_min + float(_softplus(theta[0]))
            phi = self.phi_min + (self.phi_max - self.phi_min) * float(_sigmoid(theta[1]))
            g = self.g_min + (self.g_max - self.g_min) * float(_sigmoid(theta[2]))
            return mu, phi, g

        def objective(theta: np.ndarray) -> float:
            mu, phi, g = unpack(theta)
            h = np.empty(len(y_np), dtype=float)
            h[0] = h0
            for row in range(len(y_np) - 1):
                q_row = g * y_np[row] + (1.0 - g) * h[row]
                h[row + 1] = max((1.0 - phi) * mu + phi * q_row, self.h_min)
            return self._loss_from_state(y_np, h)

        result = minimize(objective, theta0, method="L-BFGS-B")
        if not bool(result.success):
            raise RuntimeError(f"Intercept-only PGARCH initialization failed: {result.message}")

        mu, phi, g = unpack(np.asarray(result.x, dtype=float))
        return _ConstantPGARCHInitializer(
            mu_=mu,
            phi_=phi,
            g_=g,
            h0_=h0,
            h_min=self.h_min,
        )

    def _rowwise_grad_hess_from_impulse(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
        impulse_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=float)
        T = len(y_np)
        grad = np.zeros(T, dtype=float)
        hess = np.zeros(T, dtype=float)

        lam = self._backward_adjoint(y_np, state)
        local_impulse = np.asarray(state[impulse_key], dtype=float)
        grad[:-1] = lam[1:] * local_impulse[:-1]

        weight_h = self._loss_hess_weight_wrt_h(y_np, state["h"])
        future_weight = np.zeros(T, dtype=float)
        future_weight[-1] = weight_h[-1]
        for row in range(T - 2, 0, -1):
            future_weight[row] = weight_h[row] + (state["rho"][row] ** 2) * future_weight[row + 1]

        diag = (local_impulse[:-1] ** 2) * future_weight[1:]
        diag = np.where(np.isfinite(diag), diag, 0.0)
        hess[:-1] = np.clip(diag, 0.0, 1e12)
        grad[-1] = 0.0
        hess[-1] = 0.0
        return grad, hess

    def _xgb_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "booster": self.booster,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "alpha": self.reg_alpha,
            "lambda": self.reg_lambda,
            "gamma": self.gamma,
            "verbosity": self.verbosity,
            "objective": "reg:squarederror",
            "base_score": 0.0,
        }
        if self.random_state is not None:
            params["seed"] = int(self.random_state)
        return params

    def _predict_margin(self, booster: Any, X: np.ndarray, channel: str | None = None) -> np.ndarray:
        dmatrix = self._make_dmatrix(X, channel=channel)
        return np.asarray(booster.predict(dmatrix, output_margin=True), dtype=float)

    def _score_from_state(self, y: np.ndarray, h: np.ndarray, metric: str) -> float:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        y_eff = y_np[1:]
        h_eff = h_safe[1:]
        if metric == "mse":
            return float(np.mean((y_eff - h_eff) ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean((y_eff - h_eff) ** 2)))
        if metric == "qlike":
            return float(np.mean(np.log(h_eff) + y_eff / h_eff))
        raise ValueError(f"metric must be one of ['mse', 'rmse', 'qlike'], got {metric!r}.")

    def _snapshot_boosters(self) -> dict[str, Any]:
        return {
            "mu": self._clone_booster(self.booster_mu_),
            "phi": self._clone_booster(self.booster_phi_),
            "g": self._clone_booster(self.booster_g_),
        }

    def _restore_boosters(self, snapshot: dict[str, Any]) -> None:
        self.booster_mu_ = snapshot["mu"]
        self.booster_phi_ = snapshot["phi"]
        self.booster_g_ = snapshot["g"]

    def _clone_booster(self, booster: Any) -> Any:
        if booster is None:
            return None
        save_raw_kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(booster.save_raw)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "raw_format" in signature.parameters:
            save_raw_kwargs["raw_format"] = "ubj"
        raw = booster.save_raw(**save_raw_kwargs)
        return xgb.Booster(model_file=raw)
