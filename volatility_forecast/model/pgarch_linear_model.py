"""Linear Predictive GARCH (PGARCH) volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    pos = z_arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    if np.isscalar(z):
        return float(out)
    return out


def _softplus(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=float)
    out = np.log1p(np.exp(-np.abs(z_arr))) + np.maximum(z_arr, 0.0)
    if np.isscalar(z):
        return float(out)
    return out


def _logit(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    p_arr = np.asarray(p, dtype=float)
    p_arr = np.clip(p_arr, eps, 1.0 - eps)
    out = np.log(p_arr) - np.log1p(-p_arr)
    if np.isscalar(p):
        return float(out)
    return out


def _softplus_inverse(y: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    y_arr = np.asarray(y, dtype=float)
    y_arr = np.maximum(y_arr, eps)
    out = y_arr + np.log(-np.expm1(-y_arr))
    if np.isscalar(y):
        return float(out)
    return out


class PGARCHLinearModel:
    """Linear Predictive GARCH model with analytic derivatives.

    Notes
    -----
    ``predict_components(X)`` returns component values associated with each
    feature row ``X[t]``. The recursive variance path instead uses ``X[t-1]``
    together with ``y[t-1]`` to forecast ``h[t]``.

    The initial state ``h[0]`` is treated as a fixed causal warm start using
    ``max(y[0], h_min)``. Losses, gradients, Hessians, and scores therefore
    exclude ``t = 0`` and are computed only on ``t = 1, ..., T-1``.
    """

    def __init__(
        self,
        loss: str = "qlike",
        lambda_mu: float = 0.0,
        lambda_phi: float = 0.0,
        lambda_g: float = 0.0,
        mu_min: float = 1e-12,
        phi_min: float = 1e-4,
        phi_max: float = 1.0 - 1e-4,
        g_min: float = 1e-4,
        g_max: float = 1.0 - 1e-4,
        h_min: float = 1e-12,
        optimizer: str = "lbfgs",
        max_iter: int = 500,
        tol: float = 1e-8,
        n_restarts: int = 1,
        standardize_features: bool = True,
        random_state: int | None = None,
    ) -> None:
        valid_losses = {"mse", "rmse", "qlike"}
        if loss not in valid_losses:
            raise ValueError(f"loss must be one of {sorted(valid_losses)}")
        if optimizer != "lbfgs":
            raise ValueError("optimizer must be 'lbfgs'")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and > 0")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if mu_min <= 0.0 or h_min <= 0.0:
            raise ValueError("mu_min and h_min must be > 0")
        if not (0.0 < phi_min < phi_max < 1.0):
            raise ValueError("Require 0 < phi_min < phi_max < 1")
        if not (0.0 < g_min < g_max < 1.0):
            raise ValueError("Require 0 < g_min < g_max < 1")
        for name, value in {
            "lambda_mu": lambda_mu,
            "lambda_phi": lambda_phi,
            "lambda_g": lambda_g,
        }.items():
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and >= 0")

        self.loss = loss
        self.lambda_mu = float(lambda_mu)
        self.lambda_phi = float(lambda_phi)
        self.lambda_g = float(lambda_g)
        self.mu_min = float(mu_min)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.h_min = float(h_min)
        self.optimizer = optimizer
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)
        self.standardize_features = bool(standardize_features)
        self.random_state = random_state

        self.coef_mu_: np.ndarray | None = None
        self.coef_phi_: np.ndarray | None = None
        self.coef_g_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.optimization_result_: Any = None
        self.train_loss_: float | None = None

    def fit(self, y: np.ndarray, X: np.ndarray) -> "PGARCHLinearModel":
        """Fit the linear PGARCH model.

        Parameters
        ----------
        y : ndarray of shape (T,)
            Positive target series, typically squared returns.
        X : ndarray of shape (T, d)
            Exogenous feature matrix aligned so that ``X[t-1]`` forecasts
            ``h[t]``.
            The fitted objective excludes ``t = 0`` because ``h[0]`` is a fixed
            causal warm start.
        """
        y_np, X_np = self._validate_inputs(y, X)
        self.n_features_in_ = int(X_np.shape[1])
        X_std = self._standardize_fit(X_np)
        X_aug = self._add_intercept(X_std)

        base_theta = self._initialize_params(y_np, X_aug)
        rng = np.random.default_rng(self.random_state)
        best_result: OptimizeResult | None = None
        failure_messages: list[str] = []

        for restart in range(self.n_restarts):
            theta0 = base_theta.copy()
            if restart > 0:
                theta0 += rng.normal(0.0, 0.1, size=theta0.shape)

            result = minimize(
                fun=self._objective_and_gradient,
                x0=theta0,
                args=(y_np, X_aug),
                jac=True,
                method="L-BFGS-B",
                tol=self.tol,
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tol,
                    "gtol": self.tol,
                },
            )

            if bool(result.success) and np.isfinite(result.fun):
                if best_result is None or float(result.fun) < float(best_result.fun):
                    best_result = result
            else:
                failure_messages.append(f"restart={restart}: {result.message}")

        if best_result is None:
            detail = "; ".join(failure_messages) if failure_messages else "no details"
            raise RuntimeError(f"PGARCHLinearModel optimization failed: {detail}")

        self.theta_ = np.asarray(best_result.x, dtype=float)
        self.coef_mu_, self.coef_phi_, self.coef_g_ = self._unpack_params(self.theta_)
        self.optimization_result_ = best_result
        self.is_fitted_ = True

        state = self._forward_recursion(self.theta_, y_np, X_aug)
        self.train_loss_ = self._loss_from_state(y_np, state["h"])
        return self

    def predict_variance(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return the recursive variance path using the provided lagged target path."""
        self._check_is_fitted()
        y_np, X_np = self._validate_inputs(y, X)
        X_aug = self._add_intercept(self._standardize_transform(X_np))
        state = self._forward_recursion(self.theta_, y_np, X_aug)
        return state["h"]

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``mu``, ``phi``, and ``g`` for each feature row ``X[t]``."""
        self._check_is_fitted()
        X_np = self._validate_feature_matrix(X)
        X_aug = self._add_intercept(self._standardize_transform(X_np))
        mu, phi, g = self._components_from_feature_rows(X_aug)
        return {"mu": mu, "phi": phi, "g": g}

    def implied_garch_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return implied ``omega``, ``alpha``, and ``beta`` for each feature row."""
        comps = self.predict_components(X)
        mu = comps["mu"]
        phi = comps["phi"]
        g = comps["g"]
        return {
            "omega": (1.0 - phi) * mu,
            "alpha": phi * g,
            "beta": phi * (1.0 - g),
        }

    def score(self, y: np.ndarray, X: np.ndarray, metric: str = "qlike") -> float:
        """Compute a forecast loss on the recursive variance path."""
        valid_metrics = {"qlike", "mse", "rmse"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {sorted(valid_metrics)}")
        y_np, X_np = self._validate_inputs(y, X)
        h = self.predict_variance(y_np, X_np)
        y_eff = y_np[1:]
        h_eff = h[1:]
        if metric == "mse":
            return float(np.mean((y_eff - h_eff) ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean((y_eff - h_eff) ** 2)))
        h_safe = np.maximum(h_eff, self.h_min)
        return float(np.mean(np.log(h_safe) + y_eff / h_safe))

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

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=float)
        if not self.standardize_features:
            self.feature_mean_ = None
            self.feature_scale_ = None
            return X_np.copy()

        self.feature_mean_ = X_np.mean(axis=0)
        scale = X_np.std(axis=0, ddof=0)
        self.feature_scale_ = np.where(scale > 0.0, scale, 1.0)
        return (X_np - self.feature_mean_) / self.feature_scale_

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        X_np = self._validate_feature_matrix(X)
        if not self.standardize_features:
            return X_np.copy()
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError("Feature standardization parameters are not fitted.")
        return (X_np - self.feature_mean_) / self.feature_scale_

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=float)
        intercept = np.ones((X_np.shape[0], 1), dtype=float)
        return np.concatenate([intercept, X_np], axis=1)

    def _pack_params(
        self,
        w_mu: np.ndarray,
        w_phi: np.ndarray,
        w_g: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(w_mu, dtype=float),
                np.asarray(w_phi, dtype=float),
                np.asarray(w_g, dtype=float),
            ]
        )

    def _unpack_params(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_np = np.asarray(theta, dtype=float)
        if theta_np.ndim != 1 or theta_np.size % 3 != 0:
            raise ValueError("theta must be a flat vector with length divisible by 3.")
        block = theta_np.size // 3
        return (
            theta_np[:block].copy(),
            theta_np[block : 2 * block].copy(),
            theta_np[2 * block :].copy(),
        )

    def _initialize_params(self, y: np.ndarray, X_aug: np.ndarray) -> np.ndarray:
        block = int(np.asarray(X_aug, dtype=float).shape[1])
        v0 = max(float(np.mean(y)), self.h_min)
        mu0 = v0
        phi0 = float(np.clip(0.98, self.phi_min + 1e-8, self.phi_max - 1e-8))
        g0 = float(np.clip(0.05, self.g_min + 1e-8, self.g_max - 1e-8))

        a0 = _softplus_inverse(max(mu0 - self.mu_min, 1e-12))
        b0 = _logit((phi0 - self.phi_min) / (self.phi_max - self.phi_min))
        c0 = _logit((g0 - self.g_min) / (self.g_max - self.g_min))

        w_mu = np.zeros(block, dtype=float)
        w_phi = np.zeros(block, dtype=float)
        w_g = np.zeros(block, dtype=float)
        w_mu[0] = float(a0)
        w_phi[0] = float(b0)
        w_g[0] = float(c0)
        return self._pack_params(w_mu, w_phi, w_g)

    def _forward_recursion(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_aug: np.ndarray,
        compute_grad: bool = False,
        compute_hess: bool = False,
    ) -> dict[str, np.ndarray]:
        if compute_hess:
            compute_grad = True

        y_np = np.asarray(y, dtype=float)
        X_aug_np = np.asarray(X_aug, dtype=float)
        T = y_np.shape[0]
        w_mu, w_phi, w_g = self._unpack_params(theta)
        block = X_aug_np.shape[1]
        if len(w_mu) != block or len(w_phi) != block or len(w_g) != block:
            raise ValueError(
                "theta block size must match X_aug.shape[1]. "
                f"Got theta block size {len(w_mu)} and X_aug.shape[1]={block}."
            )
        total_params = 3 * block
        mu_slice = slice(0, block)
        phi_slice = slice(block, 2 * block)
        g_slice = slice(2 * block, 3 * block)
        phi_span = self.phi_max - self.phi_min
        g_span = self.g_max - self.g_min

        h = np.empty(T, dtype=float)
        h[0] = max(float(y_np[0]), self.h_min)

        mu_arr = np.full(T, np.nan, dtype=float)
        phi_arr = np.full(T, np.nan, dtype=float)
        g_arr = np.full(T, np.nan, dtype=float)
        omega_arr = np.full(T, np.nan, dtype=float)
        alpha_arr = np.full(T, np.nan, dtype=float)
        beta_arr = np.full(T, np.nan, dtype=float)

        J = np.zeros((T, total_params), dtype=float) if compute_grad else None
        H = np.zeros((T, total_params, total_params), dtype=float) if compute_hess else None

        for t in range(1, T):
            x = X_aug_np[t - 1]
            outer_x = np.outer(x, x) if compute_hess else None

            a_t = float(np.dot(w_mu, x))
            b_t = float(np.dot(w_phi, x))
            c_t = float(np.dot(w_g, x))

            sig_a = float(_sigmoid(a_t))
            sig_b = float(_sigmoid(b_t))
            sig_c = float(_sigmoid(c_t))

            mu_t = self.mu_min + float(_softplus(a_t))
            phi_t = self.phi_min + phi_span * sig_b
            g_t = self.g_min + g_span * sig_c

            mu_arr[t] = mu_t
            phi_arr[t] = phi_t
            g_arr[t] = g_t
            omega_arr[t] = (1.0 - phi_t) * mu_t
            alpha_arr[t] = phi_t * g_t
            beta_arr[t] = phi_t * (1.0 - g_t)

            y_prev = y_np[t - 1]
            h_prev = h[t - 1]
            q_t = g_t * y_prev + (1.0 - g_t) * h_prev

            if compute_grad:
                J_mu_t = np.zeros(total_params, dtype=float)
                J_phi_t = np.zeros(total_params, dtype=float)
                J_g_t = np.zeros(total_params, dtype=float)

                J_mu_t[mu_slice] = sig_a * x
                J_phi_t[phi_slice] = phi_span * sig_b * (1.0 - sig_b) * x
                J_g_t[g_slice] = g_span * sig_c * (1.0 - sig_c) * x

                J_prev = J[t - 1]

                # q_t = g_t * y_{t-1} + (1-g_t) * h_{t-1}
                J_q_t = (1.0 - g_t) * J_prev + (y_prev - h_prev) * J_g_t

                if compute_hess:
                    H_mu_t = np.zeros((total_params, total_params), dtype=float)
                    H_phi_t = np.zeros((total_params, total_params), dtype=float)
                    H_g_t = np.zeros((total_params, total_params), dtype=float)

                    H_mu_t[mu_slice, mu_slice] = sig_a * (1.0 - sig_a) * outer_x
                    H_phi_t[phi_slice, phi_slice] = (
                        phi_span * sig_b * (1.0 - sig_b) * (1.0 - 2.0 * sig_b) * outer_x
                    )
                    H_g_t[g_slice, g_slice] = (
                        g_span * sig_c * (1.0 - sig_c) * (1.0 - 2.0 * sig_c) * outer_x
                    )

                    H_prev = H[t - 1]

                    # Hessian of q_t. The cross terms come from differentiating
                    # (1-g_t) * J[t-1] and (y_{t-1} - h_{t-1}) * J_g[t].
                    H_q_t = (
                        (1.0 - g_t) * H_prev
                        + (y_prev - h_prev) * H_g_t
                        - np.outer(J_g_t, J_prev)
                        - np.outer(J_prev, J_g_t)
                    )

                # h_t = (1-phi_t) * mu_t + phi_t * q_t
                J_raw_t = (
                    (1.0 - phi_t) * J_mu_t
                    + (q_t - mu_t) * J_phi_t
                    + phi_t * J_q_t
                )

                if compute_hess:
                    H_raw_t = (
                        (1.0 - phi_t) * H_mu_t
                        + (q_t - mu_t) * H_phi_t
                        + phi_t * H_q_t
                        + np.outer(J_phi_t, J_q_t - J_mu_t)
                        + np.outer(J_q_t - J_mu_t, J_phi_t)
                    )

            h_raw_t = (1.0 - phi_t) * mu_t + phi_t * q_t
            if h_raw_t <= self.h_min:
                # The clipped state is locally constant with respect to the
                # model parameters, so both first- and second-order
                # sensitivities are zero through this transition.
                h[t] = self.h_min
                if compute_grad:
                    J[t] = 0.0
                if compute_hess:
                    H[t] = 0.0
            else:
                h[t] = h_raw_t
                if compute_grad:
                    J[t] = J_raw_t
                if compute_hess:
                    H[t] = H_raw_t

        state: dict[str, np.ndarray] = {
            "h": h,
            "mu": mu_arr,
            "phi": phi_arr,
            "g": g_arr,
            "omega": omega_arr,
            "alpha": alpha_arr,
            "beta": beta_arr,
        }
        if compute_grad and J is not None:
            state["J"] = J
        if compute_hess and H is not None:
            state["H"] = H
        return state

    def _objective(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_aug: np.ndarray,
    ) -> float:
        state = self._forward_recursion(theta, y, X_aug, compute_grad=False)
        return self._loss_from_state(y, state["h"]) + self._regularization_objective(theta)

    def _gradient(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_aug: np.ndarray,
    ) -> np.ndarray:
        _, grad = self._objective_and_gradient(theta, y, X_aug)
        return grad

    def _objective_and_gradient(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_aug: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        state = self._forward_recursion(theta, y, X_aug, compute_grad=True)
        h = state["h"]
        J = state["J"]
        loss = self._loss_from_state(y, h)
        grad_h = self._loss_grad_wrt_h(y, h)
        grad = J.T @ grad_h + self._regularization_gradient(theta)
        objective = loss + self._regularization_objective(theta)
        return float(objective), grad

    def _hessian(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_aug: np.ndarray,
    ) -> np.ndarray:
        state = self._forward_recursion(theta, y, X_aug, compute_grad=True, compute_hess=True)
        h = state["h"]
        J = state["J"]
        H_state = state["H"]
        grad_h = self._loss_grad_wrt_h(y, h)
        hess_h = self._loss_hess_wrt_h(y, h)

        hessian = np.einsum("t,tij->ij", grad_h, H_state, optimize=True)
        hessian += np.einsum("t,ti,tj->ij", hess_h, J, J, optimize=True)
        hessian += self._regularization_hessian(theta)
        return 0.5 * (hessian + hessian.T)

    def _loss_from_state(
        self,
        y: np.ndarray,
        h: np.ndarray,
    ) -> float:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        y_eff = y_np[1:]
        h_eff = h_safe[1:]
        if self._optimization_loss_name == "mse":
            return float(np.mean((y_eff - h_eff) ** 2))
        return float(np.mean(np.log(h_eff) + y_eff / h_eff))

    def _loss_grad_wrt_h(
        self,
        y: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        grad = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self._optimization_loss_name == "mse":
            grad[1:] = 2.0 * (h_safe[1:] - y_np[1:]) / n_eff
            return grad
        grad[1:] = (1.0 / h_safe[1:] - y_np[1:] / (h_safe[1:] ** 2)) / n_eff
        return grad

    def _loss_hess_wrt_h(
        self,
        y: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        y_np = np.asarray(y, dtype=float)
        h_safe = np.maximum(np.asarray(h, dtype=float), self.h_min)
        hess = np.zeros_like(h_safe)
        n_eff = float(len(y_np) - 1)
        if self._optimization_loss_name == "mse":
            hess[1:] = 2.0 / n_eff
            return hess
        hess[1:] = (-1.0 / (h_safe[1:] ** 2) + 2.0 * y_np[1:] / (h_safe[1:] ** 3)) / n_eff
        return hess

    @property
    def _optimization_loss_name(self) -> str:
        return "mse" if self.loss == "rmse" else self.loss

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.theta_ is None:
            raise ValueError("Model is not fitted.")

    def _validate_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if not np.all(np.isfinite(X_np)):
            raise ValueError("X must not contain NaN or inf.")
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_np.shape[1]}."
            )
        return X_np.astype(float, copy=False)

    def _components_from_feature_rows(
        self, X_aug: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.theta_ is None:
            raise ValueError("Model is not fitted.")
        w_mu, w_phi, w_g = self._unpack_params(self.theta_)
        a = X_aug @ w_mu
        b = X_aug @ w_phi
        c = X_aug @ w_g
        mu = self.mu_min + np.asarray(_softplus(a), dtype=float)
        phi = self.phi_min + (self.phi_max - self.phi_min) * np.asarray(
            _sigmoid(b), dtype=float
        )
        g = self.g_min + (self.g_max - self.g_min) * np.asarray(_sigmoid(c), dtype=float)
        return mu, phi, g

    def _regularization_objective(self, theta: np.ndarray) -> float:
        w_mu, w_phi, w_g = self._unpack_params(theta)
        return float(
            self.lambda_mu * np.dot(w_mu[1:], w_mu[1:])
            + self.lambda_phi * np.dot(w_phi[1:], w_phi[1:])
            + self.lambda_g * np.dot(w_g[1:], w_g[1:])
        )

    def _regularization_gradient(self, theta: np.ndarray) -> np.ndarray:
        w_mu, w_phi, w_g = self._unpack_params(theta)
        grad_mu = np.zeros_like(w_mu)
        grad_phi = np.zeros_like(w_phi)
        grad_g = np.zeros_like(w_g)
        grad_mu[1:] = 2.0 * self.lambda_mu * w_mu[1:]
        grad_phi[1:] = 2.0 * self.lambda_phi * w_phi[1:]
        grad_g[1:] = 2.0 * self.lambda_g * w_g[1:]
        return self._pack_params(grad_mu, grad_phi, grad_g)

    def _regularization_hessian(self, theta: np.ndarray) -> np.ndarray:
        block = np.asarray(theta, dtype=float).size // 3
        total = 3 * block
        hessian = np.zeros((total, total), dtype=float)
        diag_mu = np.zeros(block, dtype=float)
        diag_phi = np.zeros(block, dtype=float)
        diag_g = np.zeros(block, dtype=float)
        diag_mu[1:] = 2.0 * self.lambda_mu
        diag_phi[1:] = 2.0 * self.lambda_phi
        diag_g[1:] = 2.0 * self.lambda_g
        hessian[:block, :block] = np.diag(diag_mu)
        hessian[block : 2 * block, block : 2 * block] = np.diag(diag_phi)
        hessian[2 * block :, 2 * block :] = np.diag(diag_g)
        return hessian
