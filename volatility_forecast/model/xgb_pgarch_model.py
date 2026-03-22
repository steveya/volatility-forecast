"""XGBoost innovation-share PGARCH model."""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore

try:
    from .pgarch_linear_model import PGARCHLinearModel
except Exception:  # pragma: no cover - optional dependency / branch skew
    PGARCHLinearModel = None  # type: ignore

logger = logging.getLogger(__name__)


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


def _softplus_inverse(y: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    y_arr = np.asarray(y, dtype=float)
    y_arr = np.maximum(y_arr, eps)
    out = y_arr + np.log(-np.expm1(-y_arr))
    if np.isscalar(y):
        return float(out)
    return out


def _logit(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    p_arr = np.asarray(p, dtype=float)
    p_arr = np.clip(p_arr, eps, 1.0 - eps)
    out = np.log(p_arr) - np.log1p(-p_arr)
    if np.isscalar(p):
        return float(out)
    return out


def _coerce_X(X: Any) -> tuple[np.ndarray, list[str] | None]:
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy(dtype=float), dtype=float), list(X.columns)
    return np.asarray(X, dtype=float), None


@dataclass
class _ConstantPGARCHInitializer:
    mu_: float
    phi_: float
    g_: float
    h0_: float
    h_min: float

    def fit(self, y: np.ndarray, X: np.ndarray) -> "_ConstantPGARCHInitializer":
        del y, X
        return self

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X_np, _ = _coerce_X(X)
        n = int(X_np.shape[0])
        return {
            "mu": np.full(n, self.mu_, dtype=float),
            "phi": np.full(n, self.phi_, dtype=float),
            "g": np.full(n, self.g_, dtype=float),
        }

    def predict_variance(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y, dtype=float).reshape(-1)
        X_np, _ = _coerce_X(X)
        if len(y_np) != X_np.shape[0]:
            raise ValueError("y and X must have the same length.")
        h = np.empty(len(y_np), dtype=float)
        h[0] = max(float(self.h0_), self.h_min)
        for t in range(1, len(y_np)):
            q_t = self.g_ * y_np[t - 1] + (1.0 - self.g_) * h[t - 1]
            h[t] = max((1.0 - self.phi_) * self.mu_ + self.phi_ * q_t, self.h_min)
        return h


class XGBGPGARCHModel:
    """XGBoost innovation-share PGARCH model.

    ``predict_components(X)`` returns rowwise ``mu``, ``phi``, and ``g`` values
    associated with each feature row ``X[t]``. The recursive variance path
    instead uses row ``t`` to affect the next-step forecast ``h[t+1]``. Under
    that alignment the terminal row contributes zero gradient and zero Hessian.

    The initial state ``h[0]`` is treated as a fixed causal warm start using
    ``max(y[0], h_min)`` unless an explicit carryover state is supplied.
    Losses, gradients, Hessian weights, and scores therefore exclude ``t = 0``
    and are computed only on ``t = 1, ..., T-1``.
    """

    def __init__(
        self,
        loss: str = "qlike",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        g_min: float = 1e-4,
        g_max: float = 1.0 - 1e-4,
        h_min: float = 1e-12,
        init_method: str = "linear_pgarch",
        init_model: object | None = None,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = None,
        verbosity: int = 0,
        booster: str = "gbtree",
    ) -> None:
        valid_losses = {"mse", "rmse", "qlike"}
        valid_init_methods = {"linear_pgarch", "intercept_only_pgarch"}
        valid_boosters = {"gbtree", "gblinear"}
        if loss not in valid_losses:
            raise ValueError(f"loss must be one of {sorted(valid_losses)}")
        if booster not in valid_boosters:
            raise ValueError(f"booster must be one of {sorted(valid_boosters)}")
        if init_method not in valid_init_methods:
            raise ValueError(f"init_method must be one of {sorted(valid_init_methods)}")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
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
        if not (0.0 < g_min < g_max < 1.0):
            raise ValueError("Require 0 < g_min < g_max < 1")
        if h_min <= 0.0:
            raise ValueError("h_min must be > 0")
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive when provided")
        if eval_metric not in {None, "mse", "qlike"}:
            raise ValueError("eval_metric must be one of {None, 'mse', 'qlike'}")

        self.loss = loss
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.h_min = float(h_min)
        self.init_method = init_method
        self.init_model = init_model
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.verbosity = int(verbosity)
        self.booster = booster

        self.booster_: Any = None
        self.initializer_: object | None = None
        self.init_method_: str | None = None
        self.n_features_in_: int | None = None
        self.is_fitted_: bool = False
        self.train_loss_: float | None = None
        self.best_iteration_: int | None = None
        self.feature_names_: list[str] | None = None
        self.baseline_train_: dict[str, np.ndarray] | None = None

        self._current_h0_: float | None = None
        self._dmatrix_contexts: dict[int, dict[str, Any]] = {}
        self._last_fit_X_: np.ndarray | None = None
        self._alignment_: str = "row t affects h[t+1]; terminal row gradient/hessian are zero"

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBGPGARCHModel":
        """Fit the XGB-g-PGARCH model.

        Parameters
        ----------
        y : ndarray of shape (T,)
            Positive target series.
        X : ndarray of shape (T, d)
            Feature matrix. Row ``t`` affects the next-step forecast ``h[t+1]``.
            The fitted objective excludes ``t = 0`` because ``h[0]`` is a fixed
            causal warm start.
        eval_set : tuple of ``(y_eval, X_eval)``, optional
            Optional validation sequence used for early stopping.
        """
        self._require_xgboost()
        X_np_raw, cols = _coerce_X(X)
        y_np, X_np = self._validate_inputs(y, X_np_raw)

        self.n_features_in_ = int(X_np.shape[1])
        self.feature_names_ = cols
        self._last_fit_X_ = X_np.copy()
        self._dmatrix_contexts = {}

        initializer = self._fit_initializer(y_np, X)
        baseline_train = self._extract_baseline_sequences(y_np, X, initializer)
        c0_train = self._initialize_raw_scores(baseline_train)

        params = self._xgb_params()
        dtrain = self._make_dmatrix(X, y_np)
        dtrain.set_base_margin(c0_train)
        self._register_dmatrix_context(dtrain, y_np, baseline_train)

        evals: list[tuple[Any, str]] = [(dtrain, "train")]
        if eval_set is not None:
            y_eval_raw, X_eval_raw = eval_set
            X_eval_np_raw, eval_cols = _coerce_X(X_eval_raw)
            y_eval, X_eval_np = self._validate_inputs(y_eval_raw, X_eval_np_raw)
            self._check_schema(eval_cols, X_eval_np)
            baseline_eval = self._extract_baseline_sequences(y_eval, X_eval_raw, initializer)
            dvalid = self._make_dmatrix(X_eval_raw, y_eval)
            dvalid.set_base_margin(self._initialize_raw_scores(baseline_eval))
            self._register_dmatrix_context(dvalid, y_eval, baseline_eval)
            evals.append((dvalid, "valid"))

        train_kwargs: dict[str, Any] = {
            "params": params,
            "dtrain": dtrain,
            "num_boost_round": self.n_estimators,
            "obj": self._custom_objective,
            "evals": evals,
            "early_stopping_rounds": self.early_stopping_rounds if eval_set is not None else None,
            "verbose_eval": self.verbosity > 0,
        }
        if "custom_metric" in inspect.signature(xgb.train).parameters:
            train_kwargs["custom_metric"] = self._custom_eval
        else:
            train_kwargs["feval"] = self._custom_eval
        booster = xgb.train(**train_kwargs)

        self.booster_ = booster
        self.initializer_ = initializer
        self.baseline_train_ = baseline_train
        self.is_fitted_ = True
        if eval_set is not None and self.early_stopping_rounds is not None:
            self.best_iteration_ = int(getattr(booster, "best_iteration", 0))
        else:
            self.best_iteration_ = None

        c_train = self._predict_raw_scores(X)
        state_train = self._forward_recursion_with_h0(
            y_np,
            baseline_train["mu"],
            baseline_train["phi"],
            c_train,
            h0=float(baseline_train["h"][0]),
        )
        self.train_loss_ = self._loss_from_state(y_np, state_train["h"])
        return self

    def predict_variance(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return the recursion-aware variance path using the provided lagged target path."""
        self._check_is_fitted()
        X_np_raw, cols = _coerce_X(X)
        y_np, X_np = self._validate_inputs(y, X_np_raw)
        self._check_schema(cols, X_np)

        baseline = self._extract_baseline_sequences(y_np, X, self.initializer_)
        c = self._predict_raw_scores(X)
        state = self._forward_recursion_with_h0(
            y_np,
            baseline["mu"],
            baseline["phi"],
            c,
            h0=float(baseline["h"][0]),
        )
        return state["h"]

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return rowwise ``mu``, ``phi``, and ``g`` for each feature row ``X[t]``."""
        self._check_is_fitted()
        X_np_raw, cols = _coerce_X(X)
        X_np = self._validate_feature_matrix(X_np_raw)
        self._check_schema(cols, X_np)

        components = self.initializer_.predict_components(X)
        c = self._predict_raw_scores(X)
        return {
            "mu": np.asarray(components["mu"], dtype=float),
            "phi": np.asarray(components["phi"], dtype=float),
            "g": self._link_g(c),
        }

    def implied_garch_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return rowwise implied GARCH parameters."""
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
        """Evaluate the recursion-aware forecast path."""
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
            "c0": self._inverse_link_g(g),
        }

    def _initialize_raw_scores(
        self,
        baseline: dict[str, np.ndarray],
    ) -> np.ndarray:
        return np.asarray(baseline["c0"], dtype=float).copy()

    def _predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        X_np_raw, cols = _coerce_X(X)
        X_np = self._validate_feature_matrix(X_np_raw)
        self._check_schema(cols, X_np)
        if self.initializer_ is None:
            raise ValueError("Initializer is not available.")

        baseline_components = self.initializer_.predict_components(X)
        baseline_g = np.asarray(baseline_components["g"], dtype=float)
        c0 = self._inverse_link_g(baseline_g)

        if self.booster_ is None:
            return c0

        dmatrix = self._make_dmatrix(X)
        margin = np.asarray(self._predict_margin_from_dmatrix(dmatrix), dtype=float)
        return c0 + margin

    def _link_g(self, c: np.ndarray) -> np.ndarray:
        c_np = np.asarray(c, dtype=float)
        return self.g_min + (self.g_max - self.g_min) * np.asarray(_sigmoid(c_np), dtype=float)

    def _link_g_prime(self, c: np.ndarray) -> np.ndarray:
        c_np = np.asarray(c, dtype=float)
        sigma = np.asarray(_sigmoid(c_np), dtype=float)
        return (self.g_max - self.g_min) * sigma * (1.0 - sigma)

    def _forward_recursion(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
        c: np.ndarray,
    ) -> dict[str, np.ndarray]:
        h0 = self._current_h0_
        if h0 is None:
            y_np = np.asarray(y, dtype=float)
            h0 = max(float(y_np[0]), self.h_min)
        return self._forward_recursion_with_h0(y, mu, phi, c, h0=h0)

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
        weight[1:] = 1.0 / (h_safe[1:] ** 2)
        weight[1:] = np.clip(weight[1:] / n_eff, 1e-12, 1e12)
        return weight

    def _backward_adjoint(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> np.ndarray:
        u = self._loss_grad_wrt_h(y, state["h"])
        lam = np.zeros_like(u)
        lam[-1] = u[-1]

        # rho[t] is the state propagation from h[t] to h[t+1].
        for t in range(len(u) - 2, -1, -1):
            lam[t] = u[t] + lam[t + 1] * state["rho"][t]
        return lam

    def _rowwise_grad_hess(
        self,
        y: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=float)
        T = len(y_np)
        grad = np.zeros(T, dtype=float)
        hess = np.zeros(T, dtype=float)

        lam = self._backward_adjoint(y_np, state)
        grad[:-1] = lam[1:] * state["local_impulse_g"][:-1]

        weight_h = self._loss_hess_weight_wrt_h(y_np, state["h"])
        future_weight = np.zeros(T, dtype=float)
        future_weight[-1] = weight_h[-1]
        for t in range(T - 2, 0, -1):
            future_weight[t] = weight_h[t] + (state["rho"][t] ** 2) * future_weight[t + 1]

        diag = (state["local_impulse_g"][:-1] ** 2) * future_weight[1:]
        diag = np.where(np.isfinite(diag), diag, 1e-12)
        hess[:-1] = np.clip(diag, 1e-12, 1e12)

        # Terminal row controls no in-sample next-step forecast.
        grad[-1] = 0.0
        hess[-1] = 0.0
        return grad, hess

    def _make_dmatrix(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> xgb.DMatrix:
        self._require_xgboost()
        X_np, cols = _coerce_X(X)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_np.shape[1]}."
            )
        if cols is not None and self.feature_names_ is not None and list(cols) != list(self.feature_names_):
            raise ValueError("Feature schema mismatch.")
        label = None if y is None else np.asarray(y, dtype=float)
        feature_names = self.feature_names_ if self.feature_names_ is not None else cols
        return xgb.DMatrix(X_np, label=label, feature_names=feature_names)

    def _custom_objective(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        context = self._get_dmatrix_context(dtrain)
        baseline = context["baseline"]
        state = self._forward_recursion_with_h0(
            context["y"],
            baseline["mu"],
            baseline["phi"],
            np.asarray(preds, dtype=float),
            h0=float(baseline["h"][0]),
        )
        # Row t affects h[t+1], so the terminal in-sample row has zero
        # gradient and Hessian by construction.
        return self._rowwise_grad_hess(context["y"], state)

    def _custom_eval(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        context = self._get_dmatrix_context(dtrain)
        baseline = context["baseline"]
        state = self._forward_recursion_with_h0(
            context["y"],
            baseline["mu"],
            baseline["phi"],
            np.asarray(preds, dtype=float),
            h0=float(baseline["h"][0]),
        )
        metric_name = "mse" if self._optimization_loss_name == "mse" else "qlike"
        return metric_name, self._loss_from_state(context["y"], state["h"])

    @property
    def _optimization_loss_name(self) -> str:
        return "mse" if self.loss == "rmse" else self.loss

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.booster_ is None or self.initializer_ is None:
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

    def _check_schema(self, cols: list[str] | None, X_np: np.ndarray) -> None:
        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_np.shape[1]}."
            )
        if cols is not None and self.feature_names_ is not None and list(cols) != list(self.feature_names_):
            raise ValueError("Feature schema mismatch.")

    def _require_xgboost(self) -> None:
        if xgb is None:
            raise ImportError("xgboost is required to use XGBGPGARCHModel.")

    def _fit_intercept_only_initializer(self, y: np.ndarray) -> _ConstantPGARCHInitializer:
        y_np = np.asarray(y, dtype=float)
        if PGARCHLinearModel is not None:
            zero_features = np.empty((len(y_np), 0), dtype=float)
            model = PGARCHLinearModel(loss=self.loss, standardize_features=False, random_state=self.random_state)
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

        phi_min = 1e-4
        phi_max = 1.0 - 1e-4
        mu_min = 1e-12
        h0 = max(float(y_np[0]), self.h_min)

        theta0 = np.array(
            [
                float(_softplus_inverse(max(h0 - mu_min, 1e-12))),
                float(_logit((0.98 - phi_min) / (phi_max - phi_min))),
                float(_logit((0.05 - self.g_min) / (self.g_max - self.g_min))),
            ],
            dtype=float,
        )

        def unpack(theta: np.ndarray) -> tuple[float, float, float]:
            mu = mu_min + float(_softplus(theta[0]))
            phi = phi_min + (phi_max - phi_min) * float(_sigmoid(theta[1]))
            g = self.g_min + (self.g_max - self.g_min) * float(_sigmoid(theta[2]))
            return mu, phi, g

        def objective(theta: np.ndarray) -> float:
            mu, phi, g = unpack(theta)
            h = np.empty(len(y_np), dtype=float)
            h[0] = h0
            for t in range(1, len(y_np)):
                q_t = g * y_np[t - 1] + (1.0 - g) * h[t - 1]
                h[t] = max((1.0 - phi) * mu + phi * q_t, self.h_min)
            return self._loss_from_state(y_np, h)

        result = minimize(objective, theta0, method="L-BFGS-B")
        if not bool(result.success):
            raise RuntimeError(f"Intercept-only PGARCH initialization failed: {result.message}")

        mu, phi, g = unpack(np.asarray(result.x, dtype=float))
        return _ConstantPGARCHInitializer(mu_=mu, phi_=phi, g_=g, h0_=h0, h_min=self.h_min)

    def _inverse_link_g(self, g: np.ndarray) -> np.ndarray:
        g_np = np.asarray(g, dtype=float)
        scaled = (g_np - self.g_min) / (self.g_max - self.g_min)
        return np.asarray(_logit(scaled), dtype=float)

    def _forward_recursion_with_h0(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
        c: np.ndarray,
        *,
        h0: float,
    ) -> dict[str, np.ndarray]:
        y_np = np.asarray(y, dtype=float)
        mu_np = np.asarray(mu, dtype=float)
        phi_np = np.asarray(phi, dtype=float)
        c_np = np.asarray(c, dtype=float)
        T = len(y_np)
        for name, arr in {"mu": mu_np, "phi": phi_np, "c": c_np}.items():
            if arr.shape != (T,):
                raise ValueError(f"{name} must have shape {(T,)}, got {arr.shape}.")

        g = self._link_g(c_np)
        g_prime = self._link_g_prime(c_np)
        h = np.empty(T, dtype=float)
        rho = np.zeros(T, dtype=float)
        local_impulse_g = np.zeros(T, dtype=float)
        h[0] = max(float(h0), self.h_min)

        for row in range(T - 1):
            g_row = g[row]
            phi_row = phi_np[row]
            rho_raw = phi_row * (1.0 - g_row)
            local_impulse_raw = phi_row * (y_np[row] - h[row]) * g_prime[row]
            q_next = g_row * y_np[row] + (1.0 - g_row) * h[row]
            h_raw_next = (1.0 - phi_row) * mu_np[row] + phi_row * q_next
            if h_raw_next <= self.h_min:
                # The clipped next-step state is locally constant with respect
                # to the raw score, so both the state propagation coefficient
                # and the local impulse through this transition are zero.
                h[row + 1] = self.h_min
                rho[row] = 0.0
                local_impulse_g[row] = 0.0
            else:
                h[row + 1] = h_raw_next
                rho[row] = rho_raw
                local_impulse_g[row] = local_impulse_raw

        return {
            "h": h,
            "g": g,
            "rho": rho,
            "local_impulse_g": local_impulse_g,
            "c": c_np,
            "g_prime": g_prime,
            "mu": mu_np,
            "phi": phi_np,
        }

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

    def _register_dmatrix_context(
        self,
        dmatrix: xgb.DMatrix,
        y: np.ndarray,
        baseline: dict[str, np.ndarray],
    ) -> None:
        self._dmatrix_contexts[self._dmatrix_key(dmatrix)] = {
            "y": np.asarray(y, dtype=float),
            "baseline": baseline,
        }

    def _get_dmatrix_context(self, dmatrix: xgb.DMatrix) -> dict[str, Any]:
        key = self._dmatrix_key(dmatrix)
        if key not in self._dmatrix_contexts:
            raise KeyError("No context registered for the provided DMatrix.")
        return self._dmatrix_contexts[key]

    @staticmethod
    def _dmatrix_key(dmatrix: xgb.DMatrix) -> int:
        handle = getattr(dmatrix, "handle", None)
        handle_value = getattr(handle, "value", None)
        return int(handle_value) if handle_value is not None else id(dmatrix)

    def _predict_margin_from_dmatrix(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        if self.booster_ is None:
            raise ValueError("Booster is not fitted.")
        predict_kwargs: dict[str, Any] = {"output_margin": True}
        if self.best_iteration_ is not None:
            signature = inspect.signature(self.booster_.predict)
            if "iteration_range" in signature.parameters:
                predict_kwargs["iteration_range"] = (0, self.best_iteration_ + 1)
            elif "ntree_limit" in signature.parameters:
                predict_kwargs["ntree_limit"] = self.best_iteration_ + 1
        return np.asarray(self.booster_.predict(dmatrix, **predict_kwargs), dtype=float)
