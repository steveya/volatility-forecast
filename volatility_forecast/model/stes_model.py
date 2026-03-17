"""Smooth Transition Exponential Smoothing (STES) volatility model.

This module implements the STES model for volatility forecasting, where the
smoothing parameter alpha_t is time-varying and driven by exogenous features
through a logistic link function.

Mathematical formulation
------------------------
The STES model forecasts next-period variance using an exponential smoothing
recursion with adaptive smoothing:

.. math::

    v_{t+1} = \\alpha_t \\, r_t^2 + (1 - \\alpha_t) \\, v_t

where

- :math:`v_t` is the variance state at time *t*,
- :math:`r_t` is the return observed at time *t*,
- :math:`\\alpha_t = \\sigma(X_t^\\top \\beta)` is the time-varying smoothing
  parameter driven by feature vector :math:`X_t` via the logistic sigmoid,
- :math:`\\beta` is the parameter vector to be estimated.

The model is fit by minimising the sum of squared forecast errors:

.. math::

    \\min_{\\beta} \\; \\sum_{t=s}^{T} \\bigl(y_t - v_{t+1}(\\beta)\\bigr)^2

where :math:`y_t = r_{t+1}^2` is the next-day realised squared return (the
forecast target) and *s* is a burn-in index.

Analytical Jacobian
-------------------
``scipy.optimize.least_squares`` accepts an optional ``jac`` callable.  We
supply an **analytical Jacobian** derived via forward-mode adjoint recursion.
Denoting :math:`D_t = \\partial v_t / \\partial \\beta` (a *p*-vector):

.. math::

    D_{t+1} = \\alpha_t (1 - \\alpha_t) \\, X_t \\, (r_t^2 - v_t)
              \\;+\\; (1 - \\alpha_t) \\, D_t

with initial condition :math:`D_0 = 0` (since :math:`v_0 = r_0^2` is constant
with respect to :math:`\\beta`).  This is evaluated in a **single forward pass**
alongside the variance recursion itself.

Complexity comparison (per optimizer iteration):

- Finite-difference Jacobian: :math:`O((p+1) \\cdot T)` sequential Python loop
  evaluations.
- Analytical Jacobian: :math:`O(T \\cdot p)` in a single pass.

For *p* = 4 (baseline features) the difference is negligible, but for *p* ≥ 40
the analytical Jacobian is ~4–10× faster in practice.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import expit

from .base_model import BaseVolatilityModel

logger = logging.getLogger(__name__)

LossName = Literal["mse", "qlike"]


def _schema_hash(cols):
    payload = json.dumps(list(cols), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class STESModel(BaseVolatilityModel):
    """Smooth Transition Exponential Smoothing (STES) volatility model.

    The smoothing parameter ``alpha_t = sigma(X_t @ beta)`` is driven by
    exogenous features through a logistic link.  The model is feature-agnostic:
    any DataFrame or numpy array of predictors can be used.  Feature schema is
    recorded at fit time for safe persistence and inference-time validation.

    Parameters
    ----------
    params : array-like or None
        Pre-set parameter vector beta.  If ``None``, parameters are estimated
        during :meth:`fit`.
    l2_reg : float, default 0.0
        L2 regularisation strength.  The penalty is applied to all parameters
        except the intercept (``const`` column) and is scaled by the norm of
        the target vector to remain unit-agnostic.
    adaptive_weights : array-like or None
        Per-feature penalty multipliers for adaptive LASSO.  When provided,
        ``penalty_vec[j] = sqrt(l2_reg) * scale * mask[j] * adaptive_weights[j]``.
        Use :meth:`compute_adaptive_weights` to derive these from a ridge pilot.
    keep_result : bool, default False
        If ``True``, the full :class:`scipy.optimize.OptimizeResult` is stored
        in ``self.result`` after fitting for diagnostic inspection.
    random_state : int or None
        Seed for the random initial parameter draw.

    Attributes
    ----------
    params : ndarray of shape (p,)
        Fitted parameter vector beta.
    feature_names_ : list of str or None
        Column names recorded at fit time.
    feature_schema_hash_ : str or None
        Deterministic hash of ``feature_names_`` for fast schema checks.
    n_features_ : int or None
        Number of features (columns) used during fitting.
    init_var_ : float or None
        Initial variance state v_0 used during the last fit.
    last_var_ : float or None
        Terminal variance state after the last fit recursion (useful for
        warm-starting subsequent predictions).
    result : OptimizeResult or None
        Full optimizer result (only stored when ``keep_result=True``).
    """

    def __init__(
        self,
        params=None,
        *,
        loss: LossName = "mse",
        l2_reg: float = 0.0,
        gate_entropy_lambda: float = 0.0,
        adaptive_weights=None,
        qlike_epsilon: float = 1e-8,
        keep_result: bool = False,
        random_state: int | None = None,
    ):
        if loss not in {"mse", "qlike"}:
            raise ValueError("loss must be one of: {'mse','qlike'}")
        if not (np.isfinite(qlike_epsilon) and qlike_epsilon > 0.0):
            raise ValueError("qlike_epsilon must be finite and > 0")

        self.params = params
        self.loss: LossName = loss
        self.keep_result = keep_result
        self.random_state = random_state
        self.l2_reg = float(l2_reg)
        self.gate_entropy_lambda = float(gate_entropy_lambda)
        self.qlike_epsilon = float(qlike_epsilon)
        self.adaptive_weights = (
            np.asarray(adaptive_weights, dtype=float)
            if adaptive_weights is not None
            else None
        )

        # Schema metadata (for safe reload + inference).
        self.feature_names_: list[str] | None = None
        self.feature_schema_hash_: str | None = None
        self.n_features_: int | None = None

        # Optional diagnostic.
        self.result = None

        # Warm-start state carry-over.
        self.init_var_: float | None = None
        self.last_var_: float | None = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _coerce_X(self, X):
        """Convert *X* to numpy, extracting column names if available.

        Returns
        -------
        X_np : ndarray of shape (n, p)
        feature_names : list of str or None
        """
        if hasattr(X, "columns") and hasattr(X, "to_numpy"):
            return X.to_numpy(dtype=float), list(X.columns)
        return np.asarray(X, dtype=float), None

    def _set_schema(self, feature_names, n_features: int):
        """Record feature schema at fit time for safe persistence."""
        self.n_features_ = int(n_features)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_)]
        self.feature_names_ = list(feature_names)
        self.feature_schema_hash_ = _schema_hash(self.feature_names_)

    def _check_schema(self, X_feature_names, X_np):
        """Validate that inference-time features match the training schema."""
        if self.n_features_ is None:
            return  # not fit yet
        if X_np.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: trained {self.n_features_}, got {X_np.shape[1]}"
            )
        if X_feature_names is not None and self.feature_names_ is not None:
            if list(X_feature_names) != list(self.feature_names_):
                raise ValueError(
                    "Feature schema mismatch.\n"
                    f"Trained: {self.feature_names_[:8]}...\n"
                    f"Got:     {list(X_feature_names)[:8]}..."
                )

    # ------------------------------------------------------------------ #
    #  Objective and Jacobian                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_penalty_vec(
        n_features: int,
        feature_names: list[str] | None,
        y_window: np.ndarray,
        *,
        l2_reg: float,
        adaptive_weights: np.ndarray | None,
    ) -> np.ndarray | None:
        """Construct the smooth L2/adaptive-LASSO penalty vector."""
        if l2_reg <= 0.0:
            return None

        mask = np.ones(n_features, dtype=float)
        if feature_names and "const" in feature_names:
            mask[feature_names.index("const")] = 0.0

        scale = np.linalg.norm(y_window)
        penalty_vec = np.sqrt(l2_reg) * scale * mask
        if adaptive_weights is not None:
            penalty_vec = penalty_vec * adaptive_weights
        return penalty_vec

    @staticmethod
    def _objective(
        params,
        returns,
        features,
        y,
        burnin_size,
        os_index,
        penalty_vec,
    ):
        """Compute least-squares residual vector for the STES recursion.

        The residual for timestep *t* is:

        .. math::

            e_t = y_t - v_{t+1}

        where :math:`y_t = r_{t+1}^2` is the realised next-day squared return
        and :math:`v_{t+1} = \\alpha_t r_t^2 + (1-\\alpha_t) v_t` is the STES
        forecast made at time *t*.

        Parameters
        ----------
        params : ndarray, shape (p,)
            Current parameter vector beta.
        returns : ndarray, shape (T,)
            Return series :math:`r_t`.
        features : ndarray, shape (T, p)
            Feature matrix :math:`X`.
        y : ndarray, shape (T,)
            Target vector (next-day squared returns, aligned to time *t*).
        burnin_size : int
            Number of initial timesteps to exclude from the residual vector
            (avoids sensitivity to the :math:`v_0` initialisation).
        os_index : int
            End index (exclusive) for the residual window.
        penalty_vec : ndarray or None
            Pre-computed L2 penalty weights
            ``sqrt(lambda) * scale * mask``.  If not ``None``, penalty
            residuals ``penalty_vec * params`` are appended.

        Returns
        -------
        residuals : ndarray, shape (os_index - burnin_size [+ p],)
        """
        n = features.shape[0]
        alphas = expit(features @ params)
        returns2 = returns**2

        v_state = np.empty(n + 1)
        vhat_next = np.empty(n)
        v_state[0] = returns2[0]

        for t in range(n):
            vhat_next[t] = alphas[t] * returns2[t] + (1.0 - alphas[t]) * v_state[t]
            v_state[t + 1] = vhat_next[t]

        res = (y - vhat_next)[burnin_size:os_index]

        if penalty_vec is not None:
            return np.concatenate([res, penalty_vec * params])
        return res

    @staticmethod
    def _scalar_objective_and_grad(
        params,
        returns,
        features,
        y,
        burnin_size,
        os_index,
        penalty_vec,
        *,
        loss: LossName,
        qlike_epsilon: float,
        gate_entropy_lambda: float = 0.0,
    ):
        """Compute a scalar objective and analytical gradient.

        This is used for losses that do not fit the residual-vector interface
        required by ``scipy.optimize.least_squares``.
        """
        n, p = features.shape
        alphas = expit(features @ params)
        returns2 = returns**2

        # Pre-compute logits (used for entropy gradient) — logit(alpha_t) = features[t] @ params exactly.
        logits_all = features @ params if gate_entropy_lambda != 0.0 else None
        T_window = os_index - burnin_size

        v = float(returns2[0])
        d_v = np.zeros(p, dtype=float)

        objective = 0.0
        grad = np.zeros(p, dtype=float)

        for t in range(n):
            a = float(alphas[t])
            innovation = float(returns2[t] - v)
            d_vhat = a * (1.0 - a) * features[t] * innovation + (1.0 - a) * d_v
            vhat = a * float(returns2[t]) + (1.0 - a) * v

            if burnin_size <= t < os_index:
                if loss == "mse":
                    err = vhat - float(y[t])
                    objective += 0.5 * err * err
                    dloss_dvhat = err
                else:
                    y_safe = max(float(y[t]), qlike_epsilon)
                    vhat_safe = max(float(vhat), qlike_epsilon)
                    ratio = y_safe / vhat_safe
                    objective += ratio - np.log(ratio) - 1.0
                    dloss_dvhat = (vhat_safe - y_safe) / (vhat_safe * vhat_safe)

                grad += dloss_dvhat * d_vhat

                # Gate entropy term: loss += lambda * mean_t [a*log(a) + (1-a)*log(1-a)]
                # This equals -lambda * mean_t H(a_t), penalising saturated gates.
                # Gradient w.r.t. beta: lambda/T * logit(a_t) * a_t*(1-a_t) * X_t
                if gate_entropy_lambda != 0.0:
                    a_c = max(1e-12, min(1.0 - 1e-12, a))
                    objective += gate_entropy_lambda / T_window * (
                        a_c * np.log(a_c) + (1.0 - a_c) * np.log(1.0 - a_c)
                    )
                    logit_t = float(logits_all[t])  # type: ignore[index]
                    grad += (gate_entropy_lambda / T_window * logit_t * a_c * (1.0 - a_c)) * features[t]

            v = vhat
            d_v = d_vhat

        if penalty_vec is not None:
            objective += 0.5 * np.sum((penalty_vec * params) ** 2)
            grad += (penalty_vec**2) * params

        return float(objective), grad

    @staticmethod
    def _jacobian(
        params,
        returns,
        features,
        y,
        burnin_size,
        os_index,
        penalty_vec,
    ):
        """Analytical Jacobian of the residual vector w.r.t. beta.

        Derivation
        ----------
        The residual at time *t* is :math:`e_t = y_t - v_{t+1}(\\beta)`, so

        .. math::

            \\frac{\\partial e_t}{\\partial \\beta_j}
            = -\\frac{\\partial v_{t+1}}{\\partial \\beta_j}

        From the STES recursion
        :math:`v_{t+1} = \\alpha_t r_t^2 + (1 - \\alpha_t) v_t`:

        .. math::

            \\frac{\\partial v_{t+1}}{\\partial \\beta_j}
            = \\frac{\\partial \\alpha_t}{\\partial \\beta_j}
              \\bigl(r_t^2 - v_t\\bigr)
              + (1 - \\alpha_t)\\,
                \\frac{\\partial v_t}{\\partial \\beta_j}

        The sigmoid derivative gives:

        .. math::

            \\frac{\\partial \\alpha_t}{\\partial \\beta_j}
            = \\alpha_t\\,(1 - \\alpha_t)\\, X_{t,j}

        Denoting :math:`D_t = \\partial v_t / \\partial \\beta` (a *p*-vector):

        .. math::

            D_{t+1} = \\alpha_t (1-\\alpha_t)\\, X_t\\, (r_t^2 - v_t)
                      \\;+\\; (1 - \\alpha_t)\\, D_t

        This linear recursion is evaluated in a **single forward pass**
        alongside the variance recursion, with initial condition
        :math:`D_0 = 0` (since :math:`v_0 = r_0^2` does not depend on beta).

        For the L2 penalty residuals
        :math:`p_j = \\texttt{penalty\\_vec}_j \\cdot \\beta_j`, the Jacobian
        rows are simply :math:`\\text{diag}(\\texttt{penalty\\_vec})`.

        Parameters
        ----------
        params : ndarray, shape (p,)
        returns, features, y, burnin_size, os_index, penalty_vec :
            Same as :meth:`_objective`.

        Returns
        -------
        jac : ndarray, shape (n_residuals, p)
            Jacobian matrix :math:`J_{i,j} = \\partial (\\text{residual}_i)
            / \\partial \\beta_j`.
        """
        n, p = features.shape
        alphas = expit(features @ params)
        returns2 = returns**2

        v = returns2[0]  # scalar variance state v_t
        d_v = np.zeros(p)  # d(v_t) / d(beta), shape (p,)

        n_res = os_index - burnin_size
        n_penalty = p if penalty_vec is not None else 0
        jac = np.empty((n_res + n_penalty, p))

        for t in range(n):
            a = alphas[t]
            innovation = returns2[t] - v

            # D_{t+1} = a(1-a) X_t (r_t^2 - v_t)  +  (1-a) D_t
            d_vhat = a * (1.0 - a) * features[t] * innovation + (1.0 - a) * d_v

            # Residual Jacobian row:  d(y_t - v_{t+1})/d(beta) = -d_vhat
            if burnin_size <= t < os_index:
                jac[t - burnin_size] = -d_vhat

            # Advance state
            v = a * returns2[t] + (1.0 - a) * v
            d_v = d_vhat

        # Penalty Jacobian: d(penalty_vec * params)/d(params) = diag(penalty_vec)
        if penalty_vec is not None:
            jac[n_res:] = np.diag(penalty_vec)

        return jac

    def _optimize_params(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        r_np: np.ndarray,
        *,
        start_index: int,
        end_index: int,
        penalty_vec: np.ndarray | None,
        initial_params: np.ndarray | None = None,
        gate_entropy_lambda: float = 0.0,
    ):
        """Optimize STES parameters under the configured loss."""
        if initial_params is None:
            rng = np.random.default_rng(self.random_state)
            initial_params = rng.normal(0, 1, size=X_np.shape[1])

        common_args = (r_np, X_np, y_np, start_index, end_index, penalty_vec)

        if self.loss == "mse":
            return least_squares(
                self._objective,
                x0=initial_params,
                jac=self._jacobian,
                args=common_args,
            )

        _gate_entropy_lambda = gate_entropy_lambda

        def obj_and_grad(beta, *args):
            return self._scalar_objective_and_grad(
                beta,
                *args,
                loss=self.loss,
                qlike_epsilon=self.qlike_epsilon,
                gate_entropy_lambda=_gate_entropy_lambda,
            )

        return minimize(
            obj_and_grad,
            x0=initial_params,
            args=common_args,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500},
        )

    def _validation_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the validation score associated with the configured loss."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if self.loss == "mse":
            err = y_true - y_pred
            return float(np.mean(err * err))

        y_true_safe = np.clip(y_true, self.qlike_epsilon, None)
        y_pred_safe = np.clip(y_pred, self.qlike_epsilon, None)
        ratio = y_true_safe / y_pred_safe
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X,
        y,
        *,
        returns=None,
        start_index=0,
        end_index=None,
        perform_cv: bool = False,
        cv_grid: Optional[Iterable[Dict[str, Any]]] = None,
        cv_splits: int = 5,
    ):
        """Fit the STES model under the configured training loss.

        For ``loss='mse'``, this uses ``scipy.optimize.least_squares`` with
        the analytical Jacobian (see :meth:`_jacobian`). For scalar losses
        such as ``loss='qlike'``, this uses ``scipy.optimize.minimize`` with
        an analytical gradient.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (T, p)
            Feature matrix.  Each row :math:`X_t` contains information
            available at the end of day *t*.
        y : array-like, shape (T,)
            Target vector.  :math:`y_t = r_{t+1}^2` is the next-day squared
            return, aligned to time *t*.
        returns : array-like, shape (T,)
            Return series :math:`r_t`.
        start_index : int, default 0
            Burn-in: residuals before this index are excluded from the loss.
        end_index : int or None
            End of the residual window (exclusive).  Defaults to *T*.
        perform_cv : bool, default False
            If ``True``, run time-series cross-validation to tune ``l2_reg``
            before the final fit.
        cv_grid : iterable of dict, optional
            Parameter grid for CV (each dict must contain ``"l2_reg"``).
        cv_splits : int, default 5
            Number of ``TimeSeriesSplit`` folds for CV.

        Returns
        -------
        self
        """
        if perform_cv:
            if cv_grid is None:
                raise ValueError("If perform_cv=True, you must provide a cv_grid.")
            try:
                import sklearn.model_selection  # noqa
            except ImportError:
                raise ImportError("scikit-learn is required for CV.")

        assert returns is not None, "fit() requires returns=..."
        X_np, cols = self._coerce_X(X)
        y_np = np.asarray(y).reshape(-1)
        r_np = np.asarray(returns).reshape(-1)

        if end_index is None:
            end_index = len(X_np)

        assert len(X_np) == len(y_np) == len(r_np)

        # store feature schema for safe persistence/inference
        self._set_schema(cols, X_np.shape[1])

        # -----------------------------------------------------------------
        # AUTO-TUNING BLOCK
        # -----------------------------------------------------------------
        if perform_cv and cv_grid is not None:
            logger.info(f"Running STES auto-tuning with {cv_splits} splits...")

            # Run CV to find optimal l2_reg
            cv_results = self.run_cv(
                X_np, y_np, returns=r_np, param_grid=cv_grid, n_splits=cv_splits
            )

            # Pick winner
            best_score, best_params = cv_results[0]
            logger.info(f"STES Auto-tuning complete. Best Score: {best_score:.6f}")
            logger.info(f"Best Params: {best_params}")

            # Update self.l2_reg and self.gate_entropy_lambda with the winner
            if "l2_reg" in best_params:
                self.l2_reg = float(best_params["l2_reg"])
            if "gate_entropy_lambda" in best_params:
                self.gate_entropy_lambda = float(best_params["gate_entropy_lambda"])

            # If adaptive LASSO won, compute weights on the full training set
            if "adaptive_gamma" in best_params:
                gamma = float(best_params["adaptive_gamma"])
                # Fit a ridge pilot on the full training data
                rng_pilot = np.random.default_rng(self.random_state)
                x0_pilot = rng_pilot.normal(0, 1, size=X_np.shape[1])
                pvec_pilot = self._build_penalty_vec(
                    X_np.shape[1],
                    self.feature_names_,
                    y_np[start_index:end_index],
                    l2_reg=self.l2_reg,
                    adaptive_weights=None,
                )
                pilot = self._optimize_params(
                    X_np,
                    y_np,
                    r_np,
                    start_index=start_index,
                    end_index=end_index,
                    penalty_vec=pvec_pilot,
                    initial_params=x0_pilot,
                )
                aw = self.compute_adaptive_weights(pilot.x, gamma=gamma)
                if self.feature_names_ and "const" in self.feature_names_:
                    aw[self.feature_names_.index("const")] = 0.0
                self.adaptive_weights = aw

        rng = np.random.default_rng(self.random_state)
        initial_params = rng.normal(0, 1, size=X_np.shape[1])

        penalty_vec = self._build_penalty_vec(
            X_np.shape[1],
            self.feature_names_,
            y_np[start_index:end_index],
            l2_reg=self.l2_reg,
            adaptive_weights=self.adaptive_weights,
        )

        result = self._optimize_params(
            X_np,
            y_np,
            r_np,
            start_index=start_index,
            end_index=end_index,
            penalty_vec=penalty_vec,
            initial_params=initial_params,
            gate_entropy_lambda=self.gate_entropy_lambda,
        )

        self.params = result.x
        self.result = result if self.keep_result else None

        # Store warm-start state (terminal variance) for optional carry-over
        # We compute the recursion on the same contiguous block used in fit residuals.
        # Note: objective recursion starts from returns2[0] on the passed block.
        X_block = X_np[:end_index]
        r_block = r_np[:end_index]
        sigma2_next, _ = self.predict_with_alpha(X_block, returns=r_block)
        if len(sigma2_next) > 0:
            self.init_var_ = float((r_block[0] ** 2))
            self.last_var_ = float(sigma2_next[-1])
        return self

    # ------------------------------------------------------------------ #
    #  Predict                                                             #
    # ------------------------------------------------------------------ #

    def predict(self, X, **kwargs):
        """Forecast next-period variance.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (T, p)
        **kwargs
            ``returns`` (required) and ``init_var`` (optional) are forwarded
            to :meth:`predict_with_alpha`.

        Returns
        -------
        sigma2_next : ndarray, shape (T,)
            Forecast :math:`v_{t+1}` for each *t*.
        """
        sigma2, _ = self.predict_with_alpha(X, **kwargs)
        return sigma2

    def predict_with_alpha(self, X, **kwargs):
        """Forecast next-period variance and return the alpha series.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (T, p)
        returns : array-like, shape (T,)
            Must be passed as keyword argument.
        init_var : float, optional
            Initial variance state :math:`v_0`.  Defaults to :math:`r_0^2`.

        Returns
        -------
        sigma2_next : ndarray, shape (T,)
            One-step-ahead variance forecast.
        alphas : ndarray, shape (T,)
            Time-varying smoothing parameters.
        """
        returns = kwargs.pop("returns", None)
        init_var = kwargs.pop("init_var", None)
        assert returns is not None, "predict() requires returns=..."

        if self.params is None:
            raise ValueError("Model not fitted")

        X_np, cols = self._coerce_X(X)
        self._check_schema(cols, X_np)

        r_np = np.asarray(returns).reshape(-1)
        n = len(r_np)

        # basic safety: X should match n
        if len(X_np) != n:
            raise ValueError(f"Length mismatch: len(X)={len(X_np)} vs len(returns)={n}")

        alphas = expit(np.dot(X_np, self.params))
        returns2 = r_np**2

        # Standard recursion: v_{t+1} = alpha_t r_t^2 + (1-alpha_t) v_t
        v_state = np.zeros(n + 1)
        sigma2_next = np.zeros(n)

        # Allow warm-start: set v_0 from caller if provided, otherwise default to r_0^2
        v_state[0] = float(returns2[0]) if init_var is None else float(init_var)
        for t in range(n):
            sigma2_next[t] = alphas[t] * returns2[t] + (1.0 - alphas[t]) * v_state[t]
            v_state[t + 1] = sigma2_next[t]

        return sigma2_next, alphas

    # ------------------------------------------------------------------ #
    #  Cross-validation                                                    #
    # ------------------------------------------------------------------ #

    def run_cv(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        *,
        returns: np.ndarray,
        param_grid: Iterable[Dict[str, Any]],
        n_splits: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Run time-series cross-validation to tune regularisation.

        Uses ``TimeSeriesSplit`` from scikit-learn to create expanding-window
        folds. For each candidate in *param_grid*, fits the STES model on
        each training fold and evaluates the configured training loss on the
        validation fold.

        Adaptive LASSO is supported: when a grid entry contains
        ``"adaptive_gamma"``, a ridge pilot is first fit within each fold
        and used to compute per-feature penalty weights via
        :meth:`compute_adaptive_weights`.

        Parameters
        ----------
        X_np : ndarray, shape (T, p)
        y_np : ndarray, shape (T,)
        returns : ndarray, shape (T,)
        param_grid : iterable of dict
            Each dict must contain ``"l2_reg"`` (float).  Optionally include
            ``"adaptive_gamma"`` (float) to enable adaptive LASSO.
        n_splits : int, default 5

        Returns
        -------
        results : list of (float, dict)
            Sorted by ascending validation loss. Each entry is
            ``(mean_loss, params_dict)``.
        """
        from sklearn.model_selection import TimeSeriesSplit

        init_var_global = (
            float(np.mean((returns**2)[:500])) if len(returns) > 500 else 1e-8
        )
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        # Determine Const Index for Masking
        mask = np.ones(X_np.shape[1], dtype=float)
        if self.feature_names_ and "const" in self.feature_names_:
            mask[self.feature_names_.index("const")] = 0.0

        for params_cand in param_grid:
            l2 = float(params_cand.get("l2_reg", 0.0))
            adaptive_gamma = params_cand.get("adaptive_gamma", None)
            cand_gate_entropy_lambda = float(params_cand.get("gate_entropy_lambda", 0.0))
            fold_scores = []

            for train_idx, valid_idx in tscv.split(X_np):
                X_tr, X_va = X_np[train_idx], X_np[valid_idx]
                y_tr, y_va = y_np[train_idx], y_np[valid_idx]
                r_tr, r_va = returns[train_idx], returns[valid_idx]

                # Warm Start v0
                v0_val = (
                    float(np.mean((r_tr**2)[-20:]))
                    if len(r_tr) >= 20
                    else init_var_global
                )

                # Penalty Scale (computed on Training fold)
                p_vec_tr = self._build_penalty_vec(
                    X_tr.shape[1],
                    self.feature_names_,
                    y_tr,
                    l2_reg=l2,
                    adaptive_weights=None,
                )

                # Adaptive LASSO: fit ridge pilot, then reweight penalty
                if adaptive_gamma is not None and p_vec_tr is not None:
                    rng_pilot = np.random.default_rng(self.random_state)
                    x0_pilot = rng_pilot.normal(0, 1, size=X_tr.shape[1])
                    pilot = self._optimize_params(
                        X_tr,
                        y_tr,
                        r_tr,
                        start_index=0,
                        end_index=len(X_tr),
                        penalty_vec=p_vec_tr,
                        initial_params=x0_pilot,
                    )
                    aw = self.compute_adaptive_weights(pilot.x, gamma=adaptive_gamma)
                    # Zero out intercept weight
                    if self.feature_names_ and "const" in self.feature_names_:
                        aw[self.feature_names_.index("const")] = 0.0
                    p_vec_tr = p_vec_tr * aw

                # Fit on the fold
                rng = np.random.default_rng(self.random_state)
                x0 = rng.normal(0, 1, size=X_tr.shape[1])
                res = self._optimize_params(
                    X_tr,
                    y_tr,
                    r_tr,
                    start_index=0,
                    end_index=len(X_tr),
                    penalty_vec=p_vec_tr,
                    initial_params=x0,
                    gate_entropy_lambda=cand_gate_entropy_lambda,
                )
                beta_hat = res.x

                # Predict on Validation (Manual recursion using beta_hat)
                alphas_va = expit(np.dot(X_va, beta_hat))
                r2_va = r_va**2
                n_va = len(y_va)

                v_curr = v0_val
                yhat_va = np.zeros(n_va)

                for t in range(n_va):
                    # Forecast v_{t+1} made at t
                    v_next = alphas_va[t] * r2_va[t] + (1.0 - alphas_va[t]) * v_curr
                    yhat_va[t] = v_next
                    v_curr = v_next

                fold_scores.append(self._validation_score(y_va, yhat_va))

            avg_score = np.mean(fold_scores)
            results.append((avg_score, params_cand))

        results.sort(key=lambda x: x[0])
        return results

    # ------------------------------------------------------------------ #
    #  Adaptive LASSO utilities                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_adaptive_weights(
        ridge_params: np.ndarray,
        *,
        gamma: float = 1.0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute adaptive LASSO weights from a ridge pilot estimate.

        Returns per-parameter penalty multipliers
        :math:`w_j = 1 / \\max(|\\hat\\beta_j^{\\text{ridge}}|, \\varepsilon)^\\gamma`.

        Parameters
        ----------
        ridge_params : ndarray, shape (p,)
            Parameter estimates from a ridge (L2-only) fit.
        gamma : float, default 1.0
            Exponent controlling selection aggressiveness.  Higher values
            penalise small coefficients more heavily (e.g., 2.0 is more
            aggressive than 1.0).
        eps : float, default 1e-6
            Floor on ``|beta_j|`` to prevent blow-up near zero.

        Returns
        -------
        weights : ndarray, shape (p,)
            Adaptive penalty multipliers.
        """
        abs_beta = np.abs(np.asarray(ridge_params, dtype=float))
        return 1.0 / np.maximum(abs_beta, eps) ** gamma

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, filename: str, *, format: str = "joblib"):
        """Save model artifact to disk.

        Parameters
        ----------
        filename : str
            Output path.  A ``.joblib`` extension is appended if missing
            (when *format* is ``"joblib"``).
        format : ``"joblib"`` or ``"npy"``
            Serialisation format.  ``"joblib"`` persists the full object;
            ``"npy"`` saves only the parameter vector.
        """
        if format == "joblib":
            joblib.dump(
                self, filename if filename.endswith(".joblib") else filename + ".joblib"
            )
        elif format == "npy":
            np.save(filename + ".npy", self.params)
        else:
            raise ValueError("format must be 'joblib' or 'npy'")

    @classmethod
    def load(cls, filename: str):
        """Load a previously saved model artifact.

        Parameters
        ----------
        filename : str
            Path to the saved model.  Tries ``.joblib`` then ``.npy``
            extensions if the exact path is not found.
        """
        if filename.endswith(".joblib"):
            return joblib.load(filename)
        if filename.endswith(".npy"):
            model = cls()
            model.params = np.load(filename)
            return model
        # Convenience: try common extensions
        p_joblib = filename + ".joblib"
        if os.path.exists(p_joblib):
            return joblib.load(p_joblib)
        p_npy = filename + ".npy"
        if os.path.exists(p_npy):
            model = cls()
            model.params = np.load(p_npy)
            return model
        raise FileNotFoundError(filename)
