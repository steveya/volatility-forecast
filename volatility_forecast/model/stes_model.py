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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from .base_model import BaseVolatilityModel

logger = logging.getLogger(__name__)


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
        l2_reg: float = 0.0,
        keep_result: bool = False,
        random_state: int | None = None,
    ):
        self.params = params
        self.keep_result = keep_result
        self.random_state = random_state
        self.l2_reg = float(l2_reg)

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
        """Fit the STES model by non-linear least-squares optimisation.

        Uses ``scipy.optimize.least_squares`` with the analytical Jacobian
        (see :meth:`_jacobian`) for efficient gradient computation.

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

            # Update self.l2_reg with the winner
            if "l2_reg" in best_params:
                self.l2_reg = float(best_params["l2_reg"])

        rng = np.random.default_rng(self.random_state)
        initial_params = rng.normal(0, 1, size=X_np.shape[1])

        # --- Construct Penalty Vector ---
        penalty_vec = None
        if self.l2_reg > 0.0:
            # 1. Mask: Allow 'const' to evolve freely (avoid 0.5 bias)
            mask = np.ones(X_np.shape[1], dtype=float)
            if self.feature_names_ and "const" in self.feature_names_:
                c_idx = self.feature_names_.index("const")
                mask[c_idx] = 0.0

            # 2. Scale: Match magnitude of y (avoid 10^8 scale mismatch)
            # Scale factor S = sqrt(sum(y^2)) / sqrt(N_features) approx?
            # Actually, just matching the Frobenius norm of Y is a good heuristic
            # to make lambda=1.0 meaningful relative to the total error sum.
            scale = np.linalg.norm(y_np[start_index:end_index])

            penalty_vec = np.sqrt(self.l2_reg) * scale * mask

        # ----- Optimise with analytical Jacobian -----
        common_args = (r_np, X_np, y_np, start_index, end_index, penalty_vec)
        result = least_squares(
            self._objective,
            x0=initial_params,
            jac=self._jacobian,
            args=common_args,
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
        """Run time-series cross-validation to tune L2 regularisation.

        Uses ``TimeSeriesSplit`` from scikit-learn to create expanding-window
        folds.  For each candidate ``l2_reg`` in *param_grid*, fits the STES
        model on each training fold (using the analytical Jacobian) and
        evaluates MSE on the validation fold.

        Parameters
        ----------
        X_np : ndarray, shape (T, p)
        y_np : ndarray, shape (T,)
        returns : ndarray, shape (T,)
        param_grid : iterable of dict
            Each dict must contain ``"l2_reg"`` (float).
        n_splits : int, default 5

        Returns
        -------
        results : list of (float, dict)
            Sorted by ascending MSE.  Each entry is
            ``(mean_mse, params_dict)``.
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
                scale_tr = np.linalg.norm(y_tr)
                p_vec_tr = np.sqrt(l2) * scale_tr * mask if l2 > 0 else None

                # Fit (with analytical Jacobian)
                rng = np.random.default_rng(self.random_state)
                x0 = rng.normal(0, 1, size=X_tr.shape[1])
                res = least_squares(
                    self._objective,
                    x0=x0,
                    jac=self._jacobian,
                    args=(r_tr, X_tr, y_tr, 0, len(X_tr), p_vec_tr),
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

                mse = np.mean((y_va - yhat_va) ** 2)
                fold_scores.append(mse)

            avg_score = np.mean(fold_scores)
            results.append((avg_score, params_cand))

        results.sort(key=lambda x: x[0])
        return results

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
