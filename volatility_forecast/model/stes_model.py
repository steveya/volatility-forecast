import hashlib
import json
import numpy as np
import joblib
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
from scipy.optimize import least_squares
from scipy.special import expit
from .base_model import BaseVolatilityModel

logger = logging.getLogger(__name__)


def _schema_hash(cols):
    payload = json.dumps(list(cols), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class STESModel(BaseVolatilityModel):
    """
    STES model that remains feature-agnostic (accepts arbitrary features),
    but records feature schema so the model can be safely saved/loaded as an artifact
    and validated at inference time.

    DuckDB integration model:
      - Store the model artifact via joblib (filesystem).
      - Store metadata (model_type, params, feature_schema_hash, feature_names, etc.) in DuckDB.
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

        # schema metadata (for safe reload + inference)
        self.feature_names_ = None
        self.feature_schema_hash_ = None
        self.n_features_ = None

        # optional diagnostic
        self.result = None

        # warm-start diagnostics / state carry-over
        # init_var_ is the initial variance state used during the last fit recursion
        # last_var_ is the terminal variance state after the last fit recursion
        self.init_var_ = None
        self.last_var_ = None

    # ---------- internal helpers ----------
    def _coerce_X(self, X):
        """
        Accepts numpy array or pandas DataFrame.
        Returns (X_np, feature_names or None).
        """
        # pandas DataFrame
        if hasattr(X, "columns") and hasattr(X, "to_numpy"):
            cols = list(X.columns)
            X_np = X.to_numpy(dtype=float)
            return X_np, cols
        # numpy array
        X_np = np.asarray(X, dtype=float)
        return X_np, None

    def _set_schema(self, feature_names, n_features: int):
        self.n_features_ = int(n_features)
        if feature_names is None:
            # If caller passed numpy, we still want a stable schema representation.
            feature_names = [f"x{i}" for i in range(self.n_features_)]
        self.feature_names_ = list(feature_names)
        self.feature_schema_hash_ = _schema_hash(self.feature_names_)

    def _check_schema(self, X_feature_names, X_np):
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

    # ---------- model logic ----------
    def _objective(
        self,
        params,
        returns,
        features,
        y,
        burnin_size,
        os_index,
        penalty_vec,
    ):
        """Least-squares residuals for STES.

        Conventions (important):
        - Features X_t and returns r_t are information available at time t (end of date t).
        - Target y[t] is the *next-day* squared return r_{t+1}^2, shifted to time t.
        - We keep the STES recursion in its standard form:

              v_{t+1} = alpha_t * r_t^2 + (1 - alpha_t) * v_t

          where v_{t+1} is the forecast made at time t about the next period.
        - Therefore, the model-implied forecast aligned to row t is vhat_next[t] = v_{t+1}.
        """
        n, _ = features.shape
        alphas = expit(np.dot(features, params))
        returns2 = returns**2

        # v_t state and next-step forecast v_{t+1} aligned to row t
        v_state = np.zeros(n + 1)
        vhat_next = np.zeros(n)

        # Initialize v_0 (state before observing the first update). We use r_0^2 as a simple anchor.
        # (Callers typically use burn-in / slicing to avoid over-weighting this initialization.)
        v_state[0] = returns2[0]

        for t in range(n):
            vhat_next[t] = alphas[t] * returns2[t] + (1.0 - alphas[t]) * v_state[t]
            v_state[t + 1] = vhat_next[t]

        # Residuals compare y[t]=r_{t+1}^2 against the forecast made at time t: v_{t+1}
        res = (y - vhat_next)[burnin_size:os_index]

        # L2 Regularization (Intercept-Exempt & Scaled)
        if penalty_vec is not None:
            # penalty_vec is pre-calculated as: sqrt(lambda) * scale * mask
            # We just multiply by params to get the residual term
            p_term = penalty_vec * params
            return np.concatenate([res, p_term])

        return res

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

        result = least_squares(
            self._objective,
            x0=initial_params,
            # args passed to _objective
            args=(
                r_np,
                X_np,
                y_np,
                start_index,
                end_index,
                penalty_vec,
            ),
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

    def predict(self, X, **kwargs):
        sigma2, _ = self.predict_with_alpha(X, **kwargs)
        return sigma2

    def predict_with_alpha(self, X, **kwargs):
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

    def run_cv(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        *,
        returns: np.ndarray,
        param_grid: Iterable[Dict[str, Any]],
        n_splits: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Run Time-Series CV to tune L2 Regularization."""
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

                # Fit
                rng = np.random.default_rng(self.random_state)
                x0 = rng.normal(0, 1, size=X_tr.shape[1])
                res = least_squares(
                    self._objective,
                    x0=x0,
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

    # ---------- persistence ----------
    def save(self, filename: str, *, format: str = "joblib"):
        """
        Save model artifact. DuckDB registry will store the artifact path.
        """
        if format == "joblib":
            joblib.dump(
                self, filename if filename.endswith(".joblib") else filename + ".joblib"
            )
        elif format == "npy":
            # backwards compat if you want it
            np.save(filename + ".npy", self.params)
        else:
            raise ValueError("format must be 'joblib' or 'npy'")

    @classmethod
    def load(cls, filename: str):
        """
        Load a previously saved model artifact.
        """
        if filename.endswith(".joblib"):
            return joblib.load(filename)
        if filename.endswith(".npy"):
            model = cls()
            model.params = np.load(filename)
            return model
        # convenience
        p_joblib = filename + ".joblib"
        if os.path.exists(p_joblib):
            return joblib.load(p_joblib)
        p_npy = filename + ".npy"
        if os.path.exists(p_npy):
            model = cls()
            model.params = np.load(p_npy)
            return model
        raise FileNotFoundError(filename)
