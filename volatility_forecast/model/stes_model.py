import hashlib
import json
import numpy as np
import joblib
from scipy.optimize import least_squares
from scipy.special import expit
from .base_model import BaseVolatilityModel


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
        self, params=None, *, keep_result: bool = False, random_state: int | None = None
    ):
        self.params = params
        self.keep_result = keep_result
        self.random_state = random_state

        # schema metadata (for safe reload + inference)
        self.feature_names_ = None
        self.feature_schema_hash_ = None
        self.n_features_ = None

        # optional diagnostic
        self.result = None

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
    def _objective(self, params, returns, features, y, burnin_size, os_index):
        n, _ = features.shape
        alphas = expit(np.dot(features, params))
        returns2 = returns**2
        sigma2 = np.zeros(n)
        sigma2[0] = returns[0] ** 2
        for t in range(1, n):
            # Use alphas[t-1] so alpha_t forecasts sigma_{t+1} (consistent with our feature/target alignment)
            sigma2[t] = (
                alphas[t - 1] * returns2[t - 1] + (1 - alphas[t - 1]) * sigma2[t - 1]
            )
        return (y - sigma2)[burnin_size:os_index]

    def fit(self, X, y, **kwargs):
        returns = kwargs.pop("returns", None)
        start_index = kwargs.pop("start_index", 0)
        end_index = kwargs.pop("end_index", None)

        assert returns is not None, "fit() requires returns=..."
        X_np, cols = self._coerce_X(X)

        y_np = np.asarray(y).reshape(-1)
        r_np = np.asarray(returns).reshape(-1)

        if end_index is None:
            end_index = len(X_np)

        assert len(X_np) == len(y_np) == len(r_np)

        # store feature schema for safe persistence/inference
        self._set_schema(cols, X_np.shape[1])

        rng = np.random.default_rng(self.random_state)
        initial_params = rng.normal(0, 1, size=X_np.shape[1])

        result = least_squares(
            self._objective,
            x0=initial_params,
            args=(r_np, X_np, y_np, start_index, end_index),
        )

        self.params = result.x
        self.result = result if self.keep_result else None
        return self

    def predict(self, X, **kwargs):
        returns = kwargs.pop("returns", None)
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

        sigma2 = np.zeros(n)
        sigma2[0] = r_np[0] ** 2
        for t in range(1, n):
            # Use alphas[t-1] so alpha_t forecasts sigma_{t+1} (consistent with our feature/target alignment)
            sigma2[t] = (
                alphas[t - 1] * returns2[t - 1] + (1 - alphas[t - 1]) * sigma2[t - 1]
            )
        return sigma2

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
