from scipy.special import expit
from .stes_model import STESModel


class ESModel(STESModel):
    """Exponential Smoothing (ES) model: STES with constant alpha.

    Stores theta (unconstrained parameter) internally but always exposes
    alpha (smoothing parameter) in probability space (0,1) via the alpha_ attribute.
    """

    def __init__(self, params=None, *, keep_result=False, random_state=None):
        super().__init__(
            params=params, keep_result=keep_result, random_state=random_state
        )
        self.theta_ = None  # unconstrained parameter (internal)
        self.alpha_ = None  # probability-space alpha in (0,1) (user-facing)

    def fit(self, X, y, **kwargs):
        # Call parent fit (only uses X[["const"]])
        super().fit(X[["const"]], y, **kwargs)

        # Extract theta (raw parameter) and compute alpha in probability space
        if self.params is not None and len(self.params) > 0:
            self.theta_ = float(self.params[0])
            self.alpha_ = float(expit(self.theta_))
            assert (
                0.0 < self.alpha_ < 1.0
            ), f"alpha_ must be in (0,1), got {self.alpha_}"
        return self

    def predict(self, X, **kwargs):
        return super().predict(X[["const"]], **kwargs)

    @property
    def alpha(self):
        """Backward-compat property: returns alpha_ (probability-space)."""
        if self.alpha_ is None and self.theta_ is not None:
            self.alpha_ = float(expit(self.theta_))
        return self.alpha_
