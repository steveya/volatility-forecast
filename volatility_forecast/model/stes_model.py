import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from .base_model import BaseVolatilityModel


class STESModel(BaseVolatilityModel):
    def __init__(self, params=None):
        self.params = params
        self.fitted_results = None

    def _objective(self, params, returns, features, y):
        n, _ = features.shape
        alphas = expit(np.dot(features, params))
        sigma2 = np.zeros(n)
        sigma2[0] = returns[0] ** 2
        for t in range(1, n):
            sigma2[t] = (
                alphas[t - 1] * returns[t - 1] ** 2
                + (1 - alphas[t - 1]) * sigma2[t - 1]
            )
        return y - sigma2

    def fit(self, X, y, returns):
        assert len(X) == len(y) == len(returns)
        initial_params = np.random.normal(0, 1, size=X.shape[1])
        result = least_squares(self._objective, x0=initial_params, args=(returns, X, y))
        self.params = result.x
        return self

    def predict(self, X, returns):
        if self.params is None:
            raise ValueError("Model not fitted")

        n = len(returns)
        alphas = expit(np.dot(self.params, X))
        sigma2 = np.zeros(n)
        sigma2[0] = returns[0] ** 2
        for t in range(1, n):
            sigma2[t] = (
                alphas[t - 1] * returns[t - 1] ** 2
                + (1 - alphas[t - 1]) * sigma2[t - 1]
            )
        return sigma2
