import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from .base_model import BaseVolatilityModel


class STESModel(BaseVolatilityModel):
    def __init__(self, params=None):
        self.params = params

    def _objective(self, params, returns, features, y, burnin_size, os_index):
        n, _ = features.shape
        alphas = expit(np.dot(features, params))
        returns2 = returns**2
        sigma2 = np.zeros(n)
        sigma2[0] = returns[0] ** 2
        for t in range(1, n):
            sigma2[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * sigma2[t - 1]
        return (y - sigma2)[burnin_size:os_index]

    def fit(self, X, y, **kwargs):
        returns = kwargs.pop("returns", None)
        start_index = kwargs.pop("start_index", 0)
        end_index = kwargs.pop("end_index", len(X))

        assert returns is not None
        assert len(X) == len(y) == len(returns)

        initial_params = np.random.normal(0, 1, size=X.shape[1])
        result = least_squares(
            self._objective,
            x0=initial_params,
            args=(returns, X, y.flatten(), start_index, end_index),
        )
        self.result = result
        self.params = result.x
        return self

    def predict(self, X, **kwargs):
        returns = kwargs.pop("returns", None)

        assert returns is not None

        if self.params is None:
            raise ValueError("Model not fitted")

        n = len(returns)
        alphas = expit(np.dot(X, self.params))
        returns2 = returns**2
        sigma2 = np.zeros(n)
        sigma2[0] = returns[0] ** 2
        for t in range(1, n):
            sigma2[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * sigma2[t - 1]
        return sigma2

    def save(self, filename):
        np.save(filename + ".npy", self.params)

    @classmethod
    def load(cls, filename):
        model = cls()
        model.params = np.load(filename + ".npy")
        return model
