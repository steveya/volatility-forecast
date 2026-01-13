import numpy as np
from volatility_forecast.benchmark import run


class MeanModel:
    def fit(self, X, y, returns, is_index, os_index):
        self.mu = np.nanmean(y[:os_index])

    def predict(self, X, returns):
        return np.ones_like(X) * self.mu


def synthetic_provider():
    n = 200
    X = np.arange(n)
    y = (
        0.02
        + 0.5 * np.sin(np.linspace(0, 10, n))
        + np.random.normal(scale=0.01, size=n)
    )
    returns = None
    date = np.arange(n)
    return X, y, returns, date


def test_run_benchmark_basic():
    mf = {"mean": lambda: MeanModel()}
    res = run.run_benchmark(
        synthetic_provider,
        mf,
        is_index=0,
        os_index=150,
        oos_index=None,
        baseline="mean",
    )
    assert "metrics" in res and "dm" in res and "mz" in res
    assert not res["metrics"].empty
