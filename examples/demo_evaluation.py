"""Demo script showing how to run the benchmark runner with synthetic data.

Run: python -m examples.demo_evaluation
"""

from volatility_forecast.benchmark import run
import numpy as np


class TrivialModel:
    def fit(self, X, y, returns, is_index, os_index):
        self.mu = float(y[:os_index].mean())

    def predict(self, X, returns):
        return (X * 0.0) + self.mu


def provider():
    n = 500
    X = (np.arange(n) * 0.0).reshape(-1, 1)
    y = (
        0.02
        + 0.1 * np.sin(np.linspace(0, 10, n))
        + np.random.normal(scale=0.02, size=n)
    )
    returns = None
    date = np.arange(n)
    return X, y, returns, date


if __name__ == "__main__":
    res = run.run_benchmark(
        provider,
        {"trivial": lambda: TrivialModel()},
        is_index=0,
        os_index=400,
        baseline="trivial",
    )
    print(res["metrics"])
    print(res["dm"])
    print(res["mz"])
