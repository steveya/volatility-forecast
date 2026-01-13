import numpy as np
from volatility_forecast.evaluation import mz


def test_mz_basic():
    np.random.seed(1)
    n = 100
    yhat = np.linspace(0.1, 1.0, n)
    # generate y with slope 1.5 and intercept 0.02
    y = 0.02 + 1.5 * yhat + np.random.normal(scale=0.05, size=n)
    res = mz.mincer_zarnowitz(y, yhat)
    assert "params" in res and "t_stats" in res and "r2" in res
    assert abs(res["params"]["beta"] - 1.5) < 0.2
    assert res["n"] == n
