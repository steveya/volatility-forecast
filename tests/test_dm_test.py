import numpy as np
from volatility_forecast.evaluation import dm_test


def test_dm_basic():
    np.random.seed(0)
    n = 200
    # model A has slightly larger errors
    e_a = np.random.normal(scale=1.02, size=n)
    e_b = np.random.normal(scale=1.0, size=n)
    loss_a = e_a**2
    loss_b = e_b**2
    r = dm_test.diebold_mariano(loss_a, loss_b)
    # Expect mean loss diff positive (A worse than B)
    assert "dm_stat" in r and "p_value" in r
    assert r["n"] == n
    assert r["mean_d"] > -0.1
    assert 0.0 <= r["p_value"] <= 1.0
