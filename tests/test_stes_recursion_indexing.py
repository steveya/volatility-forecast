import numpy as np

from volatility_forecast.model.stes_model import STESModel


def test_stes_alpha_indexing_simple():
    # Construct a toy feature matrix so that alphas are easy to compute.
    # We'll make two features where params=[0, 0] => alphas=0.5 constant
    X = np.zeros((5, 2), dtype=float)
    # Dummy returns: r_0..r_4
    r = np.array([1.0, 2.0, 3.0, 0.5, 0.2], dtype=float)
    # y: we'll compute sigma2 recursion externally using alphas[t-1]
    # fixed alpha = 0.5 for all t
    alpha = 0.5

    # manual recursion with alpha_{t-1} used to produce sigma2[t]
    sigma2 = np.zeros(len(r))
    sigma2[0] = r[0] ** 2
    for t in range(1, len(r)):
        sigma2[t] = alpha * (r[t - 1] ** 2) + (1 - alpha) * sigma2[t - 1]

    # y is sigma2 so objective should be near-zero when params produce alpha=0.5
    y = sigma2.copy()

    m = STESModel()
    # set params explicitly to zeros so expit(dot) == 0.5
    m.params = np.zeros(2, dtype=float)

    preds = m.predict(X, returns=r)
    assert np.allclose(preds, sigma2)
