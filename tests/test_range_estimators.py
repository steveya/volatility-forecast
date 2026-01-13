import numpy as np
from volatility_forecast.targets import range_estimators


def test_parkinson_formula():
    hi = np.array([110.0, 101.0])
    lo = np.array([100.0, 99.0])
    loghl = np.log(hi / lo)
    p = range_estimators.compute_parkinson_from_loghl(loghl)
    assert p.shape == loghl.shape
    assert np.all(p >= 0)


def test_garman_klass_formula():
    o = np.array([100.0, 100.0])
    hi = np.array([110.0, 105.0])
    lo = np.array([95.0, 97.0])
    c = np.array([105.0, 102.0])
    log_hl = np.log(hi / lo)
    log_co = np.log(c / o)
    gk = range_estimators.compute_garman_klass_from_logs(log_hl, log_co)
    assert gk.shape == log_hl.shape
