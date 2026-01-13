import numpy as np
import pandas as pd
from scipy.special import expit

from volatility_forecast.model.es_model import ESModel


def make_dummy_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, size=n)
    # simple target: next-day squared return proxy (we keep same length and allow fit to use start/end)
    y = r**2
    X = pd.DataFrame({"const": np.ones(n)})
    return X, y, r


def test_es_alpha_always_in_probability_space():
    """ESModel.alpha_ must always be in (0,1) after fit()."""
    X, y, r = make_dummy_series(200, seed=1)
    model = ESModel(random_state=1)
    model.fit(X, y, returns=r, start_index=0, end_index=len(X))

    # Check that alpha_ was set and is in (0,1)
    assert model.alpha_ is not None, "alpha_ should be set after fit()"
    assert 0.0 < model.alpha_ < 1.0, f"alpha_ must be in (0,1), got {model.alpha_}"

    # Check consistency: theta_ should exist and expit(theta_) should equal alpha_
    assert model.theta_ is not None, "theta_ should be set after fit()"
    assert np.isclose(
        float(expit(model.theta_)), model.alpha_
    ), "alpha_ should equal expit(theta_)"


def test_es_alpha_property_backward_compat():
    """ESModel.alpha property should return alpha_ for backward compatibility."""
    X, y, r = make_dummy_series(100, seed=2)
    model = ESModel(random_state=2)
    model.fit(X, y, returns=r, start_index=0, end_index=len(X))

    # Check that the .alpha property returns alpha_
    assert model.alpha == model.alpha_, "Property .alpha should return .alpha_"
    assert (
        0.0 < model.alpha < 1.0
    ), f"alpha property must be in (0,1), got {model.alpha}"


def test_es_alpha_vs_stes_alpha_difference():
    """Verify that delta_alpha = alpha_stes - alpha_es is in valid range."""
    # Create ES run with fixed seed
    X, y, r = make_dummy_series(150, seed=10)

    model_es = ESModel(random_state=10)
    model_es.fit(X, y, returns=r, start_index=0, end_index=len(X))

    # Simulate an STES alpha (pick a random value in (0,1))
    alpha_stes_sample = 0.25

    # delta_alpha should be in [-1, 1]
    delta = alpha_stes_sample - model_es.alpha_
    assert (
        -1.0 <= delta <= 1.0
    ), f"delta_alpha must be in [-1,1], got {delta} from alpha_stes={alpha_stes_sample}, alpha_es={model_es.alpha_}"
