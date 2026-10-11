from __future__ import annotations

import numpy as np

from volatility_forecast.model.pgarch_linear_model import PGARCHLinearModel

from .pgarch_test_utils import make_synthetic_pgarch_data


def test_feature_row_t_does_not_affect_h_t() -> None:
    y, X = make_synthetic_pgarch_data(seed=30)
    model = PGARCHLinearModel(loss="qlike", random_state=30).fit(y, X)

    h_base = model.predict_variance(y, X)
    X_perturbed = X.copy()
    X_perturbed[25] += np.array([5.0, -4.0])
    h_perturbed = model.predict_variance(y, X_perturbed)

    assert np.allclose(h_base[:26], h_perturbed[:26], atol=1e-10, rtol=0.0)
    assert not np.allclose(h_base[26:], h_perturbed[26:])


def test_feature_row_t_minus_1_can_affect_h_t() -> None:
    y, X = make_synthetic_pgarch_data(seed=31)
    model = PGARCHLinearModel(loss="qlike", random_state=31).fit(y, X)

    h_base = model.predict_variance(y, X)
    X_perturbed = X.copy()
    X_perturbed[24] += np.array([5.0, -4.0])
    h_perturbed = model.predict_variance(y, X_perturbed)

    assert not np.isclose(h_base[25], h_perturbed[25])


def test_lagged_target_can_affect_next_variance_state() -> None:
    y, X = make_synthetic_pgarch_data(seed=32)
    model = PGARCHLinearModel(loss="qlike", random_state=32).fit(y, X)

    h_base = model.predict_variance(y, X)
    y_perturbed = y.copy()
    y_perturbed[24] *= 1.75
    h_perturbed = model.predict_variance(y_perturbed, X)

    assert not np.isclose(h_base[25], h_perturbed[25])
