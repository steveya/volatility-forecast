from __future__ import annotations

import numpy as np
import pytest

from volatility_forecast.model.components.channel_layout import build_channel_layout
from volatility_forecast.model.pgarch_linear_model import PGARCHLinearModel

from .pgarch_test_utils import make_synthetic_pgarch_data


def test_channel_layout_defaults_to_all_features_for_each_channel() -> None:
    layout = build_channel_layout(3, None)

    assert np.array_equal(layout.mu_aug_cols, np.array([0, 1, 2, 3]))
    assert np.array_equal(layout.phi_aug_cols, np.array([0, 1, 2, 3]))
    assert np.array_equal(layout.g_aug_cols, np.array([0, 1, 2, 3]))
    assert np.array_equal(layout.feature_columns("mu"), np.array([0, 1, 2]))


def test_channel_layout_respects_channel_specific_subsets() -> None:
    layout = build_channel_layout(
        4,
        {
            "mu": [0, 3],
            "phi": [1],
            "g": [],
        },
    )

    assert np.array_equal(layout.mu_aug_cols, np.array([0, 1, 4]))
    assert np.array_equal(layout.phi_aug_cols, np.array([0, 2]))
    assert np.array_equal(layout.g_aug_cols, np.array([0]))
    assert layout.n_mu == 3
    assert layout.n_phi == 2
    assert layout.n_g == 1


@pytest.mark.parametrize("channel", ["mu", "phi", "g"])
@pytest.mark.parametrize("indices", [[0.9], [-0.5], ["1"], [True]])
def test_channel_layout_rejects_non_integer_indices(channel, indices) -> None:
    channel_features = {"mu": [0], "phi": [1], "g": []}
    channel_features[channel] = indices

    with pytest.raises(ValueError, match="integer column indices"):
        build_channel_layout(3, channel_features)


def test_channel_layout_rejects_missing_keys_and_bad_indices() -> None:
    with pytest.raises(ValueError, match="must contain key 'g'"):
        build_channel_layout(3, {"mu": [0], "phi": [1]})

    with pytest.raises(ValueError, match="must be a 1-D sequence"):
        build_channel_layout(
            3,
            {
                "mu": [[0, 1]],
                "phi": [1],
                "g": [2],
            },
        )

    with pytest.raises(ValueError, match="indices must be in \\[0, 3\\)"):
        build_channel_layout(
            3,
            {
                "mu": [0, 3],
                "phi": [1],
                "g": [2],
            },
        )


def test_empty_channel_feature_list_yields_intercept_only_channel() -> None:
    y, X = make_synthetic_pgarch_data(seed=21)
    model = PGARCHLinearModel(
        loss="qlike",
        random_state=21,
        channel_features={
            "mu": [],
            "phi": [0, 1],
            "g": [0, 1],
        },
    ).fit(y, X)

    components = model.predict_components(X)

    assert model.coef_mu_ is not None
    assert model.coef_mu_.shape == (1,)
    assert np.allclose(components["mu"], components["mu"][0], atol=1e-10, rtol=0.0)
