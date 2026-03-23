import numpy as np

from volatility_forecast.model.pgarch_core import PGARCHBounds, PGARCHCore, PGARCHRawScores, blend_raw_scores


def test_blend_raw_scores_supports_global_weights():
    raw_1 = PGARCHRawScores(
        mu=np.array([0.0, 1.0, 2.0]),
        phi=np.array([1.0, 2.0, 3.0]),
        g=np.array([2.0, 3.0, 4.0]),
    )
    raw_2 = PGARCHRawScores(
        mu=np.array([2.0, 4.0, 6.0]),
        phi=np.array([3.0, 5.0, 7.0]),
        g=np.array([4.0, 6.0, 8.0]),
    )

    blended = blend_raw_scores([raw_1, raw_2], np.array([1.0, 3.0]))

    assert np.allclose(blended.mu, np.array([1.5, 3.25, 5.0]))
    assert np.allclose(blended.phi, np.array([2.5, 4.25, 6.0]))
    assert np.allclose(blended.g, np.array([3.5, 5.25, 7.0]))


def test_blend_raw_scores_supports_time_varying_weights():
    raw_1 = PGARCHRawScores(
        mu=np.array([0.0, 1.0, 2.0]),
        phi=np.array([1.0, 2.0, 3.0]),
        g=np.array([2.0, 3.0, 4.0]),
    )
    raw_2 = PGARCHRawScores(
        mu=np.array([2.0, 4.0, 6.0]),
        phi=np.array([3.0, 5.0, 7.0]),
        g=np.array([4.0, 6.0, 8.0]),
    )
    weights = np.array(
        [
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 3.0],
        ]
    )

    blended = blend_raw_scores([raw_1, raw_2], weights)

    assert np.allclose(blended.mu, np.array([0.5, 3.25, 5.0]))
    assert np.allclose(blended.phi, np.array([1.5, 4.25, 6.0]))
    assert np.allclose(blended.g, np.array([2.5, 5.25, 7.0]))


def test_forward_with_scores_matches_component_variance_path():
    core = PGARCHCore(loss="qlike", bounds=PGARCHBounds())
    y = np.array([0.04, 0.05, 0.045, 0.042], dtype=float)
    raw = PGARCHRawScores(
        mu=np.array([-2.0, -1.5, -1.0, -0.5], dtype=float),
        phi=np.array([2.0, 1.8, 1.6, 1.4], dtype=float),
        g=np.array([-2.2, -2.0, -1.8, -1.6], dtype=float),
    )

    state = core.forward_with_scores(y, a=raw.a, b=raw.b, c=raw.c, h0=0.04)
    h_from_components = core.variance_path_from_components(
        y,
        mu=state.mu,
        phi=state.phi,
        g=state.g,
        h0=0.04,
    )

    assert np.allclose(state.h, h_from_components)
