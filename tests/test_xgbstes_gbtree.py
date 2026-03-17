"""Tests for XGBoostSTESModel with gbtree booster (E2E mode)."""

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from volatility_forecast.model.tree_stes_model import (
    XGBoostSTESModel,
    EndToEndFitResult,
)


def make_dummy_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    feat1 = np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.1, n)
    feat2 = rng.normal(0, 1.0, n)
    feat3 = rng.uniform(-1, 1, n)

    X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

    vol = 0.01 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 100))
    returns = rng.normal(0, 1, n) * vol
    returns_series = pd.Series(returns, name="returns")
    y = pd.Series(returns**2, name="y")

    return X, y, returns_series


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestGbtreeE2EQLIKE:
    """Tests for gbtree booster with E2E training and QLIKE loss."""

    def test_gbtree_e2e_qlike_smoke(self):
        """gbtree + E2E + QLIKE should train and produce finite results."""
        X, y, returns = make_dummy_data(n=300, seed=10)

        model = XGBoostSTESModel(
            xgb_params={"booster": "gbtree", "max_depth": 2, "eta": 0.1},
            num_boost_round=30,
            fit_method="end_to_end",
            loss="qlike",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        assert model.model_ is not None
        assert isinstance(model.fit_result_, EndToEndFitResult)
        assert model.fit_result_.loss == "qlike"

    def test_gbtree_e2e_qlike_predictions_finite(self):
        """Predictions from gbtree + QLIKE model should be finite and positive."""
        X, y, returns = make_dummy_data(n=300, seed=11)

        model = XGBoostSTESModel(
            xgb_params={"booster": "gbtree", "max_depth": 2, "eta": 0.1},
            num_boost_round=30,
            fit_method="end_to_end",
            loss="qlike",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=200)
        preds = model.predict(X.iloc[200:], returns=returns.iloc[200:])

        assert len(preds) == 100
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_gbtree_e2e_qlike_nontrivial_alphas(self):
        """gbtree gates should produce nontrivial alphas (not all 0 or all 1)."""
        X, y, returns = make_dummy_data(n=300, seed=12)

        model = XGBoostSTESModel(
            xgb_params={"booster": "gbtree", "max_depth": 2, "eta": 0.1},
            num_boost_round=50,
            fit_method="end_to_end",
            loss="qlike",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)
        alphas = model.get_alphas(X)

        assert len(alphas) == 300
        assert np.all(np.isfinite(alphas))
        assert np.all((alphas >= 0) & (alphas <= 1))
        # Should not be degenerate (all same value)
        assert np.std(alphas) > 1e-6, "Alphas should have nontrivial dispersion"

    def test_gbtree_e2e_mse_smoke(self):
        """gbtree + E2E + MSE should also work."""
        X, y, returns = make_dummy_data(n=300, seed=13)

        model = XGBoostSTESModel(
            xgb_params={"booster": "gbtree", "max_depth": 2, "eta": 0.1},
            num_boost_round=30,
            fit_method="end_to_end",
            loss="mse",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        preds = model.predict(X.iloc[250:], returns=returns.iloc[250:])
        assert len(preds) == 50
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestAlternatingFitDeprecation:
    """Test that alternating-fit raises a FutureWarning."""

    def test_alternating_fit_warns(self):
        with pytest.warns(FutureWarning, match="alternating.*deprecated"):
            XGBoostSTESModel(fit_method="alternating")
