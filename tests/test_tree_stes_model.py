"""Tests for XGBoostSTESModel (tree-gated STES)."""

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
    AlternatingFitResult,
    EndToEndFitResult,
)


def make_dummy_data(n=300, seed=42):
    """Create synthetic data for testing."""
    rng = np.random.default_rng(seed)

    # Features: trend + noise
    t = np.arange(n)
    feat1 = np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.1, n)
    feat2 = rng.normal(0, 1.0, n)
    feat3 = rng.uniform(-1, 1, n)

    X = pd.DataFrame(
        {
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
        }
    )

    # Returns with time-varying volatility
    vol = 0.01 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 100))
    returns = rng.normal(0, 1, n) * vol
    returns_series = pd.Series(returns, name="returns")

    # Target: next-day squared return (simple proxy for realized variance)
    y = pd.Series(returns**2, name="y")

    return X, y, returns_series


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelBasics:
    """Basic smoke tests for XGBoostSTESModel."""

    def test_init_default(self):
        """Test model initialization with defaults."""
        model = XGBoostSTESModel()
        assert model.fit_method == "alternating"
        assert model.loss == "mse"
        assert model.num_boost_round == 200
        assert model.init_window == 500
        assert model.n_alt_iters == 3
        assert model.gate_valid_frac == 0.10
        assert model.huber_delta == 1.0
        assert model.model_ is None
        assert model.fit_result_ is None

    def test_init_custom_params(self):
        """Test model initialization with custom parameters."""
        xgb_params = {"max_depth": 5, "eta": 0.1}
        model = XGBoostSTESModel(
            xgb_params=xgb_params,
            num_boost_round=100,
            init_window=250,
            fit_method="end_to_end",
            loss="pseudohuber",
            huber_delta=2.0,
            n_alt_iters=5,
            gate_valid_frac=0.15,
            random_state=123,
        )
        assert model.xgb_params == xgb_params
        assert model.num_boost_round == 100
        assert model.init_window == 250
        assert model.fit_method == "end_to_end"
        assert model.loss == "pseudohuber"
        assert model.huber_delta == 2.0
        assert model.n_alt_iters == 5
        assert model.gate_valid_frac == 0.15
        assert model.random_state == 123

    def test_invalid_fit_method(self):
        """Test that invalid fit_method raises ValueError."""
        with pytest.raises(ValueError, match="fit_method must be one of"):
            XGBoostSTESModel(fit_method="invalid")

    def test_invalid_loss(self):
        """Test that invalid loss raises ValueError."""
        with pytest.raises(ValueError, match="loss must be one of"):
            XGBoostSTESModel(loss="invalid")

    def test_invalid_huber_delta(self):
        """Test that invalid huber_delta raises ValueError."""
        with pytest.raises(ValueError, match="huber_delta must be finite"):
            XGBoostSTESModel(huber_delta=-1.0)
        with pytest.raises(ValueError, match="huber_delta must be finite"):
            XGBoostSTESModel(huber_delta=np.inf)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelFitAlternating:
    """Tests for alternating fit method."""

    def test_fit_alternating_mse(self):
        """Test fitting with alternating method and MSE loss."""
        X, y, returns = make_dummy_data(n=300, seed=1)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="alternating",
            loss="mse",
            n_alt_iters=2,
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        # Check that model was fitted
        assert model.model_ is not None
        assert model.fit_result_ is not None
        assert isinstance(model.fit_result_, AlternatingFitResult)
        assert model.output_mode_ == "alpha"
        assert model.init_var_ is not None
        assert model.last_var_ is not None

        # Check fit result metadata
        assert model.fit_result_.fit_method == "alternating"
        assert model.fit_result_.output_mode == "alpha"
        assert model.fit_result_.loss == "mse"
        assert model.fit_result_.n_alt_iters == 2
        assert len(model.fit_result_.feature_names) == 3

    def test_fit_alternating_pseudohuber(self):
        """Test fitting with alternating method and pseudo-Huber loss."""
        X, y, returns = make_dummy_data(n=300, seed=2)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="alternating",
            loss="pseudohuber",
            huber_delta=1.5,
            n_alt_iters=2,
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        assert model.model_ is not None
        assert isinstance(model.fit_result_, AlternatingFitResult)
        assert model.fit_result_.loss == "pseudohuber"
        assert model.fit_result_.huber_delta == 1.5

    def test_predict_after_alternating_fit(self):
        """Test prediction after fitting with alternating method."""
        X, y, returns = make_dummy_data(n=300, seed=3)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="alternating",
            n_alt_iters=2,
            random_state=42,
        )

        # Fit on first 200 samples
        model.fit(X, y, returns=returns, start_index=0, end_index=200)

        # Predict on next 50 samples
        predictions = model.predict(X, returns=returns, start_index=200, end_index=250)

        assert len(predictions) == 50
        assert predictions.notna().all()
        assert (predictions > 0).all()  # Variance should be positive
        assert np.isfinite(predictions).all()

    def test_get_alphas_after_alternating_fit(self):
        """Test retrieving alphas after fitting."""
        X, y, returns = make_dummy_data(n=300, seed=4)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="alternating",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=200)
        alphas = model.get_alphas(X, start_index=0, end_index=200)

        assert len(alphas) == 200
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()
        assert alphas.notna().all()


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelFitEndToEnd:
    """Tests for end-to-end fit method."""

    def test_fit_end_to_end_mse(self):
        """Test fitting with end-to-end method and MSE loss."""
        X, y, returns = make_dummy_data(n=300, seed=5)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="end_to_end",
            loss="mse",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        # Check that model was fitted
        assert model.model_ is not None
        assert model.fit_result_ is not None
        assert isinstance(model.fit_result_, EndToEndFitResult)
        assert model.output_mode_ == "logit"
        assert model.init_var_ is not None
        assert model.last_var_ is not None

        # Check fit result metadata
        assert model.fit_result_.fit_method == "end_to_end"
        assert model.fit_result_.output_mode == "logit"
        assert model.fit_result_.loss == "mse"
        assert len(model.fit_result_.feature_names) == 3

    def test_fit_end_to_end_pseudohuber(self):
        """Test fitting with end-to-end method and pseudo-Huber loss."""
        X, y, returns = make_dummy_data(n=300, seed=6)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="end_to_end",
            loss="pseudohuber",
            huber_delta=2.0,
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        assert model.model_ is not None
        assert isinstance(model.fit_result_, EndToEndFitResult)
        assert model.fit_result_.loss == "pseudohuber"
        assert model.fit_result_.huber_delta == 2.0

    def test_predict_after_end_to_end_fit(self):
        """Test prediction after fitting with end-to-end method."""
        X, y, returns = make_dummy_data(n=300, seed=7)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="end_to_end",
            random_state=42,
        )

        # Fit on first 200 samples
        model.fit(X, y, returns=returns, start_index=0, end_index=200)

        # Predict on next 50 samples
        predictions = model.predict(X, returns=returns, start_index=200, end_index=250)

        assert len(predictions) == 50
        assert predictions.notna().all()
        assert (predictions > 0).all()
        assert np.isfinite(predictions).all()

    def test_get_alphas_after_end_to_end_fit(self):
        """Test retrieving alphas after end-to-end fit (logit output)."""
        X, y, returns = make_dummy_data(n=300, seed=8)

        model = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="end_to_end",
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=200)
        alphas = model.get_alphas(X, start_index=0, end_index=200)

        assert len(alphas) == 200
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()
        assert alphas.notna().all()


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelUtilities:
    """Tests for internal utility methods."""

    def test_initial_variance(self):
        """Test initial variance calculation."""
        returns2 = np.array([0.0001, 0.0004, 0.0009, 0.0016, 0.0025])
        init_var = XGBoostSTESModel._initial_variance(returns2, init_window=3)
        expected = np.mean(returns2[:3])
        assert np.isclose(init_var, expected)

    def test_initial_variance_empty(self):
        """Test initial variance with empty array."""
        returns2 = np.array([])
        init_var = XGBoostSTESModel._initial_variance(returns2, init_window=10)
        assert init_var == 1e-8

    def test_filter_state_and_forecast(self):
        """Test the variance recursion."""
        returns2 = np.array([0.01, 0.04, 0.09, 0.16, 0.25])
        alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        init_var = 0.05

        v_state, yhat = XGBoostSTESModel._filter_state_and_forecast(
            returns2, alpha, init_var
        )

        assert len(v_state) == 5
        assert len(yhat) == 5

        # Manually verify first step
        # v[0] = init_var = 0.05
        # v[1] = v[0] + alpha[0] * (returns2[0] - v[0])
        #      = 0.05 + 0.1 * (0.01 - 0.05) = 0.05 - 0.004 = 0.046
        assert np.isclose(v_state[0], 0.05)
        expected_v1 = 0.05 + 0.1 * (0.01 - 0.05)
        assert np.isclose(yhat[0], expected_v1)

    def test_loss_derivs_mse(self):
        """Test MSE loss derivatives."""
        yhat = np.array([0.1, 0.2, 0.3])
        y = np.array([0.15, 0.18, 0.35])

        e, w = XGBoostSTESModel._loss_derivs(yhat, y, loss="mse", huber_delta=1.0)

        expected_e = yhat - y  # gradient
        expected_w = np.ones_like(yhat)  # hessian

        assert np.allclose(e, expected_e)
        assert np.allclose(w, expected_w)

    def test_loss_derivs_pseudohuber(self):
        """Test pseudo-Huber loss derivatives."""
        yhat = np.array([0.1, 0.2, 0.3])
        y = np.array([0.15, 0.18, 0.35])

        e, w = XGBoostSTESModel._loss_derivs(
            yhat, y, loss="pseudohuber", huber_delta=1.0
        )

        # Should be finite and valid
        assert np.isfinite(e).all()
        assert np.isfinite(w).all()
        assert (w > 0).all()

    def test_loss_value_mse(self):
        """Test MSE loss value calculation."""
        yhat = np.array([0.1, 0.2, 0.3])
        y = np.array([0.15, 0.18, 0.35])

        loss_val = XGBoostSTESModel._loss_value(yhat, y, loss="mse", huber_delta=1.0)

        # _loss_value uses ℓ(u) = 0.5 u² convention (consistent with gradient ℓ'=u)
        expected = 0.5 * np.mean((yhat - y) ** 2)
        assert np.isclose(loss_val, expected)

    def test_loss_value_pseudohuber(self):
        """Test pseudo-Huber loss value calculation."""
        yhat = np.array([0.1, 0.2, 0.3])
        y = np.array([0.15, 0.18, 0.35])

        loss_val = XGBoostSTESModel._loss_value(
            yhat, y, loss="pseudohuber", huber_delta=1.0
        )

        # Should be finite and non-negative
        assert np.isfinite(loss_val)
        assert loss_val >= 0


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises error."""
        X, y, returns = make_dummy_data(n=100, seed=9)
        model = XGBoostSTESModel()

        with pytest.raises(ValueError, match="Model has not been fitted"):
            model.predict(X, returns=returns)

    def test_get_alphas_before_fit_raises(self):
        """Test that getting alphas before fitting raises error."""
        X, y, returns = make_dummy_data(n=100, seed=10)
        model = XGBoostSTESModel()

        with pytest.raises(ValueError, match="Model has not been fitted"):
            model.get_alphas(X)

    def test_fit_with_insufficient_data_raises(self):
        """Test that fitting with too little data raises error."""
        X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]})
        y = pd.Series([0.01, 0.02, 0.03])
        returns = pd.Series([0.1, 0.2, 0.3])

        model = XGBoostSTESModel()

        with pytest.raises(ValueError, match="Not enough rows"):
            model.fit(X, y, returns=returns)

    def test_predict_with_warm_start(self):
        """Test prediction with warm start from last fit."""
        X, y, returns = make_dummy_data(n=300, seed=11)

        model = XGBoostSTESModel(num_boost_round=50, random_state=42)
        model.fit(X, y, returns=returns, start_index=0, end_index=200)

        # Predict with warm start (default)
        preds_warm = model.predict(
            X,
            returns=returns,
            start_index=200,
            end_index=250,
            warm_start_from_last_fit=True,
        )

        # Predict without warm start
        preds_cold = model.predict(
            X,
            returns=returns,
            start_index=200,
            end_index=250,
            warm_start_from_last_fit=False,
        )

        # Both should be valid
        assert (preds_warm > 0).all()
        assert (preds_cold > 0).all()
        # Note: They may or may not be significantly different depending on data
        # Just verify the parameter is accepted

    def test_predict_with_custom_init_var(self):
        """Test prediction with custom initial variance."""
        X, y, returns = make_dummy_data(n=300, seed=12)

        model = XGBoostSTESModel(num_boost_round=50, random_state=42)
        model.fit(X, y, returns=returns, start_index=0, end_index=200)

        # Predict with custom init_var
        custom_init_var = 0.001
        preds = model.predict(
            X,
            returns=returns,
            start_index=200,
            end_index=250,
            init_var=custom_init_var,
            warm_start_from_last_fit=False,
        )

        assert len(preds) == 50
        assert (preds > 0).all()

    def test_monotone_constraints(self):
        """Test that monotone constraints are applied."""
        X, y, returns = make_dummy_data(n=300, seed=13)

        model = XGBoostSTESModel(
            num_boost_round=50,
            monotonic_constraints={"feat1": 1, "feat2": -1},
            random_state=42,
        )

        model.fit(X, y, returns=returns, start_index=0, end_index=250)

        # Check that constraints were applied (in params)
        assert "monotone_constraints" in model.fit_result_.params_used


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelComparison:
    """Compare alternating vs end-to-end methods."""

    def test_both_methods_produce_valid_forecasts(self):
        """Test that both fit methods produce valid forecasts."""
        X, y, returns = make_dummy_data(n=300, seed=14)

        model_alt = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="alternating",
            n_alt_iters=2,
            random_state=42,
        )

        model_e2e = XGBoostSTESModel(
            num_boost_round=50,
            fit_method="end_to_end",
            random_state=42,
        )

        # Fit both models
        model_alt.fit(X, y, returns=returns, start_index=0, end_index=200)
        model_e2e.fit(X, y, returns=returns, start_index=0, end_index=200)

        # Predict with both
        preds_alt = model_alt.predict(
            X, returns=returns, start_index=200, end_index=250
        )
        preds_e2e = model_e2e.predict(
            X, returns=returns, start_index=200, end_index=250
        )

        # Both should be valid
        assert (preds_alt > 0).all()
        assert (preds_e2e > 0).all()
        assert preds_alt.notna().all()
        assert preds_e2e.notna().all()

        # They should be different (different training methods)
        assert not np.allclose(preds_alt.values, preds_e2e.values, rtol=0.1)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestXGBoostSTESModelBackwardCompat:
    """Test backward compatibility."""

    def test_ignored_kwargs_warning(self, caplog):
        """Test that unsupported kwargs generate warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            model = XGBoostSTESModel(deprecated_param=True, another_old_param="value")
        # Should still create model successfully
        assert model is not None
        # Check that warning was logged
        assert any(
            "Ignoring unsupported" in record.message for record in caplog.records
        )
