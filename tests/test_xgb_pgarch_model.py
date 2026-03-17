import numpy as np
import pytest

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import volatility_forecast.model.xgb_pgarch_model as xgb_pgarch_module
from volatility_forecast.model.xgb_pgarch_model import XGBGPGARCHModel


def make_synthetic_pgarch_data(
    n: int = 140,
    d: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X_aug = np.column_stack([np.ones(n, dtype=float), X])

    w_mu = np.array([-2.0, 0.25, -0.15], dtype=float)[: d + 1]
    w_phi = np.array([2.1, 0.15, 0.05], dtype=float)[: d + 1]
    w_g = np.array([-2.4, -0.10, 0.20], dtype=float)[: d + 1]

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def softplus(z: float) -> float:
        return float(np.log1p(np.exp(-abs(z))) + max(z, 0.0))

    y = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    y[0] = 0.04
    h[0] = 0.04

    for t in range(1, n):
        x = X_aug[t - 1]
        mu = 1e-12 + softplus(float(np.dot(w_mu, x)))
        phi = 1e-4 + (1.0 - 2e-4) * sigmoid(float(np.dot(w_phi, x)))
        g = 1e-4 + (1.0 - 2e-4) * sigmoid(float(np.dot(w_g, x)))
        h[t] = (1.0 - phi) * mu + phi * (g * y[t - 1] + (1.0 - g) * h[t - 1])
        y[t] = max(h[t] * (0.8 + 0.4 * rng.random()), 1e-8)

    return y, X


def make_small_problem(
    loss: str,
    seed: int = 0,
    init_method: str = "intercept_only_pgarch",
) -> tuple[XGBGPGARCHModel, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    y, X = make_synthetic_pgarch_data(n=7, d=2, seed=seed)
    model = XGBGPGARCHModel(
        loss=loss,
        init_method=init_method,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=1,
        random_state=seed,
        verbosity=0,
    )
    initializer = model._fit_initializer(y, X)
    baseline = model._extract_baseline_sequences(y, X, initializer)
    model._current_h0_ = float(baseline["h"][0])
    raw_scores = baseline["c0"] + np.random.default_rng(seed + 11).normal(
        0.0, 0.05, size=len(y)
    )
    return model, y, baseline, raw_scores


def finite_difference_raw_score_gradient(
    model: XGBGPGARCHModel,
    y: np.ndarray,
    baseline: dict[str, np.ndarray],
    raw_scores: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.zeros_like(raw_scores)
    for j in range(raw_scores.size):
        step = np.zeros_like(raw_scores)
        step[j] = eps
        state_plus = model._forward_recursion(
            y, baseline["mu"], baseline["phi"], raw_scores + step
        )
        state_minus = model._forward_recursion(
            y, baseline["mu"], baseline["phi"], raw_scores - step
        )
        grad[j] = (
            model._loss_from_state(y, state_plus["h"])
            - model._loss_from_state(y, state_minus["h"])
        ) / (2.0 * eps)
    return grad


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_xgb_g_pgarch_fit_runs_mse():
    y, X = make_synthetic_pgarch_data(seed=1)
    model = XGBGPGARCHModel(
        loss="mse",
        n_estimators=20,
        learning_rate=0.1,
        max_depth=2,
        random_state=1,
    )

    fitted = model.fit(y, X)
    h = fitted.predict_variance(y, X)

    assert fitted is model
    assert model.is_fitted_
    assert model.booster_ is not None
    assert model.initializer_ is not None
    assert model.baseline_train_ is not None
    assert model.init_method_ == "linear_pgarch"
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.train_loss_)
    assert h.shape == y.shape
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_xgb_g_pgarch_fit_runs_qlike():
    y, X = make_synthetic_pgarch_data(seed=2)
    model = XGBGPGARCHModel(
        loss="qlike",
        n_estimators=20,
        learning_rate=0.1,
        max_depth=2,
        random_state=2,
    )

    model.fit(y, X)
    h = model.predict_variance(y, X)

    assert model.is_fitted_
    assert np.isfinite(model.train_loss_)
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_predict_components_bounds():
    y, X = make_synthetic_pgarch_data(seed=3)
    model = XGBGPGARCHModel(random_state=3).fit(y, X)

    components = model.predict_components(X)

    assert set(components) == {"mu", "phi", "g"}
    assert np.all(components["mu"] > 0.0)
    assert np.all((components["phi"] > 0.0) & (components["phi"] < 1.0))
    assert np.all((components["g"] > 0.0) & (components["g"] < 1.0))


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_implied_garch_params_identity():
    y, X = make_synthetic_pgarch_data(seed=4)
    model = XGBGPGARCHModel(random_state=4).fit(y, X)

    components = model.predict_components(X)
    implied = model.implied_garch_params(X)

    assert np.allclose(
        implied["alpha"] + implied["beta"],
        components["phi"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        implied["omega"],
        (1.0 - components["phi"]) * components["mu"],
        atol=1e-10,
        rtol=1e-10,
    )


def test_exact_row_gradient_matches_finite_difference_mse_small_problem():
    model, y, baseline, raw_scores = make_small_problem(loss="mse", seed=5)

    state = model._forward_recursion(y, baseline["mu"], baseline["phi"], raw_scores)
    analytic, _ = model._rowwise_grad_hess(y, state)
    fd = finite_difference_raw_score_gradient(model, y, baseline, raw_scores)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_exact_row_gradient_matches_finite_difference_qlike_small_problem():
    model, y, baseline, raw_scores = make_small_problem(loss="qlike", seed=6)

    state = model._forward_recursion(y, baseline["mu"], baseline["phi"], raw_scores)
    analytic, _ = model._rowwise_grad_hess(y, state)
    fd = finite_difference_raw_score_gradient(model, y, baseline, raw_scores)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_row_hessian_nonnegative_and_finite_mse():
    model, y, baseline, raw_scores = make_small_problem(loss="mse", seed=7)

    state = model._forward_recursion(y, baseline["mu"], baseline["phi"], raw_scores)
    _, hess = model._rowwise_grad_hess(y, state)

    assert np.all(np.isfinite(hess))
    assert np.all(hess >= 0.0)


def test_row_hessian_nonnegative_and_finite_qlike():
    model, y, baseline, raw_scores = make_small_problem(loss="qlike", seed=8)

    state = model._forward_recursion(y, baseline["mu"], baseline["phi"], raw_scores)
    _, hess = model._rowwise_grad_hess(y, state)

    assert np.all(np.isfinite(hess))
    assert np.all(hess >= 0.0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_initializer_modes_supported(monkeypatch: pytest.MonkeyPatch):
    y, X = make_synthetic_pgarch_data(seed=9)

    linear_model = XGBGPGARCHModel(init_method="linear_pgarch", random_state=9).fit(y, X)
    assert linear_model.init_method_ == "linear_pgarch"

    const_model = XGBGPGARCHModel(
        init_method="intercept_only_pgarch", random_state=9
    ).fit(y, X)
    assert const_model.init_method_ == "intercept_only_pgarch"

    monkeypatch.setattr(xgb_pgarch_module, "PGARCHLinearModel", None)
    with pytest.warns(RuntimeWarning, match="Falling back to intercept-only"):
        fallback_model = XGBGPGARCHModel(
            init_method="linear_pgarch", random_state=9
        ).fit(y, X)
    assert fallback_model.init_method_ == "intercept_only_pgarch"


def test_terminal_row_grad_hess_policy_is_consistent():
    model, y, baseline, raw_scores = make_small_problem(loss="mse", seed=10)

    state = model._forward_recursion(y, baseline["mu"], baseline["phi"], raw_scores)
    grad, hess = model._rowwise_grad_hess(y, state)

    assert len(grad) == len(y)
    assert len(hess) == len(y)
    assert grad[-1] == 0.0
    assert hess[-1] == 0.0


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_score_metrics_supported():
    y, X = make_synthetic_pgarch_data(seed=11)
    model = XGBGPGARCHModel(random_state=11).fit(y, X)

    qlike = model.score(y, X, metric="qlike")
    mse = model.score(y, X, metric="mse")
    rmse = model.score(y, X, metric="rmse")

    assert np.isfinite(qlike)
    assert np.isfinite(mse)
    assert np.isfinite(rmse)
    assert np.isclose(rmse, np.sqrt(mse))


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_invalid_input_raises():
    y, X = make_synthetic_pgarch_data(seed=12)
    model = XGBGPGARCHModel()

    with pytest.raises(ValueError, match="Length mismatch"):
        model.fit(y[:-1], X)

    y_nan = y.copy()
    y_nan[0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN or inf"):
        model.fit(y_nan, X)

    with pytest.raises(ValueError, match="At least 3 observations"):
        model.fit(y[:2], X[:2])
