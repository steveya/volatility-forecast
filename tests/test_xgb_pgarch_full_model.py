import numpy as np
import pytest

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import volatility_forecast.model.xgb_pgarch_full_model as xgb_pgarch_full_module
from volatility_forecast.model.xgb_pgarch_full_model import XGBPGARCHModel


def make_synthetic_pgarch_data(
    n: int = 140,
    d: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X_aug = np.column_stack([np.ones(n, dtype=float), X])

    w_mu = np.array([-2.0, 0.25, -0.15], dtype=float)[: d + 1]
    w_phi = np.array([2.0, 0.20, 0.05], dtype=float)[: d + 1]
    w_g = np.array([-2.3, -0.10, 0.20], dtype=float)[: d + 1]

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def softplus(z: float) -> float:
        return float(np.log1p(np.exp(-abs(z))) + max(z, 0.0))

    y = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    y[0] = 0.04
    h[0] = 0.04

    for row in range(n - 1):
        x = X_aug[row]
        mu = 1e-12 + softplus(float(np.dot(w_mu, x)))
        phi = 1e-4 + (1.0 - 2e-4) * sigmoid(float(np.dot(w_phi, x)))
        g = 1e-4 + (1.0 - 2e-4) * sigmoid(float(np.dot(w_g, x)))
        q_row = g * y[row] + (1.0 - g) * h[row]
        h[row + 1] = (1.0 - phi) * mu + phi * q_row
        y[row + 1] = max(h[row + 1] * (0.8 + 0.4 * rng.random()), 1e-8)

    return y, X


def make_small_problem(
    loss: str,
    seed: int = 0,
    init_method: str = "intercept_only_pgarch",
) -> tuple[
    XGBPGARCHModel,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    y, X = make_synthetic_pgarch_data(n=7, d=2, seed=seed)
    model = XGBPGARCHModel(
        loss=loss,
        init_method=init_method,
        n_outer_rounds=2,
        trees_per_channel_per_round=1,
        learning_rate=0.1,
        max_depth=1,
        random_state=seed,
        verbosity=0,
    )
    initializer = model._fit_initializer(y, X)
    baseline = model._extract_baseline_sequences(y, X, initializer)
    a, b, c = model._initialize_raw_scores(baseline)
    rng = np.random.default_rng(seed + 11)
    a = a + rng.normal(0.0, 0.05, size=len(y))
    b = b + rng.normal(0.0, 0.05, size=len(y))
    c = c + rng.normal(0.0, 0.05, size=len(y))
    return model, y, baseline, a, b, c, max(float(y[0]), model.h_min)


def finite_difference_raw_score_gradient(
    model: XGBPGARCHModel,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    channel: str,
    h0: float,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.zeros_like(a, dtype=float)
    for row in range(grad.size):
        step = np.zeros_like(a, dtype=float)
        step[row] = eps
        if channel == "mu":
            state_plus = model._forward_recursion_with_scores(y, a + step, b, c, h0=h0)
            state_minus = model._forward_recursion_with_scores(y, a - step, b, c, h0=h0)
        elif channel == "phi":
            state_plus = model._forward_recursion_with_scores(y, a, b + step, c, h0=h0)
            state_minus = model._forward_recursion_with_scores(y, a, b - step, c, h0=h0)
        elif channel == "g":
            state_plus = model._forward_recursion_with_scores(y, a, b, c + step, h0=h0)
            state_minus = model._forward_recursion_with_scores(y, a, b, c - step, h0=h0)
        else:
            raise ValueError(f"Unsupported channel {channel!r}.")

        grad[row] = (
            model._loss_from_state(y, state_plus["h"])
            - model._loss_from_state(y, state_minus["h"])
        ) / (2.0 * eps)
    return grad


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_full_xgb_pgarch_fit_runs_mse():
    y, X = make_synthetic_pgarch_data(seed=1)
    model = XGBPGARCHModel(
        loss="mse",
        n_outer_rounds=6,
        trees_per_channel_per_round=1,
        learning_rate=0.1,
        max_depth=2,
        random_state=1,
    )

    fitted = model.fit(y, X)
    h = fitted.predict_variance(y, X)

    assert fitted is model
    assert model.is_fitted_
    assert model.booster_mu_ is not None
    assert model.booster_phi_ is not None
    assert model.booster_g_ is not None
    assert model.initializer_ is not None
    assert model.baseline_train_ is not None
    assert model.init_method_ == "linear_pgarch"
    assert model.n_features_in_ == X.shape[1]
    assert model.a0_ is not None
    assert model.b0_ is not None
    assert model.c0_ is not None
    assert model.channel_history_
    assert np.isfinite(model.train_loss_)
    assert h.shape == y.shape
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_full_xgb_pgarch_fit_runs_qlike():
    y, X = make_synthetic_pgarch_data(seed=2)
    model = XGBPGARCHModel(
        loss="qlike",
        n_outer_rounds=6,
        trees_per_channel_per_round=1,
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
    model = XGBPGARCHModel(random_state=3).fit(y, X)

    components = model.predict_components(X)

    assert set(components) == {"mu", "phi", "g"}
    assert np.all(components["mu"] > 0.0)
    assert np.all((components["phi"] > 0.0) & (components["phi"] < 1.0))
    assert np.all((components["g"] > 0.0) & (components["g"] < 1.0))


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_implied_garch_params_identity():
    y, X = make_synthetic_pgarch_data(seed=4)
    model = XGBPGARCHModel(random_state=4).fit(y, X)

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


def test_rowwise_gradient_mu_matches_finite_difference_small_problem():
    model, y, _, a, b, c, h0 = make_small_problem(loss="mse", seed=5)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    analytic, _ = model._rowwise_grad_hess_mu(y, state)
    fd = finite_difference_raw_score_gradient(model, y, a, b, c, channel="mu", h0=h0)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_rowwise_gradient_phi_matches_finite_difference_small_problem():
    model, y, _, a, b, c, h0 = make_small_problem(loss="mse", seed=6)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    analytic, _ = model._rowwise_grad_hess_phi(y, state)
    fd = finite_difference_raw_score_gradient(model, y, a, b, c, channel="phi", h0=h0)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_rowwise_gradient_g_matches_finite_difference_small_problem():
    model, y, _, a, b, c, h0 = make_small_problem(loss="mse", seed=7)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    analytic, _ = model._rowwise_grad_hess_g(y, state)
    fd = finite_difference_raw_score_gradient(model, y, a, b, c, channel="g", h0=h0)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_rowwise_hessian_nonnegative_and_finite_mse():
    model, y, _, a, b, c, h0 = make_small_problem(loss="mse", seed=8)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    for grad_hess_fn in (
        model._rowwise_grad_hess_mu,
        model._rowwise_grad_hess_phi,
        model._rowwise_grad_hess_g,
    ):
        _, hess = grad_hess_fn(y, state)
        assert np.all(np.isfinite(hess))
        assert np.all(hess >= 0.0)


def test_rowwise_hessian_nonnegative_and_finite_qlike():
    model, y, _, a, b, c, h0 = make_small_problem(loss="qlike", seed=9)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    for grad_hess_fn in (
        model._rowwise_grad_hess_mu,
        model._rowwise_grad_hess_phi,
        model._rowwise_grad_hess_g,
    ):
        _, hess = grad_hess_fn(y, state)
        assert np.all(np.isfinite(hess))
        assert np.all(hess >= 0.0)


def test_terminal_row_grad_hess_zero_for_all_channels():
    model, y, _, a, b, c, h0 = make_small_problem(loss="mse", seed=10)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=h0)
    for grad_hess_fn in (
        model._rowwise_grad_hess_mu,
        model._rowwise_grad_hess_phi,
        model._rowwise_grad_hess_g,
    ):
        grad, hess = grad_hess_fn(y, state)
        assert len(grad) == len(y)
        assert len(hess) == len(y)
        assert grad[-1] == 0.0
        assert hess[-1] == 0.0


def test_channel_specific_overrides_merge_into_channel_params():
    model = XGBPGARCHModel(
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=0.1,
        channel_param_overrides={"phi": {"learning_rate": 0.05, "max_depth": 2, "reg_lambda": 0.5}},
        channel_trees_per_round={"g": 2},
    )

    phi_params = model._xgb_params("phi")
    g_params = model._xgb_params("g")

    assert phi_params["eta"] == 0.05
    assert phi_params["max_depth"] == 2
    assert phi_params["lambda"] == 0.5
    assert g_params["eta"] == 0.1
    assert g_params["max_depth"] == 4
    assert g_params["lambda"] == 0.1
    assert model._num_boost_round("phi") == 1
    assert model._num_boost_round("g") == 2


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_channel_specific_tree_budget_affects_booster_rounds():
    y, X = make_synthetic_pgarch_data(seed=15)
    model = XGBPGARCHModel(
        loss="qlike",
        n_outer_rounds=3,
        trees_per_channel_per_round=1,
        channel_trees_per_round={"g": 2},
        random_state=15,
    ).fit(y, X)

    assert len(model.booster_mu_.get_dump()) == 3
    assert len(model.booster_phi_.get_dump()) == 3
    assert len(model.booster_g_.get_dump()) == 6


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_initializer_modes_supported(monkeypatch: pytest.MonkeyPatch):
    y, X = make_synthetic_pgarch_data(seed=11)

    linear_model = XGBPGARCHModel(init_method="linear_pgarch", random_state=11).fit(y, X)
    assert linear_model.init_method_ == "linear_pgarch"

    const_model = XGBPGARCHModel(init_method="intercept_only_pgarch", random_state=11).fit(y, X)
    assert const_model.init_method_ == "intercept_only_pgarch"

    monkeypatch.setattr(xgb_pgarch_full_module, "PGARCHLinearModel", None)
    with pytest.warns(RuntimeWarning, match="Falling back to intercept-only"):
        fallback_model = XGBPGARCHModel(init_method="linear_pgarch", random_state=11).fit(y, X)
    assert fallback_model.init_method_ == "intercept_only_pgarch"


def test_floor_active_transition_zeros_local_impulses_and_rho():
    model = XGBPGARCHModel(
        loss="mse",
        h_min=0.1,
        mu_min=1e-12,
        phi_min=1e-4,
        phi_max=1.0 - 1e-4,
        g_min=1e-4,
        g_max=1.0 - 1e-4,
        init_method="intercept_only_pgarch",
        random_state=12,
    )
    y = np.array([1e-8, 2e-8, 3e-8], dtype=float)
    a = np.array([-30.0, -30.0, -30.0], dtype=float)
    b = np.array([8.0, 8.0, 8.0], dtype=float)
    c = np.array([8.0, 8.0, 8.0], dtype=float)

    state = model._forward_recursion_with_scores(y, a, b, c, h0=0.1)

    assert state["h"][1] == model.h_min
    assert state["floor_active"][0]
    assert state["rho"][0] == 0.0
    assert state["local_impulse_mu"][0] == 0.0
    assert state["local_impulse_phi"][0] == 0.0
    assert state["local_impulse_g"][0] == 0.0


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_score_metrics_supported():
    y, X = make_synthetic_pgarch_data(seed=13)
    model = XGBPGARCHModel(random_state=13).fit(y, X)

    qlike = model.score(y, X, metric="qlike")
    mse = model.score(y, X, metric="mse")
    rmse = model.score(y, X, metric="rmse")

    assert np.isfinite(qlike)
    assert np.isfinite(mse)
    assert np.isfinite(rmse)
    assert np.isclose(rmse, np.sqrt(mse))


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_invalid_input_raises():
    y, X = make_synthetic_pgarch_data(seed=14)
    model = XGBPGARCHModel()

    with pytest.raises(ValueError, match="Length mismatch"):
        model.fit(y[:-1], X)

    y_nan = y.copy()
    y_nan[0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN or inf"):
        model.fit(y_nan, X)

    with pytest.raises(ValueError, match="At least 3 observations"):
        model.fit(y[:2], X[:2])


def test_invalid_channel_specific_overrides_raise():
    with pytest.raises(ValueError, match="Unsupported channel override"):
        XGBPGARCHModel(channel_param_overrides={"bad": {"max_depth": 2}})

    with pytest.raises(ValueError, match="Unsupported per-channel XGBoost params"):
        XGBPGARCHModel(channel_param_overrides={"phi": {"bad_param": 1}})

    with pytest.raises(ValueError, match="channel_trees_per_round values must be positive"):
        XGBPGARCHModel(channel_trees_per_round={"phi": 0})
