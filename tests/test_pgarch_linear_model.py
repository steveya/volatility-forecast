import numpy as np
import pytest

from volatility_forecast.model.pgarch_linear_model import (
    PGARCHLinearModel,
    _sigmoid,
    _softplus,
)


def make_synthetic_pgarch_data(
    n: int = 160,
    d: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X_aug = np.column_stack([np.ones(n, dtype=float), X])

    w_mu = np.array([-2.0, 0.25, -0.15], dtype=float)[: d + 1]
    w_phi = np.array([2.0, 0.15, 0.10], dtype=float)[: d + 1]
    w_g = np.array([-2.5, -0.20, 0.20], dtype=float)[: d + 1]

    y = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    y[0] = 0.04
    h[0] = 0.04

    for t in range(1, n):
        x = X_aug[t - 1]
        mu = 1e-12 + float(_softplus(np.dot(w_mu, x)))
        phi = 1e-4 + (1.0 - 2e-4) * float(_sigmoid(np.dot(w_phi, x)))
        g = 1e-4 + (1.0 - 2e-4) * float(_sigmoid(np.dot(w_g, x)))
        h[t] = (1.0 - phi) * mu + phi * (g * y[t - 1] + (1.0 - g) * h[t - 1])
        y[t] = max(h[t] * (0.8 + 0.4 * rng.random()), 1e-8)

    return y, X


def make_small_problem(
    loss: str,
    seed: int = 0,
) -> tuple[PGARCHLinearModel, np.ndarray, np.ndarray, np.ndarray]:
    y, X = make_synthetic_pgarch_data(n=7, d=2, seed=seed)
    model = PGARCHLinearModel(
        loss=loss,
        lambda_mu=0.05,
        lambda_phi=0.07,
        lambda_g=0.09,
        standardize_features=False,
        random_state=seed,
    )
    X_aug = model._add_intercept(X)
    theta = model._initialize_params(y, X_aug)
    theta = theta + np.random.default_rng(seed + 101).normal(0.0, 0.05, size=theta.shape)
    return model, y, X_aug, theta


def finite_difference_gradient(
    model: PGARCHLinearModel,
    theta: np.ndarray,
    y: np.ndarray,
    X_aug: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.zeros_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        grad[j] = (
            model._objective(theta + step, y, X_aug)
            - model._objective(theta - step, y, X_aug)
        ) / (2.0 * eps)
    return grad


def finite_difference_hessian(
    model: PGARCHLinearModel,
    theta: np.ndarray,
    y: np.ndarray,
    X_aug: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    hessian = np.zeros((theta.size, theta.size), dtype=float)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        grad_plus = model._gradient(theta + step, y, X_aug)
        grad_minus = model._gradient(theta - step, y, X_aug)
        hessian[:, j] = (grad_plus - grad_minus) / (2.0 * eps)
    return hessian


def test_pgarch_linear_fit_runs():
    y, X = make_synthetic_pgarch_data(seed=1)
    model = PGARCHLinearModel(loss="qlike", n_restarts=2, random_state=1)

    fitted = model.fit(y, X)
    h = fitted.predict_variance(y, X)

    assert fitted is model
    assert model.is_fitted_
    assert model.optimization_result_ is not None
    assert model.optimization_result_.success
    assert model.theta_ is not None
    assert model.coef_mu_ is not None
    assert model.coef_phi_ is not None
    assert model.coef_g_ is not None
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.train_loss_)
    assert h.shape == y.shape
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)


def test_predict_components_bounds():
    y, X = make_synthetic_pgarch_data(seed=2)
    model = PGARCHLinearModel(random_state=2).fit(y, X)

    components = model.predict_components(X)

    assert set(components) == {"mu", "phi", "g"}
    assert np.all(np.isfinite(components["mu"]))
    assert np.all(np.isfinite(components["phi"]))
    assert np.all(np.isfinite(components["g"]))
    assert np.all(components["mu"] > 0.0)
    assert np.all((components["phi"] > 0.0) & (components["phi"] < 1.0))
    assert np.all((components["g"] > 0.0) & (components["g"] < 1.0))


def test_implied_garch_params_identity():
    y, X = make_synthetic_pgarch_data(seed=3)
    model = PGARCHLinearModel(random_state=3).fit(y, X)

    components = model.predict_components(X)
    implied = model.implied_garch_params(X)

    assert set(implied) == {"omega", "alpha", "beta"}
    assert np.allclose(
        implied["alpha"] + implied["beta"],
        components["phi"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        implied["omega"],
        (1.0 - components["phi"]) * components["mu"],
        rtol=1e-10,
        atol=1e-10,
    )


def test_gradient_matches_finite_difference_mse():
    model, y, X_aug, theta = make_small_problem(loss="mse", seed=4)

    analytic = model._gradient(theta, y, X_aug)
    fd = finite_difference_gradient(model, theta, y, X_aug)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_gradient_matches_finite_difference_qlike():
    model, y, X_aug, theta = make_small_problem(loss="qlike", seed=5)

    analytic = model._gradient(theta, y, X_aug)
    fd = finite_difference_gradient(model, theta, y, X_aug)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_hessian_matches_finite_difference_small_problem():
    model, y, X_aug, theta = make_small_problem(loss="qlike", seed=6)

    analytic = model._hessian(theta, y, X_aug)
    fd = finite_difference_hessian(model, theta, y, X_aug)

    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-5)


def test_intercept_only_like_constant_garch_structure():
    y, _ = make_synthetic_pgarch_data(n=120, d=2, seed=7)
    X = np.empty((len(y), 0), dtype=float)
    model = PGARCHLinearModel(loss="mse", standardize_features=False, random_state=7)

    model.fit(y, X)
    components = model.predict_components(X)
    implied = model.implied_garch_params(X)

    for key in ("mu", "phi", "g"):
        assert np.allclose(components[key], components[key][0], atol=1e-10, rtol=0.0)
    for key in ("omega", "alpha", "beta"):
        assert np.allclose(implied[key], implied[key][0], atol=1e-10, rtol=0.0)


def test_score_metrics_supported():
    y, X = make_synthetic_pgarch_data(seed=8)
    model = PGARCHLinearModel(random_state=8).fit(y, X)

    qlike = model.score(y, X, metric="qlike")
    mse = model.score(y, X, metric="mse")
    rmse = model.score(y, X, metric="rmse")

    assert np.isfinite(qlike)
    assert np.isfinite(mse)
    assert np.isfinite(rmse)
    assert np.isclose(rmse, np.sqrt(mse))


def test_invalid_input_raises():
    y, X = make_synthetic_pgarch_data(seed=9)
    model = PGARCHLinearModel()

    with pytest.raises(ValueError, match="Length mismatch"):
        model.fit(y[:-1], X)

    y_nan = y.copy()
    y_nan[0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN or inf"):
        model.fit(y_nan, X)

    with pytest.raises(ValueError, match="At least 3 observations"):
        model.fit(y[:2], X[:2])
