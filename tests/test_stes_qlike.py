import numpy as np
import pandas as pd

from volatility_forecast.model.stes_model import STESModel


def make_dummy_data(n=180, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    X = pd.DataFrame(
        {
            "feat1": np.sin(2 * np.pi * t / 30) + rng.normal(0.0, 0.1, n),
            "feat2": rng.normal(0.0, 1.0, n),
            "const": 1.0,
        }
    )

    vol = 0.01 * (1.0 + 0.4 * np.sin(2 * np.pi * t / 45))
    returns = rng.normal(0.0, 1.0, n) * vol
    y = returns**2
    return X, pd.Series(y, name="y"), pd.Series(returns, name="returns")


def test_stes_qlike_scalar_gradient_matches_finite_difference():
    X, y, returns = make_dummy_data(n=80, seed=1)
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    r_np = returns.to_numpy(dtype=float)
    params = np.array([0.2, -0.15, 0.05], dtype=float)

    obj, grad = STESModel._scalar_objective_and_grad(
        params,
        r_np,
        X_np,
        y_np,
        0,
        len(y_np),
        None,
        loss="qlike",
        qlike_epsilon=1e-8,
    )

    assert np.isfinite(obj)
    assert np.all(np.isfinite(grad))

    fd = np.zeros_like(params)
    eps = 1e-6
    for j in range(len(params)):
        step = np.zeros_like(params)
        step[j] = eps
        f_plus, _ = STESModel._scalar_objective_and_grad(
            params + step,
            r_np,
            X_np,
            y_np,
            0,
            len(y_np),
            None,
            loss="qlike",
            qlike_epsilon=1e-8,
        )
        f_minus, _ = STESModel._scalar_objective_and_grad(
            params - step,
            r_np,
            X_np,
            y_np,
            0,
            len(y_np),
            None,
            loss="qlike",
            qlike_epsilon=1e-8,
        )
        fd[j] = (f_plus - f_minus) / (2.0 * eps)

    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-5)


def test_stes_fit_predict_qlike_smoke():
    X, y, returns = make_dummy_data(n=180, seed=2)
    model = STESModel(loss="qlike", random_state=42)

    model.fit(X.iloc[:140], y.iloc[:140], returns=returns.iloc[:140])
    preds = model.predict(X.iloc[140:], returns=returns.iloc[140:])

    assert len(preds) == 40
    assert np.isfinite(preds).all()
    assert (preds > 0).all()


def test_stes_run_cv_qlike_returns_sorted_scores():
    X, y, returns = make_dummy_data(n=120, seed=3)
    model = STESModel(loss="qlike", random_state=42)
    model._set_schema(list(X.columns), X.shape[1])

    results = model.run_cv(
        X.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        returns=returns.to_numpy(dtype=float),
        param_grid=[{"l2_reg": 0.0}, {"l2_reg": 0.01}],
        n_splits=2,
    )

    assert len(results) == 2
    assert results[0][0] <= results[1][0]
    assert all(np.isfinite(score) for score, _ in results)
