# Strict Codex Prompt — Full XGB-PGARCH

You are working inside an existing Python package named `volatility-forecast`.

Your task is to implement the **full generalized XGB-PGARCH model**, in which all three PGARCH channels are learned by coordinated XGBoost models:

- level channel: raw score `a_t -> mu_t`
- persistence channel: raw score `b_t -> phi_t`
- innovation-share channel: raw score `c_t -> g_t`

This is the next-stage generalization after:
- `PGARCHLinearModel`
- `XGBGPGARCHModel` (the g-only boosted version)

Your implementation should preserve the PGARCH structure, use exact row-wise gradients, use positive diagonal Hessian surrogates, and train the three channels in **block-coordinate boosting**.

IMPORTANT CONTEXT
- The package already has a `volatility_forecast/model/` module with existing models such as:
  - `STESModel`
  - `XGBoostSTESModel`
  - `VolGRUModel`
  - `PGARCHLinearModel`
  - `XGBGPGARCHModel`
- Reuse nearby conventions, helper logic, and API design when appropriate.
- Inspect existing files before coding, especially:
  - `volatility_forecast/model/pgarch_linear_model.py`
  - `volatility_forecast/model/xgb_pgarch_model.py`
  - related tests
- Prefer reusing mathematically correct helper logic rather than duplicating it.
- Use NumPy, SciPy, and XGBoost only.
- Do not use PyTorch or JAX in this task.

---

## GOAL

Implement a new class:

## `XGBPGARCHModel`

This is the **full three-channel generalized boosted PGARCH model**.

The recursion is

\[
h_{t+1}
=
(1-\phi_t)\mu_t
+
\phi_t\Big(g_t y_t + (1-g_t)h_t\Big),
\qquad t=0,\dots,T-2,
\]

with three raw-score channels

\[
a_t = a_t^{(0)} + F_\mu(x_t),
\qquad
b_t = b_t^{(0)} + F_\phi(x_t),
\qquad
c_t = c_t^{(0)} + F_g(x_t),
\]

and links

\[
\mu_t = \mu_{\min} + \operatorname{softplus}(a_t),
\]

\[
\phi_t = \phi_{\min} + (\phi_{\max} - \phi_{\min}) \sigma(b_t),
\]

\[
g_t = g_{\min} + (g_{\max} - g_{\min}) \sigma(c_t).
\]

The model must:
- use **row \(t\) to affect \(h[t+1]\)**
- treat `h[0]` as a fixed causal warm start unless explicit carryover is provided
- exclude `t = 0` from the optimized loss
- return zero gradient and zero Hessian for the terminal row

---

## REQUIRED FILES

Create or modify these files:

1. `volatility_forecast/model/xgb_pgarch_full_model.py`
2. `volatility_forecast/model/__init__.py`
3. `tests/test_xgb_pgarch_full_model.py`

Optional helpers only if clearly justified:
- `volatility_forecast/model/_xgb_pgarch_full_utils.py`

If helpers are small, keep them inside the main file.

---

## PUBLIC CLASS TO IMPLEMENT

In `volatility_forecast/model/xgb_pgarch_full_model.py`, implement exactly this class:

```python
class XGBPGARCHModel:
    def __init__(
        self,
        loss: str = "qlike",
        n_outer_rounds: int = 25,
        trees_per_channel_per_round: int = 1,
        learning_rate: float = 0.02,
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        mu_min: float = 1e-12,
        phi_min: float = 1e-4,
        phi_max: float = 1.0 - 1e-4,
        g_min: float = 1e-4,
        g_max: float = 1.0 - 1e-4,
        h_min: float = 1e-12,
        init_method: str = "linear_pgarch",
        init_model: object | None = None,
        channel_update_order: tuple[str, ...] = ("mu", "phi", "g"),
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = None,
        verbosity: int = 0,
    ) -> None:
        ...
```

---

## REQUIRED PUBLIC METHODS

Implement exactly these public methods and signatures:

```python
def fit(
    self,
    y: np.ndarray,
    X: np.ndarray,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
) -> "XGBPGARCHModel":
    ...

def predict_variance(
    self,
    y: np.ndarray,
    X: np.ndarray,
    *,
    h0: float | None = None,
) -> np.ndarray:
    ...

def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
    ...

def implied_garch_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
    ...

def score(self, y: np.ndarray, X: np.ndarray, metric: str = "qlike") -> float:
    ...
```

---

## REQUIRED INTERNAL METHODS

Implement these internal methods with exact signatures:

```python
def _validate_inputs(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

def _fit_initializer(self, y: np.ndarray, X: np.ndarray) -> object:
    ...

def _extract_baseline_sequences(
    self,
    y: np.ndarray,
    X: np.ndarray,
    initializer: object,
) -> dict[str, np.ndarray]:
    ...

def _initialize_raw_scores(
    self,
    baseline: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

def _predict_raw_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

def _link_mu(self, a: np.ndarray) -> np.ndarray:
    ...

def _link_phi(self, b: np.ndarray) -> np.ndarray:
    ...

def _link_g(self, c: np.ndarray) -> np.ndarray:
    ...

def _link_mu_prime(self, a: np.ndarray) -> np.ndarray:
    ...

def _link_phi_prime(self, b: np.ndarray) -> np.ndarray:
    ...

def _link_g_prime(self, c: np.ndarray) -> np.ndarray:
    ...

def _inverse_link_mu(self, mu: np.ndarray) -> np.ndarray:
    ...

def _inverse_link_phi(self, phi: np.ndarray) -> np.ndarray:
    ...

def _inverse_link_g(self, g: np.ndarray) -> np.ndarray:
    ...

def _forward_recursion_with_scores(
    self,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    h0: float,
) -> dict[str, np.ndarray]:
    ...

def _loss_from_state(self, y: np.ndarray, h: np.ndarray) -> float:
    ...

def _loss_grad_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
    ...

def _loss_hess_weight_wrt_h(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
    ...

def _backward_adjoint(
    self,
    y: np.ndarray,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    ...

def _rowwise_grad_hess_mu(
    self,
    y: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ...

def _rowwise_grad_hess_phi(
    self,
    y: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ...

def _rowwise_grad_hess_g(
    self,
    y: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ...

def _make_dmatrix(
    self,
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> xgb.DMatrix:
    ...

def _fit_channel_update(
    self,
    channel: str,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h0: float,
) -> tuple[np.ndarray, Any]:
    ...
```

You may add additional private helpers, but keep these exact names.

---

## FITTED ATTRIBUTES

After `fit`, set at least these attributes:

```python
self.booster_mu_: Any
self.booster_phi_: Any
self.booster_g_: Any
self.initializer_: object
self.init_method_: str
self.n_features_in_: int
self.is_fitted_: bool
self.train_loss_: float
self.best_iteration_: int | None
self.feature_names_: list[str] | None
self.baseline_train_: dict[str, np.ndarray]
self.a0_: np.ndarray
self.b0_: np.ndarray
self.c0_: np.ndarray
self.channel_history_: list[dict[str, Any]]
```

If repo conventions differ, keep these names available anyway.

---

## FIXED DESIGN DECISIONS

These choices are already made. Do not deviate.

1. Use **row \(t\) affects \(h[t+1]\)** indexing everywhere.
2. Keep training row count equal to `X.shape[0]`.
3. Return **zero gradient and zero Hessian for the terminal row**.
4. Treat `h[0]` as a fixed causal warm start; losses exclude `t = 0`.
5. Use **exact row-wise gradients**.
6. Use **positive diagonal Hessian surrogates**, not exact raw-score Hessians.
7. Use **block-coordinate boosting** over channels.
8. Use **Fisher-style QLIKE curvature**
   \[
   w_t = \operatorname{clip}\!\left(\frac{1}{N h_t^2}, \epsilon, w_{\max}\right)
   \]
9. Default initializer should be **linear PGARCH**, with fallback to intercept-only PGARCH if needed.

---

## MATHEMATICAL REQUIREMENTS

### Recursion
Use

\[
q_t = g_t y_t + (1-g_t) h_t,
\]

\[
h_{t+1} = (1-\phi_t)\mu_t + \phi_t q_t.
\]

### Local channel derivatives

Define

\[
A_t = \frac{\partial \mu_t}{\partial a_t},
\qquad
B_t = \frac{\partial \phi_t}{\partial b_t},
\qquad
C_t = \frac{\partial g_t}{\partial c_t}.
\]

Then the local impulses are

\[
\delta_t^{(\mu)} = (1-\phi_t) A_t,
\]

\[
\delta_t^{(\phi)} = (q_t - \mu_t) B_t,
\]

\[
\delta_t^{(g)} = \phi_t (y_t - h_t) C_t.
\]

### State propagation

When the variance floor is inactive,

\[
\rho_t = \frac{\partial h_{t+1}}{\partial h_t} = \phi_t (1-g_t).
\]

If the next-step state is clipped to `h_min`, then:
- `h[t+1] = h_min`
- `rho_t = 0`
- all local impulses at row `t` are zero

You must implement this floor-consistently.

### Adjoint recursion

Let \(u_t = \partial \ell_t / \partial h_t\). Then

\[
\lambda_{T-1} = u_{T-1},
\]

\[
\lambda_t = u_t + \rho_t \lambda_{t+1},
\qquad t=T-2,\dots,0.
\]

### Exact row-wise gradients

For each channel \(z \in \{\mu,\phi,g\}\),

\[
G_t^{(z)} = \lambda_{t+1}\delta_t^{(z)},
\qquad t=0,\dots,T-2,
\]

and terminal row gradients are zero.

### Hessian surrogates

#### MSE
Use Gauss–Newton style positive diagonal curvature:

\[
H_t^{(z)}
\approx
\sum_{s=t+1}^{T-1}
\frac{2}{N}
\left(
\frac{\partial h_s}{\partial \text{raw}_t^{(z)}}
\right)^2.
\]

#### QLIKE
Use Fisher-style positive weight:

\[
w_s = \operatorname{clip}\!\left(\frac{1}{N h_s^2}, \epsilon, w_{\max}\right)
\]

and

\[
H_t^{(z)}
\approx
\sum_{s=t+1}^{T-1}
w_s
\left(
\frac{\partial h_s}{\partial \text{raw}_t^{(z)}}
\right)^2.
\]

You do NOT need the exact second derivative with respect to raw scores.

---

## TRAINING STRATEGY

Use **block-coordinate boosting**.

For each outer round:
1. update the `mu` channel
2. update the `phi` channel
3. update the `g` channel

unless a different `channel_update_order` is supplied.

Each channel update should:
- hold the other two channels fixed
- compute the current recursion state
- compute exact row-wise gradients for that channel
- compute positive diagonal Hessian surrogates for that channel
- fit one or more XGBoost trees for that channel
- update only that raw-score channel

Use one `xgb.Booster` per channel:
- `booster_mu_`
- `booster_phi_`
- `booster_g_`

The implementation may use `xgb.train` incrementally per channel update.

---

## INITIALIZATION

### Supported modes
- `init_method="linear_pgarch"`
- `init_method="intercept_only_pgarch"`

### Baseline extraction
From the initializer obtain:
- baseline `mu`
- baseline `phi`
- baseline `g`
- baseline `h`

Then convert baseline components to raw-score baselines:
- `a0 = inverse_link_mu(mu)`
- `b0 = inverse_link_phi(phi)`
- `c0 = inverse_link_g(g)`

These serve as the starting raw-score sequences before boosting.

If `PGARCHLinearModel` is unavailable and `init_method="linear_pgarch"`, fall back to intercept-only PGARCH with a clear warning.

---

## LOSS FUNCTIONS

Support:
- `"mse"`
- `"rmse"` for reporting only; optimize as MSE internally
- `"qlike"`

All optimized losses exclude `t = 0`.

### MSE
\[
\frac{1}{N}\sum_{t=1}^{T-1}(h_t-y_t)^2
\]

### QLIKE
\[
\frac{1}{N}\sum_{t=1}^{T-1}\left(\log h_t + \frac{y_t}{h_t}\right)
\]

---

## NUMERICAL STABILITY REQUIREMENTS

- Use stable sigmoid, softplus, and inverse-link implementations.
- Use clipping on Hessian weights:
  - lower bound `1e-12`
  - upper bound `1e12`
- Use floor-consistent derivatives.
- Do not let the terminal row produce nonzero grad/hess.
- Keep `h[0]` causal.
- Preserve feature schema consistency between train and predict.
- Keep `predict_variance` as a recursion-aware filtering path using the supplied lagged `y`.

---

## PREDICTION BEHAVIOR

### `predict_variance(y, X, h0=None)`
- if `h0` is provided, use it
- else use causal warm start `max(y[0], h_min)`
- compute raw scores
- run the full recursion
- return `h`

### `predict_components(X)`
Return:
- `"mu"`
- `"phi"`
- `"g"`

with arrays aligned rowwise to `X[t]`.

### `implied_garch_params(X)`
Return:
- `"omega"`
- `"alpha"`
- `"beta"`

where

\[
\omega_t = (1-\phi_t)\mu_t,
\quad
\alpha_t = \phi_t g_t,
\quad
\beta_t = \phi_t(1-g_t).
\]

### `score(y, X, metric="qlike")`
Support:
- `"qlike"`
- `"mse"`
- `"rmse"`

and exclude `t = 0`.

---

## EXPORT

Update `volatility_forecast/model/__init__.py` to export:

```python
from .xgb_pgarch_full_model import XGBPGARCHModel
```

---

## TESTS

Create `tests/test_xgb_pgarch_full_model.py`.

Implement tests for at least the following:

1. `test_full_xgb_pgarch_fit_runs_mse()`
2. `test_full_xgb_pgarch_fit_runs_qlike()`
3. `test_predict_components_bounds()`
4. `test_implied_garch_params_identity()`
5. `test_rowwise_gradient_mu_matches_finite_difference_small_problem()`
6. `test_rowwise_gradient_phi_matches_finite_difference_small_problem()`
7. `test_rowwise_gradient_g_matches_finite_difference_small_problem()`
8. `test_rowwise_hessian_nonnegative_and_finite_mse()`
9. `test_rowwise_hessian_nonnegative_and_finite_qlike()`
10. `test_terminal_row_grad_hess_zero_for_all_channels()`
11. `test_initializer_modes_supported()`
12. `test_floor_active_transition_zeros_local_impulses_and_rho()`
13. `test_score_metrics_supported()`
14. `test_invalid_input_raises()`

Finite-difference tests should be run on tiny synthetic problems and validate the exact row-wise gradients channel-by-channel.

---

## STYLE REQUIREMENTS

- Use type hints throughout
- Use docstrings for public methods
- Keep the code modular and readable
- Add inline comments for recursive derivative logic
- Follow existing repo style where reasonable
- Do not refactor unrelated models
- Do not change existing model behavior outside the new class
- Do not add notebooks or benchmarking scripts

---

## AFTER CODING

At the end, provide:
1. a concise summary of files changed
2. a concise summary of the forward recursion and the three rowwise gradient formulas
3. a concise summary of the Hessian surrogates used
4. any assumptions about causal warm starts and terminal-row alignment
5. any fallback behavior for missing `PGARCHLinearModel`

Now inspect the existing repo structure and implement the feature exactly as specified.
