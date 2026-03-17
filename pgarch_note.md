# PGARCH: Motivation, Model Construction, and Estimation

## Introduction

### Why this model family was needed

Our starting point was an empirical puzzle. In out-of-sample variance forecasting experiments, **GARCH(1,1)** repeatedly outperformed both **STES** and **XGBSTES** under the **QLIKE** evaluation metric, even when all competing models were fit by minimizing a QLIKE-based objective. At first glance this is surprising. STES and XGBSTES appear more flexible: they allow the innovation weight to vary with predictors and, in the case of XGBSTES, allow a highly nonlinear mapping from predictors to the smoothing gate. Since GARCH(1,1) may be written as a simple first-order variance recursion, one might expect it to be nested inside, or at least dominated by, the richer adaptive models.

That intuition turns out to be incomplete. The key issue is that **GARCH(1,1) separates three distinct roles that STES-type recursions tend to entangle**:

1. **long-run scale / anchor level**,  
2. **overall persistence**, and  
3. **innovation share within the persistent component**.

This decomposition is easiest to see algebraically.

A standard GARCH(1,1) recursion is

\[
h_t = \omega + \alpha y_{t-1} + \beta h_{t-1},
\]

where \(y_t\) is a nonnegative volatility target, typically \(r_t^2\) or a realized variance proxy, and \(h_t\) is the one-step-ahead conditional variance forecast.

Now define

\[
\phi = \alpha + \beta, \qquad
g = \frac{\alpha}{\alpha+\beta}, \qquad
\mu = \frac{\omega}{1-\alpha-\beta},
\]

under the usual stationary and positivity conditions

\[
\omega > 0, \qquad \alpha \ge 0, \qquad \beta \ge 0, \qquad \alpha+\beta < 1.
\]

Then GARCH(1,1) can be rewritten as

\[
h_t
=
(1-\phi)\mu
+
\phi\Big(g y_{t-1} + (1-g)h_{t-1}\Big).
\]

This representation reveals the three structural knobs:

- \(\mu\): the long-run variance anchor,
- \(\phi\): the total persistence,
- \(g\): the innovation share inside the persistent block.

By contrast, a STES-style recursion is typically of the form

\[
h_t = g_t y_{t-1} + (1-g_t)h_{t-1},
\]

or equivalently a predictor-driven exponential smoothing recursion. Such a model only varies the **shock-vs-memory split**. It does **not** separately represent:

- an independently estimated long-run anchor,
- or a separate total persistence parameter.

This distinction matters especially under **QLIKE**.

### Why QLIKE favors GARCH-style structure

We evaluate and often fit using the QLIKE loss

\[
L_{\text{QLIKE}}(h_t, y_t)
=
\log h_t + \frac{y_t}{h_t}.
\]

This loss is well known to be especially suitable for volatility forecasting because it penalizes under-forecasting of variance sharply while remaining robust to noise in realized variance proxies. However, that same property means QLIKE strongly rewards models that get the **variance scale** and **persistence structure** approximately right. A model that occasionally drives \(h_t\) too low will be punished heavily.

This explains much of the empirical success of GARCH(1,1):

- the intercept term \(\omega\), equivalently the anchor \(\mu\), prevents the forecast from collapsing toward zero;
- \(\alpha+\beta\), equivalently \(\phi\), controls persistence independently of the innovation share;
- the model is constrained in a way that yields stable filtered variance paths.

STES and XGBSTES, despite their flexibility, may underperform because they often let a single gate bear too much responsibility. In particular, a time-varying smoothing gate must simultaneously decide:

- how much to react to the latest shock,
- how much memory to retain from the previous state,
- and indirectly what long-run level the recursion drifts toward.

That is too much to ask from a single scalar mechanism.

Even when STES or XGBSTES are fit directly on a QLIKE objective, they remain structurally misspecified relative to what QLIKE rewards. The issue is not simply optimization. It is the lack of a model decomposition that distinguishes:

\[
\text{level}, \qquad \text{persistence}, \qquad \text{innovation share}.
\]

### Motivation for PGARCH

This motivates a new model family: **Predictive GARCH (PGARCH)**.

The core idea is simple:

- keep the **GARCH-like structural decomposition** that appears to be important for QLIKE performance,
- but let the structural components be **predictor-driven and time-varying**.

Instead of predicting \((\omega_t,\alpha_t,\beta_t)\) directly, we parameterize the model in terms of the more interpretable quantities

\[
\mu_t > 0, \qquad \phi_t \in (0,1), \qquad g_t \in (0,1),
\]

and define

\[
h_t
=
(1-\phi_t)\mu_t
+
\phi_t\Big(g_t y_{t-1} + (1-g_t)h_{t-1}\Big).
\]

This formulation retains the GARCH decomposition while generalizing it.

It is useful for at least three reasons:

1. **Interpretability**  
   We can separately inspect predicted long-run level, persistence, and innovation share.

2. **Nesting structure**  
   Several familiar models are recovered as special cases:
   - intercept-only PGARCH recovers a constrained GARCH(1,1)-type recursion,
   - STES appears when \(\phi_t \equiv 1\) and the anchor is absent or fixed,
   - more flexible nonlinear variants arise when \(\mu_t,\phi_t,g_t\) are produced by machine-learning models.

3. **Better alignment with QLIKE**  
   The model class can adapt to predictors while preserving the structural features that seem essential for good QLIKE behavior.

The rest of this note formalizes the model and derives the equations needed for estimation.

---

## Model

### 1. PGARCH recursion

Let \(y_t \ge 0\) denote a nonnegative volatility target. In empirical work \(y_t\) may be:

- squared return \(r_t^2\),
- a realized variance estimate,
- or another nonnegative volatility proxy.

PGARCH is defined by the recursion

\[
h_t
=
(1-\phi_t)\mu_t
+
\phi_t\Big(g_t y_{t-1} + (1-g_t)h_{t-1}\Big),
\qquad t=1,\dots,T-1,
\]

with a fixed initial state \(h_0 \ge 0\).

It is convenient to define the intermediate quantity

\[
q_t = g_t y_{t-1} + (1-g_t)h_{t-1},
\]

so that

\[
h_t = (1-\phi_t)\mu_t + \phi_t q_t.
\]

The constraints

\[
\mu_t > 0, \qquad 0 < \phi_t < 1, \qquad 0 < g_t < 1
\]

guarantee positivity and stable first-order propagation when \(y_t \ge 0\) and \(h_0 \ge 0\).

### 2. Implied time-varying GARCH representation

Define

\[
\omega_t = (1-\phi_t)\mu_t, \qquad
\alpha_t = \phi_t g_t, \qquad
\beta_t = \phi_t(1-g_t).
\]

Then

\[
h_t = \omega_t + \alpha_t y_{t-1} + \beta_t h_{t-1}.
\]

Also,

\[
\alpha_t + \beta_t = \phi_t.
\]

Thus PGARCH may be viewed as a **time-varying GARCH recursion**, but parameterized through the more interpretable triple \((\mu_t,\phi_t,g_t)\).

### 3. Special cases

#### 3.1 Intercept-only PGARCH

If \(\mu_t = \mu\), \(\phi_t = \phi\), and \(g_t = g\) are all constant, then

\[
h_t
=
(1-\phi)\mu
+
\phi\Big(g y_{t-1} + (1-g)h_{t-1}\Big),
\]

which is exactly a constrained GARCH(1,1)-form recursion with

\[
\omega=(1-\phi)\mu,\qquad
\alpha=\phi g,\qquad
\beta=\phi(1-g).
\]

The inverse mapping is

\[
\phi = \alpha+\beta,
\qquad
g = \frac{\alpha}{\alpha+\beta},
\qquad
\mu = \frac{\omega}{1-\alpha-\beta},
\]

provided \(\alpha+\beta \in (0,1)\).

#### 3.2 STES-like case

If \(\phi_t \equiv 1\) and the anchor is dropped or fixed, then

\[
h_t = g_t y_{t-1} + (1-g_t)h_{t-1},
\]

which is the STES-style exponential smoothing form.

This shows exactly why PGARCH is the natural generalization: it adds missing structure rather than replacing the recursion wholesale.

---

## Linear PGARCH

### 1. Predictor-driven latent scores

Let \(x_{t-1}\in\mathbb{R}^d\) denote the predictor vector available at time \(t-1\), aligned so that \(x_{t-1}\) forecasts \(h_t\). Define the augmented feature vector

\[
\tilde{x}_{t-1}
=
\begin{bmatrix}
1 \\ x_{t-1}
\end{bmatrix}
\in \mathbb{R}^{d+1}.
\]

We introduce three linear latent scores:

\[
a_t = w_\mu^\top \tilde{x}_{t-1}, \qquad
b_t = w_\phi^\top \tilde{x}_{t-1}, \qquad
c_t = w_g^\top \tilde{x}_{t-1}.
\]

These are transformed via link functions to satisfy the parameter constraints.

### 2. Link functions

We use

\[
\mu_t = \mu_{\min} + \operatorname{softplus}(a_t),
\]

\[
\phi_t = \phi_{\min} + (\phi_{\max}-\phi_{\min})\sigma(b_t),
\]

\[
g_t = g_{\min} + (g_{\max}-g_{\min})\sigma(c_t),
\]

where

\[
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\operatorname{softplus}(z)=\log(1+e^z).
\]

The resulting parameter vector is

\[
\theta =
\begin{bmatrix}
w_\mu \\ w_\phi \\ w_g
\end{bmatrix}
\in \mathbb{R}^{3(d+1)}.
\]

### 3. Derivatives of the link functions

Let

\[
s_a = \sigma(a_t), \qquad s_b = \sigma(b_t), \qquad s_c = \sigma(c_t).
\]

Then

\[
\frac{\partial \mu_t}{\partial a_t} = s_a,
\]

\[
\frac{\partial \phi_t}{\partial b_t}
=
(\phi_{\max}-\phi_{\min}) s_b(1-s_b),
\]

\[
\frac{\partial g_t}{\partial c_t}
=
(g_{\max}-g_{\min}) s_c(1-s_c).
\]

The second derivatives are

\[
\frac{\partial^2 \mu_t}{\partial a_t^2}
=
s_a(1-s_a),
\]

\[
\frac{\partial^2 \phi_t}{\partial b_t^2}
=
(\phi_{\max}-\phi_{\min}) s_b(1-s_b)(1-2s_b),
\]

\[
\frac{\partial^2 g_t}{\partial c_t^2}
=
(g_{\max}-g_{\min}) s_c(1-s_c)(1-2s_c).
\]

---

## Loss functions

We consider two training losses.

### 1. Mean squared error

\[
L_{\mathrm{MSE}}(\theta)
=
\frac{1}{N}
\sum_{t=1}^{T-1}
(y_t - h_t)^2.
\]

We exclude \(t=0\) because \(h_0\) is treated as a fixed causal warm start, not a parameter-driven forecast.

RMSE is

\[
L_{\mathrm{RMSE}}(\theta)
=
\sqrt{L_{\mathrm{MSE}}(\theta)},
\]

but in deterministic optimization it is preferable to optimize MSE directly.

### 2. QLIKE

\[
L_{\mathrm{QLIKE}}(\theta)
=
\frac{1}{N}
\sum_{t=1}^{T-1}
\left(
\log h_t + \frac{y_t}{h_t}
\right),
\]

with \(N=T-1\).

For a generic per-time loss \(\ell_t(h_t,y_t)\), define

\[
u_t := \frac{\partial \ell_t}{\partial h_t},
\qquad
v_t := \frac{\partial^2 \ell_t}{\partial h_t^2}.
\]

Then:

#### MSE
\[
u_t = \frac{2(h_t-y_t)}{N},
\qquad
v_t = \frac{2}{N}.
\]

#### QLIKE
\[
u_t
=
\frac{1}{N}\left(
\frac{1}{h_t} - \frac{y_t}{h_t^2}
\right),
\]

\[
v_t
=
\frac{1}{N}\left(
-\frac{1}{h_t^2} + \frac{2y_t}{h_t^3}
\right).
\]

---

## First-order derivatives for linear PGARCH

We now derive the recursive Jacobian with respect to \(\theta\).

### 1. Block derivative vectors

Define the block-embedded feature vectors

\[
d^\mu_t
=
\begin{bmatrix}
\tilde{x}_{t-1}\\0\\0
\end{bmatrix},
\qquad
d^\phi_t
=
\begin{bmatrix}
0\\\tilde{x}_{t-1}\\0
\end{bmatrix},
\qquad
d^g_t
=
\begin{bmatrix}
0\\0\\\tilde{x}_{t-1}
\end{bmatrix}.
\]

Then

\[
J^\mu_t := \frac{\partial \mu_t}{\partial \theta}
=
\frac{\partial \mu_t}{\partial a_t} d^\mu_t,
\]

\[
J^\phi_t := \frac{\partial \phi_t}{\partial \theta}
=
\frac{\partial \phi_t}{\partial b_t} d^\phi_t,
\]

\[
J^g_t := \frac{\partial g_t}{\partial \theta}
=
\frac{\partial g_t}{\partial c_t} d^g_t.
\]

### 2. Jacobian recursion for \(q_t\)

Recall

\[
q_t = g_t y_{t-1} + (1-g_t)h_{t-1}.
\]

Differentiating,

\[
J^q_t
=
\frac{\partial q_t}{\partial \theta}
=
(1-g_t)J_{t-1} + (y_{t-1}-h_{t-1})J^g_t,
\]

where

\[
J_t := \frac{\partial h_t}{\partial \theta}.
\]

### 3. Jacobian recursion for \(h_t\)

From

\[
h_t = (1-\phi_t)\mu_t + \phi_t q_t,
\]

we obtain

\[
J_t
=
(1-\phi_t)J^\mu_t
+
(q_t-\mu_t)J^\phi_t
+
\phi_t J^q_t.
\]

This is the key first-order recursion.

With fixed \(h_0\), the initial condition is

\[
J_0 = 0.
\]

---

## Second-order derivatives for linear PGARCH

Let

\[
H_t := \frac{\partial^2 h_t}{\partial \theta \partial \theta^\top}.
\]

We also define

\[
H^\mu_t
=
\frac{\partial^2 \mu_t}{\partial \theta\partial\theta^\top},
\qquad
H^\phi_t
=
\frac{\partial^2 \phi_t}{\partial \theta\partial\theta^\top},
\qquad
H^g_t
=
\frac{\partial^2 g_t}{\partial \theta\partial\theta^\top}.
\]

### 1. Link Hessians

Because each channel depends on a single scalar score, we obtain rank-one Hessians:

\[
H^\mu_t
=
\frac{\partial^2 \mu_t}{\partial a_t^2}
\, d^\mu_t (d^\mu_t)^\top,
\]

\[
H^\phi_t
=
\frac{\partial^2 \phi_t}{\partial b_t^2}
\, d^\phi_t (d^\phi_t)^\top,
\]

\[
H^g_t
=
\frac{\partial^2 g_t}{\partial c_t^2}
\, d^g_t (d^g_t)^\top.
\]

### 2. Hessian recursion for \(q_t\)

Differentiate

\[
J^q_t = (1-g_t)J_{t-1} + (y_{t-1}-h_{t-1})J^g_t.
\]

Using the product rule,

\[
H^q_t
=
(1-g_t)H_{t-1}
+
(y_{t-1}-h_{t-1})H^g_t
-
J^g_t J_{t-1}^\top
-
J_{t-1}(J^g_t)^\top.
\]

### 3. Hessian recursion for \(h_t\)

Differentiate

\[
J_t
=
(1-\phi_t)J^\mu_t
+
(q_t-\mu_t)J^\phi_t
+
\phi_t J^q_t.
\]

This yields

\[
H_t
=
(1-\phi_t)H^\mu_t
+
(q_t-\mu_t)H^\phi_t
+
\phi_t H^q_t
+
J^\phi_t (J^q_t-J^\mu_t)^\top
+
(J^q_t-J^\mu_t)(J^\phi_t)^\top.
\]

With fixed \(h_0\),

\[
H_0 = 0.
\]

---

## Gradient and Hessian of the training objectives

### 1. General formulas

By the chain rule,

\[
\nabla_\theta L
=
\sum_{t=1}^{T-1}
u_t J_t.
\]

Similarly,

\[
\nabla_\theta^2 L
=
\sum_{t=1}^{T-1}
\left(
v_t J_t J_t^\top + u_t H_t
\right).
\]

These formulas specialize directly to MSE and QLIKE.

### 2. MSE

Since

\[
u_t = \frac{2(h_t-y_t)}{N},
\qquad
v_t = \frac{2}{N},
\]

we obtain

\[
\nabla_\theta L_{\mathrm{MSE}}
=
\frac{2}{N}
\sum_{t=1}^{T-1}
(h_t-y_t)J_t,
\]

and

\[
\nabla_\theta^2 L_{\mathrm{MSE}}
=
\frac{2}{N}
\sum_{t=1}^{T-1}
\left(
J_tJ_t^\top + (h_t-y_t)H_t
\right).
\]

### 3. QLIKE

Since

\[
u_t
=
\frac{1}{N}
\left(
\frac{1}{h_t} - \frac{y_t}{h_t^2}
\right),
\]

\[
v_t
=
\frac{1}{N}
\left(
-\frac{1}{h_t^2} + \frac{2y_t}{h_t^3}
\right),
\]

we obtain

\[
\nabla_\theta L_{\mathrm{QLIKE}}
=
\frac{1}{N}
\sum_{t=1}^{T-1}
\left(
\frac{1}{h_t} - \frac{y_t}{h_t^2}
\right)J_t,
\]

and

\[
\nabla_\theta^2 L_{\mathrm{QLIKE}}
=
\frac{1}{N}
\sum_{t=1}^{T-1}
\left[
\left(
-\frac{1}{h_t^2} + \frac{2y_t}{h_t^3}
\right)
J_tJ_t^\top
+
\left(
\frac{1}{h_t} - \frac{y_t}{h_t^2}
\right)
H_t
\right].
\]

These are the equations needed for exact deterministic optimization of the linear PGARCH model, provided the active path stays away from a hard variance floor.

---

## XGB-g-PGARCH: a first boosted nonlinear extension

A full nonlinear PGARCH could let all three channels be learned by flexible machine-learning models. However, the first and most stable extension is to boost only the **innovation-share channel** \(g_t\), while holding \(\mu_t\) and \(\phi_t\) fixed from a baseline PGARCH initializer.

### 1. Model definition

Let \(\mu_t\) and \(\phi_t\) be fixed baseline sequences, obtained for example from:

- intercept-only PGARCH, or
- linear PGARCH.

Then define a raw score sequence

\[
c_t = c_t^{(0)} + F_g(x_t),
\]

where \(F_g\) is a boosted tree ensemble and \(c_t^{(0)}\) is the baseline raw score corresponding to the baseline \(g_t\).

The gate is

\[
g_t
=
g_{\min} + (g_{\max}-g_{\min})\sigma(c_t).
\]

The recursion is

\[
h_{t+1}
=
(1-\phi_t)\mu_t
+
\phi_t\Big(g_t y_t + (1-g_t)h_t\Big).
\]

This indexing is especially natural for XGBoost: **row \(t\)** produces a score that affects the **next-step forecast \(h_{t+1}\)**.

### 2. Local derivatives

Define

\[
C_t := \frac{\partial g_t}{\partial c_t}
=
(g_{\max}-g_{\min})\sigma(c_t)(1-\sigma(c_t)).
\]

The local effect of \(c_t\) on the next-step forecast is

\[
\delta_t
:=
\frac{\partial h_{t+1}}{\partial c_t}
=
\phi_t (y_t-h_t) C_t.
\]

The state-propagation coefficient is

\[
\rho_t
:=
\frac{\partial h_{t+1}}{\partial h_t}
=
\phi_t(1-g_t).
\]

### 3. Adjoint recursion

Let the per-time loss derivative with respect to \(h_t\) be \(u_t\), as defined above for MSE or QLIKE. Then define the adjoint variable

\[
\lambda_t := \frac{\partial L}{\partial h_t},
\]

including all future dependence through the recursion.

Since \(h_t\) affects the objective both directly and through \(h_{t+1}\), we obtain the backward recursion

\[
\lambda_{T-1} = u_{T-1},
\]

\[
\lambda_t = u_t + \rho_t \lambda_{t+1},
\qquad
t=T-2,\dots,0.
\]

### 4. Exact row-wise gradient

Because row \(s\) affects the loss only through \(h_{s+1}\) and later states,

\[
G_s
=
\frac{\partial L}{\partial c_s}
=
\lambda_{s+1}\delta_s,
\qquad
s=0,\dots,T-2.
\]

The terminal row has no next-step in-sample forecast contribution, so

\[
G_{T-1}=0.
\]

This yields the exact row-wise gradient needed for a custom XGBoost objective.

### 5. Row-wise Hessian approximation

Standard XGBoost custom objectives require one diagonal Hessian value per row. The exact second derivative with respect to each raw score is algebraically available in principle but cumbersome, and may be indefinite. Therefore it is natural to use a positive diagonal approximation.

#### MSE

A Gauss–Newton-type approximation is

\[
H_s^{\mathrm{GN}}
\approx
\sum_{t=s+1}^{T-1}
\frac{2}{N}
\left(
\frac{\partial h_t}{\partial c_s}
\right)^2.
\]

#### QLIKE

A useful positive curvature surrogate is Fisher-style scaling:

\[
w_t = \operatorname{clip}\!\left(\frac{1}{N h_t^2}, \varepsilon, w_{\max}\right),
\]

leading to

\[
H_s
\approx
\sum_{t=s+1}^{T-1}
w_t
\left(
\frac{\partial h_t}{\partial c_s}
\right)^2.
\]

This is not the exact raw-score Hessian of QLIKE, but it is positive, interpretable, and numerically better behaved than observed-curvature surrogates involving \(y_t/h_t^3\).

---

## Estimation strategy

### 1. Linear PGARCH

For the linear model, one may use deterministic second-order or quasi-second-order optimization:

- BFGS,
- L-BFGS,
- Newton or trust-region methods if the Hessian is implemented.

The key ingredients are:

- forward recursion for \(h_t\),
- forward recursions for \(J_t\) and \(H_t\),
- loss, gradient, and Hessian formulas above.

### 2. XGB-g-PGARCH

For the boosted model:

1. fit a baseline PGARCH initializer,
2. extract baseline \(\mu_t,\phi_t,g_t\),
3. convert baseline \(g_t\) to raw scores \(c_t^{(0)}\),
4. train XGBoost on row-wise gradients \(G_s\) and diagonal Hessian approximations \(H_s\),
5. use the fitted booster to refine the innovation-share channel.

This yields a nonlinear extension that preserves the GARCH-like decomposition and the exact recursive dependency structure while targeting the channel most analogous to STES.

---

## Concluding perspective

The central lesson behind PGARCH is that **flexibility alone is not enough** for volatility forecasting under QLIKE. What mattered empirically was not just whether a model could make its smoothing weight depend on predictors, but whether it preserved the structural decomposition that GARCH(1,1) already encoded.

PGARCH therefore arises not from abandoning GARCH, but from **understanding why GARCH worked** and then generalizing that structure carefully:

- separate long-run level,
- separate persistence,
- separate innovation share,
- then let predictors drive them.

This viewpoint explains both the weakness of single-gate STES-style models under QLIKE and the promise of predictor-driven GARCH decompositions as a broader model family.
