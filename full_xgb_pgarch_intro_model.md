# Toward Full XGB-PGARCH: Motivation and Model

## Introduction

### Empirical context from Part 5

In Part 5, we compared three models under out-of-sample volatility forecasting:

- **GARCH(1,1)**
- **PGARCH-L**
- **XGB-g-PGARCH**

The empirical result was informative in two directions at once.

First, both **PGARCH-L** and **XGB-g-PGARCH** outperformed **GARCH(1,1)** on our volatility forecasting task. This suggests that the main conceptual move from GARCH to PGARCH was useful: decomposing the variance recursion into

- long-run level,
- overall persistence,
- innovation share,

and then allowing those structural components to depend on predictors, is a better inductive bias than the classical constant-parameter recursion.

Second, **XGB-g-PGARCH underperformed PGARCH-L**, although the difference was not statistically significant under the Diebold–Mariano test. This result is equally important. It tells us that once the GARCH recursion has been generalized into the PGARCH decomposition, simply adding nonlinear flexibility to the **innovation-share channel alone** does not produce a robust improvement over the linear model.

Taken together, the Part 5 findings suggest the following interpretation:

1. The gain over GARCH came primarily from the **structural decomposition itself**, not merely from black-box flexibility.
2. The nonlinearization of the \(g_t\) channel alone is **not sufficient** to unlock the next layer of predictive improvement.
3. If further gains exist, they are more likely to come from allowing the **level channel** \(\mu_t\) and the **persistence channel** \(\phi_t\) to become nonlinear as well.

This motivates a further generalization: a **full XGB-PGARCH model**, in which all three channels are allowed to be predictor-driven through flexible nonlinear learners.

---

### Why the g-only nonlinearization was not enough

The v1 boosted model, XGB-g-PGARCH, was intentionally conservative. It kept the PGARCH decomposition but allowed only the innovation-share channel \(g_t\) to be learned nonlinearly by XGBoost, while the level and persistence channels were inherited from a baseline initializer.

This was the correct first extension for two reasons.

First, it was the closest nonlinear analogue of STES and XGBSTES, both of which primarily act through a dynamic shock-versus-memory weighting mechanism. If one hoped that the failure of STES relative to GARCH came mainly from a poor nonlinear map for the gate, then boosting \(g_t\) should have recovered that missing value.

Second, it gave a tractable and interpretable training problem. Only one recursive channel needed row-wise gradients and Hessian surrogates, making it possible to validate the mathematics and implementation before generalizing further.

The Part 5 result now tells us that this restriction is too severe. Once the PGARCH decomposition is present, the remaining misspecification is likely not in the innovation-share channel alone. In particular:

- **QLIKE is highly sensitive to scale errors**, which points directly to the importance of \(\mu_t\);
- volatility clustering and regime persistence are controlled by \(\phi_t\), so a misspecified persistence channel can dominate forecasting error even if \(g_t\) is flexible;
- allowing only \(g_t\) to move nonlinearly may force it to absorb effects that properly belong to level or persistence.

Thus the natural next step is not “more trees on \(g_t\),” but rather a full model in which all three structural components may respond nonlinearly to predictors.

---

### The rationale for the fully generalized model

The PGARCH decomposition already suggested that a variance recursion should be understood through three structurally distinct objects:

\[
\mu_t,\qquad \phi_t,\qquad g_t.
\]

These correspond, respectively, to:

- the **anchor level** of the variance process,
- the **total persistence** of the recursion,
- the **share of persistent dynamics assigned to the latest innovation**.

In Part 5, the linear PGARCH model showed that even a simple predictor-driven version of this decomposition can outperform classical GARCH. That means the decomposition itself carries meaningful signal. However, the failure of XGB-g-PGARCH to clearly dominate PGARCH-L suggests that the **remaining nonlinearity is distributed across channels**, not concentrated only in \(g_t\).

This observation motivates the fully generalized model:

- one nonlinear learner for the level channel,
- one nonlinear learner for the persistence channel,
- one nonlinear learner for the innovation-share channel.

Crucially, this is **not** a rejection of the PGARCH framework. It is the next logical step inside it. The philosophy remains the same:

> preserve the structural decomposition that made PGARCH successful, but relax the linearity assumption channel by channel.

This is exactly the kind of generalization that is justified by the Part 5 evidence. We are not introducing a flexible model merely because flexibility is available. We are doing so because the empirical comparison suggests that the decomposition is right, but the linear predictor maps are still too restrictive.

---

## Model

## 1. Full XGB-PGARCH recursion

Let \(y_t \ge 0\) denote the nonnegative volatility target, such as squared returns or a realized variance proxy. The full generalized PGARCH recursion is

\[
h_{t+1}
=
(1-\phi_t)\mu_t
+
\phi_t\Big(g_t y_t + (1-g_t) h_t\Big),
\qquad t=0,\dots,T-2,
\]

with a fixed initial state \(h_0 \ge 0\).

Define the intermediate quantity

\[
q_t = g_t y_t + (1-g_t)h_t,
\]

so that

\[
h_{t+1} = (1-\phi_t)\mu_t + \phi_t q_t.
\]

This indexing is intentionally chosen so that **row \(t\)** of the feature matrix affects the **next-step forecast \(h_{t+1}\)**. This is the most natural alignment for custom boosting objectives.

---

## 2. Three predictor-driven latent score channels

Let \(x_t \in \mathbb{R}^d\) denote the predictor vector available at row \(t\). We define three raw score sequences:

\[
a_t = F_\mu(x_t),
\qquad
b_t = F_\phi(x_t),
\qquad
c_t = F_g(x_t),
\]

where:

- \(F_\mu\) is the nonlinear learner for the level channel,
- \(F_\phi\) is the nonlinear learner for the persistence channel,
- \(F_g\) is the nonlinear learner for the innovation-share channel.

In the XGBoost implementation, each of these will be an additive boosted tree model.

---

## 3. Link functions

To enforce parameter constraints, the raw scores are passed through link functions:

\[
\mu_t = \mu_{\min} + \operatorname{softplus}(a_t),
\]

\[
\phi_t = \phi_{\min} + (\phi_{\max} - \phi_{\min}) \sigma(b_t),
\]

\[
g_t = g_{\min} + (g_{\max} - g_{\min}) \sigma(c_t),
\]

where

\[
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\operatorname{softplus}(z) = \log(1+e^z).
\]

These guarantee

\[
\mu_t > \mu_{\min} > 0,
\qquad
\phi_t \in (\phi_{\min}, \phi_{\max}) \subset (0,1),
\qquad
g_t \in (g_{\min}, g_{\max}) \subset (0,1).
\]

Hence, so long as \(y_t \ge 0\) and \(h_0 \ge 0\), the recursion remains positive and stable.

---

## 4. Implied time-varying GARCH parameters

As before, define

\[
\omega_t = (1-\phi_t)\mu_t,
\qquad
\alpha_t = \phi_t g_t,
\qquad
\beta_t = \phi_t (1-g_t).
\]

Then

\[
h_{t+1}
=
\omega_t + \alpha_t y_t + \beta_t h_t.
\]

So full XGB-PGARCH is a nonlinear, predictor-driven, time-varying GARCH recursion written in the more interpretable \((\mu_t,\phi_t,g_t)\) coordinates.

---

## 5. Local derivatives of the channels

For later estimation, define the first derivatives of each channel with respect to its raw score.

### Level channel

\[
A_t := \frac{\partial \mu_t}{\partial a_t}
= \sigma(a_t).
\]

### Persistence channel

\[
B_t := \frac{\partial \phi_t}{\partial b_t}
=
(\phi_{\max} - \phi_{\min}) \sigma(b_t)(1-\sigma(b_t)).
\]

### Innovation-share channel

\[
C_t := \frac{\partial g_t}{\partial c_t}
=
(g_{\max} - g_{\min}) \sigma(c_t)(1-\sigma(c_t)).
\]

---

## 6. Local impulses into the recursion

Because the recursion is one-step causal, each raw score at row \(t\) affects the state first through \(h_{t+1}\).

### 6.1 Level channel impulse

Since

\[
\frac{\partial h_{t+1}}{\partial \mu_t} = 1-\phi_t,
\]

the local impulse with respect to \(a_t\) is

\[
\delta_t^{(\mu)}
:=
\frac{\partial h_{t+1}}{\partial a_t}
=
(1-\phi_t) A_t.
\]

### 6.2 Persistence channel impulse

Since

\[
\frac{\partial h_{t+1}}{\partial \phi_t} = q_t - \mu_t,
\]

the local impulse with respect to \(b_t\) is

\[
\delta_t^{(\phi)}
:=
\frac{\partial h_{t+1}}{\partial b_t}
=
(q_t - \mu_t) B_t.
\]

### 6.3 Innovation-share channel impulse

Since

\[
\frac{\partial h_{t+1}}{\partial g_t}
=
\phi_t (y_t - h_t),
\]

the local impulse with respect to \(c_t\) is

\[
\delta_t^{(g)}
:=
\frac{\partial h_{t+1}}{\partial c_t}
=
\phi_t (y_t - h_t) C_t.
\]

---

## 7. State-propagation coefficient

The effect of the current variance state on the next-step forecast is

\[
\rho_t
:=
\frac{\partial h_{t+1}}{\partial h_t}
=
\phi_t (1-g_t),
\]

provided the variance floor is inactive. This coefficient propagates all recursive sensitivities forward through time.

---

## 8. Loss functions

We consider the same two loss families as before, evaluated on the forecast path \(\{h_t\}_{t=1}^{T-1}\). Since \(h_0\) is treated as a fixed causal warm start, the effective sample size is

\[
N = T-1.
\]

### 8.1 MSE

\[
L_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{t=1}^{T-1}
(h_t - y_t)^2.
\]

### 8.2 QLIKE

\[
L_{\mathrm{QLIKE}}
=
\frac{1}{N}
\sum_{t=1}^{T-1}
\left(
\log h_t + \frac{y_t}{h_t}
\right).
\]

Define the per-time derivative

\[
u_t := \frac{\partial \ell_t}{\partial h_t}.
\]

Then:

#### MSE
\[
u_t = \frac{2(h_t-y_t)}{N}.
\]

#### QLIKE
\[
u_t
=
\frac{1}{N}\left(
\frac{1}{h_t} - \frac{y_t}{h_t^2}
\right).
\]

---

## 9. Backward adjoint recursion

Let

\[
\lambda_t := \frac{\partial L}{\partial h_t},
\]

including all future recursive dependence.

Because \(h_t\) enters the loss directly through \(\ell_t\) and indirectly through \(h_{t+1}\), the adjoint recursion is

\[
\lambda_{T-1} = u_{T-1},
\]

\[
\lambda_t = u_t + \rho_t \lambda_{t+1},
\qquad
t=T-2,\dots,0.
\]

This recursion is the key object that allows efficient exact row-wise gradients.

---

## 10. Exact row-wise gradients for the three boosted channels

Since row \(s\) affects the full loss only through its local impulse into \(h_{s+1}\) and then propagates forward recursively, the row-wise gradients are simply:

### Level channel
\[
G_s^{(\mu)}
=
\frac{\partial L}{\partial a_s}
=
\lambda_{s+1}\delta_s^{(\mu)},
\qquad s=0,\dots,T-2.
\]

### Persistence channel
\[
G_s^{(\phi)}
=
\frac{\partial L}{\partial b_s}
=
\lambda_{s+1}\delta_s^{(\phi)},
\qquad s=0,\dots,T-2.
\]

### Innovation-share channel
\[
G_s^{(g)}
=
\frac{\partial L}{\partial c_s}
=
\lambda_{s+1}\delta_s^{(g)},
\qquad s=0,\dots,T-2.
\]

For the terminal row,

\[
G_{T-1}^{(\mu)} = G_{T-1}^{(\phi)} = G_{T-1}^{(g)} = 0,
\]

because row \(T-1\) has no next-step in-sample forecast contribution.

These are the exact row-wise gradients used for custom boosting.

---

## 11. Positive diagonal Hessian surrogates

XGBoost custom objectives require one Hessian value per row and per channel. The exact raw-score Hessians are available in principle but are cumbersome, involve second-order recursive terms, and need not be positive. Therefore we use **positive diagonal curvature surrogates**.

### 11.1 MSE

For any channel \(z \in \{\mu,\phi,g\}\),

\[
H_s^{(z)}
\approx
\sum_{t=s+1}^{T-1}
\frac{2}{N}
\left(
\frac{\partial h_t}{\partial \text{raw}_s^{(z)}}
\right)^2.
\]

This is a Gauss–Newton-type approximation.

### 11.2 QLIKE

For QLIKE we use a Fisher-style positive weight,

\[
w_t = \operatorname{clip}\!\left(\frac{1}{N h_t^2}, \varepsilon, w_{\max}\right),
\]

and then define

\[
H_s^{(z)}
\approx
\sum_{t=s+1}^{T-1}
w_t
\left(
\frac{\partial h_t}{\partial \text{raw}_s^{(z)}}
\right)^2.
\]

This is not the exact observed Hessian of QLIKE with respect to the raw scores, but it is positive, stable, and consistent with XGBoost’s custom-objective interface.

---

## 12. Why block-coordinate boosting is the right estimation strategy

A full nonlinear PGARCH could in principle be trained by attempting to update all three channels jointly. In practice, that is not the right first implementation.

The recursion couples \(\mu_t\), \(\phi_t\), and \(g_t\) multiplicatively, so a full joint second-order treatment would involve:

- cross-channel curvature terms,
- recursive second derivatives,
- and complex identification interactions.

By contrast, a **block-coordinate boosting strategy** is both mathematically cleaner and computationally more stable:

1. hold two channels fixed,
2. compute exact row-wise gradients and positive Hessian surrogates for the remaining channel,
3. fit one boosted tree update for that channel,
4. cycle to the next channel.

This produces the natural algorithmic roadmap:

- update the level channel,
- update the persistence channel,
- update the innovation-share channel,
- repeat.

Given the Part 5 evidence, such a design is also empirically motivated. Since \(g\)-only boosting did not clearly beat PGARCH-L, the most likely useful nonlinearities are in \(\mu_t\) and \(\phi_t\). Thus the full model should not be conceived as “more complexity everywhere,” but rather as a structured attempt to discover whether nonlinear level and nonlinear persistence contain incremental predictive information beyond the linear PGARCH benchmark.

---

## 13. Interpretation of the full model

The full XGB-PGARCH model is best understood as the end point of a sequence of increasingly structured generalizations:

1. **GARCH(1,1)**  
   constant level, persistence, and innovation share.

2. **PGARCH-L**  
   all three structural components driven linearly by predictors.

3. **XGB-g-PGARCH**  
   PGARCH decomposition retained, but only the innovation-share channel boosted.

4. **Full XGB-PGARCH**  
   all three channels learned nonlinearly in a structured block-coordinate fashion.

The Part 5 result suggests that the critical gain came in moving from (1) to (2), not from the restricted move from (2) to (3). The motivation for the present model is therefore not to abandon PGARCH-L, but to ask a sharper question:

> if the PGARCH decomposition is right, where does the remaining nonlinearity actually live?

The full model is designed to answer that question channel by channel.
