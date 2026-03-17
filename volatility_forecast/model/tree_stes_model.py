from __future__ import annotations

"""Tree-gated STES (XGBoostSTESModel).

This module contains a single model class, :class:`XGBoostSTESModel`, that combines

- a **variance recursion** (EWMA/STES-style filter)

    v_{t+1} = v_t + α_t (r_t^2 - v_t)

- a **gate model** for α_t.

Two fitting routes are supported:

1) ``fit_method='alternating'``  *(deprecated – will be removed)*
   Alternating supervision on a per-step pseudo-optimal gate α*_t, where α*_t is
   defined as the minimiser (over α∈[0,1]) of the one-step forecast loss while
   holding v_t fixed.

2) ``fit_method='end_to_end'``  *(default)*
   End-to-end optimisation of the variance forecast loss through the recursion,
   using an adjoint pass to produce gradients and a stable diagonal Gauss–Newton
   approximation for Hessians.

Notes on data alignment
-----------------------
We treat each row t as:
  - state before update: v_t
  - innovation available at t: r_t^2
  - gate decided at t: alpha_t
  - forecast made at t: yhat_t = v_{t+1|t} = v_t + alpha_t (r_t^2 - v_t)

So training labels y[t] should correspond to the realised variance for the next step
being forecast at time t (commonly: y_t = r_{t+1}^2).
"""

from dataclasses import dataclass
import logging
import warnings
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

from .base_model import BaseVolatilityModel

logger = logging.getLogger(__name__)

FitMethod = Literal["alternating", "end_to_end"]
LossName = Literal["mse", "pseudohuber", "qlike"]
OutputMode = Literal["alpha", "logit"]


@dataclass(frozen=True, slots=True)
class FitResultBase:
    booster: Any
    params_used: Dict[str, Any]
    num_boost_round: int
    feature_names: Tuple[str, ...]
    alpha_base: float
    fit_method: FitMethod
    output_mode: OutputMode


@dataclass(frozen=True, slots=True)
class AlternatingFitResult(FitResultBase):
    n_alt_iters: int
    gate_valid_frac: float
    loss: LossName
    huber_delta: float


@dataclass(frozen=True, slots=True)
class EndToEndFitResult(FitResultBase):
    loss: LossName
    huber_delta: float


class XGBoostSTESModel(BaseVolatilityModel):
    """XGBoost-gated STES volatility model."""

    def __init__(
        self,
        *,
        xgb_params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 200,
        init_window: int = 500,
        # fitting mode
        fit_method: FitMethod = "end_to_end",
        # shared loss (used for α* (alternating) and end-to-end objective)
        loss: LossName = "mse",
        huber_delta: float = 1.0,
        qlike_epsilon: float = 1e-8,
        # end-to-end diagnostics / numerics
        e2e_grad_hess_scale: float = 1.0,
        e2e_debug: bool = False,
        e2e_debug_print_once: bool = True,
        # alternating-specific knobs
        n_alt_iters: int = 3,
        gate_valid_frac: float = 0.10,
        # misc
        random_state: Optional[int] = None,
        monotonic_constraints: Optional[dict[str, int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            # Backward-compat: ignore removed constructor knobs (priors, logit-target modes, etc.)
            logger.warning(
                "Ignoring unsupported XGBoostSTESModel __init__ kwargs: %s",
                sorted(list(kwargs.keys())),
            )

        if fit_method not in {"alternating", "end_to_end"}:
            raise ValueError("fit_method must be one of: {'alternating','end_to_end'}")
        if fit_method == "alternating":
            warnings.warn(
                "fit_method='alternating' is deprecated and will be removed in a "
                "future release. Use fit_method='end_to_end' instead.",
                FutureWarning,
                stacklevel=2,
            )
        if loss not in {"mse", "pseudohuber", "qlike"}:
            raise ValueError("loss must be one of: {'mse','pseudohuber','qlike'}")
        if not (np.isfinite(huber_delta) and huber_delta > 0.0):
            raise ValueError("huber_delta must be finite and > 0")
        if not (np.isfinite(qlike_epsilon) and qlike_epsilon > 0.0):
            raise ValueError("qlike_epsilon must be finite and > 0")
        if not (np.isfinite(e2e_grad_hess_scale) and e2e_grad_hess_scale > 0.0):
            raise ValueError("e2e_grad_hess_scale must be finite and > 0")

        self.xgb_params: Dict[str, Any] = dict(xgb_params or {})
        self.num_boost_round = int(num_boost_round)
        self.init_window = int(init_window)
        self.fit_method: FitMethod = fit_method

        self.loss: LossName = loss
        self.huber_delta = float(huber_delta)
        self.qlike_epsilon = float(qlike_epsilon)

        self.e2e_grad_hess_scale = float(e2e_grad_hess_scale)
        self.e2e_debug = bool(e2e_debug)
        self.e2e_debug_print_once = bool(e2e_debug_print_once)

        self.n_alt_iters = int(n_alt_iters)
        self.gate_valid_frac = float(gate_valid_frac)

        self.random_state = random_state
        self.monotonic_constraints: dict[str, int] = monotonic_constraints or {}

        # learned artifacts
        self.model_: Any = None
        self.fit_result_: Optional[FitResultBase] = None
        self.output_mode_: Optional[OutputMode] = None
        self.init_var_: Optional[float] = None
        self.last_var_: Optional[float] = None
        self.base_margin_: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # small utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _check_xgb() -> None:
        if xgb is None:
            raise ImportError(
                "xgboost is not available. Install xgboost to use XGBoostSTESModel."
            )

    @staticmethod
    def _as_numpy(X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X.values, dtype=float, order="C")

    @staticmethod
    def _initial_variance(returns2: np.ndarray, init_window: int) -> float:
        w = int(max(1, init_window))
        arr = np.asarray(returns2, dtype=float)
        if arr.size == 0:
            return 1e-8
        head = arr[: min(w, arr.size)]
        v0 = float(np.nanmean(head))
        return v0 if np.isfinite(v0) and v0 > 0.0 else 1e-8

    @staticmethod
    def _apply_monotone_constraints(
        params: Dict[str, Any],
        feature_names: Sequence[str],
        constraints: dict[str, int],
    ) -> None:
        if not constraints:
            return
        params["monotone_constraints"] = {
            name: int(direction)
            for name, direction in constraints.items()
            if name in set(feature_names)
        }

    # ---------------------------------------------------------------------
    # recursion + loss helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _filter_state_and_forecast(
        returns2: np.ndarray, alpha: np.ndarray, init_var: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run v_{t+1} = v_t + α_t (r_t^2 - v_t).

        Returns
        -------
        v_state : np.ndarray
            v_state[t] = v_t (pre-update state at time t)
        yhat : np.ndarray
            yhat[t] = v_{t+1|t} (forecast made at time t)
        """
        n = int(len(returns2))
        v_state = np.empty(n, dtype=float)
        yhat = np.empty(n, dtype=float)
        v = float(init_var)
        for t in range(n):
            v_state[t] = v
            v = v + float(alpha[t]) * (float(returns2[t]) - v)
            yhat[t] = v
        return v_state, yhat

    @staticmethod
    def _loss_derivs(
        yhat: np.ndarray,
        y: np.ndarray,
        *,
        loss: LossName,
        huber_delta: float,
        qlike_epsilon: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return elementwise (e, w) where:

        - e = dℓ/dyhat
        - w = d²ℓ/dyhat²

        Conventions:
        - For MSE we use ℓ(u) = 0.5 * u^2, so e=u and w=1.
        - For pseudo-Huber we use ℓ(u) = δ^2 * (sqrt(1 + (u/δ)^2) - 1).
        """
        u = (yhat - y).astype(float)
        if loss == "mse":
            return u, np.ones_like(u, dtype=float)
        if loss == "qlike":
            y_safe = np.maximum(np.asarray(y, dtype=float), qlike_epsilon)
            yhat_safe = np.maximum(np.asarray(yhat, dtype=float), qlike_epsilon)
            e = (yhat_safe - y_safe) / (yhat_safe**2)
            w = (2.0 * y_safe - yhat_safe) / (yhat_safe**3)
            return e, w
        d = float(huber_delta)
        s = np.sqrt(1.0 + (u / d) ** 2)
        e = u / s
        w = 1.0 / (s**3)
        return e, w

    @staticmethod
    def _loss_value(
        yhat: np.ndarray,
        y: np.ndarray,
        *,
        loss: LossName,
        huber_delta: float,
        qlike_epsilon: float = 1e-8,
    ) -> float:
        u = (yhat - y).astype(float)
        if loss == "mse":
            # Keep reporting consistent with _loss_derivs convention:
            # ℓ(u) = 0.5 * u^2  =>  dℓ/du = u, d²ℓ/du² = 1
            return float(0.5 * np.mean(u * u))
        if loss == "qlike":
            y_safe = np.maximum(np.asarray(y, dtype=float), qlike_epsilon)
            yhat_safe = np.maximum(np.asarray(yhat, dtype=float), qlike_epsilon)
            ratio = y_safe / yhat_safe
            return float(np.mean(ratio - np.log(ratio) - 1.0))
        d = float(huber_delta)
        return float(np.mean(d * d * (np.sqrt(1.0 + (u / d) ** 2) - 1.0)))

    def _alpha_star_for_loss(
        self,
        *,
        y: np.ndarray,
        returns2: np.ndarray,
        v_state: np.ndarray,
        alpha_fallback: float,
        max_iter: int = 15,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Compute α*_t in [0,1] minimising one-step loss holding v_t fixed.

        For MSE we have a closed form:
            α* = clip((y - v)/(r^2 - v), 0, 1).

        For pseudo-Huber we do a small, robust Newton solve in α (per time step),
        using the shared loss derivatives.
        """
        n = int(len(y))
        out = np.full(n, float(alpha_fallback), dtype=float)
        eps = 1e-12

        for t in range(n):
            v = float(v_state[t])
            denom = float(returns2[t] - v)
            if (not np.isfinite(denom)) or abs(denom) < eps:
                out[t] = float(alpha_fallback)
                continue

            # MSE closed-form init (also good init for Huber)
            a = float((y[t] - v) / denom)
            a = float(np.clip(a, 0.0, 1.0))

            if self.loss in {"mse", "qlike"}:
                out[t] = a
                continue

            # pseudo-Huber: minimise ℓ(yhat(α), y), yhat = v + α*denom.
            # dℓ/dα = (dℓ/dyhat) * denom
            # d²ℓ/dα² = (d²ℓ/dyhat²) * denom²
            for _ in range(int(max_iter)):
                yhat = v + a * denom
                e, w = self._loss_derivs(
                    np.asarray([yhat]),
                    np.asarray([y[t]]),
                    loss=self.loss,
                    huber_delta=self.huber_delta,
                    qlike_epsilon=self.qlike_epsilon,
                )
                g = float(e[0]) * denom
                h = float(w[0]) * (denom * denom)
                if not (np.isfinite(g) and np.isfinite(h)) or h <= 0.0:
                    break
                step = g / h
                a_new = float(np.clip(a - step, 0.0, 1.0))
                if abs(a_new - a) < tol:
                    a = a_new
                    break
                a = a_new

            out[t] = a

        return out

    # ---------------------------------------------------------------------
    # Fit / predict API
    # ---------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        returns: pd.Series,
        start_index: int = 0,
        end_index: Optional[int] = None,
        # New CV arguments
        perform_cv: bool = False,
        cv_grid: Optional[Iterable[Dict[str, Any]]] = None,
        cv_splits: int = 5,
        # Warm-start: provide baseline z-scores as base_margin
        base_margin: Optional[np.ndarray] = None,
        **_: Any,
    ) -> "XGBoostSTESModel":
        self._check_xgb()
        if end_index is None:
            end_index = len(y)

        # Sanity check for CV
        if perform_cv:
            if cv_grid is None:
                raise ValueError("If perform_cv=True, you must provide a cv_grid.")
            # Ensure sklearn is available before starting heavy work
            try:
                import sklearn.model_selection  # noqa
            except ImportError:
                raise ImportError("scikit-learn is required for CV.")

        X_fit = X.iloc[start_index:end_index].copy()
        y_fit = y.iloc[start_index:end_index].copy()
        r_fit = returns.iloc[start_index:end_index].copy()

        idx = X_fit.index.intersection(y_fit.index).intersection(r_fit.index)
        X_fit = X_fit.loc[idx]
        y_fit = y_fit.loc[idx]
        r_fit = r_fit.loc[idx]
        if len(X_fit) < 10:
            raise ValueError("Not enough rows to fit XGBoostSTESModel.")

        # Slice base_margin to match the fitted data range
        base_margin_fit: Optional[np.ndarray] = None
        if base_margin is not None:
            bm = np.asarray(base_margin, dtype=float)
            bm_slice = bm[start_index:end_index]
            # If index intersection dropped rows, re-index via positional mapping
            if len(bm_slice) == len(idx):
                base_margin_fit = bm_slice
            else:
                # Fallback: assume base_margin aligns with X before slicing
                base_margin_fit = bm_slice[: len(idx)]
            logger.info(
                "base_margin provided (len=%d). Using base_score=0.0.",
                len(base_margin_fit),
            )

        X_np = self._as_numpy(X_fit)
        y_np = np.asarray(y_fit.values, dtype=float)
        returns2_np = np.asarray(r_fit.values, dtype=float) ** 2
        init_var = self._initial_variance(returns2_np, self.init_window)

        params = dict(self.xgb_params)
        params.setdefault("max_depth", 3)
        params.setdefault("eta", 0.05)
        params.setdefault("subsample", 1.0)
        params.setdefault("colsample_bytree", 1.0)
        params.setdefault("min_child_weight", 1.0)
        params.setdefault("lambda", 1.0)
        params.setdefault("alpha", 0.0)
        if self.random_state is not None:
            params.setdefault("seed", int(self.random_state))

        feature_names = tuple(X_fit.columns)
        self._apply_monotone_constraints(
            params, feature_names, self.monotonic_constraints
        )

        # -----------------------------------------------------------------
        # AUTO-TUNING BLOCK
        # -----------------------------------------------------------------
        if perform_cv and cv_grid is not None:
            logger.info(
                f"Running auto-tuning ({self.fit_method}) with {cv_splits} splits..."
            )

            # 1. Dispatch to the correct CV method
            if self.fit_method == "end_to_end":
                # Relies on the run_cv method added in previous step
                cv_results = self.run_cv_e2e(
                    X_fit, y_fit, returns=r_fit, param_grid=cv_grid, n_splits=cv_splits
                )
            else:
                # Relies on the run_cv_alternating method added in previous step
                cv_results = self.run_cv_alternating(
                    X_fit, y_fit, returns=r_fit, param_grid=cv_grid, n_splits=cv_splits
                )

            # 2. Pick winner (results are sorted by score ascending, so index 0 is best)
            best_score, best_params = cv_results[0]
            logger.info(f"Auto-tuning complete. Best Score: {best_score:.6f}")
            logger.info(f"Best Params: {best_params}")

            # 3. Update the params dict that will be used for the final fit below
            params.update(best_params)
            # Optional: Update the class state so user sees what was chosen
            self.xgb_params.update(best_params)

        # baseline alpha for init (best-effort)
        alpha_base = 0.06
        try:
            from .es_model import ESModel

            m = ESModel(random_state=self.random_state)
            m.fit(X_fit, y_fit, returns=r_fit, start_index=0, end_index=len(X_fit))
            alpha_base = float(getattr(m, "alpha_", alpha_base))
        except Exception:
            logger.warning("ESModel fit failed; using default alpha_base=0.06.")

        if self.fit_method == "alternating":
            params.setdefault(
                "objective",
                (
                    "reg:logistic"
                    if self.loss in {"mse", "qlike"}
                    else "reg:pseudohubererror"
                ),
            )
            # XGBoost ≥ 2.0 requires base_score ∈ (0,1) for reg:logistic.
            if params.get("objective") == "reg:logistic":
                params.setdefault("base_score", 0.5)
            booster = self._fit_alternating(
                X_np=X_np,
                y_np=y_np,
                returns2_np=returns2_np,
                init_var=init_var,
                params=params,
                alpha_base=alpha_base,
                feature_names=feature_names,
            )
            self.output_mode_ = "alpha"
            self.fit_result_ = AlternatingFitResult(
                booster=booster,
                params_used=params,
                num_boost_round=self.num_boost_round,
                feature_names=feature_names,
                alpha_base=float(alpha_base),
                fit_method="alternating",
                output_mode="alpha",
                n_alt_iters=int(self.n_alt_iters),
                gate_valid_frac=float(self.gate_valid_frac),
                loss=self.loss,
                huber_delta=float(self.huber_delta),
            )
        else:
            params.setdefault("objective", "reg:squarederror")  # ignored by custom obj
            # Only apply relaxed defaults when CV has NOT already selected
            # values for these keys.  Previously these were unconditional
            # overwrites which silently discarded CV-tuned hyperparameters.
            if params.get("booster", "gbtree") != "gblinear":
                params.setdefault("min_child_weight", 1e-8)
                params.setdefault("reg_lambda", 1e-8)
                params.setdefault("max_depth", 2)
                params.setdefault("eta", 0.1)
            params.setdefault("subsample", 1)
            params.setdefault("colsample_bytree", 1)

            booster = self._fit_end_to_end(
                X_np=X_np,
                y_np=y_np,
                returns2_np=returns2_np,
                init_var=init_var,
                params=params,
                feature_names=feature_names,
                base_margin=base_margin_fit,
            )
            self.output_mode_ = "logit"
            self.fit_result_ = EndToEndFitResult(
                booster=booster,
                params_used=params,
                num_boost_round=self.num_boost_round,
                feature_names=feature_names,
                alpha_base=float(alpha_base),
                fit_method="end_to_end",
                output_mode="logit",
                loss=self.loss,
                huber_delta=float(self.huber_delta),
            )

        self.model_ = self.fit_result_.booster

        # Store base_margin function for predict-time reconstruction
        self.base_margin_: Optional[np.ndarray] = base_margin_fit

        # store warm-start terminal variance on the fit block
        d_all = xgb.DMatrix(X_np, feature_names=list(feature_names))
        if base_margin_fit is not None:
            d_all.set_base_margin(base_margin_fit)
        score = self.model_.predict(d_all)
        alpha_fit = (
            expit(score.astype(float))
            if self.output_mode_ == "logit"
            else np.clip(score.astype(float), 0.0, 1.0)
        )

        _, yhat_fit = self._filter_state_and_forecast(returns2_np, alpha_fit, init_var)
        self.init_var_ = float(init_var)
        self.last_var_ = float(yhat_fit[-1]) if len(yhat_fit) else float(init_var)
        return self

    def _fit_alternating(
        self,
        *,
        X_np: np.ndarray,
        y_np: np.ndarray,
        returns2_np: np.ndarray,
        init_var: float,
        params: Dict[str, Any],
        alpha_base: float,
        feature_names: Tuple[str, ...],
    ) -> Any:
        """Alternating supervised loop on α*_t under the model loss."""
        n = int(len(y_np))
        alpha_pred = np.full(n, float(alpha_base), dtype=float)
        booster = None

        # params["objective"] is passed from fit()

        for _ in range(max(1, int(self.n_alt_iters))):
            v_state, _ = self._filter_state_and_forecast(
                returns2_np, alpha_pred, init_var
            )
            alpha_star = self._alpha_star_for_loss(
                y=y_np,
                returns2=returns2_np,
                v_state=v_state,
                alpha_fallback=float(alpha_base),
            )

            valid_n = int(max(1, np.floor(self.gate_valid_frac * n)))
            valid_n = min(valid_n, max(1, n - 1))
            split = max(1, n - valid_n)
            tr = np.arange(split)
            va = np.arange(split, n)

            dtrain = xgb.DMatrix(
                X_np[tr],
                label=alpha_star[tr],
                feature_names=list(feature_names),
            )
            dvalid = xgb.DMatrix(
                X_np[va],
                label=alpha_star[va],
                feature_names=list(feature_names),
            )

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                verbose_eval=False,
                xgb_model=booster,
                early_stopping_rounds=params.get("early_stopping_rounds", 10),
            )

            d_all = xgb.DMatrix(X_np, feature_names=list(feature_names))
            # Safety clip: works for reg:logistic (redundant but safe) and reg:pseudohuber (necessary)
            alpha_pred = np.clip(booster.predict(d_all).astype(float), 0.0, 1.0)

        return booster

    # -------------------------- end-to-end (B2) ---------------------------
    def _make_e2e_objective(
        self,
        *,
        returns2: np.ndarray,
        y: np.ndarray,
        init_var: float,
    ):
        """Create (obj, feval) closures for xgboost.train."""
        returns2 = np.asarray(returns2, dtype=float)
        y = np.asarray(y, dtype=float)
        n = int(len(y))
        v0 = float(init_var)

        loss = self.loss
        huber_delta = self.huber_delta
        scale = float(self.e2e_grad_hess_scale)
        debug = bool(self.e2e_debug)
        print_once = bool(self.e2e_debug_print_once)
        printed = {"done": False}

        def obj(preds: np.ndarray, dtrain: Any):
            z = preds.astype(float)
            alpha = expit(z)
            v = np.empty(n + 1, dtype=float)
            v[0] = v0
            for t in range(n):
                v[t + 1] = v[t] + alpha[t] * (returns2[t] - v[t])
            yhat = v[1:]

            e_next, w_next = self._loss_derivs(
                yhat,
                y,
                loss=loss,
                huber_delta=huber_delta,
                qlike_epsilon=self.qlike_epsilon,
            )

            g = np.zeros(n + 1, dtype=float)
            S = np.zeros(n + 1, dtype=float)
            grad = np.zeros(n, dtype=float)
            hess = np.zeros(n, dtype=float)

            for t in range(n - 1, -1, -1):
                g[t + 1] += e_next[t]
                S[t + 1] += w_next[t]

                a = float(alpha[t])
                ap = a * (1.0 - a)
                denom = float(returns2[t] - v[t])
                dv_dz = denom * ap

                grad[t] = g[t + 1] * dv_dz
                h = S[t + 1] * (dv_dz * dv_dz)
                hess[t] = h if np.isfinite(h) and h > 1e-12 else 1e-12

                g[t] += (1.0 - a) * g[t + 1]
                S[t] += (1.0 - a) ** 2 * S[t + 1]

            # Scaling knob: does not change Newton ratio grad/hess, but *does*
            # change interaction with XGBoost regularization/min_child_weight.
            if scale != 1.0:
                grad *= scale
                hess *= scale

            # Debug prints (point 5): one-time summary of magnitudes.
            if debug and (not printed["done"]):

                def _summ(x: np.ndarray) -> Tuple[float, float, float, float]:
                    return (
                        float(np.min(x)),
                        float(np.max(x)),
                        float(np.mean(x)),
                        float(np.std(x)),
                    )

                zmin, zmax, zmean, zstd = _summ(z)
                amin, amax, amean, astd = _summ(alpha)
                gmin, gmax = float(np.min(grad)), float(np.max(grad))
                gmean, gabs = float(np.mean(grad)), float(np.mean(np.abs(grad)))
                hmin, hmax, hmean = (
                    float(np.min(hess)),
                    float(np.max(hess)),
                    float(np.mean(hess)),
                )
                logger.info(
                    "[E2E] z(min,max,mean,std)=(%.4g, %.4g, %.4g, %.4g) | "
                    "alpha(min,max,mean,std)=(%.4g, %.4g, %.4g, %.4g) | "
                    "grad(min,max,mean,mean_abs)=(%.4g, %.4g, %.4g, %.4g) | "
                    "hess(min,max,mean)=(%.4g, %.4g, %.4g) | scale=%.4g",
                    zmin,
                    zmax,
                    zmean,
                    zstd,
                    amin,
                    amax,
                    amean,
                    astd,
                    gmin,
                    gmax,
                    gmean,
                    gabs,
                    hmin,
                    hmax,
                    hmean,
                    scale,
                )
                if print_once:
                    printed["done"] = True

            return grad, hess

        def feval(preds: np.ndarray, dtrain: Any):
            z = preds.astype(float)
            alpha = expit(z)
            v = float(v0)
            yhat = np.empty(n, dtype=float)
            for t in range(n):
                v = v + float(alpha[t]) * (float(returns2[t]) - v)
                yhat[t] = v
            return f"{loss}_e2e", self._loss_value(
                yhat,
                y,
                loss=loss,
                huber_delta=huber_delta,
                qlike_epsilon=self.qlike_epsilon,
            )

        return obj, feval

    def _fit_end_to_end(
        self,
        *,
        X_np: np.ndarray,
        y_np: np.ndarray,
        returns2_np: np.ndarray,
        init_var: float,
        params: Dict[str, Any],
        feature_names: Tuple[str, ...],
        base_margin: Optional[np.ndarray] = None,
    ) -> Any:
        # 1. Split Train/Valid for Early Stopping
        n = len(y_np)
        valid_n = int(max(1, np.floor(self.gate_valid_frac * n)))
        # Ensure at least 1 row for valid, but not more than n-1
        valid_n = min(valid_n, max(1, n - 1))
        split = max(1, n - valid_n)

        X_tr, X_va = X_np[:split], X_np[split:]
        y_tr, y_va = y_np[:split], y_np[split:]
        r2_tr, r2_va = returns2_np[:split], returns2_np[split:]

        # 2. Warm Start for Validation Set
        # We cannot start validation recursion from global init_var (e.g. 2000 data).
        # We use the recent history of training data to approximate v_t at split point.
        lookback = 20
        if len(r2_tr) >= lookback:
            v0_val = float(np.mean(r2_tr[-lookback:]))
        else:
            v0_val = float(init_var)

        # 3. Create Objectives
        # Training objective: drives gradient descent
        obj_train, _ = self._make_e2e_objective(
            returns2=r2_tr, y=y_tr, init_var=init_var
        )
        # Validation metric: drives early stopping (only needs feval)
        _, feval_valid = self._make_e2e_objective(
            returns2=r2_va, y=y_va, init_var=v0_val
        )

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(feature_names))
        dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=list(feature_names))

        # 4. Apply base_margin if provided (warm-start / two-stage)
        if base_margin is not None:
            bm_tr, bm_va = base_margin[:split], base_margin[split:]
            dtrain.set_base_margin(bm_tr)
            dvalid.set_base_margin(bm_va)
            # Disable default base_score to avoid double-counting
            params["base_score"] = 0.0
            logger.info(
                "[E2E] base_margin applied: train=%d, valid=%d",
                len(bm_tr),
                len(bm_va),
            )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            obj=obj_train,
            custom_metric=feval_valid,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=params.get("early_stopping_rounds", 10),
            verbose_eval=False,
        )

        # Post-fit E2E sanity: if score/alpha are constant, the booster learned no splits.
        if self.e2e_debug:
            d_all = xgb.DMatrix(X_np, feature_names=list(feature_names))
            score = booster.predict(d_all)
            alpha = expit(score.astype(float))
            s_std = float(np.std(score))
            a_std = float(np.std(alpha))
            logger.info(
                "[E2E post-fit] score(std)=%.3g, alpha(std)=%.3g, score(mean)=%.6g, alpha(mean)=%.6g",
                s_std,
                a_std,
                float(np.mean(score)),
                float(np.mean(alpha)),
            )
            if a_std < 1e-10:
                logger.warning(
                    "[E2E post-fit] alpha is essentially constant. "
                    "Try increasing e2e_grad_hess_scale (e.g., 1e6..1e9) and/or loosening "
                    "regularization/min_child_weight."
                )
        return booster

    def predict(
        self,
        X: pd.DataFrame,
        *,
        returns: pd.Series,
        start_index: int = 0,
        end_index: Optional[int] = None,
        init_var: Optional[float] = None,
        warm_start_from_last_fit: bool = True,
        base_margin: Optional[np.ndarray] = None,
        **_: Any,
    ) -> pd.Series:
        self._check_xgb()
        if self.model_ is None or self.output_mode_ is None:
            raise ValueError("Model has not been fitted.")

        if end_index is None:
            end_index = len(X)

        Xp = X.iloc[start_index:end_index].copy()
        rp = returns.iloc[start_index:end_index].copy()
        idx = Xp.index.intersection(rp.index)
        Xp = Xp.loc[idx]
        rp = rp.loc[idx]

        X_np = self._as_numpy(Xp)
        r2 = np.asarray(rp.values, dtype=float) ** 2

        feature_names = list(Xp.columns)
        dtest = xgb.DMatrix(X_np, feature_names=feature_names)
        if base_margin is not None:
            bm = np.asarray(base_margin, dtype=float)
            bm_pred = bm[start_index:end_index][: len(Xp)]
            dtest.set_base_margin(bm_pred)
        score = self.model_.predict(dtest)
        alpha = (
            expit(score.astype(float))
            if self.output_mode_ == "logit"
            else np.clip(score.astype(float), 0.0, 1.0)
        )

        if init_var is None and warm_start_from_last_fit and self.last_var_ is not None:
            v0 = float(self.last_var_)
        else:
            v0 = (
                float(init_var)
                if init_var is not None
                else self._initial_variance(r2, self.init_window)
            )

        _, yhat = self._filter_state_and_forecast(r2, alpha, v0)
        return pd.Series(yhat, index=Xp.index, name="yhat")

    def get_alphas(
        self,
        X: pd.DataFrame,
        *,
        start_index: int = 0,
        end_index: Optional[int] = None,
        base_margin: Optional[np.ndarray] = None,
    ) -> pd.Series:
        self._check_xgb()
        if self.model_ is None or self.output_mode_ is None:
            raise ValueError("Model has not been fitted.")
        if end_index is None:
            end_index = len(X)

        Xp = X.iloc[start_index:end_index]
        d = xgb.DMatrix(self._as_numpy(Xp), feature_names=list(Xp.columns))
        if base_margin is not None:
            bm = np.asarray(base_margin, dtype=float)
            bm_pred = bm[start_index:end_index][: len(Xp)]
            d.set_base_margin(bm_pred)
        score = self.model_.predict(d)
        alpha = (
            expit(score.astype(float))
            if self.output_mode_ == "logit"
            else np.clip(score.astype(float), 0.0, 1.0)
        )
        return pd.Series(alpha, index=Xp.index, name="alpha")

    def run_cv_alternating(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        returns: pd.Series,
        param_grid: Iterable[Dict[str, Any]],
        n_splits: int = 5,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Run Time-Series CV specifically for the Alternating Method.

        CRITICAL DIFFERENCE:
        - TRAINS on alpha_star (proxy target) to mimic fit_alternating behavior.
        - VALIDATES on realized variance (y) to measure true forecast quality.
        This solves the 'Proxy Trap' by optimizing hyperparameters for the
        end goal (forecasting), not the intermediate goal (mimicking the oracle).
        """
        self._check_xgb()

        if end_index is None:
            end_index = len(y)

        # 1. Align Data
        X_fit = X.iloc[start_index:end_index].copy()
        y_fit = y.iloc[start_index:end_index].copy()
        r_fit = returns.iloc[start_index:end_index].copy()
        idx = X_fit.index.intersection(y_fit.index).intersection(r_fit.index)
        X_fit, y_fit, r_fit = X_fit.loc[idx], y_fit.loc[idx], r_fit.loc[idx]

        X_np = self._as_numpy(X_fit)
        y_np = np.asarray(y_fit.values, dtype=float)
        returns2_np = np.asarray(r_fit.values, dtype=float) ** 2
        init_var_global = self._initial_variance(returns2_np, self.init_window)

        # Pre-calculate alpha_star for the whole dataset to save time
        # (We use a dummy pass to get the 'perfect' gates)
        # Note: In a strict causal sense, we should recalc alpha_star per fold
        # based on that fold's v_state, but using global alpha_star is a
        # standard approximation for speed in the alternating method.
        v_state_global, _ = self._filter_state_and_forecast(
            returns2_np,
            np.full(len(y_np), 0.05),  # Dummy alpha to get v_state
            init_var_global,
        )
        alpha_star_global = self._alpha_star_for_loss(
            y=y_np, returns2=returns2_np, v_state=v_state_global, alpha_fallback=0.05
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for params_cand in param_grid:
            p = dict(self.xgb_params)
            p.update(params_cand)
            # Match the logic in fit():
            p["objective"] = (
                "reg:logistic"
                if self.loss in {"mse", "qlike"}
                else "reg:pseudohubererror"
            )

            fold_scores = []

            for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X_np)):
                # A. Slice Data
                X_tr, X_va = X_np[train_idx], X_np[valid_idx]
                # Train Labels: ALPHA STAR (The Proxy)
                astar_tr = alpha_star_global[train_idx]
                # Valid Labels: REALIZED VAR (The Truth)
                y_va = y_np[valid_idx]
                r2_va = returns2_np[valid_idx]

                # B. Train on Proxy
                dtrain = xgb.DMatrix(X_tr, label=astar_tr)
                booster = xgb.train(
                    params=p, dtrain=dtrain, num_boost_round=self.num_boost_round
                )

                # C. Predict Alpha
                dvalid = xgb.DMatrix(X_va)
                # Clip is required if p["objective"] is reg:pseudohuber
                alpha_pred = np.clip(booster.predict(dvalid).astype(float), 0.0, 1.0)

                # D. Validate on "Truth" (Warm Start Recursion)
                # Heuristic v0 for validation segment
                v0_va = float(np.mean(returns2_np[train_idx][-20:]))

                # Run filter with predicted alphas
                _, yhat = self._filter_state_and_forecast(r2_va, alpha_pred, v0_va)

                fold_loss = self._loss_value(
                    yhat,
                    y_va,
                    loss=self.loss,
                    huber_delta=self.huber_delta,
                    qlike_epsilon=self.qlike_epsilon,
                )
                fold_scores.append(fold_loss)

            avg_score = np.mean(fold_scores)
            logger.info("Alt CV Params: %s | MSE: %.6f", params_cand, avg_score)
            results.append((avg_score, params_cand))

        results.sort(key=lambda x: x[0])
        return results

    def run_cv_e2e(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        returns: pd.Series,
        param_grid: Iterable[Dict[str, Any]],
        n_splits: int = 5,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Run Time-Series CV (End-to-End) to find optimal hyperparameters.

        Uses an expanding window strategy with 'warm start' initialization for
        validation folds to robustly test recursive volatility performance.
        """
        self._check_xgb()

        if end_index is None:
            end_index = len(y)

        # 1. Align Data
        X_fit = X.iloc[start_index:end_index].copy()
        y_fit = y.iloc[start_index:end_index].copy()
        r_fit = returns.iloc[start_index:end_index].copy()
        idx = X_fit.index.intersection(y_fit.index).intersection(r_fit.index)
        X_fit = X_fit.loc[idx]
        y_fit = y_fit.loc[idx]
        r_fit = r_fit.loc[idx]

        X_np = self._as_numpy(X_fit)
        y_np = np.asarray(y_fit.values, dtype=float)
        returns2_np = np.asarray(r_fit.values, dtype=float) ** 2
        init_var_global = self._initial_variance(returns2_np, self.init_window)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        logger.info(f"Starting E2E CV with {n_splits} splits on {len(X_np)} rows.")

        for params_cand in param_grid:
            # Merge candidate params with defaults/existing
            p = dict(self.xgb_params)
            p.update(params_cand)

            fold_scores = []

            for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X_np)):
                # A. Slice Data
                X_tr, X_va = X_np[train_idx], X_np[valid_idx]
                y_tr, y_va = y_np[train_idx], y_np[valid_idx]
                r2_tr, r2_va = returns2_np[train_idx], returns2_np[valid_idx]

                # B. Warm Start Initialization
                v0_tr = init_var_global
                # Heuristic: use mean realized var of last 20 days of train as proxy for v_t
                lookback = 20
                if len(r2_tr) >= lookback:
                    v0_va = float(np.mean(r2_tr[-lookback:]))
                else:
                    v0_va = init_var_global

                # C. Closures for this fold
                # _make_e2e_objective returns (obj, feval), we bind them to slice data
                obj_closure = self._make_e2e_objective(
                    returns2=r2_tr, y=y_tr, init_var=v0_tr
                )[0]
                feval_closure = self._make_e2e_objective(
                    returns2=r2_va, y=y_va, init_var=v0_va
                )[1]

                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dvalid = xgb.DMatrix(X_va, label=y_va)

                # D. Train
                booster = xgb.train(
                    params=p,
                    dtrain=dtrain,
                    num_boost_round=self.num_boost_round,
                    obj=obj_closure,
                    custom_metric=feval_closure,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=p.get("early_stopping_rounds", 10),
                    verbose_eval=False,
                )

                # Best score is the metric (e.g., mse_e2e) on validation
                fold_scores.append(booster.best_score)

            avg_score = np.mean(fold_scores)
            logger.info("CV Params: %s | Score: %.6f", params_cand, avg_score)
            results.append((avg_score, params_cand))

        # Return best params first
        results.sort(key=lambda x: x[0])
        return results
