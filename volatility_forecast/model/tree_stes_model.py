"""
XGBoost-STES model: a tree-ensemble transition function inside an EWMA / STES-style variance filter.

This module implements a hybrid model with the *mathematically natural* STES recursion

    v_{t+1} = v_t + α_t (r_t^2 - v_t)
    α_t = expit( F(X_t) )

where:
- X_t contains information available at time t (end of day t for daily data)
- r_t is the return over (t-1, t]
- the forecast made at time t for the next period is v_{t+1|t}

- The dataset row indexed by time t stores:
    - features X_t
    - contemporaneous return r_t (for the recursion update)
    - label y_t := r_{t+1}^2 (next-day squared return) aligned to time t
- During training and evaluation, we compare y_t against the forecast made at time t:

    ŷ_t = v_{t+1|t}

`predict()` returns a time series aligned to X.index where element t is the model's
one-step-ahead variance forecast *made at t*.

Training uses the alternating supervised scheme (stable EM-like loop):
- Given a current α_t path, compute the state v_t and forecasts ŷ_t.
- Holding v_t fixed, compute the one-step optimal α*_t for each t:

    α*_t = argmin_{α in [0,1]} ( y_t - (v_t + α (r_t^2 - v_t)) )^2
         = clip( (y_t - v_t) / (r_t^2 - v_t), 0, 1 )

- Fit XGBoost to predict logit(α*_t) from X_t.
- Repeat for a few outer iterations, warm-starting the booster each time.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from scipy.special import expit, logit

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

from .base_model import BaseVolatilityModel


@dataclass
class XGBSTESFitResult:
    booster: Any
    params_used: Dict[str, Any]
    num_boost_round: int
    n_alt_iters: int
    feature_names: list[str]
    alpha_base: float


class XGBoostSTESModel(BaseVolatilityModel):
    """
    XGBoost-based STES gate trained with the alternating 
    supervised loop.
    """

    def __init__(
        self,
        *,
        xgb_params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 200,
        init_window: int = 500,
        n_alt_iters: int = 3,
        alt_clip_eps: float = 1e-6,
        random_state: Optional[int] = None,
        monotonic_constraints: Optional[dict[str, int]] = None,
        alt_objective: str = "reg:pseudohubererror",
        residual_mode: bool = True,
        denom_quantile: float = 0.05,
        min_denom_floor: float = 1e-12,
    ) -> None:
        super().__init__()
        self.xgb_params = dict(xgb_params or {})
        self.num_boost_round = int(num_boost_round)
        self.init_window = int(init_window)
        self.n_alt_iters = int(n_alt_iters)
        self.alt_clip_eps = float(alt_clip_eps)
        self.random_state = random_state

        self.monotonic_constraints = monotonic_constraints
        self.alt_objective = alt_objective
        self.residual_mode = residual_mode
        self.denom_quantile = denom_quantile
        self.min_denom_floor = min_denom_floor
        self.base_margin_: Optional[float] = None

        self.model_: Optional[Any] = None
        self.fit_result_: Optional[XGBSTESFitResult] = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _check_xgb() -> None:
        if xgb is None:
            raise ImportError(
                "xgboost is not available. Install xgboost to use XGBoostSTESModel."
            )

    @staticmethod
    def _as_numpy(X: pd.DataFrame) -> np.ndarray:
        # Ensure stable column order and dtype (float) for xgboost.
        return np.asarray(X.values, dtype=float)

    @staticmethod
    def _initial_variance(returns2: np.ndarray, init_window: int) -> float:
        w = min(int(init_window), int(len(returns2)))
        if w <= 0:
            return float("nan")
        return float(np.mean(returns2[:w]))

    @staticmethod
    def _apply_monotone_constraints(
        params: Dict[str, Any], feature_names: list[str], constraints: dict[str, int]
    ) -> None:
        """Apply XGBoost monotone constraints in the correct column order."""
        if not constraints:
            return
        ordered = [str(int(constraints.get(c, 0))) for c in feature_names]
        # XGBoost expects a string like "(1,0,-1,...)"
        params["monotone_constraints"] = "(" + ",".join(ordered) + ")"

    @staticmethod
    def _filter_state_and_forecast(
        returns2: np.ndarray, alpha: np.ndarray, init_var: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the STES recursion with *row-aligned* alpha.

        We treat each row t as:
          - state before update: v_t
          - innovation available at t: r_t^2
          - gate decided at t: alpha_t
          - forecast made at t: yhat_t = v_{t+1|t} = v_t + alpha_t (r_t^2 - v_t)

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
            a = float(alpha[t])
            v_next = v + a * (float(returns2[t]) - v)
            yhat[t] = v_next
            v = v_next
        return v_state, yhat

    @staticmethod
    def _compute_alpha_star(
        *, y: np.ndarray, returns2: np.ndarray, v_state: np.ndarray, alpha_fallback: float
    ) -> np.ndarray:
        """Compute pseudo-optimal alpha*_t holding v_t fixed (one-step)."""
        n = int(len(y))
        out = np.full(n, float(alpha_fallback), dtype=float)

        for t in range(n):
            denom = float(returns2[t] - v_state[t])
            if (not np.isfinite(denom)) or abs(denom) < 1e-12:
                a = float(alpha_fallback)
            else:
                a = float((y[t] - v_state[t]) / denom)
            out[t] = float(np.clip(a, 0.0, 1.0))
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
        **kwargs: Any,
    ) -> "XGBoostSTESModel":
        """Fit on a contiguous time block [start_index:end_index).

        Expected alignment:
        - X[t] is info at time t
        - returns[t] is r_t (realized over (t-1,t])
        - y[t] is r_{t+1}^2 (next-day squared return) aligned to time t
        """
        self._check_xgb()

        if end_index is None:
            end_index = len(y)

        X_fit = X.iloc[start_index:end_index].copy()
        y_fit = y.iloc[start_index:end_index].copy()
        r_fit = returns.iloc[start_index:end_index].copy()

        # Strict alignment (defensive)
        idx = X_fit.index.intersection(y_fit.index).intersection(r_fit.index)
        X_fit = X_fit.loc[idx]
        y_fit = y_fit.loc[idx]
        r_fit = r_fit.loc[idx]

        if len(X_fit) < 10:
            raise ValueError("Not enough rows to fit XGBoostSTESModel.")

        X_np = self._as_numpy(X_fit)
        y_np = np.asarray(y_fit.values, dtype=float)
        returns2_np = np.asarray(r_fit.values, dtype=float) ** 2

        init_var = self._initial_variance(returns2_np, self.init_window)

        # Default XGBoost params (can be overridden)
        params = dict(self.xgb_params)
        params.setdefault("objective", self.alt_objective)
        params.setdefault("max_depth", 3)
        params.setdefault("eta", 0.05)
        params.setdefault("subsample", 1.0)
        params.setdefault("colsample_bytree", 1.0)
        params.setdefault("min_child_weight", 1.0)
        params.setdefault("lambda", 1.0)
        params.setdefault("alpha", 0.0)
        if self.random_state is not None:
            params.setdefault("seed", int(self.random_state))

        feature_names = list(X_fit.columns)
        self._apply_monotone_constraints(params, feature_names, self.monotonic_constraints or {})

        # Baseline alpha for fallback/initialization: try ES alpha if available.
        alpha_base = 0.06
        try:
            from .es_model import ESModel  # type: ignore

            m = ESModel(random_state=self.random_state)
            m.fit(X_fit, y_fit, returns=r_fit, start_index=0, end_index=len(X_fit))
            alpha_base = float(getattr(m, "alpha_", alpha_base))
        except Exception:
            pass

        # Store base margin for residual learning
        base = logit(np.clip(alpha_base, 1e-6, 1.0 - 1e-6))
        self.base_margin_ = float(base)

        booster = self._fit_alternating_supervised(
            X_np=X_np,
            y_np=y_np,
            returns2_np=returns2_np,
            init_var=init_var,
            params=params,
            alpha_base=alpha_base,
            feature_names=feature_names,
        )

        self.model_ = booster
        self.fit_result_ = XGBSTESFitResult(
            booster=booster,
            params_used=params,
            num_boost_round=self.num_boost_round,
            n_alt_iters=self.n_alt_iters,
            feature_names=feature_names,
            alpha_base=float(alpha_base),
        )
        return self

    def _fit_alternating_supervised(
        self,
        *,
        X_np: np.ndarray,
        y_np: np.ndarray,
        returns2_np: np.ndarray,
        init_var: float,
        params: Dict[str, Any],
        alpha_base: float,
        feature_names: list[str],
    ) -> Any:
        """A1: alternating supervised training with booster warm-start."""
        n = int(len(y_np))
        if n != int(len(returns2_np)):
            raise ValueError("Length mismatch: y and returns must be aligned to rows.")

        alpha_pred = np.full(n, float(alpha_base), dtype=float)
        booster = None
        eps = float(self.alt_clip_eps)

        for _ in range(max(1, int(self.n_alt_iters))):
            # 1) Compute v_t state and one-step forecasts yhat_t = v_{t+1|t}
            v_state, _ = self._filter_state_and_forecast(returns2_np, alpha_pred, init_var)

            # 2) Compute pseudo targets alpha*_t holding v_t fixed
            alpha_star = self._compute_alpha_star(
                y=y_np,
                returns2=returns2_np,
                v_state=v_state,
                alpha_fallback=float(alpha_base),
            )
            alpha_star = np.clip(alpha_star, eps, 1.0 - eps)
            target = logit(alpha_star)

            # --- Denominator masking for noisy alpha* labels ---
            den = returns2_np - v_state
            abs_den = np.abs(den)
            finite_mask = np.isfinite(target) & np.isfinite(den)
            finite_abs_den = abs_den[finite_mask]
            if finite_abs_den.size > 0:
                floor = max(self.min_denom_floor, np.quantile(finite_abs_den, self.denom_quantile))
            else:
                floor = self.min_denom_floor
            mask = np.isfinite(target) & np.isfinite(den) & (abs_den >= floor)
            if mask.sum() < max(10, int(0.05 * n)):
                mask = np.ones(n, dtype=bool)

            # 3) Fit (warm-start on the previous booster), with base_margin if enabled
            dtrain = xgb.DMatrix(
                X_np[mask],
                label=target[mask].astype(float),
                feature_names=feature_names,
            )
            if self.residual_mode and self.base_margin_ is not None:
                dtrain.set_base_margin(np.full(mask.sum(), self.base_margin_, dtype=float))

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train")],
                verbose_eval=False,
                xgb_model=booster,
            )

            # 4) Update alpha_pred from the current booster
            d_all = xgb.DMatrix(X_np, feature_names=feature_names)
            if self.residual_mode and self.base_margin_ is not None:
                d_all.set_base_margin(np.full(n, self.base_margin_, dtype=float))
            score = booster.predict(d_all)
            alpha_pred = expit(score.astype(float))

        return booster

    def predict(
        self,
        X: pd.DataFrame,
        *,
        returns: pd.Series,
        start_index: int = 0,
        end_index: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Predict one-step-ahead variance aligned to X (forecast made at time t)."""
        self._check_xgb()
        if self.model_ is None:
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
        if self.residual_mode and self.base_margin_ is not None:
            dtest.set_base_margin(np.full(len(Xp), self.base_margin_, dtype=float))
        score = self.model_.predict(dtest)
        alpha = expit(score.astype(float))

        init_var = self._initial_variance(r2, self.init_window)
        _, yhat = self._filter_state_and_forecast(r2, alpha, init_var)

        return pd.Series(yhat, index=Xp.index, name="yhat")

    def get_alphas(
        self, X: pd.DataFrame, *, start_index: int = 0, end_index: Optional[int] = None
    ) -> pd.Series:
        """Return α_t series (row-aligned; used to update from r_t^2 to v_{t+1})."""
        self._check_xgb()
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        if end_index is None:
            end_index = len(X)
        Xp = X.iloc[start_index:end_index]
        feature_names = list(Xp.columns)
        d = xgb.DMatrix(self._as_numpy(Xp), feature_names=feature_names)
        if self.residual_mode and self.base_margin_ is not None:
            d.set_base_margin(np.full(len(Xp), self.base_margin_, dtype=float))
        score = self.model_.predict(d)
        alpha = expit(score.astype(float))
        return pd.Series(alpha, index=Xp.index, name="alpha")
