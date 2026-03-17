"""Backend-agnostic VolGRU volatility model wrapper."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, Iterable

import joblib
import numpy as np

from .base_model import BaseVolatilityModel

logger = logging.getLogger(__name__)
from .volgru_config import VolGRUConfig
from .volgru_utils import (
    _check_schema,
    _coerce_X,
    _set_schema,
    align_returns_next,
    feature_group_penalty_torch,
    gate_entropy_term_torch,
    nll_gaussian_torch,
    qlike_torch,
)


class VolGRUModel(BaseVolatilityModel):
    """Configurable GRU-style volatility model (torch backend)."""

    def __init__(
        self,
        config: VolGRUConfig | None = None,
        *,
        random_state: int | None = None,
    ) -> None:
        self.config = config or VolGRUConfig()
        self.random_state = random_state

        self.feature_names_: list[str] | None = None
        self.feature_schema_hash_: str | None = None
        self.n_features_: int | None = None

        self.init_var_: float | None = None
        self.last_var_: float | None = None
        self.last_state_: np.ndarray | None = None

        self.torch_module_: Any = None
        self.params_: Any = None
        self.training_loss_history_: list[float] = []
        self.is_fitted_: bool = False

    def _validate_config(self) -> None:
        if self.config.batch_size is not None:
            raise NotImplementedError(
                "Mini-batch sequence training is not implemented. "
                "Set batch_size=None for full-sequence training."
            )

    def _ensure_torch_module(self, n_features: int) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for backend='torch'.") from exc

        from .volgru_torch import VolGRUModelTorch

        if self.torch_module_ is None:
            if self.random_state is not None:
                torch.manual_seed(int(self.random_state))
            self.torch_module_ = VolGRUModelTorch(
                config=self.config,
                n_features=n_features,
                dtype=torch.float32,
            )
            return
        if int(self.torch_module_.n_features) != int(n_features):
            raise ValueError(
                "Feature count mismatch with existing torch model: "
                f"{self.torch_module_.n_features} vs {n_features}"
            )

    def _backend_feature_count(self) -> int | None:
        if self.torch_module_ is None:
            return None
        return int(self.torch_module_.n_features)

    def set_gate_beta(self, beta: np.ndarray) -> "VolGRUModel":
        """Set gate beta explicitly (useful for STES reduction tests)."""
        beta_np = np.asarray(beta, dtype=float)
        if beta_np.ndim == 1:
            n_features = int(beta_np.shape[0])
        elif beta_np.ndim == 2:
            n_features = int(beta_np.shape[0])
        else:
            raise ValueError(f"beta must be 1D or 2D. Got shape {beta_np.shape}")

        self._ensure_torch_module(n_features=n_features)
        assert self.torch_module_ is not None
        import torch

        beta_t = torch.as_tensor(beta_np, dtype=torch.float32)
        self.torch_module_.set_gate_beta(beta_t)
        return self

    def get_gate_beta(self) -> np.ndarray:
        """Get current gate beta vector."""
        if self.torch_module_ is None:
            raise ValueError(
                "Model not initialized. Call fit() or set_gate_beta() first."
            )
        beta_t = self.torch_module_.get_gate_beta()
        return beta_t.detach().cpu().numpy()

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        returns: Any = None,
        start_index: int = 0,
        end_index: int | None = None,
        perform_cv: bool = False,
        cv_grid: Iterable[Dict[str, Any]] | None = None,
        cv_splits: int = 3,
        beta_ref: np.ndarray | None = None,
        init_var: float | None = None,
        **_: Any,
    ) -> "VolGRUModel":
        """Fit the VolGRU model on a contiguous fixed-split sequence.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (T, p)
            Feature matrix.
        y : array-like, shape (T,)
            Target vector (next-day squared return).
        returns : array-like, shape (T,)
            Return series.
        start_index : int, default 0
            Burn-in: residuals before this index are excluded from the loss.
        end_index : int or None
            End of the residual window (exclusive).  Defaults to *T*.
        perform_cv : bool, default False
            If True, run time-series cross-validation over ``cv_grid``
            before the final fit.
        cv_grid : iterable of dict, optional
            Parameter grid for CV.  Each dict may contain any subset of
            ``VolGRUConfig`` training hyperparameters (e.g. ``lr``,
            ``weight_decay_gate``, ``weight_decay_candidate``,
            ``beta_stay_close_lambda``, ``gate_entropy_lambda``,
            ``max_epochs``, ``early_stopping_patience``).
        cv_splits : int, default 3
            Number of ``TimeSeriesSplit`` folds for CV.
        beta_ref : ndarray or None
            Reference gate vector for stay-close regularisation.
        init_var : float or None
            Initial variance.  Defaults to ``r[0]**2``.

        Returns
        -------
        self
        """
        self._validate_config()
        if perform_cv:
            if cv_grid is None:
                raise ValueError("If perform_cv=True, you must provide a cv_grid.")
        if returns is None:
            raise ValueError("fit() requires returns=...")

        X_np, cols = _coerce_X(X)
        y_np = np.asarray(y, dtype=float).reshape(-1)
        r_np = np.asarray(returns, dtype=float).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if end_index is None:
            end_index = int(len(X_np))
        if not (0 <= int(start_index) < int(end_index) <= int(len(X_np))):
            raise ValueError(
                "Invalid [start_index, end_index) window: "
                f"start_index={start_index}, end_index={end_index}, n={len(X_np)}."
            )
        if not (len(X_np) == len(y_np) == len(r_np)):
            raise ValueError("X, y, returns must have the same length.")

        _set_schema(self, cols, X_np.shape[1])
        beta_ref_np = None if beta_ref is None else np.asarray(beta_ref, dtype=float)
        if beta_ref_np is not None and beta_ref_np.shape[0] != X_np.shape[1]:
            raise ValueError(
                f"beta_ref leading dimension mismatch: expected {X_np.shape[1]}, got {beta_ref_np.shape[0]}"
            )

        # -----------------------------------------------------------------
        # Cross-validation block
        # -----------------------------------------------------------------
        if perform_cv and cv_grid is not None:
            logger.info("Running VolGRU CV with %d splits...", cv_splits)
            cv_results = self.run_cv(
                X_np,
                y_np,
                returns=r_np,
                param_grid=cv_grid,
                n_splits=cv_splits,
                start_index=start_index,
                beta_ref=beta_ref_np,
            )
            best_score, avg_epochs, best_params = cv_results[0]
            logger.info(
                "VolGRU CV complete. Best score=%.6e, avg_epochs=%d, params=%s",
                best_score,
                avg_epochs,
                best_params,
            )

            # Apply winning hyperparameters to self.config
            for key, val in best_params.items():
                if key in self._CV_TUNABLE_FIELDS:
                    setattr(self.config, key, val)

            # Mirror XGBoost CV styling: apply the average stopping limit from CV
            # then run on 100% of data (val_fraction=0) to be comparable with STES framework.
            self.config.val_fraction = 0.0
            self.config.max_epochs = avg_epochs

            # Reset backend modules so the final fit starts fresh
            self.torch_module_ = None
            self.params_ = None

        X_block = X_np[:end_index]
        y_block = y_np[:end_index]
        r_block = r_np[:end_index]
        init_var_fit = float(r_block[0] ** 2) if init_var is None else float(init_var)

        self._ensure_torch_module(n_features=X_block.shape[1])
        self._fit_torch(
            X=X_block,
            y=y_block,
            returns=r_block,
            start_index=int(start_index),
            init_var=init_var_fit,
            beta_ref=beta_ref_np,
        )

        sigma2_next, _, _ = self.predict_with_gates(
            X_block, returns=r_block, init_var=init_var_fit
        )
        self.init_var_ = init_var_fit
        self.last_var_ = float(sigma2_next[-1]) if len(sigma2_next) else init_var_fit
        self.last_state_ = None
        self.is_fitted_ = True
        return self

    # -----------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------

    _CV_TUNABLE_FIELDS = frozenset(
        {
            "lr",
            "weight_decay_gate",
            "weight_decay_candidate",
            "beta_stay_close_lambda",
            "gate_entropy_lambda",
            "gate_feature_group_lambda",
            "max_epochs",
            "early_stopping_patience",
        }
    )

    def run_cv(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        *,
        returns: np.ndarray,
        param_grid: Iterable[Dict[str, Any]],
        n_splits: int = 3,
        start_index: int = 0,
        beta_ref: np.ndarray | None = None,
    ) -> list[tuple[float, int, Dict[str, Any]]]:
        """Time-series cross-validation over ``VolGRUConfig`` hyperparameters.

        Uses ``sklearn.model_selection.TimeSeriesSplit`` with expanding
        windows, matching the protocol in ``STESModel.run_cv``. Also tracks
        and returns the average optimal stopping epoch across folds.

        Parameters
        ----------
        X_np : ndarray, shape (T, p)
        y_np : ndarray, shape (T,)
        returns : ndarray, shape (T,)
        param_grid : iterable of dict
            Each dict maps tunable ``VolGRUConfig`` field names to values.
        n_splits : int, default 3
        start_index : int, default 0
            Burn-in passed to each fold's ``fit()``.
        beta_ref : ndarray or None
            Passed through to each fold's ``fit()``.

        Returns
        -------
        list of (float, int, dict)
            ``(mean_mse, mean_epochs, params)`` tuples sorted ascending by score.
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results: list[tuple[float, int, Dict[str, Any]]] = []

        for params_cand in param_grid:
            fold_scores: list[float] = []
            fold_epochs: list[int] = []

            for train_idx, valid_idx in tscv.split(X_np):
                X_tr = X_np[train_idx]
                y_tr = y_np[train_idx]
                r_tr = returns[train_idx]
                X_va = X_np[valid_idx]
                y_va = y_np[valid_idx]
                r_va = returns[valid_idx]

                # Build per-fold config with candidate overrides
                fold_cfg = copy.deepcopy(self.config)
                for key, val in params_cand.items():
                    if key in self._CV_TUNABLE_FIELDS:
                        setattr(fold_cfg, key, val)

                fold_model = VolGRUModel(
                    config=fold_cfg, random_state=self.random_state
                )

                # Fit on the training fold
                si = min(int(start_index), max(len(X_tr) - 1, 0))
                fold_model.fit(
                    X_tr,
                    y_tr,
                    returns=r_tr,
                    start_index=si,
                    end_index=len(X_tr),
                    beta_ref=beta_ref,
                )

                if fold_model.training_loss_history_ is not None:
                    fold_epochs.append(len(fold_model.training_loss_history_))
                else:
                    fold_epochs.append(fold_model.config.max_epochs)

                # Predict on validation fold with warm-started init_var
                tail = min(20, len(r_tr))
                v0_val = float(np.mean(r_tr[-tail:] ** 2)) if tail > 0 else 1e-8
                sigma2_va = fold_model.predict(X_va, returns=r_va, init_var=v0_val)

                if self.config.loss_mode == "qlike":
                    y_safe = np.clip(y_va, self.config.eps, None)
                    pred_safe = np.clip(sigma2_va, self.config.eps, None)
                    ratio = y_safe / pred_safe
                    fold_score = float(np.mean(ratio - np.log(ratio) - 1.0))
                else:
                    fold_score = float(np.mean((y_va - sigma2_va) ** 2))
                fold_scores.append(fold_score)

            avg_score = float(np.mean(fold_scores))
            avg_epochs = int(np.round(np.mean(fold_epochs)))
            results.append((avg_score, avg_epochs, dict(params_cand)))

        results.sort(key=lambda x: x[0])
        return results

    def _fit_torch(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray,
        start_index: int,
        init_var: float,
        beta_ref: np.ndarray | None,
    ) -> None:
        import torch

        assert self.torch_module_ is not None
        self.torch_module_.train()

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        r_t = torch.as_tensor(returns, dtype=torch.float32)
        init_var_t = torch.tensor(float(init_var), dtype=torch.float32)
        r_next_t = torch.as_tensor(align_returns_next(returns), dtype=torch.float32)

        gate_params = list(self.torch_module_.gate_parameters())
        cand_params = list(self.torch_module_.candidate_parameters())
        param_groups: list[dict[str, Any]] = []
        if gate_params:
            param_groups.append(
                {
                    "params": gate_params,
                    "weight_decay": float(self.config.weight_decay_gate),
                }
            )
        if cand_params:
            param_groups.append(
                {
                    "params": cand_params,
                    "weight_decay": float(self.config.weight_decay_candidate),
                }
            )
        if not param_groups:
            raise ValueError("No trainable parameters found for torch backend.")

        optimizer = torch.optim.AdamW(param_groups, lr=float(self.config.lr))
        beta_ref_t = (
            None if beta_ref is None else torch.as_tensor(beta_ref, dtype=torch.float32)
        )

        best_loss = float("inf")
        best_state = copy.deepcopy(self.torch_module_.state_dict())
        no_improve = 0
        history: list[float] = []
        val_history: list[float] = []

        # ----- validation split for early stopping -----
        n_total = len(y) - int(start_index)
        vf = float(self.config.val_fraction)
        if vf > 0.0 and n_total > 1:
            n_val = max(1, int(n_total * vf))
            val_split = len(y) - n_val
            train_slice = slice(int(start_index), val_split)
            val_slice = slice(val_split, len(y))
            use_val = True
        else:
            train_slice = slice(int(start_index), len(y))
            val_slice = None
            use_val = False

        for _epoch in range(int(self.config.max_epochs)):
            optimizer.zero_grad(set_to_none=True)
            sigma2_next, z_t, _v_cand, _final_var = self.torch_module_.forward_sequence(
                X=X_t,
                returns=r_t,
                init_var=init_var_t,
            )

            pred = sigma2_next[train_slice]
            target = y_t[train_slice]
            if self.config.loss_mode == "mse_r2":
                loss = torch.mean((target - pred) ** 2)
            elif self.config.loss_mode == "nll_gaussian":
                loss = nll_gaussian_torch(
                    returns_next=r_next_t[train_slice],
                    sigma2_next=pred,
                    eps=self.config.eps,
                )
            elif self.config.loss_mode == "qlike":
                loss = qlike_torch(
                    y_true=target,
                    y_pred=pred,
                    eps=self.config.eps,
                )
            else:
                raise ValueError(f"Unsupported loss_mode={self.config.loss_mode!r}")

            if beta_ref_t is not None and self.config.beta_stay_close_lambda > 0.0:
                beta_now = self.torch_module_.get_gate_beta()
                if beta_ref_t.ndim == 1 and beta_now.ndim == 2:
                    beta_ref_use = beta_ref_t.unsqueeze(1).expand_as(beta_now)
                else:
                    beta_ref_use = beta_ref_t
                loss = loss + float(self.config.beta_stay_close_lambda) * torch.sum(
                    (beta_now - beta_ref_use) ** 2
                )
            if self.config.gate_entropy_lambda != 0.0:
                entropy_term = gate_entropy_term_torch(
                    z_t[train_slice],
                    eps=self.config.eps,
                )
                loss = loss + float(self.config.gate_entropy_lambda) * entropy_term
            if self.config.gate_feature_group_lambda > 0.0:
                beta_now_grp = self.torch_module_.get_gate_beta()
                loss = loss + float(
                    self.config.gate_feature_group_lambda
                ) * feature_group_penalty_torch(beta_now_grp, eps=self.config.eps)

            loss.backward()
            optimizer.step()

            train_loss_val = float(loss.detach().cpu().item())
            history.append(train_loss_val)

            # ----- early-stopping metric -----
            if use_val:
                with torch.no_grad():
                    val_pred = sigma2_next[val_slice]
                    val_target = y_t[val_slice]
                    if self.config.loss_mode == "mse_r2":
                        es_metric = float(
                            torch.mean((val_target - val_pred) ** 2).item()
                        )
                    elif self.config.loss_mode == "nll_gaussian":
                        es_metric = float(
                            nll_gaussian_torch(
                                returns_next=r_next_t[val_slice],
                                sigma2_next=val_pred,
                                eps=self.config.eps,
                            ).item()
                        )
                    else:
                        es_metric = float(
                            qlike_torch(
                                y_true=val_target,
                                y_pred=val_pred,
                                eps=self.config.eps,
                            ).item()
                        )
                val_history.append(es_metric)
            else:
                es_metric = train_loss_val

            if es_metric < best_loss - 1e-12:
                best_loss = es_metric
                best_state = copy.deepcopy(self.torch_module_.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= int(self.config.early_stopping_patience):
                break

        self.torch_module_.load_state_dict(best_state)
        self.training_loss_history_ = history
        self.validation_loss_history_ = val_history if use_val else None

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict next-step variance sequence."""
        sigma2_next, _, _ = self.predict_with_gates(X, **kwargs)
        return sigma2_next

    def predict_with_gates(
        self, X: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict variance and return update gates and candidate variances."""
        returns = kwargs.pop("returns", None)
        init_var = kwargs.pop("init_var", None)
        if returns is None:
            raise ValueError("predict_with_gates() requires returns=...")

        X_np, cols = _coerce_X(X)
        r_np = np.asarray(returns, dtype=float).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_np.shape}.")
        if len(X_np) != len(r_np):
            raise ValueError(
                f"Length mismatch: len(X)={len(X_np)} vs len(returns)={len(r_np)}"
            )

        _check_schema(self, cols, X_np)
        backend_n_features = self._backend_feature_count()
        if backend_n_features is not None and X_np.shape[1] != backend_n_features:
            raise ValueError(
                f"Feature count mismatch with initialized backend: {backend_n_features} vs {X_np.shape[1]}"
            )

        init_var_pred = float(r_np[0] ** 2) if init_var is None else float(init_var)

        if self.torch_module_ is None:
            raise ValueError(
                "Model not initialized. Call fit() or set_gate_beta() first."
            )
        sigma2_next, z_t, v_cand_t, final_var = self._predict_torch(
            X=X_np,
            returns=r_np,
            init_var=init_var_pred,
        )

        final_state = np.asarray(final_var, dtype=float)
        self.last_state_ = final_state.copy()
        self.last_var_ = (
            float(np.mean(final_state)) if final_state.size else float(init_var_pred)
        )
        if self.init_var_ is None:
            self.init_var_ = float(init_var_pred)
        return sigma2_next, z_t, v_cand_t

    def _predict_torch(
        self,
        *,
        X: np.ndarray,
        returns: np.ndarray,
        init_var: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch

        assert self.torch_module_ is not None
        self.torch_module_.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        r_t = torch.as_tensor(returns, dtype=torch.float32)
        init_var_t = torch.tensor(float(init_var), dtype=torch.float32)
        with torch.no_grad():
            sigma2_next, z_t, v_cand_t, final_var = self.torch_module_.forward_sequence(
                X=X_t,
                returns=r_t,
                init_var=init_var_t,
            )
        return (
            sigma2_next.detach().cpu().numpy(),
            z_t.detach().cpu().numpy(),
            v_cand_t.detach().cpu().numpy(),
            final_var.detach().cpu().numpy(),
        )

    def save(self, filename: str, *, format: str = "joblib") -> None:
        """Save model artifact to disk."""
        if format != "joblib":
            raise ValueError("format must be 'joblib'")
        path = filename if filename.endswith(".joblib") else filename + ".joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, filename: str) -> "VolGRUModel":
        """Load a previously saved VolGRU model artifact."""
        if filename.endswith(".joblib"):
            return joblib.load(filename)
        path = filename + ".joblib"
        if os.path.exists(path):
            return joblib.load(path)
        raise FileNotFoundError(filename)
