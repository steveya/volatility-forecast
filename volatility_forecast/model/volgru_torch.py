"""Torch backend for VolGRU volatility models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .volgru_config import VolGRUConfig
from .volgru_utils import positive_transform_torch


class VolGRUCellTorch(nn.Module):
    """Single-step VolGRU cell for scalar or vector variance state updates."""

    def __init__(
        self,
        config: VolGRUConfig,
        n_features: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.config = config
        self.n_features = int(n_features)
        self.state_dim = int(config.state_dim)
        self.dtype = dtype

        self.gate_beta = nn.Parameter(torch.zeros(self.n_features, self.state_dim, dtype=dtype))
        if self.config.gate_mode == "gru_linear":
            self.gate_state_weight = nn.Parameter(torch.zeros(self.state_dim, dtype=dtype))
        else:
            self.register_parameter("gate_state_weight", None)

        if self.config.use_reset_gate:
            self.reset_gamma = nn.Parameter(torch.zeros(self.n_features, self.state_dim, dtype=dtype))
            self.reset_state_weight = nn.Parameter(torch.zeros(self.state_dim, dtype=dtype))
        else:
            self.register_parameter("reset_gamma", None)
            self.register_parameter("reset_state_weight", None)

        if self.config.candidate_mode == "stes_r2":
            self.candidate_linear = None
            self.candidate_mlp = None
        elif self.config.candidate_mode == "linear_pos":
            self.candidate_linear = nn.Linear(4 + self.state_dim, self.state_dim, bias=True).to(dtype=dtype)
            self.candidate_mlp = None
        elif self.config.candidate_mode == "mlp_pos":
            hidden = int(self.config.mlp_hidden_dim)
            self.candidate_linear = None
            self.candidate_mlp = nn.Sequential(
                nn.Linear(3 + self.state_dim, hidden, bias=True),
                nn.Tanh(),
                nn.Linear(hidden, self.state_dim, bias=True),
            ).to(dtype=dtype)
        else:
            raise ValueError(f"Unsupported candidate_mode={self.config.candidate_mode!r}")

    def gate_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.gate_beta]
        if self.gate_state_weight is not None:
            params.append(self.gate_state_weight)
        if self.reset_gamma is not None:
            params.append(self.reset_gamma)
        if self.reset_state_weight is not None:
            params.append(self.reset_state_weight)
        return params

    def candidate_parameters(self) -> list[nn.Parameter]:
        if self.config.candidate_mode == "stes_r2":
            return []
        if self.candidate_linear is not None:
            return list(self.candidate_linear.parameters())
        if self.candidate_mlp is not None:
            return list(self.candidate_mlp.parameters())
        return []

    def _compute_update_gate(self, X_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        gate_logit = X_t @ self.gate_beta
        if self.gate_state_weight is not None:
            gate_logit = gate_logit + self.gate_state_weight * v_t
        return torch.sigmoid(gate_logit)

    def _compute_reset_gate(
        self, X_t: torch.Tensor, v_t: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.config.use_reset_gate:
            return None
        assert self.reset_gamma is not None
        reset_logit = X_t @ self.reset_gamma
        assert self.reset_state_weight is not None
        reset_logit = reset_logit + self.reset_state_weight * v_t
        return torch.sigmoid(reset_logit)

    def _compute_candidate(
        self,
        r_t: torch.Tensor,
        v_t: torch.Tensor,
        reset_gate_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.config.candidate_mode == "stes_r2":
            return torch.ones_like(v_t) * (r_t * r_t)

        state_for_cand = v_t if reset_gate_t is None else reset_gate_t * v_t
        abs_r = torch.abs(r_t)
        r2 = r_t * r_t

        if self.config.candidate_mode == "linear_pos":
            assert self.candidate_linear is not None
            base = torch.stack([torch.ones_like(r_t), r_t, abs_r, r2])
            x = torch.cat([base, state_for_cand], dim=0)
            raw = self.candidate_linear(x.unsqueeze(0)).reshape(self.state_dim)
            return positive_transform_torch(raw, self.config.positive_transform)

        if self.config.candidate_mode == "mlp_pos":
            assert self.candidate_mlp is not None
            base = torch.stack([r_t, abs_r, r2])
            x = torch.cat([base, state_for_cand], dim=0)
            raw = self.candidate_mlp(x.unsqueeze(0)).reshape(self.state_dim)
            return positive_transform_torch(raw, self.config.positive_transform)

        raise ValueError(f"Unsupported candidate_mode={self.config.candidate_mode!r}")

    def forward_step(
        self,
        v_t: torch.Tensor,
        r_t: torch.Tensor,
        X_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Single recursion step: return v_{t+1}, z_t, v_cand_t, reset_t."""
        z_t = self._compute_update_gate(X_t, v_t)
        reset_t = self._compute_reset_gate(X_t, v_t)
        v_cand_t = self._compute_candidate(r_t, v_t, reset_t)
        v_next = (1.0 - z_t) * v_t + z_t * v_cand_t
        v_next = torch.clamp(v_next, min=float(self.config.eps))
        return v_next, z_t, v_cand_t, reset_t

    def forward_sequence(
        self,
        X: torch.Tensor,
        returns: torch.Tensor,
        init_var: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full sequence and return sigma2_next, z, v_cand, final_var."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {tuple(X.shape)}")
        if returns.ndim != 1:
            raise ValueError(f"returns must be 1D. Got shape {tuple(returns.shape)}")
        if X.shape[0] != returns.shape[0]:
            raise ValueError(
                f"Length mismatch: len(X)={X.shape[0]} vs len(returns)={returns.shape[0]}"
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature mismatch: cell expects {self.n_features}, got {X.shape[1]}"
            )

        n = int(returns.shape[0])
        sigma2_next = torch.empty(n, dtype=returns.dtype, device=returns.device)
        z_all = torch.empty((n, self.state_dim), dtype=returns.dtype, device=returns.device)
        cand_all = torch.empty((n, self.state_dim), dtype=returns.dtype, device=returns.device)

        if init_var is None:
            v_t = torch.full(
                (self.state_dim,),
                float((returns[0] * returns[0]).item()),
                dtype=returns.dtype,
                device=returns.device,
            )
        else:
            if torch.is_tensor(init_var):
                init_var_t = init_var.to(dtype=returns.dtype, device=returns.device)
            else:
                init_var_t = torch.tensor(float(init_var), dtype=returns.dtype, device=returns.device)
            if init_var_t.ndim == 0:
                v_t = torch.full(
                    (self.state_dim,),
                    float(init_var_t.item()),
                    dtype=returns.dtype,
                    device=returns.device,
                )
            else:
                v_t = init_var_t.reshape(self.state_dim)

        for t in range(n):
            v_t, z_t, cand_t, _ = self.forward_step(v_t, returns[t], X[t])
            sigma2_next[t] = v_t.mean()
            z_all[t] = z_t
            cand_all[t] = cand_t

        if self.state_dim == 1:
            z_all = z_all.reshape(n)
            cand_all = cand_all.reshape(n)

        return sigma2_next, z_all, cand_all, v_t


class VolGRUModelTorch(nn.Module):
    """Torch module wrapper around :class:`VolGRUCellTorch`."""

    def __init__(
        self,
        config: VolGRUConfig,
        n_features: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_features = int(n_features)
        self.cell = VolGRUCellTorch(config=config, n_features=n_features, dtype=dtype)

    def forward_sequence(
        self,
        X: torch.Tensor,
        returns: torch.Tensor,
        init_var: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cell.forward_sequence(X=X, returns=returns, init_var=init_var)

    def gate_parameters(self) -> list[nn.Parameter]:
        return self.cell.gate_parameters()

    def candidate_parameters(self) -> list[nn.Parameter]:
        return self.cell.candidate_parameters()

    def set_gate_beta(self, beta: torch.Tensor) -> None:
        if beta.ndim == 1:
            if beta.shape[0] != self.n_features:
                raise ValueError(
                    f"beta length mismatch: expected {self.n_features}, got {beta.shape[0]}"
                )
            beta = beta.unsqueeze(1).repeat(1, self.config.state_dim)
        elif beta.ndim == 2:
            if tuple(beta.shape) != (self.n_features, self.config.state_dim):
                raise ValueError(
                    "beta shape mismatch: expected "
                    f"({self.n_features}, {self.config.state_dim}), got {tuple(beta.shape)}"
                )
        else:
            raise ValueError(f"beta must be 1D or 2D. Got shape {tuple(beta.shape)}")
        with torch.no_grad():
            self.cell.gate_beta.copy_(beta.to(dtype=self.cell.gate_beta.dtype))

    def get_gate_beta(self) -> torch.Tensor:
        beta = self.cell.gate_beta.detach().clone()
        if self.config.state_dim == 1:
            return beta.reshape(-1)
        return beta
