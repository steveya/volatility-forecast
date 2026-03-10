"""Configuration schema for the VolGRU volatility model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(slots=True)
class VolGRUConfig:
    """Configuration for backend-agnostic VolGRU models."""

    backend: Literal["torch", "jax"] = "torch"
    gate_mode: Literal["stes_linear", "gru_linear"] = "stes_linear"
    candidate_mode: Literal["stes_r2", "linear_pos", "mlp_pos"] = "stes_r2"
    state_dim: int = 1
    use_reset_gate: bool = False
    positive_transform: Literal["softplus", "exp"] = "softplus"
    loss_mode: Literal["mse_r2", "nll_gaussian"] = "mse_r2"

    weight_decay_gate: float = 0.0
    weight_decay_candidate: float = 0.0
    beta_stay_close_lambda: float = 0.0
    gate_entropy_lambda: float = 0.0

    lr: float = 1e-2
    max_epochs: int = 200
    batch_size: Optional[int] = None
    early_stopping_patience: int = 20

    # Fraction of the training window held out for validation-based early
    # stopping.  When > 0 the last ``val_fraction`` of the
    # ``[start_index, end_index)`` loss window is used as a validation set;
    # gradients are back-propagated only through the earlier portion while
    # early-stopping monitors the held-out validation loss.
    val_fraction: float = 0.0

    # Tiny MLP width for candidate_mode="mlp_pos".
    mlp_hidden_dim: int = 8
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1.")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0.")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1.")
        if self.mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be >= 1.")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if self.weight_decay_gate < 0.0 or self.weight_decay_candidate < 0.0:
            raise ValueError("Weight decay values must be >= 0.")
        if self.beta_stay_close_lambda < 0.0:
            raise ValueError("beta_stay_close_lambda must be >= 0.")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1).")
