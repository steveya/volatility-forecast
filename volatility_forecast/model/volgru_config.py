"""Configuration schema for the VolGRU volatility model."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Literal, Optional

_LEGACY_DTYPE_BACKFILLED_IDS: set[int] = set()


@dataclass(slots=True)
class VolGRUConfig:
    """Configuration for backend-agnostic VolGRU models.

    Attributes
    ----------
    dtype : {"float32", "float64"}, default "float32"
        Floating-point precision used by the Torch backend parameters and
        state recursion. ``"float32"`` preserves backward-compatible training and
        default predictions; ``"float64"`` gives double-precision STES
        equivalence (``rtol=0, atol=1e-12`` when variance stays above ``eps``).
    """

    backend: Literal["torch"] = "torch"
    gate_mode: Literal["stes_linear", "gru_linear"] = "stes_linear"
    candidate_mode: Literal["stes_r2", "linear_pos", "mlp_pos"] = "stes_r2"
    state_dim: int = 1
    use_reset_gate: bool = False
    positive_transform: Literal["softplus", "exp"] = "softplus"
    loss_mode: Literal["mse_r2", "nll_gaussian", "qlike"] = "mse_r2"
    dtype: Literal["float32", "float64"] = "float32"

    weight_decay_gate: float = 0.0
    weight_decay_candidate: float = 0.0
    beta_stay_close_lambda: float = 0.0
    gate_entropy_lambda: float = 0.0
    gate_feature_group_lambda: float = 0.0

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
        if self.gate_entropy_lambda < 0.0:
            raise ValueError("gate_entropy_lambda must be >= 0.")
        if self.gate_feature_group_lambda < 0.0:
            raise ValueError("gate_feature_group_lambda must be >= 0.")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1).")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'.")
        _LEGACY_DTYPE_BACKFILLED_IDS.discard(id(self))

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
        elif isinstance(state, dict):
            dict_state, slot_state = None, state
        else:
            raise TypeError(f"Unsupported VolGRUConfig pickle state: {type(state)!r}")

        restored: dict[str, Any] = {}
        if dict_state:
            restored.update(dict_state)
        if slot_state:
            restored.update(slot_state)

        missing_dtype = "dtype" not in restored
        for f in fields(self):
            if f.name in restored:
                object.__setattr__(self, f.name, restored[f.name])
            elif f.default is not MISSING:
                object.__setattr__(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                object.__setattr__(self, f.name, f.default_factory())

        if missing_dtype:
            _LEGACY_DTYPE_BACKFILLED_IDS.add(id(self))
        else:
            _LEGACY_DTYPE_BACKFILLED_IDS.discard(id(self))

    def __del__(self, _ids: set[int] = _LEGACY_DTYPE_BACKFILLED_IDS) -> None:
        _ids.discard(id(self))
