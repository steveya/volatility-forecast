"""Channel-head planning helpers for PGARCH-family models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pgarch_core import VALID_PGARCH_CHANNELS

CHANNEL_XGB_PARAM_MAP = {
    "booster": "booster",
    "learning_rate": "eta",
    "max_depth": "max_depth",
    "min_child_weight": "min_child_weight",
    "subsample": "subsample",
    "colsample_bytree": "colsample_bytree",
    "reg_alpha": "alpha",
    "reg_lambda": "lambda",
    "gamma": "gamma",
}


@dataclass(slots=True)
class XGBChannelHeadPlan:
    booster: str = "gbtree"
    learning_rate: float = 0.02
    max_depth: int = 3
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    trees_per_channel_per_round: int = 1
    channel_param_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    channel_trees_per_round: dict[str, int] = field(default_factory=dict)
    random_state: int | None = None
    verbosity: int = 0

    def __post_init__(self) -> None:
        self.channel_param_overrides = {
            channel: dict(overrides)
            for channel, overrides in self.channel_param_overrides.items()
        }
        self.channel_trees_per_round = {
            channel: int(num_rounds)
            for channel, num_rounds in self.channel_trees_per_round.items()
        }
        self.validate()

    def validate(self) -> None:
        valid_boosters = {"gbtree", "gblinear"}
        if self.booster not in valid_boosters:
            raise ValueError(f"booster must be one of {sorted(valid_boosters)}")
        if self.trees_per_channel_per_round <= 0:
            raise ValueError("trees_per_channel_per_round must be positive.")
        if self.learning_rate <= 0.0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and > 0.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        if self.min_child_weight <= 0.0 or not np.isfinite(self.min_child_weight):
            raise ValueError("min_child_weight must be finite and > 0.")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1].")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must be in (0, 1].")
        if self.reg_alpha < 0.0 or self.reg_lambda < 0.0 or self.gamma < 0.0:
            raise ValueError("reg_alpha, reg_lambda, and gamma must be >= 0.")

        for channel, overrides in self.channel_param_overrides.items():
            if channel not in VALID_PGARCH_CHANNELS:
                raise ValueError(f"Unsupported channel override {channel!r}.")
            unknown = set(overrides) - set(CHANNEL_XGB_PARAM_MAP)
            if unknown:
                raise ValueError(f"Unsupported per-channel XGBoost params: {sorted(unknown)!r}.")
            self._validate_channel_override_params(overrides)

        for channel, num_rounds in self.channel_trees_per_round.items():
            if channel not in VALID_PGARCH_CHANNELS:
                raise ValueError(f"Unsupported channel tree-budget override {channel!r}.")
            if int(num_rounds) <= 0:
                raise ValueError("channel_trees_per_round values must be positive.")

    def params_for(self, channel: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "booster": self.booster,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "alpha": self.reg_alpha,
            "lambda": self.reg_lambda,
            "gamma": self.gamma,
            "verbosity": self.verbosity,
            "objective": "reg:squarederror",
            "base_score": 0.0,
        }
        if channel is not None:
            for key, value in self.channel_param_overrides.get(channel, {}).items():
                params[CHANNEL_XGB_PARAM_MAP[key]] = value
        if self.random_state is not None:
            params["seed"] = int(self.random_state)
        return params

    def num_boost_round(self, channel: str) -> int:
        return int(self.channel_trees_per_round.get(channel, self.trees_per_channel_per_round))

    @staticmethod
    def _validate_channel_override_params(overrides: dict[str, Any]) -> None:
        if "booster" in overrides and overrides["booster"] not in {"gbtree", "gblinear"}:
            raise ValueError("Per-channel booster must be one of ['gblinear', 'gbtree'].")
        if "learning_rate" in overrides:
            value = float(overrides["learning_rate"])
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("Per-channel learning_rate must be finite and > 0.")
        if "max_depth" in overrides and int(overrides["max_depth"]) < 0:
            raise ValueError("Per-channel max_depth must be >= 0.")
        if "min_child_weight" in overrides:
            value = float(overrides["min_child_weight"])
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("Per-channel min_child_weight must be finite and > 0.")
        if "subsample" in overrides:
            value = float(overrides["subsample"])
            if not (0.0 < value <= 1.0):
                raise ValueError("Per-channel subsample must be in (0, 1].")
        if "colsample_bytree" in overrides:
            value = float(overrides["colsample_bytree"])
            if not (0.0 < value <= 1.0):
                raise ValueError("Per-channel colsample_bytree must be in (0, 1].")
        for key in ("reg_alpha", "reg_lambda", "gamma"):
            if key in overrides and float(overrides[key]) < 0.0:
                raise ValueError(f"Per-channel {key} must be >= 0.")
