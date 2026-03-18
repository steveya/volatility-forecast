"""Pluggable refit policies for the horserace DAG."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from volatility_forecast.horserace.types import DataBundle
from volatility_forecast.storage import VolForecastStore

logger = logging.getLogger(__name__)


class RefitPolicy(ABC):
    """Base class for refit-decision logic."""

    @abstractmethod
    def should_refit(
        self,
        store: VolForecastStore,
        model_name: str,
        bundle: DataBundle,
    ) -> bool:
        """Return True if *model_name* should be retrained."""
        ...


class EveryNDaysPolicy(RefitPolicy):
    """Retrain if N calendar days have passed since the model was last trained."""

    def __init__(self, n_days: int):
        self.n_days = n_days

    def should_refit(
        self, store: VolForecastStore, model_name: str, bundle: DataBundle
    ) -> bool:
        model_id = store.latest_model_id(model_name)
        if model_id is None:
            return True
        row = store.get_model_row(model_id)
        if row is None:
            return True
        created = pd.Timestamp(row["created_utc"])
        if created.tzinfo is None:
            created = created.tz_localize("UTC")
        age_days = (pd.Timestamp.now("UTC") - created).days
        return age_days >= self.n_days


class ExpandingWindowPolicy(RefitPolicy):
    """Always retrain — the training window expands each run."""

    def should_refit(
        self, store: VolForecastStore, model_name: str, bundle: DataBundle
    ) -> bool:
        return True


class AlwaysLoadPolicy(RefitPolicy):
    """Never retrain — always load the existing model (used with force_retrain override)."""

    def should_refit(
        self, store: VolForecastStore, model_name: str, bundle: DataBundle
    ) -> bool:
        return store.latest_model_id(model_name) is None


class ForceRefitPolicy(RefitPolicy):
    """Always retrain (used when --retrain flag is set)."""

    def should_refit(
        self, store: VolForecastStore, model_name: str, bundle: DataBundle
    ) -> bool:
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_POLICY_REGISTRY = {
    "every_n_days": lambda params: EveryNDaysPolicy(n_days=params.get("n_days", 30)),
    "expanding_window": lambda params: ExpandingWindowPolicy(),
}


def build_refit_policy(cfg: dict) -> RefitPolicy:
    """Build a RefitPolicy from a config dict like {"type": "every_n_days", "params": {"n_days": 30}}."""
    policy_type = cfg.get("type", "every_n_days")
    params = cfg.get("params", {})
    factory = _POLICY_REGISTRY.get(policy_type)
    if factory is None:
        raise ValueError(
            f"Unknown refit policy type: {policy_type!r}. "
            f"Available: {list(_POLICY_REGISTRY)}"
        )
    return factory(params)


def resolve_refit_policy(config: dict, model_name: str, force_retrain: bool) -> RefitPolicy:
    """Resolve the effective refit policy for a model.

    Priority: force_retrain flag > per-model config > global config.
    """
    if force_retrain:
        return ForceRefitPolicy()

    # Per-model override
    model_cfg = config.get("models", {}).get(model_name, {})
    if "refit_policy" in model_cfg:
        return build_refit_policy(model_cfg["refit_policy"])

    # Global default
    global_cfg = config.get("training", {}).get("refit_policy")
    if global_cfg:
        return build_refit_policy(global_cfg)

    # Fallback
    return EveryNDaysPolicy(n_days=30)
