# volatility_forecast/model_manager.py
from __future__ import annotations
from typing import Optional, Any
import pandas as pd

from volatility_forecast.storage import VolForecastStore


class ModelManager:
    def __init__(self, store: VolForecastStore):
        self.store = store

    def save_model(
        self,
        model_type: str,
        model_obj: Any,
        performance_metric: float,
        params: dict,
        trained_start=None,
        trained_end=None,
    ) -> str:
        return self.store.register_model(
            model_type=model_type,
            model_obj=model_obj,
            trained_start=trained_start,
            trained_end=trained_end,
            metric_name="rmse",
            metric_value=float(performance_metric),
            params=params,
        )

    def load_latest_model(self, model_type: str):
        return self.store.load_latest_model(model_type)

    def get_model_history(self, model_type: str):
        return self.store.model_history(model_type)
