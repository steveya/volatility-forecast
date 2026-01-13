"""Smoke script for SPY 2020Q1 evaluation using benchmark runner.

Requires Tiingo API key in .env (TIINGO_API). Uses new scalable dataset API.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from volatility_forecast.pipeline import (
    VolDatasetSpec,
    build_default_ctx,
    build_vol_dataset,
)
from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    JoinPolicy,
    MissingnessPolicy,
)
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget
from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel
from volatility_forecast.model.xgboost_stes_model import XGBoostSTESModel
from volatility_forecast.benchmark import run as bench
from volatility_forecast.evaluation import metrics, dm_test, mz


def spy_provider():
    # Ensure TIINGO_API from .env is available
    load_dotenv()
    api = os.getenv("TIINGO_API")
    ctx = build_default_ctx(tiingo_api_key=api)

    # Define feature and target requests (scalable)
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": 5,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": 5,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": 5,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={
            "source": "tiingo",
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1.0,
        },
        horizon=1,
        name="y",
    )

    spec = VolDatasetSpec(
        universe=UniverseSpec(entities=["SPY"]),
        time=TimeSpec(
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-03-31", tz="UTC"),
            calendar="XNYS",
            grid="B",
            asof=None,
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )

    X, y, returns, catalog = build_vol_dataset(ctx, spec, persist=False)
    # extract single-ticker panel as DataFrame/Series for models that expect DF
    X1 = X.xs("SPY", level="entity_id").sort_index().copy()
    # Ensure constant feature is present for ESModel
    if "const" not in X1.columns:
        X1["const"] = 1.0
    y1 = y.xs("SPY", level="entity_id").sort_index()
    r1 = returns.xs("SPY", level="entity_id").sort_index()
    return X1, y1, r1, X1.index


if __name__ == "__main__":
    models = {
        "es": lambda: ESModel(),
        "stes": lambda: STESModel(),
        "xgb": lambda: XGBoostSTESModel(
            **{
                "num_boost_round": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        ),
    }

    # fetch once to compute a sensible split index
    X1, y1, r1, dates = spy_provider()
    os_index = int(0.8 * len(y1))

    def provider_closure():
        return X1, y1, r1, dates

    res = bench.run_benchmark(
        provider_closure,
        models,
        is_index=0,
        os_index=os_index,
        oos_index=None,
        metric_fns={"rmse": metrics.rmse, "qlike": metrics.qlike},
        baseline="es",
    )

    print("Metrics:\n", res["metrics"])
    print("DM vs ES baseline:\n", res["dm"])
    print("MZ summary:\n", res["mz"])
