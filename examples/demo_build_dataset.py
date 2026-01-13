"""Demo: Build a lag-feature vol dataset using alphaforge integration.

Run from your project root after installing alphaforge (editable is fine):
    python -m volatility_forecast.af.examples.demo_build_dataset

You'll need TIINGO_API in env, or pass api_key to build_default_ctx.
"""

import os
import pandas as pd

from volatility_forecast.pipeline import (
    VolDatasetSpec,
    build_default_ctx,
    build_vol_dataset,
    to_numpy_1d_bundle,
)
from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
)
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget


def main():
    ctx = build_default_ctx(
        tiingo_api_key=os.getenv("TIINGO_API"), store_root=".af_store_demo"
    )

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
            "scale": 1e4,
        },
        horizon=1,
        name="y",
    )
    spec = VolDatasetSpec(
        universe=UniverseSpec(entities=("SPY",)),
        time=TimeSpec(
            start=pd.Timestamp("2018-01-01"),
            end=pd.Timestamp("2023-12-31"),
            calendar="XNYS",
            grid="B",
        ),
        features=features,
        target=target,
        meta={"name": "demo_build_dataset"},
    )

    X, y, r, catalog = build_vol_dataset(ctx, spec, persist=True)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("returns shape:", r.shape)
    print("catalog head:\n", catalog.head())

    Xn, yn, rn, dates = to_numpy_1d_bundle(X, y, r, ticker="SPY")
    print("1D bundle:", Xn.shape, yn.shape, rn.shape, len(dates))


if __name__ == "__main__":
    main()
