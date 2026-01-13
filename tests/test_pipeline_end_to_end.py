import os
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.time.calendar import TradingCalendar
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

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
from volatility_forecast.sources.tiingo_eod import TiingoEODSource
from volatility_forecast.pipeline import VolDatasetSpec, build_vol_dataset


def test_build_vol_dataset_smoke(tmp_path):
    api = os.getenv("TIINGO_API")
    if not api:
        return  # or pytest.skip("TIINGO_API not set")

    cal = TradingCalendar("XNYS", tz="America/New_York")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    src = TiingoEODSource(api_key=api)

    ctx = DataContext(sources={"tiingo": src}, calendars={"XNYS": cal}, store=store)

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

    X, y, r, catalog = build_vol_dataset(ctx, spec, persist=True)
    assert len(X.columns) > 0
    assert y.notna().any()
    assert r.notna().any()
    assert len(catalog) == len(X.columns)
