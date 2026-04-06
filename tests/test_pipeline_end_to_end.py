import os
import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from alphaforge.data.adapter import SourceAdapterBase
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.data.types import FetchResult
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
    LagAbsLogReturnTemplate,
    LagLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget
from volatility_forecast.sources.tiingo_eod import TiingoEODSource
from volatility_forecast.pipeline import VolDatasetSpec, build_vol_dataset


class TinyMarketAdapter(SourceAdapterBase):
    source_name = "market"
    datasets = frozenset({"market.ohlcv"})

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def fetch(self, query: Query, *, max_staleness=None) -> FetchResult:
        frame = self._frame.copy()
        if query.entities is not None:
            frame = frame[frame["series_key"].isin(query.entities)]

        obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.start is not None:
            frame = frame[obs >= pd.Timestamp(query.start)]
            obs = pd.to_datetime(frame["obs_date"], utc=True)
        if query.end is not None:
            frame = frame[obs <= pd.Timestamp(query.end)]

        keep = ["series_key", "obs_date"] + [
            column for column in query.columns if column in frame.columns
        ]
        return FetchResult(
            data=frame[keep].reset_index(drop=True),
            source=self.source_name,
            dataset=query.table,
            is_pit=False,
            cached_at=None,
        )


def _market_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="B", tz="UTC")
    closes = [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 106.0, 108.0]
    rows = [
        {
            "series_key": "AAA",
            "obs_date": obs_date,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000.0,
        }
        for obs_date, close in zip(dates, closes, strict=True)
    ]
    return pd.DataFrame(rows)


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


def test_build_vol_dataset_adapter_only_context(tmp_path):
    calendar = TradingCalendar("XNYS", tz="UTC")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    ctx = DataContext.from_adapters(
        TinyMarketAdapter(_market_frame()),
        calendars={"XNYS": calendar},
        store=store,
    )

    spec = VolDatasetSpec(
        universe=UniverseSpec(entities=("AAA",)),
        time=TimeSpec(
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-01-10", tz="UTC"),
            calendar="XNYS",
            grid="B",
        ),
        features=(
            FeatureRequest(
                template=LagAbsLogReturnTemplate(),
                params={
                    "lags": 0,
                    "source": "market",
                    "table": "market.ohlcv",
                    "price_col": "close",
                },
            ),
        ),
        target=TargetRequest(
            template=NextDaySquaredReturnTarget(),
            params={
                "source": "market",
                "table": "market.ohlcv",
                "price_col": "close",
                "scale": 1.0,
            },
            horizon=1,
            name="y",
        ),
    )

    X, y, r, _ = build_vol_dataset(ctx, spec, persist=False)

    assert not X.empty
    assert y.notna().any()
    assert r.notna().any()
    assert "AAA" in X.index.get_level_values("entity_id")

    aligned_dates = X.index.get_level_values("ts_utc").unique().sort_values()
    expected_dates = [
        calendar.session_close_utc(pd.Timestamp(obs_date))
        for obs_date in pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    ]
    assert list(aligned_dates) == expected_dates
