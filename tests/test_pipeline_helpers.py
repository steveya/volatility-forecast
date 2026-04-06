from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from alphaforge.data.context import DataContext
from alphaforge.data.adapter import SourceAdapterBase
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource
from alphaforge.data.types import FetchResult
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.calendar import TradingCalendar

from volatility_forecast.pipeline import build_default_ctx, to_numpy_1d_bundle
from volatility_forecast.sources.simulated_garch import SimulatedGARCHAdapter
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget


@dataclass
class TinyEODSource(DataSource):
    name: str = "tiingo"

    def schemas(self) -> Dict[str, TableSchema]:
        return {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=["open", "high", "low", "close", "volume"],
                canonical_columns=["open", "high", "low", "close", "volume"],
                entity_column="entity_id",
                time_column="date",
                native_freq="B",
                expected_cadence_days=1,
            )
        }

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != "market.ohlcv":
            raise KeyError(q.table)
        if q.start is None or q.end is None:
            raise ValueError("start/end required")
        entities = [str(x) for x in (q.entities or [])]
        if not entities:
            entities = ["SPY"]
        dates = pd.date_range(pd.Timestamp(q.start), pd.Timestamp(q.end), freq="B", tz="UTC")
        rows = []
        for e in entities:
            # simple price path 100, 101, 102, ...
            px = 100.0 + np.arange(len(dates))
            for d, p in zip(dates, px):
                rows.append(
                    {
                        "date": d,
                        "entity_id": e,
                        "open": p,
                        "high": p,
                        "low": p,
                        "close": p,
                        "volume": 1_000.0,
                    }
                )
        return pd.DataFrame(rows)


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
    rows = []
    closes = [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 106.0, 108.0]
    for obs_date, close in zip(dates, closes, strict=True):
        rows.append(
            {
                "series_key": "AAA",
                "obs_date": obs_date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_to_numpy_1d_bundle_shapes_and_alignment():
    # Build a small 2-date panel for ticker AAA
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).tz_localize("UTC"), ["AAA"]],
        names=["ts_utc", "entity_id"],
    )
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=idx)
    y = pd.Series([0.1, 0.2], index=idx, name="y")
    r = pd.Series([0.01, 0.02], index=idx, name="ret")

    Xn, yn, rn, dates = to_numpy_1d_bundle(X, y, r, ticker="AAA")
    assert Xn.shape == (2, 2)
    assert yn.shape == (2, 1)
    assert rn.shape == (2,)
    assert list(dates) == list(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).tz_localize("UTC"))


def test_build_default_ctx_registers_adapter_and_env_resolution(tmp_path, monkeypatch):
    """build_default_ctx registers TiingoAdapter alongside legacy source."""
    monkeypatch.setenv("TIINGO_API", "env_tiingo_key")
    ctx = build_default_ctx(store_root=str(tmp_path / "af_store"))

    # Legacy source path works
    assert "tiingo" in ctx.sources
    assert ctx.sources["tiingo"].api_key == "env_tiingo_key"
    assert str(ctx.store.root_path).endswith("af_store")

    # New unified adapter is registered
    assert ctx.adapters is not None
    assert "tiingo" in ctx.adapters
    adapter = ctx.adapters["tiingo"]
    assert adapter._api_key == "env_tiingo_key"
    assert adapter.source_name == "tiingo"
    assert "market.ohlcv" in adapter.datasets

    # Default source mapping
    assert ctx.default_sources == {"market.ohlcv": "tiingo"}

    # _resolve_source should find it
    resolved = ctx._resolve_source("market.ohlcv")
    assert resolved is adapter


def test_build_default_ctx_registers_extra_adapters():
    sim = SimulatedGARCHAdapter(entity_id="SIM", random_state=7, n_periods=64)
    ctx = build_default_ctx(extra_adapters=(sim,))

    assert ctx.adapters is not None
    assert "simulated_garch" in ctx.adapters

    result = ctx.load(
        "market.ohlcv",
        columns=["close"],
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2020-03-31", tz="UTC"),
        entities=["SIM"],
        source="simulated_garch",
    )

    assert not result.data.empty
    assert set(result.data.columns) >= {"series_key", "obs_date", "close"}
    assert np.isfinite(pd.to_numeric(result.data["close"])).all()


def test_ctx_fetch_adapter_path(tmp_path):
    """ctx.fetch() routes to TiingoAdapter for market.ohlcv queries."""
    from unittest.mock import patch

    from alphaforge.data.sources.tiingo import TiingoAdapter
    from alphaforge.data.types import FetchResult

    cal = TradingCalendar("XNYS", tz="UTC")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    src = TinyEODSource()

    # Create adapter with mock fetch
    adapter = TiingoAdapter(api_key="dummy")
    mock_result = FetchResult(
        data=pd.DataFrame(
            {"series_key": ["SPY"], "obs_date": [pd.Timestamp("2020-01-02")], "close": [300.0]}
        ),
        source="tiingo",
        dataset="market.ohlcv",
        is_pit=False,
        cached_at=None,
    )

    ctx = DataContext(
        sources={"tiingo": src},
        calendars={"XNYS": cal},
        store=store,
        adapters={"tiingo": adapter},
        default_sources={"market.ohlcv": "tiingo"},
    )

    with patch.object(adapter, "fetch", return_value=mock_result) as mock_fetch:
        q = Query(
            table="market.ohlcv",
            columns=["close"],
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-01-03"),
            entities=["SPY"],
        )
        result = ctx.fetch(q)

    assert result.source == "tiingo"
    assert result.dataset == "market.ohlcv"
    assert not result.is_pit
    mock_fetch.assert_called_once()


def test_nextday_squared_return_target_with_tiny_source(tmp_path):
    # Context with tiny source and calendar
    cal = TradingCalendar("XNYS", tz="UTC")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    src = TinyEODSource()
    ctx = DataContext(sources={"tiingo": src}, calendars={"XNYS": cal}, store=store)

    tgt = NextDaySquaredReturnTarget()

    from alphaforge.features.template import SliceSpec

    sl = SliceSpec(
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2020-01-06", tz="UTC"),
        entities=["SPY"],
        asof=None,
        grid="B",
    )

    ff = tgt.transform(
        ctx,
        params={
            "source": "tiingo",
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1.0,
        },
        slice=sl,
        state=None,
    )

    # ff is a FeatureFrame with 1 column containing y; verify non-empty and expected length (one fewer due to shift)
    assert hasattr(ff, "X") and isinstance(ff.X, pd.DataFrame)
    # Expect at least two non-NaNs for the next-day target in this short range
    assert ff.X.shape[1] == 1
    assert ff.X.notna().sum().sum() >= 2
