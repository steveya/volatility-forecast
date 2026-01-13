from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass
from typing import Dict

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.calendar import TradingCalendar

from volatility_forecast.pipeline import to_numpy_1d_bundle
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
        dates = pd.date_range(
            pd.Timestamp(q.start), pd.Timestamp(q.end), freq="B", tz="UTC"
        )
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


def test_to_numpy_1d_bundle_shapes_and_alignment():
    # Build a small 2-date panel for ticker AAA
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).tz_localize("UTC"), ["AAA"]],
        names=["ts_utc", "entity_id"],
    )
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=idx)
    y = pd.Series([0.1, 0.2], index=idx, name="y")
    r = pd.Series([0.01, 0.02], index=idx, name="ret")

    from volatility_forecast.pipeline import to_numpy_1d_bundle

    Xn, yn, rn, dates = to_numpy_1d_bundle(X, y, r, ticker="AAA")
    assert Xn.shape == (2, 2)
    assert yn.shape == (2, 1)
    assert rn.shape == (2,)
    assert list(dates) == list(
        pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).tz_localize("UTC")
    )


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


def test_xgb_accepts_pandas_inputs_if_available():
    try:
        import xgboost  # noqa: F401
    except Exception:
        pytest.skip("xgboost not available")

    from volatility_forecast.model.xgboost_stes_model import XGBoostSTESModel

    # small synthetic dataset
    n = 50
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series(np.random.rand(n), name="y")

    m = XGBoostSTESModel(max_depth=2, eta=0.1, subsample=0.8, colsample_bytree=0.8)
    m.fit(X, y, start_index=0, end_index=n)
    yhat = m.predict(X)
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape[0] == n
