from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphaforge")

from dataclasses import dataclass
from typing import Dict

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.calendar import TradingCalendar

from alphaforge.features.dataset_spec import FeatureRequest, TargetRequest
from alphaforge.features.dataset_spec import UniverseSpec, TimeSpec

from volatility_forecast.pipeline import (
    build_vol_dataset,
    VolDatasetSpec,
    to_numpy_1d_bundle,
)
from volatility_forecast.features.return_features import LagAbsLogReturnTemplate
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


def test_lag_abslogret_aligned_with_nextday_target(tmp_path):
    # Build context
    cal = TradingCalendar("XNYS", tz="UTC")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    src = TinyEODSource()
    ctx = DataContext(sources={"tiingo": src}, calendars={"XNYS": cal}, store=store)

    # Vol dataset: short date range to make expectations easy
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp(
        "2020-01-10", tz="UTC"
    )  # extend range to ensure multiple aligned rows

    features = (
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": 0,
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
        universe=UniverseSpec(entities=("SPY",)),
        time=TimeSpec(start=start, end=end, calendar="XNYS", grid="B"),
        features=features,
        target=target,
    )

    X, y, ret, catalog = build_vol_dataset(ctx, spec, persist=False)

    # Convert to single-ticker 1D arrays
    Xn, yn, rn, dates = to_numpy_1d_bundle(X, y, ret, ticker="SPY")

    # Sanity: we should have at least two rows to check next-day relation
    assert len(dates) >= 2

    # Recompute log returns from the raw source to form expectations
    q = Query(
        table="market.ohlcv",
        columns=["close"],
        start=start,
        end=end,
        entities=["SPY"],
        asof=None,
        grid="B",
    )
    panel = ctx.fetch_panel("tiingo", q)
    px = panel.df["close"].astype(float)
    logret = np.log(px).groupby(level="entity_id").diff()
    logret_xs = logret.xs("SPY", level="entity_id").sort_index()

    # For all pairs of consecutive rows present after alignment, assert the relationship
    # feature at date t == |logret_t| and target at date t == (logret_{t+1})^2
    if len(dates) < 2:
        pytest.skip("Not enough aligned rows to test next-day relation")

    for i in range(len(dates) - 1):
        dt0 = dates[i]
        dt1 = dates[i + 1]

        expected_feature = abs(logret_xs.loc[dt0])
        assert np.isfinite(expected_feature)
        assert Xn[i, 0] == pytest.approx(expected_feature)

        expected_target = (logret_xs.loc[dt1]) ** 2
        assert np.isfinite(expected_target)
        assert yn[i, 0] == pytest.approx(expected_target)
