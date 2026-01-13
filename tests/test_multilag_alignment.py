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

from alphaforge.features.dataset_spec import FeatureRequest, TargetRequest
from alphaforge.features.dataset_spec import UniverseSpec, TimeSpec

from volatility_forecast.pipeline import (
    build_vol_dataset,
    VolDatasetSpec,
    to_numpy_1d_bundle,
)
from volatility_forecast.features.return_features import LagLogReturnTemplate
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
            # simple price path 100, 102, 104, ... even steps so logrets alternate
            px = 100.0 + 2 * np.arange(len(dates))
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


def test_multilag_alignment(tmp_path):
    cal = TradingCalendar("XNYS", tz="UTC")
    store = DuckDBParquetStore(root=str(tmp_path / "store"))
    src = TinyEODSource()
    ctx = DataContext(sources={"tiingo": src}, calendars={"XNYS": cal}, store=store)

    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-01-20", tz="UTC")

    # Request lags=2 means k in {0,1,2}
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": 2,
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

    X, y, r, catalog = build_vol_dataset(ctx, spec, persist=False)

    Xn, yn, rn, dates = to_numpy_1d_bundle(X, y, r, ticker="SPY")

    # Check that for date t, feature k corresponds to logret_{t-k}
    # and target at t corresponds to (logret_{t+1})^2
    # Need at least 4 aligned rows so dt+1 is available in checks
    if len(dates) < 4:
        pytest.skip("Not enough rows for multilag verification")

    # Recompute raw log returns
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

    # For each t, map features k=0..2 to logret_{t-k}
    # iterate up to len(dates)-1 so dt+1 is valid
    for i in range(2, len(dates) - 1):
        # date t is dates[i]
        dt = dates[i]
        # features columns ordered by feature_id which include k in name; find them
        # feature ordering: k=0, k=1, k=2
        f0 = Xn[i, 0]
        f1 = Xn[i, 1]
        f2 = Xn[i, 2]

        # Align using the aligned 'dates' array to avoid calendar gaps
        assert f0 == pytest.approx(logret_xs.loc[dt])
        assert f1 == pytest.approx(logret_xs.loc[dates[i - 1]])
        assert f2 == pytest.approx(logret_xs.loc[dates[i - 2]])

        # target at dt should equal (logret at the next aligned date)^2
        expected_target = (logret_xs.loc[dates[i + 1]]) ** 2
        assert yn[i, 0] == pytest.approx(expected_target)
