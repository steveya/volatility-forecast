from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query

# New scalable dataset API
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    JoinPolicy,
    MissingnessPolicy,
)
from alphaforge.features.dataset_builder import build_dataset

from .sources.tiingo_eod import TiingoEODSource


@dataclass(frozen=True)
class VolDatasetSpec:
    """
    Scalable dataset spec for volatility experiments.

    - arbitrary feature list via FeatureRequest tuples
    - arbitrary target via TargetRequest
    - universe/time compatible with PIT/asof and grid selection
    - join/missingness policies configurable
    """

    universe: UniverseSpec
    time: TimeSpec

    # requests
    features: Tuple[FeatureRequest, ...] = field(default_factory=tuple)
    target: TargetRequest | None = None  # type: ignore

    # policies
    join_policy: JoinPolicy = field(default_factory=JoinPolicy)
    missingness: MissingnessPolicy = field(default_factory=MissingnessPolicy)

    # optional metadata (unused by builder but helpful to callers)
    meta: Dict[str, Any] = field(default_factory=dict)


def build_default_ctx(
    tiingo_api_key: Optional[str] = None,
    store_root: str = ".alphaforge_store",
) -> DataContext:
    """Construct a default Alphaforge DataContext for the vol domain.

    Uses:
    - TiingoEODSource for daily adjusted OHLCV
    - LocalParquetStore for caching materializations
    """
    from alphaforge.time.calendar import TradingCalendar
    from alphaforge.store.duckdb_parquet import DuckDBParquetStore

    cal = TradingCalendar("XNYS", tz="UTC")
    src = TiingoEODSource(api_key=tiingo_api_key)
    store = DuckDBParquetStore(root=".af_store")

    return DataContext(
        sources={"tiingo": src},
        calendars={"XNYS": cal},
        store=store,
        universe=None,
        entity_meta=None,
    )


def build_vol_dataset(
    ctx: DataContext,
    spec: VolDatasetSpec,
    persist: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Build (X, y, returns, catalog) using the new scalable dataset API.

    - Uses alphaforge.features.dataset_builder.build_dataset
    - Computes raw daily log-returns as auxiliary output for models (ES/STES)
    """

    if spec.target is None:
        raise ValueError("VolDatasetSpec.target must be provided")

    # Map VolDatasetSpec -> core DatasetSpec
    ds = DatasetSpec(
        universe=spec.universe,
        time=spec.time,
        target=spec.target,
        features=list(spec.features),
        join_policy=spec.join_policy,
        missingness=spec.missingness,
        name=spec.meta.get("name", "vol_dataset"),
        tags=spec.meta.get("tags", {}),
    )

    artifact = build_dataset(ctx, ds, persist=persist)
    X = artifact.X
    y = artifact.y.rename("y") if artifact.y.name is None else artifact.y
    catalog = artifact.catalog

    # Compute raw log returns from target params (source/table/price_col).
    # We assume the target's params reflect the primary market source/table to compute returns from.
    tgt_params = spec.target.params or {}
    source = tgt_params.get("source", "tiingo")
    table = tgt_params.get("table", "market.ohlcv")
    price_col = tgt_params.get("price_col", "close")

    q = Query(
        table=table,
        columns=[price_col],
        start=spec.time.start,
        end=spec.time.end,
        entities=list(spec.universe.entities),
        asof=spec.time.asof,
        grid=spec.time.grid,
    )
    panel = ctx.fetch_panel(source, q)
    px = panel.df[price_col].astype(float)
    ret = np.log(px).groupby(level="entity_id").diff().rename("logret")

    # Align returns to the final dataset index, then drop any rows with NaNs across all
    # Note: artifact already applied missingness policy across X and y. We ensure returns align.
    if len(X) > 0:
        ret = ret.reindex(X.index)
        combined = pd.concat([X, y.rename("__y__"), ret.rename("__ret__")], axis=1)
        combined = combined.dropna(axis=0, how="any")
        X = combined.drop(columns=["__y__", "__ret__"]).copy()
        y = combined["__y__"].copy()
        ret = combined["__ret__"].copy()
    else:
        # no features; still align y and ret to common index
        idx = y.index.intersection(ret.index)
        y = y.loc[idx]
        ret = ret.loc[idx]

    return X, y, ret, catalog



def build_wide_dataset(ctx: DataContext, spec: VolDatasetSpec, *, entity_id: str):
    """Build a wide dataset for one entity."""
    X, y, returns, catalog = build_vol_dataset(ctx, spec, persist=False)

    X1 = X.xs(entity_id, level="entity_id").sort_index().copy()
    y1 = y.xs(entity_id, level="entity_id").sort_index()
    r1 = returns.xs(entity_id, level="entity_id").sort_index()

    if "const" not in X1.columns:
        X1["const"] = 1.0

    # strict alignment
    idx = X1.index.intersection(y1.index).intersection(r1.index)
    X1, y1, r1 = X1.loc[idx], y1.loc[idx], r1.loc[idx]

    return X1, y1, r1, catalog


def to_numpy_1d_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    ticker: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Convert MultiIndex panel outputs to 1D arrays for single-asset models.

    This matches the way your existing STES/ES models are written (time series arrays).

    Returns
    -------
    Xn : (T, p) float array
    yn : (T, 1) float array
    rn : (T,) float array
    dates : DatetimeIndex of length T
    """
    ticker = str(ticker)
    X1 = X.xs(ticker, level="entity_id").sort_index()
    y1 = y.xs(ticker, level="entity_id").sort_index()
    r1 = returns.xs(ticker, level="entity_id").sort_index()

    # ensure aligned on time
    idx = X1.index.intersection(y1.index).intersection(r1.index)
    X1, y1, r1 = X1.loc[idx], y1.loc[idx], r1.loc[idx]

    return (
        X1.to_numpy(dtype=float),
        y1.to_numpy(dtype=float).reshape(-1, 1),
        r1.to_numpy(dtype=float),
        pd.DatetimeIndex(idx),
    )
