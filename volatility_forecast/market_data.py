from __future__ import annotations

from typing import Sequence

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query


def _ts_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _empty_market_frame(columns: Sequence[str]) -> pd.DataFrame:
    empty_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], tz="UTC"), pd.Index([], dtype="object")],
        names=["ts_utc", "entity_id"],
    )
    out = pd.DataFrame(index=empty_index)
    for column in columns:
        out[column] = pd.Series(dtype="float64")
    out["asof_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
    return out


def _resolve_calendar_name(grid: str | None) -> str:
    if grid and ":" in grid:
        _, calendar_name = grid.split(":", 1)
        return calendar_name
    return "XNYS"


def _sessionize_daily_index(
    ctx: DataContext,
    times: pd.Series,
    *,
    grid: str | None,
) -> pd.DatetimeIndex:
    calendar_name = _resolve_calendar_name(grid)
    calendar = ctx.calendars.get(calendar_name)
    if calendar is None:
        return pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return pd.DatetimeIndex([calendar.session_close_utc(ts) for ts in times])


def _infer_entity_time_columns(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("series_key", "obs_date"),
        ("entity_id", "date"),
        ("entity_id", "ts_utc"),
        ("series_key", "ts_utc"),
        ("entity_id", "ts"),
        ("series_key", "ts"),
    ]
    for entity_col, time_col in candidates:
        if {entity_col, time_col}.issubset(frame.columns):
            return entity_col, time_col
    raise ValueError(
        "Could not infer entity/time columns from frame. Expected one of "
        "('series_key', 'obs_date'), ('entity_id', 'date'), "
        "('entity_id', 'ts_utc'), or ('entity_id', 'ts')."
    )


def _normalize_market_frame(
    ctx: DataContext,
    frame: pd.DataFrame,
    *,
    dataset: str,
    columns: Sequence[str],
    start=None,
    end=None,
    entities: Sequence[str] | None = None,
    asof=None,
    grid: str | None = None,
) -> pd.DataFrame:
    requested_columns = list(dict.fromkeys(columns))
    if frame.empty:
        return _empty_market_frame(requested_columns)

    if isinstance(frame.index, pd.MultiIndex) and {"ts_utc", "entity_id"} <= set(frame.index.names):
        out = frame.copy()
        if "ts_utc" != out.index.names[0] or "entity_id" != out.index.names[1]:
            out.index = out.index.set_names(["ts_utc", "entity_id"])
        out = out.sort_index()
    else:
        entity_col, time_col = _infer_entity_time_columns(frame)
        out = frame.copy()
        out[time_col] = pd.to_datetime(out[time_col], utc=True)
        if dataset == "market.ohlcv" and time_col in {"obs_date", "date"}:
            out["ts_utc"] = _sessionize_daily_index(ctx, out[time_col], grid=grid)
        else:
            out["ts_utc"] = pd.DatetimeIndex(out[time_col])
        out["entity_id"] = out[entity_col].astype(str)
        out = out.set_index(["ts_utc", "entity_id"]).sort_index()

    missing = [column for column in requested_columns if column not in out.columns]
    if missing:
        raise ValueError(
            f"Market frame for '{dataset}' is missing required columns: {sorted(missing)}"
        )

    if "asof_utc" in out.columns:
        out["asof_utc"] = pd.to_datetime(out["asof_utc"], utc=True)
    else:
        out["asof_utc"] = _ts_utc(asof) if asof is not None else pd.NaT

    if start is not None:
        out = out[out.index.get_level_values("ts_utc") >= _ts_utc(start)]
    if end is not None:
        out = out[out.index.get_level_values("ts_utc") <= _ts_utc(end)]
    if asof is not None:
        out = out[out.index.get_level_values("ts_utc") <= _ts_utc(asof)]
    if entities is not None:
        allowed = {str(entity) for entity in entities}
        out = out[out.index.get_level_values("entity_id").isin(allowed)]

    keep = requested_columns.copy()
    if "asof_utc" not in keep:
        keep.append("asof_utc")
    out = out.loc[:, keep]
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_market_frame(
    ctx: DataContext,
    *,
    dataset: str,
    columns: Sequence[str],
    start=None,
    end=None,
    entities: Sequence[str] | None = None,
    asof=None,
    grid: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    load_error: Exception | None = None
    if hasattr(ctx, "load") and getattr(ctx, "adapters", None):
        try:
            result = ctx.load(
                dataset,
                columns=list(columns),
                start=start,
                end=end,
                entities=entities,
                asof=asof,
                grid=grid,
                source=source,
            )
            return _normalize_market_frame(
                ctx,
                result.data,
                dataset=dataset,
                columns=columns,
                start=start,
                end=end,
                entities=entities,
                asof=asof,
                grid=grid,
            )
        except Exception as exc:
            load_error = exc

    if source is not None and hasattr(ctx, "fetch_panel"):
        query = Query(
            table=dataset,
            columns=list(columns),
            start=start,
            end=end,
            entities=list(entities) if entities is not None else None,
            asof=asof,
            grid=grid,
        )
        panel = ctx.fetch_panel(source, query)
        return _normalize_market_frame(
            ctx,
            panel.df,
            dataset=dataset,
            columns=columns,
            start=start,
            end=end,
            entities=entities,
            asof=asof,
            grid=grid,
        )

    if load_error is not None:
        raise load_error
    raise KeyError(
        f"Unable to load dataset '{dataset}'. Register adapters or provide a legacy source."
    )
