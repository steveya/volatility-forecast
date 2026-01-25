from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

import logging

from alphaforge.data.cache import CacheBackend
from alphaforge.store.raw_data_store import RawDataStore

from alphaforge.data.source import DataSource
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema


# Canonical columns exposed to Alphaforge
CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "asof_utc"]


def _ts_utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class TiingoEODSource(DataSource):
    """Alphaforge DataSource adapter for Tiingo end-of-day OHLCV.

    Notes
    -----
    - Exposes one canonical table: market.ohlcv
    - Returns long form: [date, entity_id, open, high, low, close, volume]
    - Uses Tiingo "prices" endpoint; prefers adjusted OHLCV fields if present.
    """

    name: str = "tiingo"
    api_key: Optional[str] = None
    use_adjusted: bool = True
    base_url: str = "https://api.tiingo.com/tiingo/daily"
    cache_backends: Sequence[CacheBackend] = field(default_factory=tuple)
    cache_mode: str = "use"
    include_asof: bool = True
    asof_column: str = "asof_utc"
    raw_store: Optional[RawDataStore] = None

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def schemas(self) -> Dict[str, TableSchema]:
        return {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=["open", "high", "low", "close", "volume"],
                canonical_columns=CANONICAL_COLUMNS,
                entity_column="entity_id",
                time_column="date",
                native_freq="B",
                expected_cadence_days=1,
            )
        }

    def _cache_key(self, ticker: str) -> str:
        return f"{ticker}|adj={self.use_adjusted}"

    def _normalize_cached(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
        if self.asof_column in out.columns:
            out[self.asof_column] = pd.to_datetime(
                out[self.asof_column], utc=True
            ).dt.tz_convert("UTC")
        return out.sort_values(["date", "entity_id"]).reset_index(drop=True)

    def _slice_cached(
        self, cached: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        mask = (cached["date"] >= start) & (cached["date"] <= end)
        return cached.loc[mask].reset_index(drop=True)

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if self.raw_store is not None:
            try:
                df = self.raw_store.get(key)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                self._logger.debug("Raw store get failed (%s): %s", key, exc)
        for backend in self.cache_backends:
            try:
                df = backend.get(key)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                self._logger.debug("Cache get failed (%s): %s", key, exc)
        return None

    def _cache_set(self, key: str, df: pd.DataFrame) -> None:
        if self.raw_store is not None:
            try:
                self.raw_store.set(key, df)
            except Exception as exc:
                self._logger.debug("Raw store set failed (%s): %s", key, exc)
        for backend in self.cache_backends:
            try:
                backend.set(key, df)
            except Exception as exc:
                self._logger.debug("Cache set failed (%s): %s", key, exc)

    def _fetch_one_raw(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("TiingoEODSource.api_key must be provided")

        url = f"{self.base_url}/{ticker}/prices"
        params = {
            "startDate": start.date().isoformat(),
            "endDate": end.date().isoformat(),
            "format": "json",
            "resampleFreq": "daily",
            "token": self.api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame(columns=["date", "entity_id"] + CANONICAL_COLUMNS)

        df = pd.DataFrame(data)

        # Tiingo usually returns ISO timestamps; force tz-aware UTC
        df["date"] = pd.to_datetime(
            df["date"], utc=True
        ).dt.normalize()  # midnight UTC label
        df["entity_id"] = str(ticker)

        # Prefer adjusted fields if present
        if self.use_adjusted:
            mapping = {
                "open": "adjOpen" if "adjOpen" in df.columns else "open",
                "high": "adjHigh" if "adjHigh" in df.columns else "high",
                "low": "adjLow" if "adjLow" in df.columns else "low",
                "close": "adjClose" if "adjClose" in df.columns else "close",
                # volume: Tiingo sometimes has adjVolume; otherwise volume
                "volume": "adjVolume" if "adjVolume" in df.columns else "volume",
            }
        else:
            mapping = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }

        if self.include_asof:
            df[self.asof_column] = pd.Timestamp.now(tz="UTC")
        else:
            df[self.asof_column] = pd.NaT
        mapping[self.asof_column] = self.asof_column

        out = df[
            ["date", "entity_id"] + [mapping[c] for c in CANONICAL_COLUMNS]
        ].rename(columns={v: k for k, v in mapping.items()})
        out = out.sort_values(["date", "entity_id"]).reset_index(drop=True)
        return out

    def _fetch_one(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cache_key = self._cache_key(ticker)
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached = self._normalize_cached(cached)

        cache_mode = self.cache_mode.lower().strip()
        if cache_mode not in {"use", "refresh", "update"}:
            raise ValueError("cache_mode must be one of: use, refresh, update")

        if cache_mode == "use" and cached is not None:
            cached_min = cached["date"].min()
            cached_max = cached["date"].max()
            if cached_min <= start and cached_max >= end:
                return self._slice_cached(cached, start, end)

        if cache_mode == "update" and cached is not None:
            cached_min = cached["date"].min()
            cached_max = cached["date"].max()
            frames: List[pd.DataFrame] = [cached]

            if start < cached_min:
                left_end = cached_min - pd.Timedelta(days=1)
                try:
                    frames.append(self._fetch_one_raw(ticker, start, left_end))
                except requests.HTTPError as exc:
                    self._logger.warning(
                        "Tiingo fetch failed (left update, %s). Using cached data.",
                        exc,
                    )

            if end > cached_max:
                right_start = cached_max + pd.Timedelta(days=1)
                try:
                    frames.append(self._fetch_one_raw(ticker, right_start, end))
                except requests.HTTPError as exc:
                    self._logger.warning(
                        "Tiingo fetch failed (right update, %s). Using cached data.",
                        exc,
                    )

            merged = (
                pd.concat(frames, ignore_index=True)
                .sort_values(["date", "entity_id"])
                .drop_duplicates(subset=["date", "entity_id"], keep="last")
                .reset_index(drop=True)
            )
            self._cache_set(cache_key, merged)
            return self._slice_cached(merged, start, end)

        try:
            out = self._fetch_one_raw(ticker, start, end)
            self._cache_set(cache_key, out)
            return out
        except requests.HTTPError as exc:
            if cached is not None and not cached.empty:
                self._logger.warning(
                    "Tiingo fetch failed (%s). Serving cached data.", exc
                )
                return self._slice_cached(cached, start, end)
            raise

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != "market.ohlcv":
            raise KeyError(
                f"TiingoEODSource only supports table=market.ohlcv, got {q.table}"
            )

        tickers = [str(x) for x in (q.entities or [])]
        if not tickers:
            raise ValueError("Query.entities must be provided for market.ohlcv")

        if q.start is None or q.end is None:
            raise ValueError("Query.start and Query.end are required.")

        start = _ts_utc(q.start)
        end = _ts_utc(q.end)

        cols = list(q.columns) if q.columns else CANONICAL_COLUMNS
        for c in cols:
            if c not in CANONICAL_COLUMNS:
                raise ValueError(
                    f"Unsupported column {c}. Supported: {CANONICAL_COLUMNS}"
                )

        frames: List[pd.DataFrame] = []
        for tkr in tickers:
            df = self._fetch_one(tkr, start, end)
            frames.append(df)

        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["date", "entity_id"] + CANONICAL_COLUMNS)
        )

        # Select only requested columns
        out = (
            out[["date", "entity_id"] + cols]
            .sort_values(["date", "entity_id"])
            .reset_index(drop=True)
        )
        return out
