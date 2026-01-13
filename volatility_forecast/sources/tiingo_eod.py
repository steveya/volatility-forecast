from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests

from alphaforge.data.source import DataSource
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema


# Canonical columns exposed to Alphaforge
CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


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

    def schemas(self) -> Dict[str, TableSchema]:
        return {
            "market.ohlcv": TableSchema(
                name="market.ohlcv",
                required_columns=CANONICAL_COLUMNS,
                canonical_columns=CANONICAL_COLUMNS,
                entity_column="entity_id",
                time_column="date",
                native_freq="B",
                expected_cadence_days=1,
            )
        }

    def _fetch_one(
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

        out = df[
            ["date", "entity_id"] + [mapping[c] for c in CANONICAL_COLUMNS]
        ].rename(columns={v: k for k, v in mapping.items()})
        out = out.sort_values(["date", "entity_id"]).reset_index(drop=True)
        return out

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
