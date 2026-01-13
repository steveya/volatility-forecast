from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import pandas as pd


OHLCV_REQUIRED = ("open", "high", "low", "close", "volume")
OHLCV_OPTIONAL = ("adj_close",)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of df with a clean DatetimeIndex."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Expected DataFrame with DatetimeIndex")
    # Make timezone-naive for consistent joins/serialization.
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


@dataclass(frozen=True)
class MarketData:
    """
    Canonical single-asset market data container.

    Contract:
      - df index is a tz-naive DatetimeIndex, sorted, unique
      - columns are canonical (open/high/low/close/volume) plus optional adj_close
    """

    symbol: str
    df: pd.DataFrame
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "df", _ensure_datetime_index(self.df))

    def require(self, cols: Sequence[str]) -> None:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Have: {list(self.df.columns)}"
            )

    def slice(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> "MarketData":
        return MarketData(
            symbol=self.symbol,
            df=self.df.loc[start:end],
            source=self.source,
            meta=dict(self.meta),
        )
