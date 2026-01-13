from __future__ import annotations

from typing import Iterable, Mapping, Optional
import pandas as pd


CANONICAL_OHLCV_REQUIRED = ("open", "high", "low", "close", "volume")
CANONICAL_OHLCV_OPTIONAL = ("adj_close",)


def rename_columns(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Rename columns using a mapping; leaves unknown columns untouched."""
    out = df.copy()
    out.columns = [mapping.get(c, c) for c in out.columns]
    return out


def canonicalize_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure:
      - DatetimeIndex (or convert from a date column)
      - tz-naive
      - sorted, unique index
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col])
            out = out.set_index(date_col)
        else:
            raise TypeError("Expected a DatetimeIndex or a 'date' column for index construction.")

    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def canonicalize_ohlcv(
    df: pd.DataFrame,
    required: Iterable[str] = CANONICAL_OHLCV_REQUIRED,
    *,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Canonicalize OHLCV table (vendor-agnostic).

    This function does NOT attempt vendor-specific renaming.
    Instead, each loader should map its vendor columns to the canonical set
    and then call this function to enforce index rules and required columns.
    """
    out = canonicalize_index(df)

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing canonical OHLCV columns: {missing}. "
            f"Have: {list(out.columns)}. "
            "Hint: perform vendor->canonical renaming in your loader before calling canonicalize_ohlcv()."
        )

    if not allow_extra_columns:
        keep = list(required) + [c for c in CANONICAL_OHLCV_OPTIONAL if c in out.columns]
        out = out.loc[:, keep]

    return out
