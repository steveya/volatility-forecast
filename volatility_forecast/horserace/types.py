"""Typed data containers for the horserace execution DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class DataBundle:
    """Output of the data-fetch stage — one ticker's aligned dataset."""

    ticker: str
    X: pd.DataFrame  # (T, p) feature matrix
    y: pd.Series  # (T,) target (e.g. next-day squared return)
    returns: pd.Series  # (T,) log returns
    dates: pd.DatetimeIndex  # (T,) date index


@dataclass(frozen=True)
class SplitSpec:
    """Index boundaries for train / OOS partitioning.

    All indices refer to positions in the DataBundle arrays.
    """

    burn_in: int  # recursion warm-up start (is_index)
    train_end: int  # exclusive end of training data (os_index)
    oos_start: int  # first index in the OOS region (used for leaderboard filtering)
    oos_end: int  # exclusive end of OOS region (= len(dates))


@dataclass(frozen=True)
class FitResult:
    """Output of the fit-or-load stage."""

    model_id: str  # store ID, e.g. "STES:0042"
    fitted_model: Any  # the fitted model object
    was_retrained: bool  # True if we actually retrained this run
    train_metric: Optional[float]  # in-sample loss (qlike), None if loaded
