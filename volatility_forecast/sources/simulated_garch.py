"""
Simulated GARCH(1,1) data source for backtesting and research.

Provides a DataSource adapter that generates contaminated GARCH(1,1) returns
and exposes them as an Alphaforge table for use in VolDatasetSpec pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd

from alphaforge.data.source import DataSource
from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema


CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class SimulatedGARCHSource(DataSource):
    """Alphaforge DataSource adapter for simulated contaminated GARCH(1,1).

    Generates a synthetic GARCH(1,1) return series with outlier shocks.
    Exposes one canonical table: market.ohlcv (with close = cumulative returns).

    Parameters
    ----------
    name : str
        Source name (default: "simulated_garch")
    n_periods : int
        Number of periods to simulate (default: 2500)
    mu : float
        Mean return (default: 0.0)
    omega : float
        GARCH intercept (default: 0.02)
    alpha : float
        GARCH shock coefficient (default: 0.11)
    beta : float
        GARCH persistence (default: 0.87)
    eta : float
        Outlier shock magnitude (default: 4.0)
    shock_prob : float
        Probability of outlier shock (default: 0.005)
    random_state : int, optional
        Random seed for reproducibility
    start_date : pd.Timestamp
        Start date of simulation (default: 2020-01-01)
    entity_id : str
        Entity name (default: "SIMULATED")
    """

    name: str = "simulated_garch"
    n_periods: int = 2500
    mu: float = 0.0
    omega: float = 0.02
    alpha: float = 0.11
    beta: float = 0.87
    eta: float = 4.0
    shock_prob: float = 0.005
    random_state: Optional[int] = None
    start_date: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp("2020-01-01", tz="UTC")
    )
    entity_id: str = "SIMULATED"

    # Cached data
    _cache: Dict[str, pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )

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

    def _simulate_garch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Simulate contaminated GARCH(1,1) process.

        Returns
        -------
        returns : np.ndarray
            Simulated returns
        sigma2s : np.ndarray
            Conditional variances
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        returns = np.zeros(n)
        sigma2s = np.zeros(n)
        shocks = (np.random.uniform(0, 1, n) < self.shock_prob).astype(float)

        for t in range(1, n):
            sigma2s[t] = (
                self.omega
                + self.alpha * returns[t - 1] ** 2
                + self.beta * sigma2s[t - 1]
            )
            returns[t] = (
                np.random.normal(self.mu, np.sqrt(sigma2s[t])) + self.eta * shocks[t]
            )

        return returns, sigma2s

    def _generate_ohlcv(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Generate OHLCV data from simulated returns.

        For a returns series, we create a synthetic price path and OHLCV bars.
        For simplicity: open = close of previous bar, close = cumulative return.
        """
        # Generate business days
        dates = pd.bdate_range(start=start.normalize(), end=end.normalize(), tz="UTC")
        n_dates = len(dates)

        if n_dates > self.n_periods:
            n_dates = self.n_periods

        # Simulate returns
        returns, sigma2s = self._simulate_garch(n_dates)

        # Create OHLCV: cumulative log returns as close
        cum_returns = np.cumprod(1 + returns / 100) - 1
        prices = 100 * (1 + cum_returns)  # Base at 100

        df = pd.DataFrame(
            {
                "date": dates[:n_dates],
                "entity_id": self.entity_id,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": np.full(n_dates, 1_000_000),  # constant volume
            }
        )

        return df[["date", "entity_id", "open", "high", "low", "close", "volume"]]

    def fetch(self, q: Query) -> pd.DataFrame:
        """Fetch simulated GARCH data matching the query."""
        if q.table != "market.ohlcv":
            raise KeyError(
                f"SimulatedGARCHSource only supports table=market.ohlcv, got {q.table}"
            )

        if q.start is None or q.end is None:
            raise ValueError("Query.start and Query.end are required.")

        start = (
            pd.Timestamp(q.start).tz_localize("UTC")
            if pd.Timestamp(q.start).tz is None
            else pd.Timestamp(q.start).tz_convert("UTC")
        )
        end = (
            pd.Timestamp(q.end).tz_localize("UTC")
            if pd.Timestamp(q.end).tz is None
            else pd.Timestamp(q.end).tz_convert("UTC")
        )

        # Generate or retrieve from cache
        cache_key = f"{start.isoformat()}_{end.isoformat()}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._generate_ohlcv(start, end)

        df = self._cache[cache_key].copy()

        # Filter by requested entities (if any)
        if q.entities:
            requested_entities = [str(e) for e in q.entities]
            df = df[df["entity_id"].isin(requested_entities)]

        # Select requested columns
        cols = list(q.columns) if q.columns else CANONICAL_COLUMNS
        df = df[["date", "entity_id"] + cols]

        return df.sort_values(["date", "entity_id"]).reset_index(drop=True)
