from dataclasses import dataclass
from typing import Dict, Sequence
import pandas as pd


@dataclass(frozen=True)
class DataBundle:
    """
    Multi-table container for experiments.
    Each table is a DataFrame indexed by DatetimeIndex (sorted, unique).
    Example tables:
      - "market": OHLCV for SPY
      - "macro": macro panel aligned/ffilled to business days
    """

    tables: Dict[str, pd.DataFrame]

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Missing table '{name}'. Available: {list(self.tables)}")
        return self.tables[name]

    def require(self, name: str, cols: Sequence[str]) -> None:
        df = self.get(name)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Table '{name}' missing required columns: {missing}")
