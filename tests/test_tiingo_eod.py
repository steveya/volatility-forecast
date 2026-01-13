import importlib.util
import pathlib
import pandas as pd
from alphaforge.data.query import Query

# load module directly to avoid importing the top-level package which has heavy imports
_spec_path = (
    pathlib.Path(__file__).resolve().parents[1]
    / "volatility_forecast"
    / "sources"
    / "tiingo_eod.py"
)
# load module with full package name to give dataclasses a proper module context
spec = importlib.util.spec_from_file_location(
    "volatility_forecast.sources.tiingo_eod", str(_spec_path)
)
mod = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
TiingoEODSource = mod.TiingoEODSource


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_tiingo_eod_parses_dates_and_adjusted(monkeypatch):
    # sample Tiingo-like JSON with adjusted fields
    sample = [
        {
            "date": "2021-01-04T00:00:00.000Z",
            "adjOpen": 99.0,
            "adjHigh": 101.0,
            "adjLow": 98.0,
            "adjClose": 100.0,
            "adjVolume": 1000000,
        },
        {
            "date": "2021-01-05T00:00:00.000Z",
            "adjOpen": 100.0,
            "adjHigh": 102.0,
            "adjLow": 99.0,
            "adjClose": 101.0,
            "adjVolume": 1100000,
        },
    ]

    def fake_get(url, params=None, timeout=None):
        return DummyResponse(sample)

    monkeypatch.setattr("requests.get", fake_get)

    src = TiingoEODSource(api_key="fake", use_adjusted=True)
    q = Query(
        table="market.ohlcv",
        columns=["close", "volume"],
        entities=["AAPL"],
        start=pd.Timestamp("2021-01-01"),
        end=pd.Timestamp("2021-01-10"),
    )

    out = src.fetch(q)
    assert set(out.columns) >= {"date", "entity_id", "close", "volume"}

    # dates should be tz-aware UTC and normalized to midnight
    assert pd.DatetimeIndex(out["date"]).tz is not None
    assert pd.DatetimeIndex(out["date"]).tz == pd.Timestamp("2021-01-01T00:00:00Z").tz
    assert out["entity_id"].unique().tolist() == ["AAPL"]
    assert out["close"].tolist() == [100.0, 101.0]
