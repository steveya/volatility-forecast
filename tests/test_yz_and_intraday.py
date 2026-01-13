import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from volatility_forecast.targets import range_estimators


def test_compute_yang_zhang_basic():
    n = 10
    # small synthetic series
    log_oo = np.random.normal(scale=0.01, size=n)
    log_co = np.random.normal(scale=0.01, size=n)
    log_hl = np.random.normal(scale=0.01, size=n)
    yz = range_estimators.compute_yang_zhang_from_logs(log_oo, log_co, log_hl)
    assert yz.shape == log_oo.shape
    assert np.all(np.isfinite(yz))


class DummyPanel:
    def __init__(self, df):
        self.df = df


class DummyCtx:
    def __init__(self, panel):
        self._panel = panel

    def fetch_panel(self, source, query):
        return self._panel


def test_intraday_realized_variance_target():
    # Build synthetic intraday prices for 1 entity across one trading day
    base = datetime(2020, 1, 2, 9, 30)
    times = [base + timedelta(minutes=30 * i) for i in range(6)]
    prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])

    df = pd.DataFrame(
        {
            "entity_id": ["SPY"] * len(times),
            "ts": pd.to_datetime(times).tz_localize("UTC"),
            "price": prices,
        }
    )

    panel = DummyPanel(df)
    ctx = DummyCtx(panel)

    target = range_estimators.IntradayRealizedVarianceTarget()
    params = {
        "source": "tiingo",
        "table": "market.bars",
        "price_col": "price",
        "scale": 1.0,
    }

    # slice placeholders
    class Slice:
        start = None
        end = None
        entities = ["SPY"]
        asof = None
        grid = None

    ff = target.transform(ctx, params, Slice(), None)
    # since we only have one date/entity, expect a single entry
    assert ff.X.shape[0] == 1
    val = ff.X.iloc[0, 0]
    # manual compute
    lr = np.diff(np.log(prices))
    expected = float(np.sum(lr**2))
    assert np.allclose(val, expected)
