"""Simple benchmark runner for models.

This file provides a minimal runner which:
- takes a data provider callable returning X, y, returns, date
- a list of model factory callables
- split specs and evaluation metrics
- returns a tidy DataFrame of results and runs DM vs baseline
"""

from __future__ import annotations

from typing import Callable, Dict, List, Any, Optional
import pandas as pd
import numpy as np

from volatility_forecast.evaluation import metrics, dm_test, mz


def run_benchmark(
    data_provider: Callable[[], tuple],
    model_factories: Dict[str, Callable[[], Any]],
    is_index: int,
    os_index: int,
    oos_index: Optional[int] = None,
    metric_fns: Dict[str, Callable] = None,
    baseline: Optional[str] = None,
):
    if metric_fns is None:
        metric_fns = {"rmse": metrics.rmse, "qlike": metrics.qlike}

    X, y, returns, date = data_provider()
    results = []
    forecasts = {}
    for name, factory in model_factories.items():
        model = factory()
        # Fit: be flexible with model API (some tests/models don't accept start/end/returns)
        try:
            model.fit(X, y, returns=returns, start_index=is_index, end_index=os_index)
        except TypeError:
            # Try common positional signature used in tests: (X, y, returns, is_index, os_index)
            try:
                model.fit(X, y, returns, is_index, os_index)
            except TypeError:
                try:
                    model.fit(X, y, returns=returns)
                except TypeError:
                    model.fit(X, y)

        # Predict: likewise be flexible with signature
        try:
            yhat = model.predict(X, returns=returns)
        except TypeError:
            yhat = model.predict(X)
        # align with date (many models return yhat length one less etc.) - assume same shape
        res = {"model": name}
        for mname, fn in metric_fns.items():
            res[mname] = float(fn(y[os_index:oos_index], yhat[os_index:oos_index]))
        results.append(res)
        forecasts[name] = yhat

    df = pd.DataFrame(results)

    dm_table = None
    if baseline is not None and baseline in forecasts:
        baseline_forecast = forecasts[baseline]
        dm_rows = []
        for name, yhat in forecasts.items():
            if name == baseline:
                continue
            d = dm_test.diebold_mariano((y - yhat) ** 2, (y - baseline_forecast) ** 2)
            d["model"] = name
            dm_rows.append(d)
        dm_table = pd.DataFrame(dm_rows)

    # MZ regressions
    mz_rows = []
    mz_plots = {}
    for name, yhat in forecasts.items():
        r = mz.mincer_zarnowitz(y[os_index:oos_index], yhat[os_index:oos_index])
        mz_rows.append({"model": name, **r["params"], "r2": r["r2"]})
        try:
            ax = mz.plot_mincer_zarnowitz(
                y[os_index:oos_index], yhat[os_index:oos_index]
            )
            mz_plots[name] = ax.get_figure()
        except Exception:
            mz_plots[name] = None
    mz_table = pd.DataFrame(mz_rows)

    return {"metrics": df, "dm": dm_table, "mz": mz_table, "mz_plots": mz_plots}
