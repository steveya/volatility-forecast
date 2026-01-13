from __future__ import annotations

import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from alphaforge.data.context import DataContext
from alphaforge.time.calendar import TradingCalendar
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

from volatility_forecast.sources.tiingo_eod import TiingoEODSource
from volatility_forecast.pipeline import VolDatasetSpec, build_vol_dataset
from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
)
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget

from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel

try:
    from volatility_forecast.model.xgboost_stes_model import XGBoostSTESModel

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def main():
    load_dotenv()

    api = os.getenv("TIINGO_API")
    if not api:
        raise RuntimeError("Set TIINGO_API env var first (export TIINGO_API=...)")

    # --- Alphaforge runtime ---
    cal = TradingCalendar("XNYS", tz="America/New_York")
    store = DuckDBParquetStore(
        root=str("./.af_store")
    )  # local cache for panels/materializations
    src = TiingoEODSource(api_key=api)

    ctx = DataContext(sources={"tiingo": src}, calendars={"XNYS": cal}, store=store)

    # --- Experiment spec (new scalable API) ---
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": 10,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": 10,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": 10,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
    )
    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={
            "source": "tiingo",
            "table": "market.ohlcv",
            "price_col": "close",
            "scale": 1e4,
        },
        horizon=1,
        name="y",
    )
    spec = VolDatasetSpec(
        universe=UniverseSpec(entities=["SPY"]),
        time=TimeSpec(
            start=pd.Timestamp("2018-01-01", tz="UTC"),
            end=pd.Timestamp("2020-12-31", tz="UTC"),
            calendar="XNYS",
            grid="B",
            asof=None,
        ),
        features=features,
        target=target,
    )

    X, y, r, catalog = build_vol_dataset(ctx, spec, persist=True)

    # Basic cleaning: keep rows with complete X/y/r
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y).reshape(-1), index=X.index)
    if not isinstance(r, pd.Series):
        r = pd.Series(np.asarray(r).reshape(-1), index=X.index)

    mask = X.notna().all(axis=1) & y.notna() & r.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    r = r.loc[mask].copy()

    if len(X) < 100:
        raise RuntimeError(f"Not enough data after cleaning (n={len(X)})")

    # --- Train/test split (simple last-20% holdout) ---
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    r_tr, r_te = r.iloc[:split], r.iloc[split:]

    burn = min(spec.lags, max(1, len(X_tr) // 20))

    print(f"Built dataset: X={X.shape}, y={y.shape}, returns={r.shape}")
    print(f"Train={len(X_tr)}, Test={len(X_te)}, burn={burn}")
    print("-" * 80)

    # --- ESModel (constant only) ---
    Xc_tr = pd.DataFrame({"const": 1.0}, index=X_tr.index)
    Xc_te = pd.DataFrame({"const": 1.0}, index=X_te.index)

    es = ESModel()
    es.fit(Xc_tr, y_tr, returns=r_tr, start_index=0, end_index=len(Xc_tr))
    yhat_es = es.predict(Xc_te, returns=r_te)
    print(f"ESModel RMSE:   {rmse(y_te.values, yhat_es):.6f}")

    # --- STESModel ---
    stes = STESModel()
    stes.fit(X_tr, y_tr, returns=r_tr, start_index=burn, end_index=len(X_tr))
    yhat_stes = stes.predict(X_te, returns=r_te)
    print(f"STESModel RMSE: {rmse(y_te.values, yhat_stes):.6f}")

    # --- XGBoostSTESModel (optional) ---
    if _HAS_XGB:
        # model expects numpy arrays (it uses X[:, 1] etc.)
        X_tr_np = X_tr.to_numpy(dtype=float)
        X_te_np = X_te.to_numpy(dtype=float)
        y_tr_np = y_tr.to_numpy(dtype=float).reshape(-1)
        y_te_np = y_te.to_numpy(dtype=float).reshape(-1)
        r_tr_np = r_tr.to_numpy(dtype=float).reshape(-1)
        r_te_np = r_te.to_numpy(dtype=float).reshape(-1)

        xgb = XGBoostSTESModel(
            max_depth=3,
            eta=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            seed=0,
        )
        xgb.fit(
            X_tr_np, y_tr_np, returns=r_tr_np, start_index=burn, end_index=len(X_tr_np)
        )
        yhat_xgb = xgb.predict(X_te_np, returns=r_te_np)
        print(f"XGBoostSTES RMSE: {rmse(y_te_np, yhat_xgb):.6f}")
    else:
        print("XGBoost not available; skipping XGBoostSTESModel.")

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
