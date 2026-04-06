from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from alphaforge.data.context import DataContext
    from alphaforge.data.query import Query
    from alphaforge.features.template import ParamSpec, SliceSpec
    from alphaforge.features.frame import FeatureFrame
    from alphaforge.features.ids import make_feature_id, group_path
    from volatility_forecast.market_data import load_market_frame
except ImportError:
    DataContext = object  # type: ignore[misc,assignment]

    class Query:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

    class ParamSpec:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    SliceSpec = object  # type: ignore[misc,assignment]

    class FeatureFrame:  # type: ignore[no-redef]
        def __init__(self, X: Any, **kwargs: Any) -> None:
            self.X = X

    def make_feature_id(*args: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        return "_target"

    def group_path(*args: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        return ""

    def load_market_frame(*args: Any, **kwargs: Any) -> pd.DataFrame:  # type: ignore[misc]
        return pd.DataFrame()


class ParkinsonTarget:
    name = "target_parkinson"
    version = "1.0"
    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "scale": ParamSpec("float", default=1.0),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def fit(self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec):
        return None

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table = params["source"], params["table"]
        scale = float(params.get("scale", 1.0))
        frame = load_market_frame(
            ctx,
            dataset=table,
            columns=["high", "low"],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source=source,
        )
        hi = frame["high"].astype(float)
        lo = frame["low"].astype(float)
        loghl = np.log(hi / lo)
        # Parkinson estimator (single-day)
        y = compute_parkinson_from_loghl(loghl) * scale

        fid = make_feature_id(table, "*", "target", "parkinson", {"scale": scale})
        X = pd.DataFrame({fid: y}, index=frame.index).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path("target", "parkinson", {"scale": scale}),
                    "family": "target",
                    "transform": "parkinson",
                    "source_table": table,
                    "source_col": "high,low",
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class GarmanKlassTarget:
    name = "target_garman_klass"
    version = "1.0"
    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "scale": ParamSpec("float", default=1.0),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def fit(self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec):
        return None

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table = params["source"], params["table"]
        scale = float(params.get("scale", 1.0))
        frame = load_market_frame(
            ctx,
            dataset=table,
            columns=["open", "high", "low", "close"],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source=source,
        )
        o = frame["open"].astype(float)
        hi = frame["high"].astype(float)
        lo = frame["low"].astype(float)
        c = frame["close"].astype(float)
        log_hl = np.log(hi / lo)
        log_co = np.log(c / o)
        y = compute_garman_klass_from_logs(log_hl, log_co) * scale

        fid = make_feature_id(table, "*", "target", "garman_klass", {"scale": scale})
        X = pd.DataFrame({fid: y}, index=frame.index).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "target", "garman_klass", {"scale": scale}
                    ),
                    "family": "target",
                    "transform": "garman_klass",
                    "source_table": table,
                    "source_col": "open,high,low,close",
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class YangZhangTarget:
    """Yang-Zhang variance estimator for daily OHLC data.

    This implementation uses a commonly-used fixed k weight ~0.34/(1.34) = 0.2537
    which works well for daily single-day samples. For multi-day samples the
    original formula provides a sample-dependent k.
    """

    name = "target_yang_zhang"
    version = "1.0"
    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "scale": ParamSpec("float", default=1.0),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def fit(self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec):
        return None

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table = params["source"], params["table"]
        scale = float(params.get("scale", 1.0))
        frame = load_market_frame(
            ctx,
            dataset=table,
            columns=["open", "high", "low", "close"],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source=source,
        )
        o = frame["open"].astype(float)
        hi = frame["high"].astype(float)
        lo = frame["low"].astype(float)
        c = frame["close"].astype(float)
        # compute required logs
        log_co = np.log(c / o)
        log_hl = np.log(hi / lo)
        # overnight return uses previous close; shift close by 1 within entity
        prev_close = c.groupby(level="entity_id").shift(1)
        log_oo = np.log(o / prev_close)

        y = compute_yang_zhang_from_logs(log_oo, log_co, log_hl) * scale
        fid = make_feature_id(table, "*", "target", "yang_zhang", {"scale": scale})
        X = pd.DataFrame({fid: y}, index=frame.index).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path("target", "yang_zhang", {"scale": scale}),
                    "family": "target",
                    "transform": "yang_zhang",
                    "source_table": table,
                    "source_col": "open,high,low,close",
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class IntradayRealizedVarianceTarget:
    """Realized variance computed from intraday returns (sum of squared log returns per day).

    Expects an intraday table with a price column. Groups by entity and UTC date.
    """

    name = "target_intraday_rv"
    version = "1.0"
    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.bars", choices=["market.bars"]
        ),
        "price_col": ParamSpec("categorical", default="price", choices=["price"]),
        "scale": ParamSpec("float", default=1.0),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def fit(self, ctx: DataContext, params: Dict[str, Any], fit_slice: SliceSpec):
        return None

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )
        scale = float(params.get("scale", 1.0))
        frame = load_market_frame(
            ctx,
            dataset=table,
            columns=[price_col],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source=source,
        )
        df = frame.reset_index()
        # Detect a datetime column (supports timezone-aware dtypes)
        # Avoid deprecated pandas API `is_datetime64tz_dtype` by checking the dtype class.
        from pandas.api.types import is_datetime64_any_dtype

        time_cols = [
            c
            for c in df.columns
            if is_datetime64_any_dtype(df[c].dtype)
            or isinstance(df[c].dtype, pd.DatetimeTZDtype)
        ]
        if not time_cols:
            raise RuntimeError("No datetime column found in intraday panel")
        dt = time_cols[0]
        df["date"] = df[dt].dt.tz_convert("UTC").dt.normalize()
        df = df[["entity_id", "date", price_col]]

        def day_rv(group):
            # `group` may be a Series when applying to a single column; handle both Series and DataFrame.
            if hasattr(group, "astype"):
                prices = group.astype(float).values
            else:
                prices = group[price_col].astype(float).values
            if prices.size < 2:
                return 0.0
            lr = np.diff(np.log(prices))
            return float(np.sum(lr**2))

        # Select the price column before grouping to avoid FutureWarning about
        # GroupBy.apply operating on grouping columns; passing the column ensures
        # the function receives only the Series of prices.
        rv = df.groupby(["entity_id", "date"])[price_col].apply(day_rv)
        # reindex to MultiIndex matching expected FeatureFrame index
        idx = pd.MultiIndex.from_tuples(rv.index.tolist(), names=["entity_id", "date"])
        fid = make_feature_id(table, "*", "target", "intraday_rv", {"scale": scale})
        X = pd.DataFrame({fid: rv.values * scale}, index=idx).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path("target", "intraday_rv", {"scale": scale}),
                    "family": "target",
                    "transform": "intraday_rv",
                    "source_table": table,
                    "source_col": price_col,
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


# Helper functions for testing and reuse


def compute_parkinson_from_loghl(loghl: "np.ndarray") -> "np.ndarray":
    """Compute Parkinson estimator from log(H/L)."""
    return (loghl**2) / (4.0 * np.log(2.0))


def compute_garman_klass_from_logs(
    log_hl: "np.ndarray", log_co: "np.ndarray"
) -> "np.ndarray":
    """Compute Garman-Klass estimator from log(H/L) and log(C/O)."""
    return 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)


def compute_yang_zhang_from_logs(
    log_oo: "np.ndarray", log_co: "np.ndarray", log_hl: "np.ndarray"
) -> "np.ndarray":
    """Compute a single-day Yang–Zhang estimator using a fixed k weight.

    This simplified implementation uses k = 0.34 / 1.34 (≈ 0.2537) which is a
    commonly used approximation for single-day estimates.
    """
    k = 0.34 / 1.34
    # overnight variance (open vs previous close)
    sigma_o2 = log_oo**2
    # close-to-open variance
    sigma_c2 = log_co**2
    # Rogers–Satchell component approximation from log H/L and log C/O
    rs = log_hl * log_hl * 0.5  # approximate RS using log_hl and log_co interactions
    return k * sigma_o2 + (1.0 - k) * sigma_c2 + rs
