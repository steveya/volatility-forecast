from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from alphaforge.data.context import DataContext
    from alphaforge.data.query import Query
    from alphaforge.features.dataset_spec import TargetRequest
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

    class TargetRequest:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.template = kwargs.get("template")
            self.params = kwargs.get("params")
            self.horizon = kwargs.get("horizon")
            self.name = kwargs.get("name")

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


class NextDaySquaredReturnTarget:
    """y_t = scale * (logret_{t+1})^2.

    This matches the typical evaluation target for 1-day ahead variance proxy.
    """

    name = "target_nextday_sqret"
    version = "1.0"
    param_space = {
        "source": ParamSpec(
            "categorical", default="tiingo", choices=["tiingo", "simulated_garch"]
        ),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
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
        px = frame[price_col].astype(float)
        logret = np.log(px).groupby(level="entity_id").diff()
        y = (logret.groupby(level="entity_id").shift(-1) ** 2) * scale  # next-day

        fid = make_feature_id(table, "*", "target", "nextday_sqret", {"scale": scale})
        X = pd.DataFrame({fid: y}, index=frame.index).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "target", "nextday_sqret", {"scale": scale}
                    ),
                    "family": "target",
                    "transform": "nextday_sqret",
                    "source_table": table,
                    "source_col": price_col,
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


def _entity_shift(series: pd.Series, periods: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex) and "entity_id" in series.index.names:
        return series.groupby(level="entity_id").shift(periods)
    return series.shift(periods)


def forward_realized_variance_from_log_returns(
    logret: pd.Series,
    *,
    horizon_bars: int,
    annualize_to: float = 252.0,
) -> pd.Series:
    """Annualized forward realized variance indexed at forecast origin t.

    Computes

        annualize_to / H * sum_{j=1}^H r_{t+j}^2

    where r_{t+j} is the close-to-close log return. The final H rows are NaN
    because a full future window is required. The value stored at index t uses
    only future returns r_{t+1} through r_{t+H}; neither the contemporaneous
    return at t nor any prior return contributes to the target.
    """

    horizon = int(horizon_bars)
    if horizon <= 0:
        raise ValueError("horizon_bars must be positive")
    annualize = float(annualize_to)
    if annualize <= 0.0:
        raise ValueError("annualize_to must be positive")

    squared = pd.Series(np.square(logret.astype(float)), index=logret.index)
    future_terms = [_entity_shift(squared, -step) for step in range(1, horizon + 1)]
    future_panel = pd.concat(future_terms, axis=1)
    future_sum = future_panel.sum(axis=1, min_count=horizon)
    return future_sum * (annualize / float(horizon))


class ForwardRealizedVarianceTarget:
    """Annualized forward realized variance indexed at forecast origin t.

    The target at row t is

        annualize_to / H * sum_{j=1}^H r_{t+j}^2

    based on future close-to-close daily log returns.

    The final ``horizon_bars`` rows are undefined because the full future
    window is not available. Downstream dataset builders may safely drop those
    rows after target construction without introducing look-ahead, provided the
    feature matrix is aligned on the same forecast-origin index.
    """

    name = "target_forward_realized_variance"
    version = "1.0"
    param_space = {
        "source": ParamSpec(
            "categorical", default="tiingo", choices=["tiingo", "simulated_garch"]
        ),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
        "horizon_bars": ParamSpec("int", default=63, low=1, high=252),
        "annualize_to": ParamSpec("float", default=252.0, low=1.0),
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
        horizon_bars = int(params.get("horizon_bars", 63))
        annualize_to = float(params.get("annualize_to", 252.0))

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
        px = frame[price_col].astype(float)
        logret = np.log(px).groupby(level="entity_id").diff()
        y = forward_realized_variance_from_log_returns(
            logret,
            horizon_bars=horizon_bars,
            annualize_to=annualize_to,
        )

        fid = make_feature_id(
            table,
            "*",
            "target",
            "forward_realized_variance",
            {"horizon_bars": horizon_bars, "annualize_to": annualize_to},
        )
        X = pd.DataFrame({fid: y}, index=frame.index).sort_index()
        catalog = pd.DataFrame(
            [
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "target",
                        "forward_realized_variance",
                        {"horizon_bars": horizon_bars, "annualize_to": annualize_to},
                    ),
                    "family": "target",
                    "transform": "forward_realized_variance",
                    "source_table": table,
                    "source_col": price_col,
                    "horizon_bars": horizon_bars,
                    "annualize_to": annualize_to,
                }
            ]
        ).set_index("feature_id")
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )
