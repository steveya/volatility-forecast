from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.features.template import ParamSpec, SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path


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

        panel = ctx.fetch_panel(
            source,
            Query(
                table=table,
                columns=[price_col],
                start=slice.start,
                end=slice.end,
                entities=slice.entities,
                asof=slice.asof,
                grid=slice.grid,
            ),
        )
        px = panel.df[price_col].astype(float)
        logret = np.log(px).groupby(level="entity_id").diff()
        y = (logret.groupby(level="entity_id").shift(-1) ** 2) * scale  # next-day

        fid = make_feature_id(table, "*", "target", "nextday_sqret", {"scale": scale})
        X = pd.DataFrame({fid: y}, index=panel.df.index).sort_index()
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
