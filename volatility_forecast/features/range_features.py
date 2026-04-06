from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from alphaforge import DataContext, Query
from alphaforge.features.template import ParamSpec, SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path

from volatility_forecast.market_data import load_market_frame


class HighLowRangeTemplate:
    version = "1.0"
    name = "highlow_range"

    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "high_col": ParamSpec("categorical", default="high", choices=["high"]),
        "low_col": ParamSpec("categorical", default="low", choices=["low"]),
        "include_parkinson": ParamSpec(
            "categorical", default="yes", choices=["yes", "no"]
        ),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table = params["source"], params["table"]
        high_col, low_col = params["high_col"], params["low_col"]
        include_parkinson = str(params["include_parkinson"]).lower() == "yes"

        frame = load_market_frame(
            ctx,
            dataset=table,
            columns=[high_col, low_col],
            start=slice.start,
            end=slice.end,
            entities=slice.entities,
            asof=slice.asof,
            grid=slice.grid,
            source=source,
        )

        hi = frame[high_col].astype(float)
        lo = frame[low_col].astype(float)

        hl = np.log(hi) - np.log(lo)

        X_cols = {}
        cat = []

        fid = make_feature_id(table, "*", "range", "hl_log", {})
        X_cols[fid] = hl
        cat.append(
            {
                "feature_id": fid,
                "group_path": group_path("range", "hl_log", {}),
                "family": "range",
                "transform": "hl_log",
                "source_table": table,
                "source_col": f"{high_col},{low_col}",
            }
        )

        if include_parkinson:
            fid2 = make_feature_id(table, "*", "range", "parkinson_var", {})
            X_cols[fid2] = (hl**2) / (4.0 * np.log(2.0))
            cat.append(
                {
                    "feature_id": fid2,
                    "group_path": group_path("range", "parkinson_var", {}),
                    "family": "range",
                    "transform": "parkinson_var",
                    "source_table": table,
                    "source_col": f"{high_col},{low_col}",
                }
            )

        X = pd.DataFrame(X_cols, index=frame.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )
