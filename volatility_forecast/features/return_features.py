from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from alphaforge import DataContext, Query
from alphaforge.features.template import ParamSpec, SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path


class _BaseLagTemplate:
    # no version here
    base_param_space = {
        "lags": ParamSpec("int", default=5, low=1, high=252),
        # allow simulated source for tests/examples
        "source": ParamSpec(
            "categorical", default="tiingo", choices=["tiingo", "simulated_garch"]
        ),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
    }

    # subclasses can override/extend
    param_space = base_param_space

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def _fetch_price_panel(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec
    ):
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )
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
        return panel, table, price_col

    def _logret(self, panel, price_col: str) -> pd.Series:
        px = panel.df[price_col].astype(float)
        return np.log(px).groupby(level="entity_id").diff()


class LagLogReturnTemplate(_BaseLagTemplate):
    version = "1.0"
    name = "lag_logret"

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        lags = int(params["lags"])
        panel, table, price_col = self._fetch_price_panel(ctx, params, slice)
        logret = self._logret(panel, price_col)

        X_cols, cat = {}, []
        # Use k starting at 0 so that shift(k) is explicit: k=0 -> today's return, k=1 -> yesterday, etc.
        # Loop from 0..lags inclusive so callers can request the max lag via the "lags" param.
        for k in range(0, lags + 1):
            fid = make_feature_id(table, "*", "lag", "logret", {"k": k})
            X_cols[fid] = logret.groupby(level="entity_id").shift(k)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path("lag", "logret", {"lags": lags}),
                    "family": "lag",
                    "transform": "logret",
                    "source_table": table,
                    "source_col": price_col,
                    "lag": k,
                }
            )

        X = pd.DataFrame(X_cols, index=panel.df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class LagAbsLogReturnTemplate(_BaseLagTemplate):
    name = "lag_abslogret"
    version = "1.0"

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        lags = int(params["lags"])
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )

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
        logret = np.log(px).groupby(level="entity_id").diff().abs()

        X_cols = {}
        cat = []
        # Use k starting at 0 so that shift(k) is explicit: k=0 -> today's abs-return
        for k in range(0, lags + 1):
            fid = make_feature_id(table, "*", "lag", "abslogret", {"k": k})
            X_cols[fid] = logret.groupby(level="entity_id").shift(k)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path("lag", "abslogret", {"lags": lags}),
                    "family": "lag",
                    "transform": "abslogret",
                    "source_table": table,
                    "source_col": price_col,
                    "lag": k,
                }
            )

        X = pd.DataFrame(X_cols, index=panel.df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class LagSquaredLogReturnTemplate(_BaseLagTemplate):
    name = "lag_sqlogret"
    version = "1.0"

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        lags = int(params["lags"])
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )

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
        logret2 = np.log(px).groupby(level="entity_id").diff() ** 2

        X_cols = {}
        cat = []
        # Use k starting at 0 so that shift(k) is explicit: k=0 -> today's squared return
        for k in range(0, lags + 1):
            fid = make_feature_id(table, "*", "lag", "sqlogret", {"k": k})
            X_cols[fid] = logret2.groupby(level="entity_id").shift(k)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path("lag", "sqlogret", {"lags": lags}),
                    "family": "lag",
                    "transform": "sqlogret",
                    "source_table": table,
                    "source_col": price_col,
                    "lag": k,
                }
            )

        X = pd.DataFrame(X_cols, index=panel.df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()
        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class OffsetLogReturnTemplate:
    version = "1.0"
    name = "offset_logret"

    param_space = {
        "k": ParamSpec("int", default=1, low=1, high=252),
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        k = int(params["k"])
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )

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
        lp = np.log(px)

        # log P_t - log P_{t-k}
        off = lp.groupby(level="entity_id").diff(k)

        fid = make_feature_id(table, "*", "ret", "offset_logret", {"k": k})
        X = pd.DataFrame({fid: off}, index=panel.df.index).sort_index()
        catalog = (
            pd.DataFrame(
                [
                    {
                        "feature_id": fid,
                        "group_path": group_path("ret", "offset_logret", {"k": k}),
                        "family": "ret",
                        "transform": "offset_logret",
                        "source_table": table,
                        "source_col": price_col,
                        "k": k,
                    }
                ]
            )
            .set_index("feature_id")
            .sort_index()
        )

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )


class SquaredLogReturnTemplate:
    version = "1.0"
    name = "sq_logret"

    param_space = {
        "source": ParamSpec("categorical", default="tiingo", choices=["tiingo"]),
        "table": ParamSpec(
            "categorical", default="market.ohlcv", choices=["market.ohlcv"]
        ),
        "price_col": ParamSpec("categorical", default="close", choices=["close"]),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        source, table, price_col = (
            params["source"],
            params["table"],
            params["price_col"],
        )

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
        logret2 = logret**2

        fid = make_feature_id(table, "*", "ret", "sqlogret", {})
        X = pd.DataFrame({fid: logret2}, index=panel.df.index).sort_index()
        catalog = (
            pd.DataFrame(
                [
                    {
                        "feature_id": fid,
                        "group_path": group_path("ret", "sqlogret", {}),
                        "family": "ret",
                        "transform": "sqlogret",
                        "source_table": table,
                        "source_col": price_col,
                    }
                ]
            )
            .set_index("feature_id")
            .sort_index()
        )

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )
