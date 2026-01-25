from __future__ import annotations

from typing import Any, Dict, List, Tuple
import importlib
import numpy as np
import pandas as pd
from sktime.transformations.panel.signature_based import SignatureTransformer

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.features.template import ParamSpec, SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.features.ids import make_feature_id, group_path

from .return_features import _PriceTemplate


def _patch_sktime_sklearn_tags() -> None:
    try:
        from sklearn.base import BaseEstimator
    except Exception:
        return

    def _default_tags(self):
        return BaseEstimator().__sklearn_tags__()

    def _patch_cls(cls):
        if cls is not None and not hasattr(cls, "__sklearn_tags__"):
            setattr(cls, "__sklearn_tags__", _default_tags)

    candidates = [
        (
            "sktime.transformations.panel.signature_based._augmentations",
            ["_AddTime", "_BasePoint", "_CumulativeSum"],
        ),
        (
            "sktime.transformations.panel.signature_based._signature_transform",
            ["_WindowSignatureTransform"],
        ),
        (
            "sktime.transformations.panel.signature_based._window_signature",
            ["_WindowSignatureTransform"],
        ),
        (
            "sktime.transformations.panel.signature_based._compute",
            ["_WindowSignatureTransform"],
        ),
        (
            "sktime.transformations.panel.signature_based._signature_method",
            ["_WindowSignatureTransform"],
        ),
    ]

    for module_name, class_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for class_name in class_names:
            _patch_cls(getattr(module, class_name, None))


def _patch_instance_sklearn_tags(obj) -> None:
    try:
        from sklearn.base import BaseEstimator
    except Exception:
        return

    def _default_tags(self):
        return BaseEstimator().__sklearn_tags__()

    seen = set()

    def _walk(value):
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        try:
            cls = value.__class__
            if not hasattr(cls, "__sklearn_tags__"):
                setattr(cls, "__sklearn_tags__", _default_tags)
        except Exception:
            pass

        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                _walk(v)
        else:
            for v in getattr(value, "__dict__", {}).values():
                _walk(v)


_patch_sktime_sklearn_tags()


class SignatureFeaturesTemplate(_PriceTemplate):
    version = "2.0"
    name = "signature"

    param_space = _PriceTemplate.param_space | {
        # Match the rest of the codebase: "lags" means the maximum lag / lookback window.
        "lags": ParamSpec("int", default=5, low=1, high=252),
        "sig_level": ParamSpec("int", default=2, low=1, high=5),
        # High-level preset selector. See `_resolve_augmentation_pipeline`.
        "augmentation_list": ParamSpec(
            "categorical",
            default="all",
            choices=["all", "none", "time", "basepoint"],
        ),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    _ALLOWED_AUGS = {"basepoint", "addtime", "leadlag", "ir", "cumsum"}

    def _parse_augmentations(self, v) -> Tuple[str, ...] | None:
        """
        Normalize ordered augmentation specification for sktime SignatureTransformer.
        - None / "" / "none" => None (no augmentation)
        - list/tuple preserves order; set is sorted deterministically
        - str supports separators: ',', '+', '->'
        - alias: "time" -> "addtime"
        - duplicates removed (first occurrence kept)
        """
        if v is None:
            return None

        if isinstance(v, (list, tuple)):
            parts = [str(x).strip().lower() for x in v if str(x).strip()]
        elif isinstance(v, set):
            parts = sorted(str(x).strip().lower() for x in v if str(x).strip())
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in {"", "none", "null"}:
                return None
            s = s.replace("->", ",").replace("+", ",")
            parts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            raise TypeError(
                f"augmentations must be str|list|tuple|set|None, got {type(v)}"
            )

        parts = ["addtime" if p == "time" else p for p in parts]

        bad = [p for p in parts if p not in self._ALLOWED_AUGS]
        if bad:
            raise ValueError(
                f"Unknown augmentations: {bad}. Allowed: {sorted(self._ALLOWED_AUGS)}"
            )

        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                out.append(p)
                seen.add(p)

        return tuple(out) if out else None

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        lags = int(params["lags"])
        sig_level = int(params["sig_level"])
        augmentation_tag = params.get("augmentation_list", "all")

        panel, table, price_col = self._fetch_price_panel(ctx, params, slice)

        logret = self._logret(panel, price_col)

        # Build rolling path as (lookback+1) timepoints per sample
        # We keep this because it gives us one row per (entity_id, time) and avoids ambiguity
        # about what sktime's "sliding" window output index should be for our FeatureFrame.
        path_df = pd.concat(
            {
                f"lag_{i}": logret.groupby(level="entity_id").shift(i)
                for i in range(lags, -1, -1)
            },
            axis=1,
        ).dropna()

        # ---- 2) Ordered augmentation pipeline (order matters)
        augmentation_list = self._resolve_augmentation_pipeline(augmentation_tag)

        # SignatureTransformer
        # We already constructed a single fixed-length window per sample, so use window_name="global".
        # (SignatureTransformer supports window_name choices, but "sliding" would not naturally give
        # us one row per timestamp in our FeatureFrame without extra index bookkeeping.)  [oai_citation:3‡sktime.net](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.signature_based.SignatureTransformer.html)
        sig_transformer = SignatureTransformer(
            augmentation_list=augmentation_list,
            depth=sig_level,
            window_name="global",
            window_depth=None,
            window_length=None,
            window_step=None,
            rescaling="post",
            sig_tfm="signature",
            backend="esig",
        )
        _patch_instance_sklearn_tags(sig_transformer)

        # Feed as Panel np3D: (instance, channel, time)
        # Here channel=1 (just the logret channel). If you add more channels later,
        # stack them on axis=1.
        paths = path_df.to_numpy(dtype=float)  # (n_samples, n_timepoints)
        X_panel = paths[:, None, :]  # (n_samples, 1, n_timepoints)

        sig_df = sig_transformer.fit_transform(
            X_panel
        )  # pd.DataFrame, rows correspond to instances  [oai_citation:4‡sktime.net](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.signature_based.SignatureTransformer.html)
        sig_df.index = path_df.index  # align with your (entity_id, time) index

        # Convert to FeatureFrame schema
        X_cols = {}
        cat = []
        aug_repr = str(augmentation_tag).strip().lower()
        for j, col in enumerate(sig_df.columns):
            fid = make_feature_id(table, "*", "sig", str(sig_level), {"term": j})
            X_cols[fid] = sig_df[col].to_numpy()
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "sig",
                        str(sig_level),
                        {"lags": lags, "augmentation_list": aug_repr},
                    ),
                    "family": "signature",
                    "transform": "signature",
                    "source_table": table,
                    "source_col": price_col,
                    "lag": lags,
                    "sig_level": sig_level,
                    "term": j,
                    "augmentation_list": aug_repr,
                }
            )

        X = pd.DataFrame(X_cols, index=sig_df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )

    def _resolve_augmentation_pipeline(self, augmentation_tag) -> Tuple[str, ...] | None:
        """Map high-level `augmentation_list` preset into sktime augmentation_list."""
        tag = str(augmentation_tag).strip().lower() if augmentation_tag is not None else "all"
        if tag in {"", "none", "null"}:
            return None
        if tag == "all":
            # Reasonable default (order matters)
            return ("cumsum", "basepoint", "addtime", "leadlag")
        # Delegate to the parser for single-augmentation presets (e.g., "time" -> "addtime")
        return self._parse_augmentations(tag)
