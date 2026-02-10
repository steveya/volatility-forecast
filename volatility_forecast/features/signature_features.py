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
    version = "2.1"
    name = "signature"

    param_space = _PriceTemplate.param_space | {
        # Match the rest of the codebase: "lags" means the maximum lag / lookback window.
        "lags": ParamSpec("int", default=5, low=1, high=252),
        "sig_level": ParamSpec("int", default=2, low=1, high=5),
        "sig_tfm": ParamSpec(
            "categorical",
            default="signature",
            choices=["signature", "logsignature"],
        ),
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
        sig_tfm = str(params.get("sig_tfm", "signature")).strip().lower()
        if sig_tfm not in {"signature", "logsignature"}:
            raise ValueError("sig_tfm must be 'signature' or 'logsignature'")

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
        rescaling = "post" if sig_tfm == "signature" else None
        sig_transformer = SignatureTransformer(
            augmentation_list=augmentation_list,
            depth=sig_level,
            window_name="global",
            window_depth=None,
            window_length=None,
            window_step=None,
            rescaling=rescaling,
            sig_tfm=sig_tfm,
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
            fid = make_feature_id(
                table, "*", "sig", str(sig_level), {"term": j, "sig_tfm": sig_tfm}
            )
            X_cols[fid] = sig_df[col].to_numpy()
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "sig",
                        str(sig_level),
                        {
                            "lags": lags,
                            "augmentation_list": aug_repr,
                            "sig_tfm": sig_tfm,
                        },
                    ),
                    "family": "signature",
                    "transform": sig_tfm,
                    "source_table": table,
                    "source_col": price_col,
                    "lag": lags,
                    "sig_level": sig_level,
                    "term": j,
                    "augmentation_list": aug_repr,
                    "sig_tfm": sig_tfm,
                }
            )

        X = pd.DataFrame(X_cols, index=sig_df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )

    def _resolve_augmentation_pipeline(
        self, augmentation_tag
    ) -> Tuple[str, ...] | None:
        """Map high-level `augmentation_list` preset into sktime augmentation_list."""
        tag = (
            str(augmentation_tag).strip().lower()
            if augmentation_tag is not None
            else "all"
        )
        if tag in {"", "none", "null"}:
            return None
        if tag == "all":
            # Reasonable default (order matters)
            return ("cumsum", "basepoint", "addtime", "leadlag")
        # Delegate to the parser for single-augmentation presets (e.g., "time" -> "addtime")
        return self._parse_augmentations(tag)


class FadingMemorySignatureTemplate(_PriceTemplate):
    version = "1.1"
    name = "fd-signature"

    param_space = _PriceTemplate.param_space | {
        # Lookback window (number of past steps included in the *state path* fed to the signature).
        "lags": ParamSpec("int", default=30, low=1, high=252),
        # Comma-separated list of half-lives (trading days) used to build the EWMA bank.
        # Example: "2,5,10,20".
        "halflives": ParamSpec(
            "categorical",
            default="2,5,10,20",
            choices=[
                "2,5,10,20",
                "2,5,10,20,60",
                "5,10,20",
                "10,20,60",
                "5",
            ],
        ),
        # Base channels to EWMA-filter. "default" includes leverage proxy.
        "channels": ParamSpec(
            "categorical",
            default="default",
            choices=["default", "basic"],
        ),
        "sig_level": ParamSpec("int", default=2, low=1, high=5),
        "sig_tfm": ParamSpec(
            "categorical",
            default="logsignature",
            choices=["signature", "logsignature"],
        ),
        "augmentation_list": ParamSpec(
            "categorical",
            default="all",
            choices=["all", "none", "basepoint", "addtime", "leadlag", "ir", "cumsum"],
        ),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Query]]:
        return []

    _ALLOWED_AUGS = {"basepoint", "addtime", "leadlag", "ir", "cumsum"}

    def _parse_halflives(self, v) -> Tuple[int, ...]:
        """Parse halflife specification into a deterministic tuple of positive ints."""
        if v is None:
            return (5,)
        if isinstance(v, (list, tuple)):
            parts = [str(x).strip() for x in v if str(x).strip()]
        else:
            s = str(v).strip()
            parts = [p.strip() for p in s.split(",") if p.strip()]
        out: List[int] = []
        seen = set()
        for p in parts:
            try:
                hl = int(float(p))
            except Exception as e:
                raise ValueError(f"Invalid halflife value: {p!r}") from e
            if hl <= 0:
                raise ValueError(f"halflife must be positive, got {hl}")
            if hl not in seen:
                out.append(hl)
                seen.add(hl)
        return tuple(out) if out else (5,)

    def _build_base_channels(self, logret: pd.Series, mode: str) -> pd.DataFrame:
        """Build base per-timepoint channels from log returns.

        Returns a DataFrame indexed like `logret` (MultiIndex with entity_id/time) with columns:
        - r, abs_r, r2
        - neg_r2 (only in default mode)
        """
        r = logret.astype(float)
        data: Dict[str, pd.Series] = {
            "r": r,
            "abs_r": r.abs(),
            "r2": r.pow(2),
        }
        if str(mode).strip().lower() in {"default", "all"}:
            data["neg_r2"] = data["r2"].where(r < 0.0, 0.0)
        return pd.DataFrame(data).sort_index()

    def _ewma_bank_states(self, base_df: pd.DataFrame, halflives: Tuple[int, ...]) -> pd.DataFrame:
        """Compute an EWMA-bank over base channels for each entity.

        Uses pandas ewm(halflife=hl, adjust=False).mean() per entity.
        Output columns are named like '{channel}_hl{hl}'.
        """
        out_frames: List[pd.DataFrame] = []
        for hl in halflives:
            ew = (
                base_df.groupby(level="entity_id", group_keys=False)
                .apply(lambda g: g.ewm(halflife=hl, adjust=False).mean())
            )
            ew = ew.rename(columns={c: f"{c}_hl{hl}" for c in ew.columns})
            out_frames.append(ew)
        return pd.concat(out_frames, axis=1).sort_index()

    def _build_state_path_panel(self, state_df: pd.DataFrame, lags: int) -> Tuple[np.ndarray, pd.Index]:
        """Build a fixed-length state-path panel suitable for sktime SignatureTransformer.

        Returns:
        - X_panel: np.ndarray of shape (n_samples, n_channels, n_timepoints)
        - index: index aligned to the sample end-time (entity_id, time)

        Construction:
        For each sample end-time t, the path consists of [t-lags, ..., t] states.
        """
        # MultiIndex columns: (lag, feature)
        stacked = pd.concat(
            {i: state_df.groupby(level="entity_id").shift(i) for i in range(lags, -1, -1)},
            axis=1,
        ).dropna()

        n_timepoints = lags + 1
        n_channels = state_df.shape[1]

        arr = stacked.to_numpy(dtype=float)
        # stacked column order: lag blocks (lags..0) each containing all features
        arr = arr.reshape(arr.shape[0], n_timepoints, n_channels)
        X_panel = np.transpose(arr, (0, 2, 1))
        return X_panel, stacked.index

    def transform(
        self, ctx: DataContext, params: Dict[str, Any], slice: SliceSpec, state
    ):
        lags = int(params.get("lags", 30))
        sig_level = int(params["sig_level"])
        augmentation_tag = params.get("augmentation_list", "all")
        sig_tfm = str(params.get("sig_tfm", "logsignature")).strip().lower()
        if sig_tfm not in {"signature", "logsignature"}:
            raise ValueError("sig_tfm must be 'signature' or 'logsignature'")

        # ---- 1) Fetch prices and compute log returns
        panel, table, price_col = self._fetch_price_panel(ctx, params, slice)
        logret = self._logret(panel, price_col)

        # ---- 2) Build base channels and EWMA-bank states (fading-memory features)
        channels_mode = str(params.get("channels", "default")).strip().lower()
        base_df = self._build_base_channels(logret, channels_mode)

        halflives = self._parse_halflives(params.get("halflives", "2,5,10,20"))
        state_df = self._ewma_bank_states(base_df, halflives)

        # ---- 3) Build a fixed-length state-path panel: (instance, channel, time)
        X_panel, out_index = self._build_state_path_panel(state_df, lags=lags)

        # ---- 4) Ordered augmentation pipeline (order matters)
        augmentation_list = self._resolve_augmentation_pipeline(augmentation_tag)

        # ---- 5) SignatureTransformer
        # We already constructed a single fixed-length window per sample, so use window_name="global".
        rescaling = "post" if sig_tfm == "signature" else None
        sig_transformer = SignatureTransformer(
            augmentation_list=augmentation_list,
            depth=sig_level,
            window_name="global",
            window_depth=None,
            window_length=None,
            window_step=None,
            rescaling=rescaling,
            sig_tfm=sig_tfm,
            backend="esig",
        )
        _patch_instance_sklearn_tags(sig_transformer)

        sig_df = sig_transformer.fit_transform(X_panel)
        sig_df.index = out_index

        # ---- 6) Convert to FeatureFrame schema
        X_cols: Dict[str, np.ndarray] = {}
        cat: List[Dict[str, Any]] = []
        aug_repr = str(augmentation_tag).strip().lower()
        halflives_repr = ",".join(str(x) for x in halflives)
        for j, col in enumerate(sig_df.columns):
            fid = make_feature_id(
                table,
                "*",
                "fdsig",
                str(sig_level),
                {"term": j, "sig_tfm": sig_tfm},
            )
            X_cols[fid] = sig_df[col].to_numpy()
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": group_path(
                        "fdsig",
                        str(sig_level),
                        {
                            "lags": lags,
                            "halflives": halflives_repr,
                            "channels": channels_mode,
                            "augmentation_list": aug_repr,
                            "sig_tfm": sig_tfm,
                        },
                    ),
                    "family": "fading_memory_signature",
                    "transform": sig_tfm,
                    "source_table": table,
                    "source_col": price_col,
                    "lag": lags,
                    "sig_level": sig_level,
                    "term": j,
                    "augmentation_list": aug_repr,
                    "sig_tfm": sig_tfm,
                    "halflives": halflives_repr,
                    "channels": channels_mode,
                }
            )

        X = pd.DataFrame(X_cols, index=sig_df.index).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )

    def _resolve_augmentation_pipeline(
        self, augmentation_tag
    ) -> Tuple[str, ...] | None:
        """Map high-level `augmentation_list` preset into sktime augmentation_list."""
        tag = (
            str(augmentation_tag).strip().lower()
            if augmentation_tag is not None
            else "all"
        )
        if tag in {"", "none", "null"}:
            return None
        if tag == "all":
            # Reasonable default (order matters)
            return ("cumsum", "basepoint", "addtime", "leadlag")
        # Delegate to the parser for single-augmentation presets (e.g., "time" -> "addtime")
        return self._parse_augmentations(tag)
