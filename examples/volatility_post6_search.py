"""Canonical post-6 search driver for future frontier volatility models.

This driver keeps the post-6 workflow in the owning code repository instead of
inside stale notebook-side scripts. It treats Part 6 as the accepted frontier,
uses the fixed expanded-feature snapshot as its comparison frame input, and
keeps structural generalization in scope as a disciplined search family rather
than as an unrestricted architecture hunt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SNAPSHOT = (
    WORKSPACE_ROOT / "codex-workflows" / "volatility-forecasts-7-cache" / "expanded_dataset_snapshot.pkl"
)
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "post6_search"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTDIR / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from volatility_forecast.evaluation.dm_test import diebold_mariano
from volatility_forecast.evaluation.metrics import qlike
from volatility_forecast.model.pgarch_linear_model import PGARCHLinearModel
from volatility_forecast.model.xgb_pgarch_full_model import XGBPGARCHModel

try:
    from arch import arch_model
except Exception as exc:  # pragma: no cover - environment dependent
    arch_model = None
    ARCH_IMPORT_ERROR = exc
else:
    ARCH_IMPORT_ERROR = None

try:
    from volatility_forecast.model.volgru_config import VolGRUConfig
    from volatility_forecast.model.volgru_model import VolGRUModel
except Exception as exc:  # pragma: no cover - environment dependent
    VolGRUConfig = None
    VolGRUModel = None
    VOLGRU_IMPORT_ERROR = exc
else:
    VOLGRU_IMPORT_ERROR = None


SEED = 42
ACCEPTED_K = 5
VAL_LEN = 600
BASE_REG = 0.01
ROLLING_TRAIN_LEN = 4000
ROLLING_OOS_STARTS = (4000, 4400, 4800, 5200, 5600)
ROLLING_OOS_LEN = 400

ACCEPTED_XGB_PARAMS = {
    "loss": "qlike",
    "trees_per_channel_per_round": 1,
    "early_stopping_rounds": 4,
    "eval_metric": "qlike",
    "random_state": SEED,
    "verbosity": 0,
    "booster": "gbtree",
    "n_outer_rounds": 30,
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_child_weight": 0.0001,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "gamma": 0.0,
}

ASYM_SCREEN_CANDIDATES = (
    {"name": "pgarch-kphi3-kg5", "k_phi": 3, "k_g": 5, "worktree_hint": "codex/vol7-asym-screening"},
    {"name": "pgarch-kphi5-kg3", "k_phi": 5, "k_g": 3, "worktree_hint": "codex/vol7-asym-screening"},
    {"name": "pgarch-kphi5-kg7", "k_phi": 5, "k_g": 7, "worktree_hint": "codex/vol7-asym-screening"},
    {"name": "pgarch-kphi7-kg5", "k_phi": 7, "k_g": 5, "worktree_hint": "codex/vol7-asym-screening"},
    {"name": "pgarch-kphi3-kg7", "k_phi": 3, "k_g": 7, "worktree_hint": "codex/vol7-asym-screening"},
    {"name": "pgarch-kphi7-kg3", "k_phi": 7, "k_g": 3, "worktree_hint": "codex/vol7-asym-screening"},
)

G_PRIORITY_CANDIDATES = (
    {
        "name": "xgb-g-only-loose",
        "active_channels": ("g",),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "accepted"},
        "worktree_hint": "codex/vol7-g-only-nonlinear",
    },
    {
        "name": "xgb-g-then-phi-loose",
        "active_channels": ("g", "phi"),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "accepted"},
        "worktree_hint": "codex/vol7-g-only-nonlinear",
    },
    {
        "name": "xgb-g-only-moderate",
        "active_channels": ("g",),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 25,
            "learning_rate": 0.05,
            "max_depth": 3,
            "reg_lambda": 0.5,
        },
        "screening": {"source": "accepted"},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-only-loose",
        "active_channels": ("g",),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-g-only-nonlinear",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-loose",
        "active_channels": ("g", "phi"),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-g-only-nonlinear",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-loose-phi-mid",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "channel_param_overrides": {
                "phi": {
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "reg_lambda": 0.3,
                }
            },
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-mid-phi-loose",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "channel_param_overrides": {
                "g": {
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "reg_lambda": 0.3,
                }
            },
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-then-g-loose",
        "active_channels": ("g", "phi", "g"),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-g-then-phi-loose",
        "active_channels": ("g", "g", "phi"),
        "params": ACCEPTED_XGB_PARAMS,
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-mid-lr0075-lam02",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 28,
            "learning_rate": 0.075,
            "max_depth": 4,
            "reg_lambda": 0.2,
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-mid-lr0075-lam03",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 28,
            "learning_rate": 0.075,
            "max_depth": 4,
            "reg_lambda": 0.3,
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-mid-d3-lam02",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 30,
            "learning_rate": 0.05,
            "max_depth": 3,
            "reg_lambda": 0.2,
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-mid-d4-lr005",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 30,
            "learning_rate": 0.05,
            "max_depth": 4,
            "reg_lambda": 0.2,
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
    {
        "name": "xgb-asym-kphi5-kg7-g-then-phi-moderate",
        "active_channels": ("g", "phi"),
        "params": {
            **ACCEPTED_XGB_PARAMS,
            "n_outer_rounds": 25,
            "learning_rate": 0.05,
            "max_depth": 3,
            "reg_lambda": 0.5,
        },
        "screening": {"source": "asymmetric", "k_phi": 5, "k_g": 7},
        "worktree_hint": "codex/vol7-channel-capacity",
    },
)

STRUCTURAL_CANDIDATES = (
    {
        "name": "volgru-screened-stes1",
        "config": {
            "backend": "torch",
            "gate_mode": "stes_linear",
            "candidate_mode": "linear_pos",
            "state_dim": 1,
            "loss_mode": "qlike",
            "lr": 1e-2,
            "max_epochs": 60,
            "early_stopping_patience": 15,
            "val_fraction": 0.15,
            "weight_decay_gate": 1e-4,
            "weight_decay_candidate": 1e-4,
        },
        "worktree_hint": "codex/vol7-structural-generalization",
    },
    {
        "name": "volgru-screened-gru1",
        "config": {
            "backend": "torch",
            "gate_mode": "gru_linear",
            "candidate_mode": "linear_pos",
            "state_dim": 1,
            "loss_mode": "qlike",
            "lr": 5e-3,
            "max_epochs": 80,
            "early_stopping_patience": 20,
            "val_fraction": 0.15,
            "weight_decay_gate": 1e-4,
            "weight_decay_candidate": 1e-4,
        },
        "worktree_hint": "codex/vol7-structural-generalization",
    },
    {
        "name": "volgru-screened-gru3",
        "config": {
            "backend": "torch",
            "gate_mode": "gru_linear",
            "candidate_mode": "linear_pos",
            "state_dim": 3,
            "loss_mode": "qlike",
            "lr": 5e-3,
            "max_epochs": 100,
            "early_stopping_patience": 20,
            "val_fraction": 0.15,
            "weight_decay_gate": 1e-4,
            "weight_decay_candidate": 1e-4,
        },
        "worktree_hint": "codex/vol7-structural-generalization",
    },
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SnapshotData:
    X_train: pd.DataFrame
    X_all: pd.DataFrame
    y_train: pd.Series
    y_train_scaled: pd.Series
    y_all: pd.Series
    y_all_scaled: pd.Series
    r_train: pd.Series
    r_all: pd.Series
    actual_os: np.ndarray
    scale_factor: float

    @property
    def test_offset(self) -> int:
        return int(len(self.y_train))

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


@dataclass(slots=True)
class BaselineBundle:
    garch_os: np.ndarray
    linear_os: np.ndarray
    nonlinear_os: np.ndarray
    accepted_cf: dict[str, list[int]]
    accepted_phi_names: list[str]
    accepted_g_names: list[str]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    family: str
    name: str
    model_type: str
    rationale: str
    params: dict[str, Any]
    worktree_hint: str


@dataclass(frozen=True, slots=True)
class WindowSpec:
    train_start: int
    train_end: int
    test_end: int

    @property
    def test_start(self) -> int:
        return self.train_end

    @property
    def label(self) -> str:
        return f"{self.train_start}:{self.train_end}->{self.test_start}:{self.test_end}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical post-6 volatility search families against the Part 6 frontier.",
    )
    parser.add_argument(
        "--family",
        choices=("plan", "asym_screening", "g_priority_nonlinear", "structural_generalization", "all"),
        default="plan",
        help="Which search family to run. Default is a lightweight plan/manifest pass.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the fixed expanded-feature snapshot used for post-6 comparison.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for JSON manifests and family result summaries.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Optional candidate name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--val-len",
        type=int,
        default=VAL_LEN,
        help="Validation holdout length for boosted post-6 searches.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write JSON summaries to --outdir in addition to printing a console summary.",
    )
    parser.add_argument(
        "--rolling-candidate",
        action="append",
        default=[],
        help="Optional candidate name(s) to validate on rolling windows after search.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_snapshot(path: Path) -> SnapshotData:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with path.open("rb") as f:
        snapshot = pickle.load(f)

    x_train = pd.DataFrame(snapshot["X_tr_exp_s"]).copy()
    x_all = pd.DataFrame(snapshot["X_all_exp_s"]).copy()
    y_train = pd.Series(snapshot["y_tr"]).copy()
    y_train_scaled = pd.Series(snapshot["y_tr_scaled"]).copy()
    r_train = pd.Series(snapshot["r_tr"]).copy()
    r_all = pd.Series(snapshot["r_all"]).copy()
    actual_os = np.asarray(snapshot["actual_os"], dtype=float)
    scale_factor = float(snapshot["SCALE_FACTOR"])

    y_all = pd.Series(
        np.concatenate([y_train.to_numpy(dtype=float), actual_os]),
        index=x_all.index,
        name="y",
        dtype=float,
    )
    y_all_scaled = pd.Series(
        np.concatenate([y_train_scaled.to_numpy(dtype=float), actual_os * (scale_factor**2)]),
        index=x_all.index,
        name="y_scaled",
        dtype=float,
    )

    return SnapshotData(
        X_train=x_train,
        X_all=x_all,
        y_train=y_train.astype(float),
        y_train_scaled=y_train_scaled.astype(float),
        y_all=y_all,
        y_all_scaled=y_all_scaled,
        r_train=r_train.astype(float),
        r_all=r_all.astype(float),
        actual_os=actual_os,
        scale_factor=scale_factor,
    )


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    y_arr = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(y_pred, dtype=float), epsilon)
    return np.log(pred) + y_arr / pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_arr - pred) ** 2)))


def dm_row(y_true: np.ndarray, pred_candidate: np.ndarray, pred_baseline: np.ndarray) -> dict[str, float]:
    result = diebold_mariano(qlike_loss(y_true, pred_candidate), qlike_loss(y_true, pred_baseline), h=1)
    return {
        "stat": float(result["dm_stat"]),
        "p_value": float(result["p_value"]),
        "mean_d": float(result["mean_d"]),
    }


def shared_channel_features(n_features: int) -> dict[str, list[int]]:
    all_idx = list(range(n_features))
    return {"mu": [], "phi": all_idx, "g": all_idx}


def fit_screen_ranker(data: SnapshotData, *, val_len: int) -> tuple[pd.Series, pd.Series]:
    fit_end = len(data.X_train) - val_len
    if fit_end <= 0:
        raise ValueError(f"val_len={val_len} leaves no fit window.")

    channel_features = shared_channel_features(data.X_train.shape[1])
    ranker = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    ranker.fit(data.y_train.iloc[:fit_end].to_numpy(dtype=float), data.X_train.iloc[:fit_end].to_numpy(dtype=float))

    phi = pd.Series(np.abs(np.asarray(ranker.coef_phi_[1:], dtype=float)), index=data.X_train.columns).sort_values(
        ascending=False
    )
    g = pd.Series(np.abs(np.asarray(ranker.coef_g_[1:], dtype=float)), index=data.X_train.columns).sort_values(
        ascending=False
    )
    return phi, g


def build_screened_channel_features(
    columns: list[str],
    phi_ranking: pd.Series,
    g_ranking: pd.Series,
    *,
    k_phi: int,
    k_g: int,
) -> dict[str, list[int]]:
    pos = {name: idx for idx, name in enumerate(columns)}
    phi_names = list(phi_ranking.index[:k_phi])
    g_names = list(g_ranking.index[:k_g])
    return {
        "mu": [],
        "phi": [pos[name] for name in phi_names],
        "g": [pos[name] for name in g_names],
    }


def channel_feature_names(columns: list[str], channel_features: dict[str, list[int]]) -> dict[str, list[str]]:
    return {
        "mu": [columns[idx] for idx in channel_features["mu"]],
        "phi": [columns[idx] for idx in channel_features["phi"]],
        "g": [columns[idx] for idx in channel_features["g"]],
    }


def build_candidate_registry(
    *,
    accepted_union: list[str],
) -> dict[str, list[CandidateSpec]]:
    registry = {
        "asym_screening": [
            CandidateSpec(
                family="asym_screening",
                name=spec["name"],
                model_type="PGARCH-L",
                rationale="Narrow structural deviation from the accepted K=5 screened frontier.",
                params={"k_phi": spec["k_phi"], "k_g": spec["k_g"]},
                worktree_hint=spec["worktree_hint"],
            )
            for spec in ASYM_SCREEN_CANDIDATES
        ],
        "g_priority_nonlinear": [
            CandidateSpec(
                family="g_priority_nonlinear",
                name=spec["name"],
                model_type="XGBPGARCH",
                rationale="Keep the accepted screened tier fixed and give nonlinear capacity to g first.",
                params={
                    "active_channels": list(spec["active_channels"]),
                    "xgb_params": spec["params"],
                    "screening": spec["screening"],
                },
                worktree_hint=spec["worktree_hint"],
            )
            for spec in G_PRIORITY_CANDIDATES
        ],
        "structural_generalization": [
            CandidateSpec(
                family="structural_generalization",
                name=spec["name"],
                model_type="VolGRU",
                rationale=(
                    "Hold the screened feature tier fixed and allow only a low-capacity structural relaxation. "
                    "A failed STES -> 1dGRU branch prunes that branch, not the whole structural search space."
                ),
                params={"feature_union": accepted_union, "config": spec["config"]},
                worktree_hint=spec["worktree_hint"],
            )
            for spec in STRUCTURAL_CANDIDATES
        ],
    }
    return registry


def fit_linear_candidate(data: SnapshotData, channel_features: dict[str, list[int]]) -> tuple[PGARCHLinearModel, np.ndarray]:
    model = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    model.fit(data.y_train.to_numpy(dtype=float), data.X_train.to_numpy(dtype=float))
    pred_all = model.predict_variance(data.y_all.to_numpy(dtype=float), data.X_all.to_numpy(dtype=float))
    return model, np.asarray(pred_all[data.test_offset :], dtype=float)


def fit_xgb_candidate(
    data: SnapshotData,
    channel_features: dict[str, list[int]],
    *,
    xgb_params: dict[str, Any],
    active_channels: tuple[str, ...],
    val_len: int,
) -> tuple[XGBPGARCHModel, np.ndarray, dict[str, Any]]:
    fit_end = len(data.X_train) - val_len
    if fit_end <= 0:
        raise ValueError(f"val_len={val_len} leaves no fit window.")

    x_fit = data.X_train.iloc[:fit_end]
    x_val = data.X_train.iloc[fit_end:]
    y_fit = data.y_train_scaled.iloc[:fit_end]
    y_val = data.y_train_scaled.iloc[fit_end:]

    init_val = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    init_val.fit(y_fit.to_numpy(dtype=float), x_fit.to_numpy(dtype=float))

    model_val = XGBPGARCHModel(init_model=init_val, channel_features=channel_features, **xgb_params)
    model_val.channel_update_order = tuple(active_channels)
    model_val.fit(y_fit.to_numpy(dtype=float), x_fit.to_numpy(dtype=float), eval_set=(y_val.to_numpy(dtype=float), x_val.to_numpy(dtype=float)))

    best_rounds = xgb_params["n_outer_rounds"] if model_val.best_iteration_ is None else int(model_val.best_iteration_) + 1

    init_full = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    init_full.fit(data.y_train_scaled.to_numpy(dtype=float), data.X_train.to_numpy(dtype=float))

    refit_params = dict(xgb_params)
    refit_params["n_outer_rounds"] = best_rounds
    refit_params.pop("early_stopping_rounds", None)
    refit_params.pop("eval_metric", None)

    model = XGBPGARCHModel(init_model=init_full, channel_features=channel_features, **refit_params)
    model.channel_update_order = tuple(active_channels)
    model.fit(data.y_train_scaled.to_numpy(dtype=float), data.X_train.to_numpy(dtype=float))

    pred_all_scaled = model.predict_variance(data.y_all_scaled.to_numpy(dtype=float), data.X_all.to_numpy(dtype=float))
    pred_os = np.asarray(pred_all_scaled[data.test_offset :], dtype=float) / (data.scale_factor**2)
    return model, pred_os, {"active_channels": list(active_channels), "best_rounds": int(best_rounds)}


def fit_garch_baseline(data: SnapshotData) -> np.ndarray:
    if arch_model is None:
        raise RuntimeError(f"arch is required for the GARCH baseline: {ARCH_IMPORT_ERROR}")

    r_train_pct = data.r_train.to_numpy(dtype=float) * data.scale_factor
    r_all_pct = data.r_all.to_numpy(dtype=float) * data.scale_factor
    fitted = arch_model(r_train_pct, vol="GARCH", p=1, q=1, mean="Zero", rescale=False).fit(disp="off")
    filtered = arch_model(r_all_pct, vol="GARCH", p=1, q=1, mean="Zero", rescale=False).fix(fitted.params)
    pred_all = (np.asarray(filtered.conditional_volatility, dtype=float) ** 2) / (data.scale_factor**2)
    return pred_all[data.test_offset :]


def fit_garch_filtered(r_train: pd.Series, r_window: pd.Series, scale_factor: float) -> np.ndarray:
    if arch_model is None:
        raise RuntimeError(f"arch is required for the GARCH baseline: {ARCH_IMPORT_ERROR}")

    r_train_pct = np.asarray(r_train, dtype=float) * scale_factor
    r_window_pct = np.asarray(r_window, dtype=float) * scale_factor
    fitted = arch_model(r_train_pct, vol="GARCH", p=1, q=1, mean="Zero", rescale=False).fit(disp="off")
    filtered = arch_model(r_window_pct, vol="GARCH", p=1, q=1, mean="Zero", rescale=False).fix(fitted.params)
    return (np.asarray(filtered.conditional_volatility, dtype=float) ** 2) / (scale_factor**2)


def make_structural_frame(frame: pd.DataFrame, feature_union: list[str]) -> pd.DataFrame:
    selected = frame.loc[:, feature_union].copy()
    return pd.concat(
        [
            pd.DataFrame({"const": np.ones(len(selected), dtype=float)}, index=selected.index),
            selected,
        ],
        axis=1,
    )


def fit_volgru_candidate(
    data: SnapshotData,
    *,
    feature_union: list[str],
    config_kwargs: dict[str, Any],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    if VolGRUConfig is None or VolGRUModel is None:
        raise RuntimeError(f"VolGRU is unavailable in this environment: {VOLGRU_IMPORT_ERROR}")

    x_train = make_structural_frame(data.X_train, feature_union)
    x_all = make_structural_frame(data.X_all, feature_union)
    config = VolGRUConfig(**config_kwargs)
    model = VolGRUModel(config=config, random_state=SEED)
    model.fit(
        x_train,
        data.y_train.to_numpy(dtype=float),
        returns=data.r_train.to_numpy(dtype=float),
        start_index=1,
    )
    pred_all = model.predict(x_all, returns=data.r_all.to_numpy(dtype=float))
    pred_os = np.asarray(pred_all[data.test_offset :], dtype=float)
    return model, pred_os, {"feature_count": len(feature_union), "state_dim": int(config.state_dim)}


def evaluate_candidate(
    *,
    family: str,
    name: str,
    model_type: str,
    data: SnapshotData,
    pred_os: np.ndarray,
    baselines: BaselineBundle,
    params: dict[str, Any],
) -> dict[str, Any]:
    actual = np.asarray(data.actual_os, dtype=float)
    dm_garch = dm_row(actual, pred_os, baselines.garch_os)
    dm_linear = dm_row(actual, pred_os, baselines.linear_os)
    dm_nonlinear = dm_row(actual, pred_os, baselines.nonlinear_os)
    phi_overlap = len(set(baselines.accepted_phi_names) & set(baselines.accepted_g_names))

    row = {
        "family": family,
        "candidate": name,
        "model_type": model_type,
        "os_qlike": float(qlike(actual, pred_os)),
        "os_rmse": rmse(actual, pred_os),
        "qlike_gain_vs_garch": float(qlike(actual, baselines.garch_os) - qlike(actual, pred_os)),
        "qlike_gain_vs_part6_linear": float(qlike(actual, baselines.linear_os) - qlike(actual, pred_os)),
        "qlike_gain_vs_part6_nonlinear": float(qlike(actual, baselines.nonlinear_os) - qlike(actual, pred_os)),
        "dm_vs_garch": dm_garch["stat"],
        "p_vs_garch": dm_garch["p_value"],
        "dm_vs_part6_linear": dm_linear["stat"],
        "p_vs_part6_linear": dm_linear["p_value"],
        "dm_vs_part6_nonlinear": dm_nonlinear["stat"],
        "p_vs_part6_nonlinear": dm_nonlinear["p_value"],
        "accepted_channel_overlap": phi_overlap,
        "params": params,
    }
    return row


def build_part6_baselines(
    data: SnapshotData,
    *,
    phi_ranking: pd.Series,
    g_ranking: pd.Series,
    val_len: int,
) -> BaselineBundle:
    columns = data.feature_names
    accepted_cf = build_screened_channel_features(columns, phi_ranking, g_ranking, k_phi=ACCEPTED_K, k_g=ACCEPTED_K)
    accepted_names = channel_feature_names(columns, accepted_cf)
    _, linear_os = fit_linear_candidate(data, accepted_cf)
    _, nonlinear_os, _ = fit_xgb_candidate(
        data,
        accepted_cf,
        xgb_params=ACCEPTED_XGB_PARAMS,
        active_channels=("phi", "g"),
        val_len=val_len,
    )
    garch_os = fit_garch_baseline(data)
    return BaselineBundle(
        garch_os=garch_os,
        linear_os=linear_os,
        nonlinear_os=nonlinear_os,
        accepted_cf=accepted_cf,
        accepted_phi_names=accepted_names["phi"],
        accepted_g_names=accepted_names["g"],
    )


def run_asym_screening(
    data: SnapshotData,
    *,
    phi_ranking: pd.Series,
    g_ranking: pd.Series,
    baselines: BaselineBundle,
    specs: list[CandidateSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        params = spec.params
        cf = build_screened_channel_features(
            data.feature_names,
            phi_ranking,
            g_ranking,
            k_phi=int(params["k_phi"]),
            k_g=int(params["k_g"]),
        )
        names = channel_feature_names(data.feature_names, cf)
        started = time.perf_counter()
        _, pred_os = fit_linear_candidate(data, cf)
        elapsed = time.perf_counter() - started
        row = evaluate_candidate(
            family=spec.family,
            name=spec.name,
            model_type=spec.model_type,
            data=data,
            pred_os=pred_os,
            baselines=baselines,
            params={
                **params,
                "phi_features": names["phi"],
                "g_features": names["g"],
                "overlap": len(set(names["phi"]) & set(names["g"])),
                "runtime_seconds": round(elapsed, 3),
            },
        )
        rows.append(row)
    return rows


def run_g_priority_nonlinear(
    data: SnapshotData,
    *,
    phi_ranking: pd.Series,
    g_ranking: pd.Series,
    baselines: BaselineBundle,
    specs: list[CandidateSpec],
    val_len: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        screening = spec.params["screening"]
        if screening["source"] == "accepted":
            channel_features = baselines.accepted_cf
        elif screening["source"] == "asymmetric":
            channel_features = build_screened_channel_features(
                data.feature_names,
                phi_ranking,
                g_ranking,
                k_phi=int(screening["k_phi"]),
                k_g=int(screening["k_g"]),
            )
        else:
            raise ValueError(f"Unsupported screening source: {screening['source']!r}")

        channel_names = channel_feature_names(data.feature_names, channel_features)
        started = time.perf_counter()
        _, pred_os, meta = fit_xgb_candidate(
            data,
            channel_features,
            xgb_params=spec.params["xgb_params"],
            active_channels=tuple(spec.params["active_channels"]),
            val_len=val_len,
        )
        elapsed = time.perf_counter() - started
        row = evaluate_candidate(
            family=spec.family,
            name=spec.name,
            model_type=spec.model_type,
            data=data,
            pred_os=pred_os,
            baselines=baselines,
            params={
                **spec.params,
                **meta,
                "phi_features": channel_names["phi"],
                "g_features": channel_names["g"],
                "runtime_seconds": round(elapsed, 3),
            },
        )
        rows.append(row)
    return rows


def run_structural_generalization(
    data: SnapshotData,
    *,
    baselines: BaselineBundle,
    specs: list[CandidateSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        started = time.perf_counter()
        _, pred_os, meta = fit_volgru_candidate(
            data,
            feature_union=list(spec.params["feature_union"]),
            config_kwargs=dict(spec.params["config"]),
        )
        elapsed = time.perf_counter() - started
        row = evaluate_candidate(
            family=spec.family,
            name=spec.name,
            model_type=spec.model_type,
            data=data,
            pred_os=pred_os,
            baselines=baselines,
            params={**spec.params, **meta, "runtime_seconds": round(elapsed, 3)},
        )
        rows.append(row)
    return rows


def select_specs(specs: list[CandidateSpec], requested: list[str]) -> list[CandidateSpec]:
    if not requested:
        return specs
    requested_set = set(requested)
    return [spec for spec in specs if spec.name in requested_set]


def rolling_window_specs(total_len: int) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for start in ROLLING_OOS_STARTS:
        if start > total_len:
            break
        train_start = start - ROLLING_TRAIN_LEN
        if train_start < 0:
            continue
        test_end = min(start + ROLLING_OOS_LEN, total_len)
        specs.append(WindowSpec(train_start=train_start, train_end=start, test_end=test_end))
    return specs


def fit_part6_linear_on_window(
    x_train: pd.DataFrame,
    x_window: pd.DataFrame,
    y_train: pd.Series,
    y_window: pd.Series,
    *,
    channel_features: dict[str, list[int]],
) -> np.ndarray:
    model = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    model.fit(y_train.to_numpy(dtype=float), x_train.to_numpy(dtype=float))
    pred_window = model.predict_variance(y_window.to_numpy(dtype=float), x_window.to_numpy(dtype=float))
    return np.asarray(pred_window[len(x_train) :], dtype=float)


def fit_xgb_on_window(
    x_train: pd.DataFrame,
    x_window: pd.DataFrame,
    y_train_scaled: pd.Series,
    y_window_scaled: pd.Series,
    *,
    channel_features: dict[str, list[int]],
    xgb_params: dict[str, Any],
    active_channels: tuple[str, ...],
    val_len: int,
    scale_factor: float,
) -> np.ndarray:
    fit_end = len(x_train) - val_len
    if fit_end <= 0:
        raise ValueError(f"val_len={val_len} leaves no fit window.")

    x_fit = x_train.iloc[:fit_end]
    x_val = x_train.iloc[fit_end:]
    y_fit_scaled = y_train_scaled.iloc[:fit_end]
    y_val_scaled = y_train_scaled.iloc[fit_end:]

    init_val = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    init_val.fit(y_fit_scaled.to_numpy(dtype=float), x_fit.to_numpy(dtype=float))
    model_val = XGBPGARCHModel(init_model=init_val, channel_features=channel_features, **xgb_params)
    model_val.channel_update_order = tuple(active_channels)
    model_val.fit(
        y_fit_scaled.to_numpy(dtype=float),
        x_fit.to_numpy(dtype=float),
        eval_set=(y_val_scaled.to_numpy(dtype=float), x_val.to_numpy(dtype=float)),
    )
    best_rounds = xgb_params["n_outer_rounds"] if model_val.best_iteration_ is None else int(model_val.best_iteration_) + 1

    init_full = PGARCHLinearModel(
        loss="qlike",
        dynamic_mu=False,
        lambda_mu=BASE_REG,
        lambda_phi=BASE_REG,
        lambda_g=BASE_REG,
        channel_features=channel_features,
        standardize_features=False,
        random_state=SEED,
    )
    init_full.fit(y_train_scaled.to_numpy(dtype=float), x_train.to_numpy(dtype=float))
    refit_params = dict(xgb_params)
    refit_params["n_outer_rounds"] = best_rounds
    refit_params.pop("early_stopping_rounds", None)
    refit_params.pop("eval_metric", None)

    model = XGBPGARCHModel(init_model=init_full, channel_features=channel_features, **refit_params)
    model.channel_update_order = tuple(active_channels)
    model.fit(y_train_scaled.to_numpy(dtype=float), x_train.to_numpy(dtype=float))

    pred_window_scaled = model.predict_variance(
        y_window_scaled.to_numpy(dtype=float),
        x_window.to_numpy(dtype=float),
    )
    return np.asarray(pred_window_scaled[len(x_train) :], dtype=float) / (scale_factor**2)


def resolve_g_priority_channel_features(
    data: SnapshotData,
    *,
    phi_ranking: pd.Series,
    g_ranking: pd.Series,
    baselines: BaselineBundle,
    screening: dict[str, Any],
) -> dict[str, list[int]]:
    if screening["source"] == "accepted":
        return baselines.accepted_cf
    if screening["source"] == "asymmetric":
        return build_screened_channel_features(
            data.feature_names,
            phi_ranking,
            g_ranking,
            k_phi=int(screening["k_phi"]),
            k_g=int(screening["k_g"]),
        )
    raise ValueError(f"Unsupported screening source: {screening['source']!r}")


def run_rolling_validation(
    data: SnapshotData,
    *,
    candidate_spec: CandidateSpec,
    candidate_channel_features: dict[str, list[int]],
    baselines: BaselineBundle,
    val_len: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in rolling_window_specs(len(data.X_all)):
        train_slice = slice(window.train_start, window.train_end)
        test_slice = slice(window.test_start, window.test_end)
        window_slice = slice(window.train_start, window.test_end)

        x_train = data.X_all.iloc[train_slice]
        x_window = data.X_all.iloc[window_slice]
        y_train = data.y_all.iloc[train_slice]
        y_window = data.y_all.iloc[window_slice]
        y_train_scaled = data.y_all_scaled.iloc[train_slice]
        y_window_scaled = data.y_all_scaled.iloc[window_slice]
        r_train = data.r_all.iloc[train_slice]
        r_window = data.r_all.iloc[window_slice]
        actual = data.y_all.iloc[test_slice].to_numpy(dtype=float)

        garch_window = fit_garch_filtered(r_train, r_window, data.scale_factor)
        garch_os = garch_window[len(x_train) :]
        linear_os = fit_part6_linear_on_window(
            x_train,
            x_window,
            y_train,
            y_window,
            channel_features=baselines.accepted_cf,
        )
        nonlinear_os = fit_xgb_on_window(
            x_train,
            x_window,
            y_train_scaled,
            y_window_scaled,
            channel_features=baselines.accepted_cf,
            xgb_params=ACCEPTED_XGB_PARAMS,
            active_channels=("phi", "g"),
            val_len=val_len,
            scale_factor=data.scale_factor,
        )

        if candidate_spec.family == "g_priority_nonlinear":
            candidate_os = fit_xgb_on_window(
                x_train,
                x_window,
                y_train_scaled,
                y_window_scaled,
                channel_features=candidate_channel_features,
                xgb_params=candidate_spec.params["xgb_params"],
                active_channels=tuple(candidate_spec.params["active_channels"]),
                val_len=val_len,
                scale_factor=data.scale_factor,
            )
        elif candidate_spec.family == "asym_screening":
            candidate_os = fit_part6_linear_on_window(
                x_train,
                x_window,
                y_train,
                y_window,
                channel_features=candidate_channel_features,
            )
        else:
            raise ValueError(f"Rolling validation not implemented for family {candidate_spec.family!r}")

        scores = {
            "candidate": float(qlike(actual, candidate_os)),
            "garch": float(qlike(actual, garch_os)),
            "part6_linear": float(qlike(actual, linear_os)),
            "part6_nonlinear": float(qlike(actual, nonlinear_os)),
        }
        ordered = sorted(scores, key=scores.get)
        dm_linear = dm_row(actual, candidate_os, linear_os)
        dm_nonlinear = dm_row(actual, candidate_os, nonlinear_os)
        rows.append(
            {
                "window": window.label,
                "candidate": candidate_spec.name,
                "os_qlike_candidate": scores["candidate"],
                "os_qlike_garch": scores["garch"],
                "os_qlike_part6_linear": scores["part6_linear"],
                "os_qlike_part6_nonlinear": scores["part6_nonlinear"],
                "rank_candidate": int(ordered.index("candidate") + 1),
                "rank_garch": int(ordered.index("garch") + 1),
                "rank_part6_linear": int(ordered.index("part6_linear") + 1),
                "rank_part6_nonlinear": int(ordered.index("part6_nonlinear") + 1),
                "gain_vs_part6_linear": float(scores["part6_linear"] - scores["candidate"]),
                "gain_vs_part6_nonlinear": float(scores["part6_nonlinear"] - scores["candidate"]),
                "dm_vs_part6_linear": dm_linear["stat"],
                "p_vs_part6_linear": dm_linear["p_value"],
                "dm_vs_part6_nonlinear": dm_nonlinear["stat"],
                "p_vs_part6_nonlinear": dm_nonlinear["p_value"],
            }
        )
    return rows


def build_manifest(data: SnapshotData, *, val_len: int, snapshot_path: Path) -> dict[str, Any]:
    phi_ranking, g_ranking = fit_screen_ranker(data, val_len=val_len)
    accepted_cf = build_screened_channel_features(data.feature_names, phi_ranking, g_ranking, k_phi=ACCEPTED_K, k_g=ACCEPTED_K)
    accepted_names = channel_feature_names(data.feature_names, accepted_cf)
    accepted_union = list(dict.fromkeys(accepted_names["phi"] + accepted_names["g"]))
    registry = build_candidate_registry(accepted_union=accepted_union)

    return {
        "series_frontier": {
            "linear": "PGARCH-L (screened K=5, constant mu)",
            "nonlinear": "XGBPGARCH [gbtree-loose, screened K=5, constant mu]",
            "benchmark": "GARCH(1,1)",
        },
        "snapshot_path": str(snapshot_path),
        "screening": {
            "accepted_k_phi": ACCEPTED_K,
            "accepted_k_g": ACCEPTED_K,
            "accepted_phi_features": accepted_names["phi"],
            "accepted_g_features": accepted_names["g"],
            "accepted_feature_union": accepted_union,
            "phi_top10": phi_ranking.head(10).to_dict(),
            "g_top10": g_ranking.head(10).to_dict(),
        },
        "candidate_registry": {
            family: [asdict(spec) for spec in specs]
            for family, specs in registry.items()
        },
        "search_note": (
            "Structural generalization remains in scope, but it should hold the screened feature tier fixed. "
            "A failed STES -> 1dGRU branch does not eliminate later low-capacity structural relaxations."
        ),
    }


def summarise_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No candidates matched the requested filter.")
        return
    ordered = sorted(rows, key=lambda row: row["os_qlike"])
    for row in ordered:
        print(
            f"{row['candidate']}: "
            f"OS QLIKE={row['os_qlike']:.6f}, "
            f"gain vs part6 nonlinear={row['qlike_gain_vs_part6_nonlinear']:.6f}, "
            f"gain vs part6 linear={row['qlike_gain_vs_part6_linear']:.6f}"
        )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    data = load_snapshot(args.snapshot)
    manifest = build_manifest(data, val_len=args.val_len, snapshot_path=args.snapshot)
    print(f"Loaded snapshot: {args.snapshot}")
    print(
        "Accepted screened tier: "
        f"phi={manifest['screening']['accepted_phi_features']} "
        f"g={manifest['screening']['accepted_g_features']}"
    )

    if args.write_json:
        outdir = ensure_outdir(args.outdir)
        manifest_path = outdir / "post6_candidate_manifest.json"
        write_json(manifest_path, manifest)
        print(f"Wrote manifest: {manifest_path}")

    if args.family == "plan":
        print("Plan mode only. No candidate family executed.")
        return

    phi_ranking, g_ranking = fit_screen_ranker(data, val_len=args.val_len)
    baselines = build_part6_baselines(data, phi_ranking=phi_ranking, g_ranking=g_ranking, val_len=args.val_len)

    accepted_registry = build_candidate_registry(accepted_union=manifest["screening"]["accepted_feature_union"])
    rows: list[dict[str, Any]] = []

    if args.family in {"asym_screening", "all"}:
        specs = select_specs(accepted_registry["asym_screening"], args.candidate)
        rows.extend(
            run_asym_screening(
                data,
                phi_ranking=phi_ranking,
                g_ranking=g_ranking,
                baselines=baselines,
                specs=specs,
            )
        )

    if args.family in {"g_priority_nonlinear", "all"}:
        specs = select_specs(accepted_registry["g_priority_nonlinear"], args.candidate)
        rows.extend(
            run_g_priority_nonlinear(
                data,
                phi_ranking=phi_ranking,
                g_ranking=g_ranking,
                baselines=baselines,
                specs=specs,
                val_len=args.val_len,
            )
        )

    if args.family in {"structural_generalization", "all"}:
        specs = select_specs(accepted_registry["structural_generalization"], args.candidate)
        rows.extend(
            run_structural_generalization(
                data,
                baselines=baselines,
                specs=specs,
            )
        )

    summarise_rows(rows)

    if args.write_json:
        outdir = ensure_outdir(args.outdir)
        family_path = outdir / f"{args.family}_results.json"
        write_json(
            family_path,
            {
                "family": args.family,
                "candidate_filter": args.candidate,
                "snapshot": str(args.snapshot),
                "rows": rows,
            },
        )
        print(f"Wrote results: {family_path}")

    if args.rolling_candidate:
        candidate_map = {
            spec.name: spec
            for family_specs in accepted_registry.values()
            for spec in family_specs
        }
        outdir = ensure_outdir(args.outdir) if args.write_json else args.outdir
        for candidate_name in args.rolling_candidate:
            if candidate_name not in candidate_map:
                raise ValueError(f"Unknown rolling candidate: {candidate_name}")
            spec = candidate_map[candidate_name]
            if spec.family == "g_priority_nonlinear":
                candidate_cf = resolve_g_priority_channel_features(
                    data,
                    phi_ranking=phi_ranking,
                    g_ranking=g_ranking,
                    baselines=baselines,
                    screening=spec.params["screening"],
                )
            elif spec.family == "asym_screening":
                candidate_cf = build_screened_channel_features(
                    data.feature_names,
                    phi_ranking,
                    g_ranking,
                    k_phi=int(spec.params["k_phi"]),
                    k_g=int(spec.params["k_g"]),
                )
            else:
                raise ValueError(f"Rolling validation not supported for family {spec.family!r}")

            rolling_rows = run_rolling_validation(
                data,
                candidate_spec=spec,
                candidate_channel_features=candidate_cf,
                baselines=baselines,
                val_len=args.val_len,
            )
            wins = sum(
                1
                for row in rolling_rows
                if row["gain_vs_part6_linear"] > 0.0 and row["gain_vs_part6_nonlinear"] > 0.0
            )
            avg_gain_linear = float(np.mean([row["gain_vs_part6_linear"] for row in rolling_rows]))
            avg_gain_nonlinear = float(np.mean([row["gain_vs_part6_nonlinear"] for row in rolling_rows]))
            print(
                f"Rolling validation {candidate_name}: "
                f"wins_vs_both={wins}/{len(rolling_rows)}, "
                f"avg_gain_vs_part6_linear={avg_gain_linear:.6f}, "
                f"avg_gain_vs_part6_nonlinear={avg_gain_nonlinear:.6f}"
            )
            if args.write_json:
                rolling_path = outdir / f"{candidate_name}_rolling_validation.json"
                write_json(
                    rolling_path,
                    {
                        "candidate": candidate_name,
                        "rows": rolling_rows,
                        "wins_vs_both": wins,
                        "avg_gain_vs_part6_linear": avg_gain_linear,
                        "avg_gain_vs_part6_nonlinear": avg_gain_nonlinear,
                    },
                )
                print(f"Wrote rolling validation: {rolling_path}")


if __name__ == "__main__":
    main()
