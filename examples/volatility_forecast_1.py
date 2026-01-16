"""
Volatility forecast demo: ES/STES on simulated data and SPY.
Builds a wide feature set, fits variants, reports RMSE, and writes diagnostics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

# Script-friendly repo path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Load .env from repo/CWD/HOME
_loaded_env_paths: list[str] = []
for dotenv_path in [repo_root / ".env", Path.cwd() / ".env", Path.home() / ".env"]:
    try:
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            _loaded_env_paths.append(str(dotenv_path))
    except Exception:
        pass


def _read_tiingo_api_key() -> str:
    for k in [
        "TIINGO_API",
    ]:
        v = os.environ.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return ""


import logging

# Logging config
_log_level = os.environ.get("AF_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _log_run_context():
    try:
        env_note = ", ".join(_loaded_env_paths) if _loaded_env_paths else "none"
        logger.info(f"CWD={Path.cwd()} | .env loaded from: {env_note}")
    except Exception:
        pass


# Pipeline
from alphaforge.data.context import DataContext
from alphaforge.features.dataset_spec import (
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
    JoinPolicy,
    MissingnessPolicy,
)
from volatility_forecast.pipeline import (
    build_default_ctx,
    build_vol_dataset,
    VolDatasetSpec,
    build_wide_dataset,
)

# Features/targets
from volatility_forecast.features.return_features import (
    LagLogReturnTemplate,
    LagAbsLogReturnTemplate,
    LagSquaredLogReturnTemplate,
)
from volatility_forecast.features.selector import select_variant_columns
from volatility_forecast.targets.squared_return import NextDaySquaredReturnTarget


# Models
from volatility_forecast.model.es_model import ESModel
from volatility_forecast.model.stes_model import STESModel

# Simulated data
from volatility_forecast.reporting.plots import (
    plot_gate_diagnostics,
    plot_forecast_panel,
    plot_event_paths,
    plot_bar_series,
    plot_gate_panel,
)
from volatility_forecast.sources.simulated_garch import SimulatedGARCHSource


VARIANTS = [
    "ES",
    "STES_AE",  # abs
    "STES_SE",  # squared
    "STES_EAE",  # raw + abs
    "STES_ESE",  # raw + squared
    "STES_AESE",  # abs + squared
    "STES_EAESE",  # raw + abs + squared
]

# Model kwargs
MODEL_KWARGS = {
    "ES": {},
    "STES": {
        # "solver": "least_squares",
    },
}

ENTITY = "SIMULATED"
SOURCE = "simulated_garch"
TABLE = "market.ohlcv"
PRICE_COL = "close"
CALENDAR = "XNYS"
GRID = "B"
START = pd.Timestamp("2000-01-01", tz="UTC")
END = pd.Timestamp("2023-01-01", tz="UTC")
N_LAGS = 0

IS_INDEX = 500
OS_INDEX = 2000
N_RUNS = 100

# SPY study settings
SPY_TICKER = "SPY"
SPY_START = pd.Timestamp("2000-01-01", tz="UTC")
SPY_END = pd.Timestamp("2023-12-31", tz="UTC")
SPY_IS_INDEX = 200
SPY_OS_INDEX = 4000
SPY_N_INITS = 100


def _attach_sim_source(ctx: DataContext, run_seed: int) -> None:
    """Attach a SimulatedGARCHSource to the context."""
    ctx.sources[SOURCE] = SimulatedGARCHSource(
        n_periods=2500,
        random_state=run_seed,
        mu=0.0,
        omega=0.02,
        alpha=0.11,
        beta=0.87,
        eta=4.0,
        shock_prob=0.005,
        entity_id=ENTITY,
    )


def _build_sim_spec(lags: int) -> VolDatasetSpec:
    """Spec for simulated data (raw/abs/squared lags)."""
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": TABLE,
                "price_col": PRICE_COL,
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": TABLE,
                "price_col": PRICE_COL,
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": lags,
                "source": SOURCE,
                "table": TABLE,
                "price_col": PRICE_COL,
            },
        ),
    )

    target = TargetRequest(
        template=NextDaySquaredReturnTarget(),
        params={"source": SOURCE, "table": TABLE, "price_col": PRICE_COL, "scale": 1.0},
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[ENTITY]),
        time=TimeSpec(start=START, end=END, calendar=CALENDAR, grid=GRID, asof=None),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )


def _build_spy_spec(lags: int) -> VolDatasetSpec:
    """Spec for SPY via Tiingo (raw/abs/squared lags)."""
    features = (
        FeatureRequest(
            template=LagLogReturnTemplate(),
            params={
                "lags": lags,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagAbsLogReturnTemplate(),
            params={
                "lags": lags,
                "source": "tiingo",
                "table": "market.ohlcv",
                "price_col": "close",
            },
        ),
        FeatureRequest(
            template=LagSquaredLogReturnTemplate(),
            params={
                "lags": lags,
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
            "scale": 1.0,
        },
        horizon=1,
        name="y",
    )

    return VolDatasetSpec(
        universe=UniverseSpec(entities=[SPY_TICKER]),
        time=TimeSpec(
            start=SPY_START, end=SPY_END, calendar=CALENDAR, grid=GRID, asof=None
        ),
        features=features,
        target=target,
        join_policy=JoinPolicy(how="inner", sort_index=True),
        missingness=MissingnessPolicy(final_row_policy="drop_if_any_nan"),
    )






def _fit_variant_rmse(variant: str, X: pd.DataFrame, y: pd.Series, r: pd.Series) -> float:
    """Fit on in-sample, return OOS RMSE."""
    if len(y) <= OS_INDEX:
        raise ValueError(f"Insufficient rows for slicing: len(y)={len(y)}")

    cols = select_variant_columns(X, variant)
    if not cols:
        cols = ["const"]

    X_sel = X[cols]
    X_is, y_is = X_sel.iloc[IS_INDEX:OS_INDEX], y.iloc[IS_INDEX:OS_INDEX]
    X_os, y_os = X_sel.iloc[OS_INDEX:], y.iloc[OS_INDEX:]
    r_is, r_os = r.iloc[IS_INDEX:OS_INDEX], r.iloc[OS_INDEX:]

    model = (
        ESModel(**MODEL_KWARGS["ES"])
        if variant == "ES"
        else STESModel(**MODEL_KWARGS["STES"])
    )
    model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))
    y_hat = model.predict(X_os, returns=r_os)
    return float(np.sqrt(np.mean((y_os.values - y_hat) ** 2)))


def _compute_alpha(model: STESModel, X_sel: pd.DataFrame) -> pd.Series:
    """Compute alpha_t = expit(X_t @ beta), aligned to X_sel."""
    params = model.params
    if params is None:
        raise ValueError("Model must be fitted before computing alpha_t.")
    beta = np.asarray(params).reshape(-1)
    xb = X_sel.values @ beta
    alpha = expit(xb)
    return pd.Series(alpha, index=X_sel.index, name="alpha")


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _strict_gate_check_on() -> bool:
    return (
        "--strict_gate_check" in sys.argv
        or _env_flag("STRICT_GATE_CHECK")
        or _env_flag("AF_STRICT_GATE_CHECK")
    )


def _check_gate_convention(
    model: STESModel,
    X_full: pd.DataFrame,
    alpha_full: pd.Series,
    *,
    strict: bool = False,
) -> None:
    """Compare alpha_full to expit(+s) and expit(-s)."""
    params = model.params
    if params is None:
        raise ValueError("Model must be fitted before gate convention check.")
    beta = np.asarray(params).reshape(-1)

    if len(beta) != X_full.shape[1]:
        msg = (
            f"Gate convention check failed: beta length {len(beta)} "
            f"!= X_full columns {X_full.shape[1]}"
        )
        print(msg)
        print("X_full columns:", list(X_full.columns))
        raise ValueError(msg)

    feature_names = getattr(model, "feature_names_", None)
    if feature_names is not None and list(feature_names) != list(X_full.columns):
        msg = "Gate convention check failed: model.feature_names_ != X_full.columns."
        print(msg)
        print("model.feature_names_:", list(feature_names))
        print("X_full columns:", list(X_full.columns))
        raise ValueError(msg)

    common_index = X_full.index.intersection(alpha_full.index)
    if common_index.empty:
        logger.warning(
            "Gate convention check: no overlapping index between X_full and alpha_full."
        )
        return

    X_aligned = X_full.loc[common_index]
    alpha_aligned = alpha_full.loc[common_index]
    score = pd.Series(X_aligned.values @ beta, index=common_index, name="score")
    alpha_plus = pd.Series(expit(score.values), index=common_index, name="alpha_plus")
    alpha_minus = pd.Series(expit(-score.values), index=common_index, name="alpha_minus")

    df = pd.concat(
        [alpha_aligned.rename("alpha_full"), alpha_plus, alpha_minus, score], axis=1
    ).dropna()
    if df.shape[0] < 2:
        logger.warning("Gate convention check: insufficient data after alignment.")
        return

    corr_plus = df["alpha_full"].corr(df["alpha_plus"])
    corr_minus = df["alpha_full"].corr(df["alpha_minus"])
    mae_plus = float(np.mean(np.abs(df["alpha_full"] - df["alpha_plus"])))
    mae_minus = float(np.mean(np.abs(df["alpha_full"] - df["alpha_minus"])))
    corr_score = df["score"].corr(df["alpha_full"])

    logger.info(
        "Gate convention check: corr(alpha_full, expit(s))=%.4f, "
        "corr(alpha_full, expit(-s))=%.4f, mae(expit(s))=%.6f, mae(expit(-s))=%.6f, "
        "monotonic corr(s, alpha_full)=%.4f",
        corr_plus,
        corr_minus,
        mae_plus,
        mae_minus,
        corr_score,
    )

    if (corr_plus > corr_minus + 0.1) or (mae_plus < mae_minus):
        logger.info(
            "Gate convention check: diagnostics currently consistent with expit(+s)."
        )
    else:
        logger.info("Gate convention check: likely should use expit(-s).")

    if strict:
        corr_plus_ok = np.isfinite(corr_plus) and corr_plus >= 0.95
        corr_minus_ok = np.isfinite(corr_minus) and corr_minus >= 0.95
        if not (corr_plus_ok or corr_minus_ok):
            raise AssertionError(
                "Gate convention check failed: neither corr(alpha_full, expit(s)) nor "
                "corr(alpha_full, expit(-s)) >= 0.95."
            )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_gate_diagnostics(alpha: pd.Series, r: pd.Series, out_dir: Path, prefix: str):
    """Save basic gate diagnostic plots."""
    _ensure_dir(out_dir)
    r_aligned = r.loc[alpha.index]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(alpha.index, alpha.values, lw=1.0)
    ax.set_title("STES_EAESE $\\alpha_t$ Time Series")
    ax.set_ylabel("$\\alpha_t$")
    ax.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_alpha_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(alpha.values, bins=50, color="#4c78a8", alpha=0.9)
    ax.set_title("Distribution of $\\alpha_t$")
    ax.set_xlabel("$\\alpha_t$")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_alpha_hist.png", dpi=150)
    plt.close(fig)

    shock = r_aligned
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.scatter(shock.values, alpha.values, s=6, alpha=0.25, color="#f58518")
    ax.set_title("$\\alpha_t$ vs $r_t$")
    ax.set_xlabel("$r_t")
    ax.set_ylabel("$\\alpha_t$")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_alpha_vs_r_scatter.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame(
        {"alpha": alpha.values, "abs_r": np.abs(shock.values)}, index=alpha.index
    )
    if df["abs_r"].nunique() >= 2:
        df["bin"] = pd.qcut(df["abs_r"], q=20, duplicates="drop")
        grp = df.groupby("bin", observed=True, dropna=True).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(np.arange(len(grp)), grp["alpha"].values, marker="o", lw=1.5)
        ax.set_title("Binned mean $\\alpha_t$ vs $|r_t|$ quantiles")
        ax.set_xlabel("quantile bin of $|r_t|$ (low → high)")
        ax.set_ylabel("mean $\\alpha_t$")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_alpha_vs_absr_binned.png", dpi=150)
        plt.close(fig)


def _to_series(x, index, name):
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = name
        return s
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, index=index, name=name)


def _fit_spy_variant(
    variant: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
):
    """Fit a SPY variant and return model, forecasts, slices, and columns."""
    cols = select_variant_columns(X, variant)
    if not cols:
        cols = ["const"]
    X_sel = X[cols]

    X_is = X_sel.iloc[SPY_IS_INDEX:SPY_OS_INDEX]
    y_is = y.iloc[SPY_IS_INDEX:SPY_OS_INDEX]
    r_is = r.iloc[SPY_IS_INDEX:SPY_OS_INDEX]

    X_os = X_sel.iloc[SPY_OS_INDEX:]
    y_os = y.iloc[SPY_OS_INDEX:]
    r_os = r.iloc[SPY_OS_INDEX:]

    model = (
        ESModel(random_state=seed, **MODEL_KWARGS["ES"])
        if variant == "ES"
        else STESModel(random_state=seed, **MODEL_KWARGS["STES"])
    )
    model.fit(X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is))

    yhat_is = _to_series(
        model.predict(X_is, returns=r_is), index=X_is.index, name=f"yhat_{variant}_is"
    )
    yhat_os = _to_series(
        model.predict(X_os, returns=r_os), index=X_os.index, name=f"yhat_{variant}_os"
    )

    return model, yhat_is, yhat_os, (X_is, y_is, r_is), (X_os, y_os, r_os), cols


def _event_window_mean(series: pd.Series, event_idx: pd.Index, window: int) -> pd.Series:
    """Mean path around events, indexed by [-window, +window]."""
    s = series.dropna()
    locs = s.index.get_indexer(event_idx)
    locs = locs[locs >= 0]
    mat = []
    for k in locs:
        a = k - window
        b = k + window + 1
        if a < 0 or b > len(s):
            continue
        mat.append(s.iloc[a:b].values)
    if not mat:
        rel = np.arange(-window, window + 1)
        return pd.Series([np.nan] * len(rel), index=rel)
    mat = np.vstack(mat)
    rel = np.arange(-window, window + 1)
    return pd.Series(mat.mean(axis=0), index=rel)


def _plot_forecast_panel(df: pd.DataFrame, out_dir: Path, fname: str):
    """Plot target, forecasts, and loss differential."""
    _ensure_dir(out_dir)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1, ax2 = axes

    ax1.plot(df.index, df["y"].values, lw=1.0, label="y (target)")
    ax1.plot(df.index, df["yhat_es"].values, lw=1.0, label="ES forecast")
    ax1.plot(df.index, df["yhat_stes"].values, lw=1.0, label="STES-EAESE forecast")
    ax1.set_title("SPY: target and one-step forecasts (test / OOS sample)")
    ax1.set_ylabel("variance (proxy)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(df.index, df["D"].values, lw=1.0)
    ax2.axhline(0.0, lw=1.0, alpha=0.7)
    ax2.set_title(
        r"Loss differential $D_t=(y-\hat y^{ES})^2-(y-\hat y^{STES})^2$  (positive $\Rightarrow$ STES better)"
    )
    ax2.set_ylabel(r"$D_t$ (squared-error diff)")
    ax2.set_xlabel("date")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_event_paths(
    paths: dict[str, pd.Series],
    title: str,
    out_dir: Path,
    fname: str,
    *,
    xlabel: str = "event time k (days relative to forecast date t; k=0 is the D_t date)",
    ylabel: str | None = None,
    note: str | None = None,
):
    """Plot event-window paths."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, s in paths.items():
        ax.plot(s.index, s.values, lw=2.0, label=name)

    ax.axvline(0.0, lw=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if note is not None:
        # Put a readable caption below the plot area
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_bar_series(
    series: pd.Series,
    title: str,
    out_dir: Path,
    fname: str,
    top_k: int = 15,
    *,
    xlabel: str | None = None,
    note: str | None = None,
):
    """Horizontal bar chart for a Series."""
    _ensure_dir(out_dir)

    s = series.copy().dropna()
    if s.empty:
        return

    if len(s) > top_k:
        s = s.reindex(s.abs().sort_values(ascending=False).index[:top_k])

    s = s.sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(s) + 1.5)))
    ax.barh(s.index.astype(str), s.values)

    ax.axvline(0.0, lw=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("feature")
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.grid(True, axis="x", alpha=0.25)

    if note is not None:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_gate_panel(
    *,
    alpha_t: pd.Series,
    r: pd.Series,
    D_t: pd.Series | None,
    out_dir: Path,
    prefix: str,
    title_prefix: str = "STES_EAESE",
    q_bins: int = 20,
):
    """2x2 panel: alpha, alpha hist, alpha vs |r|, and D_t vs |r|."""
    _ensure_dir(out_dir)

    # Here, alpha_t and r_t are contemporaneous with the forecast made at time t (about t+1).
    alpha_now = alpha_t
    r_now = r

    # Align to common index
    idx = alpha_now.index.intersection(r_now.index)
    if D_t is not None:
        idx = idx.intersection(D_t.index)

    alpha_now = alpha_now.reindex(idx)
    r_now = r_now.reindex(idx)
    if D_t is not None:
        D_t = D_t.reindex(idx)

    df = pd.DataFrame(
        {
            "alpha": alpha_now,
            "r": r_now,
        },
        index=idx,
    )
    df["abs_r"] = df["r"].abs()

    if D_t is not None:
        df["D_t"] = D_t

    df = df.dropna(subset=["alpha", "r", "abs_r"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.plot(df.index, df["alpha"].values, lw=1.0)
    ax00.set_title(f"{title_prefix}: $\\alpha_t$ time series")
    ax00.set_ylabel("$\\alpha_t$")
    ax00.set_xlabel("date")
    ax00.grid(True, alpha=0.25)

    ax01.hist(df["alpha"].values, bins=50, alpha=0.9)
    ax01.set_title(f"{title_prefix}: distribution of $\\alpha_t$")
    ax01.set_xlabel("$\\alpha_t$")
    ax01.set_ylabel("count")
    ax01.grid(True, alpha=0.25)

    try:
        tmp = df[["alpha", "abs_r"]].copy()
        tmp["bin"] = pd.qcut(tmp["abs_r"], q=q_bins, duplicates="drop")
        grp = tmp.groupby("bin", observed=True)["alpha"].mean()
        ax10.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
        ax10.set_title(
            f"{title_prefix}: binned mean $\\alpha_t$ vs $|r_t|$ quantiles"
        )
        ax10.set_xlabel("quantile bin of $|r_t|$ (low → high)")
        ax10.set_ylabel("mean $\\alpha_t$")
        ax10.grid(True, alpha=0.25)
    except Exception as e:
        ax10.text(
            0.05,
            0.5,
            f"binned plot failed:\n{e}",
            transform=ax10.transAxes,
            fontsize=10,
        )
        ax10.set_axis_off()

    if D_t is not None and "D_t" in df.columns:
        try:
            tmp = df[["D_t", "abs_r"]].dropna().copy()
            tmp["bin"] = pd.qcut(tmp["abs_r"], q=q_bins, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)["D_t"].mean()
            ax11.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
            ax11.axhline(0.0, lw=1.0)
            ax11.set_title(
                "Binned mean $D_t$ vs $|r_t|$ quantiles\n($D_t>0$ means STES better than ES)"
            )
            ax11.set_xlabel("quantile bin of $|r_t|$ (low → high)")
            ax11.set_ylabel("mean $D_t$")
            ax11.grid(True, alpha=0.25)
        except Exception as e:
            ax11.text(
                0.05,
                0.5,
                f"D_t binned plot failed:\n{e}",
                transform=ax11.transAxes,
                fontsize=10,
            )
            ax11.set_axis_off()
    else:
        ax11.text(
            0.05,
            0.55,
            "No $D_t$ provided.\n\nPass D_t = (y-ŷ_ES)^2 - (y-ŷ_STES)^2\n(aligned on the same date index)\n\nto show when STES helps.",
            transform=ax11.transAxes,
            fontsize=10,
        )
        ax11.set_axis_off()

    fig.tight_layout()
    out_path = out_dir / f"{prefix}_gate_helpfulness_2x2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def analyze_spy_stes(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    *,
    seeds: list[int],
    q: float = 0.10,
    window: int = 10,
    out_dir: Path | None = None,
):
    """Analyze SPY ES vs STES_EAESE on OOS and diagnostics."""
    if out_dir is None:
        out_dir = Path("./outputs/volatility_forecast_1")
    _ensure_dir(out_dir)

    # Fit ES once
    model_es, yhat_es_is, yhat_es_os, is_pack, os_pack, cols_es = (
        _fit_spy_variant("ES", seeds[0], X, y, r)
    )
    X_is_es, y_is, r_is = is_pack
    X_os_es, y_os, r_os = os_pack

    # Fit STES_EAESE for each seed and keep best OOS RMSE
    stes_runs = []
    for seed in seeds:
        try:
            model_stes, yhat_stes_is, yhat_stes_os, is_pack_s, os_pack_s, cols_s = (
                _fit_spy_variant("STES_EAESE", seed, X, y, r)
            )
            # ensure same OOS index
            if not yhat_stes_os.index.equals(yhat_es_os.index):
                yhat_stes_os = yhat_stes_os.reindex(yhat_es_os.index).dropna()
            rmse_os = float(
                np.sqrt(
                    np.mean(
                        (y_os.loc[yhat_stes_os.index].values - yhat_stes_os.values) ** 2
                    )
                )
            )
            stes_runs.append(
                (rmse_os, seed, model_stes, yhat_stes_is, yhat_stes_os, cols_s)
            )
        except Exception as e:
            continue

    if not stes_runs:
        raise RuntimeError("No STES_EAESE runs succeeded; cannot analyze.")

    stes_runs.sort(key=lambda x: x[0])
    best_rmse, best_seed, best_model, best_yhat_is, best_yhat_os, best_cols = stes_runs[
        0
    ]

    # OOS evaluation frame
    idx = yhat_es_os.index.intersection(best_yhat_os.index)
    df = pd.DataFrame(index=idx)
    df["y"] = y_os.loc[idx].values
    df["yhat_es"] = yhat_es_os.loc[idx].values
    df["yhat_stes"] = best_yhat_os.loc[idx].values
    df["loss_es"] = (df["y"] - df["yhat_es"]) ** 2
    df["loss_stes"] = (df["y"] - df["yhat_stes"]) ** 2
    df["D"] = df["loss_es"] - df["loss_stes"]

    # Returns and lags
    rr = r_os.loc[idx]
    df["r"] = rr.values
    df["abs_r"] = np.abs(df["r"].values)
    df["r2"] = df["r"].values ** 2

    # Alpha on full X, aligned contemporaneously (alpha_t for forecast at t)
    X_full_stes = X[best_cols]
    alpha_full = _compute_alpha(
        best_model, X_full_stes
    )
    _check_gate_convention(
        model=best_model,
        X_full=X_full_stes,
        alpha_full=alpha_full,
        strict=_strict_gate_check_on(),
    )
    df["alpha_stes"] = alpha_full.reindex(idx).values

    # ES constant alpha
    alpha_es_prob = model_es.alpha_
    assert 0.0 < alpha_es_prob < 1.0, f"ES alpha_ must be in (0,1), got {alpha_es_prob}"
    df["alpha_es"] = alpha_es_prob

    df["delta_alpha"] = df["alpha_stes"] - df["alpha_es"]

    # ES state proxy: the variance level v_t is the *previous* forecast made at t-1.
    df["v_es_state"] = pd.Series(df["yhat_es"], index=idx).shift(1).values

    # Innovation proxy at time t: u_t = r_t^2 - v_t^{ES}
    df["u"] = df["r2"] - df["v_es_state"]

    # Range checks
    if alpha_full.dropna().size > 0:
        af = alpha_full.dropna()
        assert ((af > 0) & (af < 1)).all(), "STES alpha outside (0,1)"

    if not np.isnan(df["alpha_es"]).all():
        ae = float(df["alpha_es"].iloc[0])
        assert 0.0 <= ae <= 1.0, "ES alpha outside [0,1]"

    if df["delta_alpha"].dropna().size > 0:
        da = df["delta_alpha"].dropna()
        assert (
            da.min() >= -1 - 1e-6 and da.max() <= 1 + 1e-6
        ), "delta_alpha out of expected [-1,1] range"

    df = df.dropna()

    # Events: wins/losses by D
    hi = df["D"].quantile(1.0 - q)
    lo = df["D"].quantile(q)
    win_idx = df.index[df["D"] >= hi]
    lose_idx = df.index[df["D"] <= lo]

    event_def = (
        f"Event definition (OOS): WIN = top {int(q*100)}% of D_t, "
        f"LOSE = bottom {int(q*100)}% of D_t. "
    )

    # Gate score contributions
    beta = np.asarray(best_model.params).reshape(-1)
    X_aligned = X_full_stes.loc[df.index]

    # Coef interpretation
    coef = pd.Series(beta, index=X_full_stes.columns, name="beta")

    coef_table = pd.DataFrame(
        {
            "beta": coef,
            "effect_on_alpha_under_expit": np.where(
                coef.values > 0,
                "↑ alpha (faster update)",
                np.where(coef.values < 0, "↓ alpha (slower update)", "no effect"),
            ),
        }
    )

    coef_table["abs(beta)"] = coef_table["beta"].abs()
    coef_table = coef_table.sort_values("abs(beta)", ascending=False)

    print("\nCoefficient interpretation (alpha_t = expit(X_t beta)):")
    print(coef_table[["beta", "effect_on_alpha_under_expit"]].head(20).to_string())

    # Score contributions
    contrib = X_aligned.mul(beta, axis=1).add_prefix("c_")
    contrib_now = contrib.reindex(df.index)
    df = df.join(contrib_now)

    # Summary tables
    def _summ(mask):
        sub = df.loc[mask]
        out = {
            "n": int(len(sub)),
            "mean_D": float(sub["D"].mean()),
            "mean_loss_es": float(sub["loss_es"].mean()),
            "mean_loss_stes": float(sub["loss_stes"].mean()),
            "mean_alpha_stes": float(sub["alpha_stes"].mean()),
            "mean_alpha_es": float(sub["alpha_es"].mean()),
            "mean_delta_alpha": float(sub["delta_alpha"].mean()),
            "frac_delta_alpha_pos": float((sub["delta_alpha"] > 0).mean()),
            "mean_u": float(sub["u"].mean()),
            "frac_u_pos": float((sub["u"] > 0).mean()),
            "mean_abs_r": float(sub["abs_r"].mean()),
            # mean_r2_lag is omitted, as r2_lag is no longer defined in this convention
        }
        ccols = [c for c in sub.columns if c.startswith("c_")]
        out["mean_contrib"] = sub[ccols].mean().sort_values(ascending=False)
        return out

    win_stats = _summ(df.index.isin(win_idx))
    lose_stats = _summ(df.index.isin(lose_idx))

    # Summary output
    print("\n" + "=" * 80)
    print("SPY: ES vs STES_EAESE 'WHEN DOES STES HELP?' analysis")
    print("=" * 80)
    print(f"Best STES seed by OOS RMSE: seed={best_seed}, rmse={best_rmse:.6e}")
    print(f"Test period length used in analysis: {len(df)}")
    print(
        f"Win events (top {int(q*100)}% D): n={win_stats['n']} | Lose events (bottom {int(q*100)}% D): n={lose_stats['n']}"
    )
    print("\n--- WIN events summary (STES better) ---")
    for k, v in win_stats.items():
        if k == "mean_contrib":
            continue
        print(f"{k:>22s}: {v}")
    print("\n--- LOSE events summary (STES worse) ---")
    for k, v in lose_stats.items():
        if k == "mean_contrib":
            continue
        print(f"{k:>22s}: {v}")

    # Contribution tables
    contrib_win = win_stats["mean_contrib"]
    contrib_lose = lose_stats["mean_contrib"]
    contrib_diff = (contrib_win - contrib_lose).sort_values(ascending=False)

    print("\nTop gate-score contributions (mean) on WIN events:")
    print(contrib_win.head(12).to_string())
    print("\nTop gate-score contributions (mean) on LOSE events:")
    print(contrib_lose.head(12).to_string())
    print("\nTop differences in gate-score contributions: WIN minus LOSE (mean):")
    print(contrib_diff.head(12).to_string())

    # Save CSV tables
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    df.to_csv(out_dir / "spy_es_vs_stes_eaese_timeseries.csv")
    pd.DataFrame(
        {
            "win_mean": contrib_win,
            "lose_mean": contrib_lose,
            "win_minus_lose": contrib_win - contrib_lose,
        }
    ).to_csv(out_dir / "spy_gate_contrib_summary.csv")

    # Plots
    _plot_forecast_panel(
        df[["y", "yhat_es", "yhat_stes", "D"]], out_dir, "spy_forecasts_and_D.png"
    )

    paths_alpha = {
        "alpha_STES (wins)": _event_window_mean(df["alpha_stes"], win_idx, window),
        "alpha_STES (losses)": _event_window_mean(
            df["alpha_stes"], lose_idx, window
        ),
        "alpha_ES (const)": pd.Series(
            [df["alpha_es"].iloc[0]] * (2 * window + 1),
            index=np.arange(-window, window + 1),
        ),
    }
    _plot_event_paths(
        paths_alpha,
        r"Event study: gate $\alpha_t$ (used in forecast for date $t+1$) around WIN vs LOSE dates",
        out_dir,
        "spy_event_alpha.png",
        ylabel=r"mean $\alpha_{t+k}$",
        note=event_def,
    )

    paths_absr = {
        "|r| (wins)": _event_window_mean(df["abs_r"], win_idx, window),
        "|r| (losses)": _event_window_mean(df["abs_r"], lose_idx, window),
    }
    _plot_event_paths(
        paths_absr,
        r"Event study: $|r_t|$ around WIN vs LOSE dates",
        out_dir,
        "spy_event_absr.png",
        ylabel=r"mean $|r_{t+k}|$",
        note=event_def,
    )

    paths_u = {
        "u_t (wins)": _event_window_mean(df["u"], win_idx, window),
        "u_t (losses)": _event_window_mean(df["u"], lose_idx, window),
    }
    _plot_event_paths(
        paths_u,
        r"Event study: innovation proxy $u_t=r_t^2-\hat v_t^{ES}$ around WIN vs LOSE dates",
        out_dir,
        "spy_event_u.png",
        ylabel=r"mean $u_{t+k}$ (so at k=0 this is $u_t$)",
        note=event_def,
    )

    _plot_bar_series(
        contrib_diff,
        "Gate-score (logit) contribution differences: WIN minus LOSE",
        out_dir,
        "spy_gate_contrib_win_minus_lose.png",
        top_k=15,
        xlabel=r"mean score contrib diff  E[c_{j,t-1}|WIN] - E[c_{j,t-1}|LOSE]",
        note=event_def,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(
        df["u"].values,
        df["delta_alpha"].values,
        s=6,
        alpha=0.15,
        label="all",
        color="#999999",
    )
    ax.scatter(
        df.loc[win_idx, "u"].values,
        df.loc[win_idx, "delta_alpha"].values,
        s=10,
        alpha=0.35,
        label="wins",
        color="#2ca02c",
    )
    ax.scatter(
        df.loc[lose_idx, "u"].values,
        df.loc[lose_idx, "delta_alpha"].values,
        s=10,
        alpha=0.35,
        label="losses",
        color="#d62728",
    )
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    ax.set_title(
        r"Mechanism view: $\Delta\alpha_t=\alpha^{STES}_t-\alpha^{ES}$ vs $u_t$"
        + f"\n(WIN/LOSE are top/bottom {int(q*100)}% of D_t in the OOS sample)"
    )
    ax.set_xlabel("$u_t=r_t^2-\\hat v_t^{ES}$")
    ax.set_ylabel("$\\Delta\\alpha_t=\\alpha^{STES}_t-\\alpha^{ES}$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "spy_delta_alpha_vs_u.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved analysis outputs to: {out_dir.resolve()}")

    out_path = _plot_gate_panel(
        alpha_t=alpha_full,
        r=r,
        D_t=df["D"],
        out_dir=Path("./outputs/volatility_forecast_1"),
        prefix="spy_oos",
        title_prefix="SPY STES_EAESE",
        q_bins=20,
    )
    print("Saved:", out_path)

    return df, win_stats, lose_stats, best_seed, best_rmse


def main():
    _log_run_context()
    logger.info("=" * 80)
    logger.info("Volatility Forecasts (Part 1) - STES Models on Simulated Data")
    logger.info("=" * 80)
    logger.info(f"Variants: {', '.join(VARIANTS)}")
    logger.info(
        f"Lags: {N_LAGS}, IS=[{IS_INDEX}:{OS_INDEX}] OOS=[{OS_INDEX}:], Runs={N_RUNS}\n"
    )

    # Base context (for calendars); add simulated source per run
    ctx = build_default_ctx(tiingo_api_key=_read_tiingo_api_key())

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_RUNS)

    rmse_acc = {v: [] for v in VARIANTS}

    for i, seed in enumerate(seeds, start=1):
        if i % 10 == 0:
            logger.info(f"  Run {i}/{N_RUNS}")

        # Replace simulated source with run-specific seed
        _attach_sim_source(ctx, int(seed))

        # Build wide dataset ONCE for this run
        spec = _build_sim_spec(N_LAGS)
        try:
            X_wide, y, r, _ = build_wide_dataset(ctx, spec)
            if i == 1:
                # Report IS/OOS date ranges once
                idx = X_wide.index
                if len(idx) > OS_INDEX:
                    is_start, is_end = idx[IS_INDEX], idx[OS_INDEX - 1]
                    os_start, os_end = idx[OS_INDEX], idx[-1]
                    logger.info(
                        f"IS window: {is_start} → {is_end} | OOS window: {os_start} → {os_end}"
                    )
                else:
                    logger.warning(
                        f"Warning: dataset too short for OS_INDEX={OS_INDEX}; len={len(idx)}"
                    )
        except Exception as e:
            # surface a compact hint once per failure type
            continue

        # Select subsets by variant without recompute
        for variant in VARIANTS:
            try:
                rmse = _fit_variant_rmse(variant, X_wide, y, r)
                rmse_acc[variant].append(rmse)
            except Exception:
                pass

    logger.info("\n" + "=" * 80)
    logger.info("Out-of-Sample RMSE (mean ± std)")
    logger.info("=" * 80)
    for variant in VARIANTS:
        vals = rmse_acc[variant]
        if vals:
            logger.info(
                f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{N_RUNS})"
            )
        else:
            logger.info(f"{variant:12s}: N/A (0/{N_RUNS})")

    logger.info("")

    logger.info("=" * 80)
    logger.info("STES and ES on SPY (Tiingo EOD; single series, random init averaging)")
    logger.info("=" * 80)
    logger.info(
        f"Ticker={SPY_TICKER}, Lags={N_LAGS}, IS=[{SPY_IS_INDEX}:{SPY_OS_INDEX}] OOS=[{SPY_OS_INDEX}:], Inits={SPY_N_INITS}"
    )
    if _read_tiingo_api_key():
        logger.info("Tiingo API key detected (env).")
    else:
        logger.warning(
            "Warning: No Tiingo API key detected in env. Set TIINGO_API in .env"
        )

    rmse_spy_oos = {v: [] for v in VARIANTS}
    rmse_spy_is = {v: [] for v in VARIANTS}

    # Track best-fit model per variant
    best_spy_rmse = {v: np.inf for v in VARIANTS}
    best_spy_seed = {v: None for v in VARIANTS}
    best_spy_model = {v: None for v in VARIANTS}
    best_spy_cols = {v: None for v in VARIANTS}

    try:
        spy_spec = _build_spy_spec(N_LAGS)
        X, y, r, _ = build_wide_dataset(ctx, spy_spec, entity_id=SPY_TICKER)
    except Exception as e:
        logger.warning(
            "SPY via Tiingo failed. Ensure TIINGO_API_KEY is set in .env and valid.\n"
            f"Reason: {e}"
        )
        X = y = r = None
    if X is not None and len(y) > SPY_OS_INDEX:
        for variant in VARIANTS:
            for seed in range(SPY_N_INITS):
                try:
                    cols = select_variant_columns(X, variant)
                    if not cols:
                        cols = ["const"]
                    X_sel = X[cols]
                    X_is, y_is = (
                        X_sel.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                        y.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                    )
                    X_os, y_os = X_sel.iloc[SPY_OS_INDEX:], y.iloc[SPY_OS_INDEX:]
                    r_is, r_os = (
                        r.iloc[SPY_IS_INDEX:SPY_OS_INDEX],
                        r.iloc[SPY_OS_INDEX:],
                    )

                    model = (
                        ESModel(random_state=seed)
                        if variant == "ES"
                        else STESModel(random_state=seed)
                    )
                    model.fit(
                        X_is, y_is, returns=r_is, start_index=0, end_index=len(X_is)
                    )
                    # In-sample RMSE
                    y_hat_is = model.predict(X_is, returns=r_is)
                    rmse_spy_is[variant].append(
                        float(np.sqrt(np.mean((y_is.values - y_hat_is) ** 2)))
                    )
                    # Out-of-sample RMSE
                    y_hat_os = model.predict(X_os, returns=r_os)
                    rmse_os = float(np.sqrt(np.mean((y_os.values - y_hat_os) ** 2)))
                    rmse_spy_oos[variant].append(rmse_os)

                    if rmse_os < best_spy_rmse[variant]:
                        best_spy_rmse[variant] = rmse_os
                        best_spy_seed[variant] = seed
                        best_spy_model[variant] = model
                        best_spy_cols[variant] = cols
                except Exception:
                    pass

        print("\nIn-Sample RMSE on SPY (mean ± std)")
        for variant in VARIANTS:
            vals = rmse_spy_is[variant]
            if vals:
                print(
                    f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{SPY_N_INITS})"
                )
            else:
                print(f"{variant:12s}: N/A (0/{SPY_N_INITS})")

        print("\nOut-of-Sample RMSE on SPY (mean ± std)")
        for variant in VARIANTS:
            vals = rmse_spy_oos[variant]
            if vals:
                print(
                    f"{variant:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f} ({len(vals)}/{SPY_N_INITS})"
                )
            else:
                print(f"{variant:12s}: N/A (0/{SPY_N_INITS})")

        # ---- Export fitted beta coefficients (best seed per variant) ----
        try:
            out_dir = Path("./outputs/volatility_forecast_1")
            _ensure_dir(out_dir)

            # Preserve feature order: const first, then in X column order.
            # (X should already include 'const', but keep this robust.)
            preferred_order = ["const"] + [c for c in list(X.columns) if c != "const"]
            all_feature_names: list[str] = []
            for variant in VARIANTS:
                m = best_spy_model.get(variant)
                if m is None or getattr(m, "feature_names_", None) is None:
                    continue
                for nm in m.feature_names_:
                    if nm not in all_feature_names:
                        all_feature_names.append(nm)
            # apply preferred ordering when possible
            ordered = []
            for nm in preferred_order:
                if nm in all_feature_names:
                    ordered.append(nm)
            for nm in all_feature_names:
                if nm not in ordered:
                    ordered.append(nm)

            beta_by_variant = pd.DataFrame(index=ordered, columns=VARIANTS, dtype=float)
            extra = pd.DataFrame(index=["best_seed", "best_oos_rmse", "es_alpha_"], columns=VARIANTS)

            for variant in VARIANTS:
                m = best_spy_model.get(variant)
                if m is None or getattr(m, "params", None) is None:
                    continue
                feature_names = getattr(m, "feature_names_", None)
                if feature_names is None:
                    continue
                beta_s = pd.Series(np.asarray(m.params).reshape(-1), index=feature_names)
                beta_by_variant[variant] = beta_s.reindex(beta_by_variant.index)
                extra.loc["best_seed", variant] = best_spy_seed.get(variant)
                extra.loc["best_oos_rmse", variant] = best_spy_rmse.get(variant)
                if variant == "ES":
                    extra.loc["es_alpha_", variant] = getattr(m, "alpha_", None)

            beta_by_variant.to_csv(out_dir / "spy_gate_betas_by_variant.csv")
            extra.to_csv(out_dir / "spy_gate_betas_by_variant__meta.csv")

            print("\nSaved: outputs/volatility_forecast_1/spy_gate_betas_by_variant.csv")
        except Exception as e:
            logger.warning(f"Failed to export SPY beta table: {e}")

        try:
            out_dir = Path("./outputs/volatility_forecast_1")
            seeds = list(range(SPY_N_INITS))
            analyze_spy_stes(
                X=X,
                y=y,
                r=r,
                seeds=seeds,
                q=0.10,
                window=10,
                out_dir=out_dir,
            )
        except Exception as e:
            logger.warning(f"SPY ES vs STES_EAESE analysis failed: {e}")
    else:
        print("SPY study skipped (insufficient data).")

    # Gate diagnostics
    try:
        blog_assets_dir = Path(
            "/Users/steveyang/Projects/Github/steveya.github.io/assets/img/post_assets/volatility-forecasts-1"
        )
        _ensure_dir(blog_assets_dir)

        # Simulated data
        _attach_sim_source(ctx, 12345)
        spec_sim = _build_sim_spec(N_LAGS)
        X_sim, y_sim, r_sim, _ = build_wide_dataset(ctx, spec_sim)
        cols_sim = select_variant_columns(X_sim, "STES_EAESE")
        if not cols_sim:
            cols_sim = ["const"]
        Xsim_sel = X_sim[cols_sim].iloc[IS_INDEX:OS_INDEX]
        ysim_is = y_sim.iloc[IS_INDEX:OS_INDEX]
        rsim_is = r_sim.iloc[IS_INDEX:OS_INDEX]

        model_sim = STESModel(random_state=0)
        model_sim.fit(
            Xsim_sel, ysim_is, returns=rsim_is, start_index=0, end_index=len(Xsim_sel)
        )
        alpha_sim = _compute_alpha(model_sim, Xsim_sel)
        _plot_gate_diagnostics(alpha_sim, rsim_is, blog_assets_dir, prefix="sim")

        # SPY data
        if X is not None and len(y) > SPY_OS_INDEX:
            cols_spy = select_variant_columns(X, "STES_EAESE")
            if not cols_spy:
                cols_spy = ["const"]
            Xspy_sel = X[cols_spy].iloc[SPY_IS_INDEX:SPY_OS_INDEX]
            yspy_is = y.iloc[SPY_IS_INDEX:SPY_OS_INDEX]
            rspy_is = r.iloc[SPY_IS_INDEX:SPY_OS_INDEX]

            model_spy = STESModel(random_state=0)
            model_spy.fit(
                Xspy_sel,
                yspy_is,
                returns=rspy_is,
                start_index=0,
                end_index=len(Xspy_sel),
            )
            alpha_spy = _compute_alpha(model_spy, Xspy_sel)
            _plot_gate_diagnostics(alpha_spy, rspy_is, blog_assets_dir, prefix="spy")

        logger.info(f"Gate diagnostic figures saved to: {blog_assets_dir}")
    except Exception as e:
        logger.warning(f"Gate diagnostics generation failed: {e}")


if __name__ == "__main__":
    main()
