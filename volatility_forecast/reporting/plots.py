"Plotting utilities for volatility forecast analysis."

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_gate_diagnostics(alpha: pd.Series, r: pd.Series, out_dir: Path, prefix: str):
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
        {"alpha": alpha.values, "abs_r": np.abs(shock.values)},
        index=alpha.index
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


def plot_forecast_panel(df: pd.DataFrame, out_dir: Path, fname: str):
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
        r"Loss differential $D_t=(y-\\hat y^{ES})^2-(y-\\hat y^{STES})^2$  (positive $\\Rightarrow$ STES better)"
    )
    ax2.set_ylabel(r"$D_t$ (squared-error diff)")
    ax2.set_xlabel("date")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_event_paths(
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


def plot_bar_series(
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


def plot_gate_panel(
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

    # Align to t-1 inputs
    alpha_lag = alpha_t.shift(1)
    r_lag = r.shift(1)

    # Align to common index
    idx = alpha_lag.index.intersection(r_lag.index)
    if D_t is not None:
        idx = idx.intersection(D_t.index)

    alpha_lag = alpha_lag.reindex(idx)
    r_lag = r_lag.reindex(idx)
    if D_t is not None:
        D_t = D_t.reindex(idx)

    df = pd.DataFrame(
        {
            "alpha_lag": alpha_lag,
            "r_lag": r_lag,
        },
        index=idx,
    )
    df["abs_r_lag"] = df["r_lag"].abs()

    if D_t is not None:
        df["D_t"] = D_t

    df = df.dropna(subset=["alpha_lag", "r_lag", "abs_r_lag"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.plot(df.index, df["alpha_lag"].values, lw=1.0)
    ax00.set_title(f"{title_prefix}: $\\alpha_{{t-1}}$ time series")
    ax00.set_ylabel("$\\alpha_{t-1}$")
    ax00.set_xlabel("date")
    ax00.grid(True, alpha=0.25)

    ax01.hist(df["alpha_lag"].values, bins=50, alpha=0.9)
    ax01.set_title(f"{title_prefix}: distribution of $\\alpha_{{t-1}}$")
    ax01.set_xlabel("$\\alpha_{t-1}$")
    ax01.set_ylabel("count")
    ax01.grid(True, alpha=0.25)

    try:
        tmp = df[[ "alpha_lag", "abs_r_lag"]].copy()
        tmp["bin"] = pd.qcut(tmp["abs_r_lag"], q=q_bins, duplicates="drop")
        grp = tmp.groupby("bin", observed=True)[ "alpha_lag"].mean()
        ax10.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
        ax10.set_title(
            f"{title_prefix}: binned mean $\\alpha_{{t-1}}$ vs $|r_{{t-1}}|$ quantiles"
        )
        ax10.set_xlabel("quantile bin of $|r_{t-1}|$ (low → high)")
        ax10.set_ylabel("mean $\\alpha_{t-1}$")
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
            tmp = df[[ "D_t", "abs_r_lag"]].dropna().copy()
            tmp["bin"] = pd.qcut(tmp["abs_r_lag"], q=q_bins, duplicates="drop")
            grp = tmp.groupby("bin", observed=True)[ "D_t"].mean()
            ax11.plot(np.arange(len(grp)), grp.values, marker="o", lw=1.5)
            ax11.axhline(0.0, lw=1.0)
            ax11.set_title(
                "Binned mean $D_t$ vs $|r_{t-1}|$ quantiles\n($D_t>0$ means STES better than ES)"
            )
            ax11.set_xlabel("quantile bin of $|r_{t-1}|$ (low → high)")
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
