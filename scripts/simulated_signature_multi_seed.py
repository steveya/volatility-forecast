import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

from volatility_forecast.pipeline import build_default_ctx, build_vol_dataset

SCRIPT_PATH = Path("examples/signature_in_volatility_forecast.py")
SPEC = importlib.util.spec_from_file_location("sig_script", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

N_RUNS = 20
AUGMENTATION = "all"


def main() -> None:
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_RUNS)
    rows = []

    for seed in seeds:
        ctx = build_default_ctx()
        MOD.add_simulated_source(ctx, run_seed=int(seed))
        spec_ds = MOD.build_simulated_spec_with_signatures(
            n_lags=MOD.SIM_N_LAGS,
            sig_lags=MOD.SIG_LAGS,
            sig_level=MOD.SIG_LEVEL,
            augmentation=AUGMENTATION,
        )
        X, y, returns, _ = build_vol_dataset(ctx, spec_ds, persist=False)
        X = X.xs(MOD.SIMULATED_ENTITY, level="entity_id").sort_index()
        y = y.xs(MOD.SIMULATED_ENTITY, level="entity_id").sort_index()
        returns = returns.xs(MOD.SIMULATED_ENTITY, level="entity_id").sort_index()

        if "const" not in X.columns:
            X["const"] = 1.0

        results = MOD.evaluate_models_with_signatures(
            X,
            y,
            returns,
            MOD.SIMULATED_IS_INDEX,
            MOD.SIMULATED_OS_INDEX,
            seed=int(seed),
            include_baseline=True,
        )

        for model, metrics in results.items():
            rows.append({"seed": int(seed), "model": model, **metrics})

    out = pd.DataFrame(rows)
    summary = out.groupby("model").agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        MedAE_mean=("MedAE", "mean"),
        MedAE_std=("MedAE", "std"),
        n=("RMSE", "count"),
    )

    out_dir = Path("outputs/signature_volatility")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "simulated_signature_multi_seed_summary.csv")
    out.to_csv(out_dir / "simulated_signature_multi_seed_raw.csv", index=False)

    print(summary.sort_values("RMSE_mean"))


if __name__ == "__main__":
    main()
