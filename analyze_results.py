import pandas as pd
import numpy as np

df = pd.read_csv("outputs/signature_test/augmentation_comparison.csv")

print("=" * 80)
print("SIGNATURE FEATURES ANALYSIS - AUGMENTATION COMPARISON")
print("=" * 80)
print()

# 1. Feature counts
print("1. FEATURE COUNTS BY AUGMENTATION:")
print("-" * 40)
fc = df.groupby("augmentation")[["n_features", "n_sig_features"]].first()
print(fc)
print()

# 2. Best by metric
print("2. BEST PERFORMANCE BY METRIC:")
print("-" * 40)
for m in ["RMSE", "MAE", "MedAE"]:
    b = df.loc[df[m].idxmin()]
    print(f'{m:8s}: {b["model"]:12s} with {b["augmentation"]:10s} = {b[m]:.6f}')
print()

# 3. Model averages
print("3. AVERAGE PERFORMANCE BY MODEL:")
print("-" * 40)
print(df.groupby("model")[["RMSE", "MAE", "MedAE"]].mean())
print()

# 4. Augmentation averages
print("4. AVERAGE PERFORMANCE BY AUGMENTATION:")
print("-" * 40)
print(df.groupby("augmentation")[["RMSE", "MAE", "MedAE"]].mean())
print()

# 5. XGBSTES details
print("5. XGBSTES PERFORMANCE BY AUGMENTATION:")
print("-" * 40)
xgb = df[df["model"] == "XGBSTES_SIG"].sort_values("RMSE")
print(xgb[["augmentation", "n_sig_features", "RMSE", "MAE", "MedAE"]])
print()

# 6. Improvements
print("6. RMSE IMPROVEMENT OVER BASELINE (ES with none):")
print("-" * 40)
baseline = df[(df["model"] == "ES") & (df["augmentation"] == "none")]["RMSE"].values[0]
for _, row in xgb.iterrows():
    imp = (baseline - row["RMSE"]) / baseline * 100
    print(f"{row['augmentation']:10s}: {imp:+6.2f}%")
