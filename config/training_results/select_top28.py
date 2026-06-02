#!/usr/bin/env python3
"""
Select top 28 trains using Scheme A:
  1. Hard filter: exclude LES < 0 OR episodes < 1000
  2. From remaining 29, drop the lowest by composite = LES + RSA + LRS
"""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\soqqle\ml-agents\config\training_results")
df = pd.read_csv(BASE / "v3_metrics_wide_table.csv")

print("=" * 90)
print("Phase 2: Select 28 trains (Scheme A)")
print("=" * 90)

# Step 1: Hard filter
hard_excluded = df[(df["les"] < 0) | (df["episodes"] < 1000)].copy()
remaining = df[(df["les"] >= 0) & (df["episodes"] >= 1000)].copy()

print(f"\n[Hard filter] Excluded {len(hard_excluded)} trains (LES<0 or episodes<1000):")
for _, row in hard_excluded.iterrows():
    print(f"  - {row['run']}/{row['train_id']}: LES={row['les']:.4f}, episodes={int(row['episodes'])}")

print(f"\n[Hard filter] Remaining: {len(remaining)} trains")

# Step 2: Composite score for ranking
remaining["composite"] = remaining["les"] + remaining["rsa"] + remaining["lrs"]
remaining = remaining.sort_values("composite", ascending=False).reset_index(drop=True)

print("\n[Ranking] Remaining 29 trains sorted by composite (LES+RSA+LRS):")
print(remaining[["run", "train_id", "les", "rsa", "lrs", "composite"]].to_string(index=False))

# Drop the lowest 1 to get exactly 28
dropped = remaining.iloc[-1:].copy()
selected = remaining.iloc[:-1].copy()

print(f"\n[Final cut] Dropped lowest composite:")
for _, row in dropped.iterrows():
    print(f"  - {row['run']}/{row['train_id']}: composite={row['composite']:.4f}")

print(f"\n[Result] Selected {len(selected)} trains for CFA:")
print(selected[["run", "train_id", "les", "cas_ratio", "lrs", "rsa", "rgec", "rtge", "episodes"]].to_string(index=False))

# Save selected and excluded lists
selected_path = BASE / "v3_metrics_selected_28.csv"
selected[["run", "train_id", "les", "cas_ratio", "lrs", "rsa", "rgec", "rtge", "episodes"]].to_csv(selected_path, index=False)
print(f"\n[OK] Selected 28 saved to: {selected_path}")

excluded_path = BASE / "v3_metrics_excluded_9.csv"
excluded = pd.concat([hard_excluded, dropped], ignore_index=True)
excluded[["run", "train_id", "les", "cas_ratio", "lrs", "rsa", "rgec", "rtge", "episodes"]].to_csv(excluded_path, index=False)
print(f"[OK] Excluded 9 saved to: {excluded_path}")
