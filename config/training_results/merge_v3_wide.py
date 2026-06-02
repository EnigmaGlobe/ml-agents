#!/usr/bin/env python3
"""
Merge all v3 metrics CSVs into a wide table and show summary for selection.
"""
import csv
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\soqqle\ml-agents\config\training_results")

records = []
for run_dir in sorted(BASE.glob("run_*")):
    if not run_dir.is_dir():
        continue
    for train_dir in sorted(run_dir.glob("train*")):
        if not train_dir.is_dir():
            continue
        v3_csv = train_dir / "metric" / "learning improvement" / f"learning_metrics_v3_{train_dir.name}.csv"
        if not v3_csv.exists():
            continue

        # Read the V3 CFA SUMMARY (KEY-VALUE) section
        kv = {}
        with v3_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            in_kv = False
            for row in reader:
                if len(row) >= 1 and row[0].strip() == "=== V3 CFA SUMMARY (KEY-VALUE) ===":
                    in_kv = True
                    continue
                if in_kv and len(row) >= 2:
                    k = row[0].strip()
                    v = row[1].strip()
                    if k.startswith("==="):
                        break
                    kv[k] = v

        def get_float(k):
            v = kv.get(k, "")
            try:
                return float(v)
            except ValueError:
                return float("nan")

        def get_bool(k):
            v = kv.get(k, "").lower().strip()
            return v == "true"

        records.append({
            "run": run_dir.name,
            "train_id": train_dir.name,
            "les": get_float("learning_exposure_score"),
            "cas_ratio": get_float("competence_arrival_ratio"),
            "cas_reached": get_bool("competence_arrival_reached"),
            "lrs": get_float("learning_retention_score"),
            "rsa": get_float("rolling_success_attainment"),
            "rgec": get_float("rolling_goal_error_consistency"),
            "rtge": get_float("rolling_time_to_goal_efficiency_score"),
            "episodes": get_float("episodes"),
        })

df = pd.DataFrame(records)

# Add derived columns for filtering
df["les_negative"] = df["les"] < 0
df["episodes_low"] = df["episodes"] < 1000

print("=" * 100)
print(f"V3 Metrics Wide Table — {len(df)} trains total")
print("=" * 100)
print(df.to_string(index=False))
print("\n" + "=" * 100)
print("Summary for selection:")
print(f"  LES < 0          : {df['les_negative'].sum()} trains")
print(f"  Episodes < 1000  : {df['episodes_low'].sum()} trains")
print(f"  CAS not reached  : {(~df['cas_reached']).sum()} trains")

# Show descriptive stats
print("\nDescriptive statistics:")
print(df[["les", "cas_ratio", "lrs", "rsa", "rgec", "rtge", "episodes"]].describe().round(4).to_string())

out_path = BASE / "v3_metrics_wide_table.csv"
df.to_csv(out_path, index=False)
print(f"\n[OK] Wide table saved to: {out_path}")
