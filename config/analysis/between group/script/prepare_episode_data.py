#!/usr/bin/env python3
"""
Merge episode-level data from half_goal and 1.5x for LMM analysis.
Output: combined_episodes_lmm.csv
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path(r"C:\Soqqle\ml-agents\config\analysis\between group\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_group(base_dir: Path, condition: str):
    records = []
    for train_dir in sorted(base_dir.glob("train_*")):
        if not train_dir.is_dir():
            continue
        li_dir = train_dir / "metric" / "learning improvement"
        if not li_dir.exists():
            continue
        csvs = sorted(li_dir.glob("learning_improvement_*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[-1])
        df["condition"] = condition
        df["train_id_unified"] = f"{base_dir.parent.name}_{train_dir.name}"
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# Load half_goal
half_base = Path(r"C:\soqqle\ml-agents\config\results_half\run_01")
df_half = load_group(half_base, "half_goal")
print(f"[Load] half_goal: {len(df_half)} episodes from {df_half['train_id_unified'].nunique()} trains")

# Load 1.5x
x15_base = Path(r"C:\soqqle\ml-agents\config\results_1.5x\run_01")
df_15x = load_group(x15_base, "1.5x")
print(f"[Load] 1.5x: {len(df_15x)} episodes from {df_15x['train_id_unified'].nunique()} trains")

# Combine
df = pd.concat([df_half, df_15x], ignore_index=True)

# Keep only columns needed for LMM
cols = ["episode_id", "training_step", "normalized_task_progress", "success",
        "final_goal_zone_error_xz", "time_to_goal", "condition", "train_id_unified"]
df = df[cols].copy()

# Clean
df["condition"] = df["condition"].astype("category")
df["train_id_unified"] = df["train_id_unified"].astype("category")

# Scale training_step to thousands of steps for nicer coefficients
df["training_step_k"] = df["training_step"] / 1000.0

out_path = OUT_DIR / "combined_episodes_lmm.csv"
df.to_csv(out_path, index=False)
print(f"[OK] Saved {len(df)} episodes to {out_path}")
print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
print(f"  Trains: {df['train_id_unified'].nunique()}")
print(f"  normalized_task_progress range: [{df['normalized_task_progress'].min():.3f}, {df['normalized_task_progress'].max():.3f}]")
