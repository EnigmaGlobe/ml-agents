#!/usr/bin/env python3
"""
Batch run learning_metrics_v3.py for all 37 trains in training_results.
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(r"C:\soqqle\ml-agents\config\training_results")
SCRIPT = Path(r"C:\soqqle\ml-agents\config\learning_metrics_v3.py")
MAX_EPISODE_STEP = 5000

# Map: run_dir -> list of (train_name, episode_csv_path)
trains = []

for run_dir in sorted(BASE.glob("run_*")):
    if not run_dir.is_dir():
        continue
    for train_dir in sorted(run_dir.glob("train*")):
        if not train_dir.is_dir():
            continue
        li_dir = train_dir / "metric" / "learning improvement"
        if not li_dir.exists():
            continue
        csvs = sorted(li_dir.glob("learning_improvement_*.csv"))
        if not csvs:
            continue
        # Use the latest (or only) file
        episode_csv = csvs[-1]
        trains.append((run_dir.name, train_dir.name, episode_csv))

print(f"[Batch] Found {len(trains)} trains to process.")

success = 0
fail = 0
for run_name, train_name, episode_csv in trains:
    out_dir = BASE / run_name / train_name / "metric" / "learning improvement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"learning_metrics_v3_{train_name}.csv"

    cmd = [
        sys.executable, str(SCRIPT),
        str(episode_csv), str(out_csv),
        "--max-episode-step", str(MAX_EPISODE_STEP)
    ]
    print(f"[Batch] Processing {run_name}/{train_name} ...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  -> OK: {out_csv}")
        success += 1
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: {run_name}/{train_name}")
        print(e.stderr)
        fail += 1

print(f"\n[Batch] Done. Success: {success}, Failed: {fail}")
