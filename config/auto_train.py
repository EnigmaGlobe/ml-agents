#!/usr/bin/env python3
"""
Auto-run ml-agents training with organized output directories.

Directory layout per training run:
    results/run_XX/train_YY/
        _ml_temp/          ← ML-Agents 原始输出，保留
        metric/
            learning improvement/
        recordings/
        tensor/            ← TensorBoard CSV exports
        train_log/
            train_YY.log

Usage:
    python auto_train.py --config config/ppo/PushBlock.yaml --run-id 1 --runs 3
    python auto_train.py --config config/ppo/PushBlock.yaml --run-id 1  (runs until stopped)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONDA_EXE = r"C:\Users\infra\anaconda3\condabin\conda.bat"
UNITY_EXE = r"C:\soqqle\ml-agents\config\PushBlockHeadless\UnityEnvironment.exe"
EXPORT_SCRIPT = Path(r"C:\soqqle\ml-agents\config\export_tensor.py")
RESULTS_BASE = Path(r"C:\soqqle\ml-agents\config\results")


def setup_train_dirs(run_num: int, train_num: int):
    """Create the full directory tree for a training run."""
    run_dir = RESULTS_BASE / f"run_{run_num:02d}"
    train_dir = run_dir / f"train_{train_num:02d}"
    subdirs = [
        train_dir / "_ml_temp",
        train_dir / "metric" / "learning improvement",
        train_dir / "recordings",
        train_dir / "tensor",
        train_dir / "train_log",
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)
    return train_dir


def find_tfevents_file(ml_temp: Path) -> Path | None:
    """Find the latest .tfevents file inside _ml_temp/train_YY/."""
    event_files = list(ml_temp.rglob("events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=lambda f: f.stat().st_mtime)


def export_tensors(tfevents_file: Path, tensor_dir: Path) -> bool:
    """Run export_tensor.py to extract CSVs into tensor directory."""
    if not EXPORT_SCRIPT.exists():
        print(f"[ERROR] export_tensor.py not found at {EXPORT_SCRIPT}")
        return False

    cmd = (
        f'"{CONDA_EXE}" activate mlagents && '
        f'python "{EXPORT_SCRIPT}" --event-file "{tfevents_file}" --out-dir "{tensor_dir}" --all'
    )
    print(f"[EXPORT] Running tensor export to {tensor_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode == 0


def _copy_learning_improvement_csv(train_dir: Path):
    """Copy Unity's learning_improvement CSV from _ml_temp to metric/learning improvement/."""
    ml_temp = train_dir / "_ml_temp"
    if not ml_temp.exists():
        return
    for f in ml_temp.rglob("learning_improvement_*.csv"):
        dest = train_dir / "metric" / "learning improvement" / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(f), str(dest))
        print(f"[OK] Copied {f.name} to metric/learning improvement/")


def _generate_learning_metrics(train_dir: Path, train_id: str):
    """Find the latest learning_improvement CSV and compute detailed metrics."""
    metric_dir = train_dir / "metric" / "learning improvement"
    if not metric_dir.exists():
        return

    csv_files = sorted(metric_dir.glob("learning_improvement_*.csv"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    if not csv_files:
        return

    input_csv = csv_files[0]
    output_csv = metric_dir / f"learning_metrics_{train_id}.csv"

    metrics_script = Path(__file__).parent / "learning_metrics.py"
    if not metrics_script.exists():
        return

    cmd = [sys.executable, str(metrics_script), str(input_csv), str(output_csv)]
    print(f"[METRICS] Computing learning-improvement metrics from {input_csv.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode == 0:
        print(f"[OK] Detailed metrics saved to {output_csv}")


def is_training_complete(line: str) -> bool:
    """Detect training completion signals in stdout."""
    success_signals = [
        r"Learning was finished",
        r"saved model and learning statistics",
        r"Copying results for run-id",
    ]
    return any(re.search(pattern, line) for pattern in success_signals)


def is_training_failed(line: str) -> bool:
    """Detect training failure signals in stdout."""
    failure_signals = [
        r"UnityTrainerException",
        r"UnityWorkerInUseException",
        r"Traceback.*most recent call last",
        r"Error.*socket",
        r"Failed to bind",
    ]
    return any(re.search(pattern, line) for pattern in failure_signals)


def run_training(yaml_config: str, run_num: int, train_num: int) -> int:
    """Run mlagents-learn and return the process exit code."""
    train_folder = f"train_{train_num:02d}"
    train_dir = setup_train_dirs(run_num, train_num)
    # Make the run-id globally unique by including the run number
    run_id = f"run_{run_num:02d}_{train_folder}"
    ml_temp = train_dir / "_ml_temp"
    log_file = train_dir / "train_log" / f"{train_folder}.log"
    metric_li_dir = train_dir / "metric" / "learning improvement"

    print(f"\n{'='*60}")
    print(f"[TRAIN] Starting training: run=run_{run_num:02d}, id={run_id}")
    print(f"[TRAIN] Config: {yaml_config}")
    print(f"[TRAIN] Output: {train_dir}")
    print(f"{'='*60}\n")

    results_dir_posix = ml_temp.as_posix()
    metric_li_posix = metric_li_dir.as_posix()
    recordings_posix = (train_dir / "recordings").as_posix()

    batch_content = (
        f'@echo off\n'
        f'set PYTHONUNBUFFERED=1\n'
        f'set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n'
        f'set PUSHBLOCK_CSV_DIR={metric_li_posix}\n'
        f'set PUSHBLOCK_METADATA_DIR={recordings_posix}\n'
        f'set PUSHBLOCK_RUN_ID={run_id}\n'
        f'call "{CONDA_EXE}" activate mlagents\n'
        f'mlagents-learn "{yaml_config}" --run-id {run_id} --force --env "{UNITY_EXE}" --results-dir "{results_dir_posix}"\n'
    )
    batch_file = train_dir / "_run.bat"
    batch_file.write_text(batch_content)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PUSHBLOCK_CSV_DIR'] = str(metric_li_dir)
    env['PUSHBLOCK_METADATA_DIR'] = str(train_dir / "recordings")
    env['PUSHBLOCK_RUN_ID'] = run_id
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    process = subprocess.Popen(
        f'cmd /c "{batch_file}" > "{log_file}" 2>&1',
        shell=True,
        env=env,
    )

    failed = False
    last_step_seen = -1
    try:
        while process.poll() is None:
            # Poll log every 30 minutes to avoid flooding the terminal
            time.sleep(1800)
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.rstrip()
                        if not line:
                            continue
                        step_match = re.search(r"Step:\s*(\d+)", line)
                        if step_match:
                            step = int(step_match.group(1))
                            if step <= last_step_seen:
                                continue
                            last_step_seen = step
                        print(line)
                        if is_training_complete(line):
                            print("\n[TRAIN] Training completion detected")
                        if is_training_failed(line):
                            print("\n[TRAIN] Training failure detected")
                            failed = True
            except FileNotFoundError:
                pass
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user")
        process.terminate()
        process.wait()
        return -1

    process.wait()
    if failed:
        return 1
    return process.returncode


def post_training(run_num: int, train_num: int) -> bool:
    """Export tensors from _ml_temp, copy CSVs, compute metrics."""
    train_id = f"train_{train_num:02d}"
    train_dir = RESULTS_BASE / f"run_{run_num:02d}" / train_id
    ml_temp = train_dir / "_ml_temp"
    tensor_dir = train_dir / "tensor"

    # Export tensors directly from _ml_temp
    tfevents = find_tfevents_file(ml_temp)
    if tfevents:
        success = export_tensors(tfevents, tensor_dir)
        if not success:
            print(f"[WARN] Tensor export failed for {train_id}")
    else:
        print(f"[WARN] No tfevents file found for {train_id}")

    # Copy learning_improvement CSV
    _copy_learning_improvement_csv(train_dir)

    # Generate learning metrics
    _generate_learning_metrics(train_dir, train_id)

    return True


def find_next_run_number() -> int:
    """Scan results/ for existing run_XX dirs and return the next available number."""
    if not RESULTS_BASE.exists():
        return 1
    existing = []
    for entry in RESULTS_BASE.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            m = re.match(r"run_(\d+)$", entry.name)
            if m:
                existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-run ml-agents training with organized output directories"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--run-id", type=int, default=0,
        help="Starting run number (run_XX). 0 = auto-find next available"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Number of training runs to execute (0 = infinite until stopped)",
    )
    args = parser.parse_args()

    yaml_config = args.config
    run_num = args.run_id if args.run_id > 0 else find_next_run_number()
    train_num = 1  # Always start from train_01 within each run
    run_count = 0
    max_runs = args.runs

    print(f"[AUTO-RUN] Using run_{run_num:02d}")

    while True:
        if max_runs > 0 and run_count >= max_runs:
            print(f"[DONE] Completed {max_runs} runs")
            break

        # Run training
        exit_code = run_training(yaml_config, run_num, train_num)

        if exit_code == -1:
            print("[STOP] Training was interrupted, stopping auto-run")
            break

        if exit_code != 0:
            train_id = f"train_{train_num:02d}"
            print(f"[STOP] Training failed for {train_id}, stopping auto-run")
            break

        # Post-training: export tensors and organize outputs
        success = post_training(run_num, train_num)
        if not success:
            train_id = f"train_{train_num:02d}"
            print(f"[WARN] Post-training organization failed for {train_id}")

        train_num += 1
        run_count += 1

        # Brief pause before next run
        if max_runs == 0 or run_count < max_runs:
            print(f"\n[NEXT] Preparing train_{train_num:02d} in 10 seconds...")
            print("[NEXT] Make sure Unity is ready before the next run starts")
            time.sleep(10)

    print(f"\n[AUTO-RUN] Finished. Completed {run_count} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
