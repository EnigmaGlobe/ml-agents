#!/usr/bin/env python3
"""
Auto-run ml-agents training with automatic run-id increment and tensor export.

Usage:
    python auto_train.py --config config/ppo/3DBall.yaml --run-id 1 --runs 3
    python auto_train.py --config config/ppo/3DBall.yaml --run-id 1  (runs until stopped)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONDA_EXE = r"C:\tools\Anaconda3\condabin\conda.bat"
UNITY_EXE = r"C:\soqqle\ml-agents\config\PushBlockHeadless\UnityEnvironment.exe"
EXPORT_SCRIPT = Path(r"C:\soqqle\ml-agents\config\export_tensor.py")
OUTPUT_DIR = Path(r"C:\soqqle\ml-agents\config\Training Outputs")
RESULTS_DIR = Path(r"C:\soqqle\ml-agents\config\ppo\results")
METRICS_DIR = Path(r"C:\soqqle\ml-agents\config\metrics\learning_improvement")
LOG_DIR = Path(r"C:\soqqle\ml-agents\config\logs")


def find_results_dir(run_id: str) -> Path | None:
    """Find the results directory for a given run_id."""
    if not RESULTS_DIR.exists():
        return None
    for child in RESULTS_DIR.iterdir():
        if child.is_dir() and child.name == run_id:
            return child
    return None


def find_tfevents_file(results_dir: Path) -> Path | None:
    """Find the latest .tfevents file in the results directory."""
    event_files = list(results_dir.glob("events.out.tfevents.*"))
    if not event_files:
        # Check subdirectories
        for child in results_dir.iterdir():
            if child.is_dir():
                event_files.extend(child.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=lambda f: f.stat().st_mtime)


def export_tensors(run_id: str, tfevents_file: Path) -> bool:
    """Run export_tensor.py to extract CSVs into Training Outputs."""
    out_dir = OUTPUT_DIR / run_id
    if not EXPORT_SCRIPT.exists():
        print(f"[ERROR] export_tensor.py not found at {EXPORT_SCRIPT}")
        return False

    conda_exe = r"C:\tools\Anaconda3\condabin\conda.bat"
    cmd = (
        f'"{conda_exe}" activate mlagents && '
        f'python "{EXPORT_SCRIPT}" --event-file "{tfevents_file}" --out-dir "{out_dir}" --all'
    )
    print(f"[EXPORT] Running tensor export to {out_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result.returncode == 0


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


def run_training(yaml_config: str, run_id: str) -> int:
    """Run mlagents-learn and return the process exit code."""
    conda_exe = r"C:\tools\Anaconda3\condabin\conda.bat"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"train_{run_id}.log"

    # Ensure the log file starts fresh (never overwrite or append)
    if log_file.exists():
        # Rename old log to preserve history: train_1.log -> train_1.log.bak_1
        counter = 1
        while log_file.with_suffix(f".bak_{counter}").exists():
            counter += 1
        log_file.rename(log_file.with_suffix(f".bak_{counter}"))

    print(f"\n{'='*60}")
    print(f"[TRAIN] Starting training: run-id={run_id}")
    print(f"[TRAIN] Config: {yaml_config}")
    print(f"[TRAIN] Log: {log_file}")
    print(f"{'='*60}\n")

    # Write a batch file to handle conda activation properly
    results_dir = RESULTS_DIR.as_posix()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    batch_content = (
        f'@echo off\n'
        f'set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n'
        f'set PUSHBLOCK_CSV_DIR={METRICS_DIR}\n'
        f'call "{conda_exe}" activate mlagents\n'
        f'mlagents-learn "{yaml_config}" --run-id {run_id} --force --env "{UNITY_EXE}" --results-dir "{results_dir}"\n'
    )
    batch_file = LOG_DIR / f"_run_{run_id}.bat"
    batch_file.write_text(batch_content)

    # Build environment with PUSHBLOCK_CSV_DIR set for Unity
    import os
    env = os.environ.copy()
    env['PUSHBLOCK_CSV_DIR'] = str(METRICS_DIR)
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    # Run and redirect output to log file
    process = subprocess.Popen(
        f'cmd /c "{batch_file}" > "{log_file}" 2>&1',
        shell=True,
        env=env,
    )

    last_line = ""
    failed = False
    last_step_seen = -1
    try:
        # Tail the log file while process runs
        while process.poll() is None:
            import time as _time
            _time.sleep(2)
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.rstrip()
                        if line:
                            # Deduplicate step lines by step number
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


def _generate_learning_metrics(run_id: str):
    """Find the latest learning_improvement CSV from Unity and compute detailed metrics."""
    if not METRICS_DIR.exists():
        print(f"[WARN] Metrics directory not found: {METRICS_DIR}")
        return

    # Find the most recent learning_improvement CSV
    csv_files = sorted(METRICS_DIR.glob("learning_improvement_*.csv"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    if not csv_files:
        print(f"[WARN] No learning_improvement CSV found for {run_id}")
        return

    input_csv = csv_files[0]
    output_csv = METRICS_DIR / f"learning_metrics_{run_id}.csv"

    # Run the metrics computation script
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "learning_metrics.py"),
        str(input_csv),
        str(output_csv),
    ]
    print(f"[METRICS] Computing learning-improvement metrics from {input_csv.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode == 0:
        print(f"[OK] Detailed metrics saved to {output_csv}")
    else:
        print(f"[WARN] Metrics computation failed for {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-run ml-agents training with incrementing run-ids"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--run-id", type=int, required=True, help="Starting run-id number"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Number of runs to execute (0 = infinite until stopped)",
    )
    args = parser.parse_args()

    yaml_config = args.config
    run_number = args.run_id
    run_count = 0
    max_runs = args.runs

    while True:
        if max_runs > 0 and run_count >= max_runs:
            print(f"[DONE] Completed {max_runs} runs")
            break

        run_id = f"train_{run_number}"

        # Run training
        exit_code = run_training(yaml_config, run_id)

        if exit_code == -1:
            print("[STOP] Training was interrupted, stopping auto-run")
            break

        if exit_code != 0:
            print(f"[STOP] Training failed for {run_id}, stopping auto-run")
            break

        # Find and export tensors
        results_dir = find_results_dir(run_id)
        if results_dir:
            tfevents = find_tfevents_file(results_dir)
            if tfevents:
                success = export_tensors(run_id, tfevents)
                if success:
                    print(f"[OK] Tensors exported for {run_id}")
                else:
                    print(f"[WARN] Tensor export failed for {run_id}")
            else:
                print(f"[WARN] No tfevents file found for {run_id}")
        else:
            print(f"[WARN] No results directory found for {run_id}")

        # Generate learning-improvement metrics from Unity episode CSV
        _generate_learning_metrics(run_id)

        run_number += 1
        run_count += 1

        # Brief pause before next run
        if max_runs == 0 or run_count < max_runs:
            print(f"\n[NEXT] Preparing run {run_number} in 10 seconds...")
            print("[NEXT] Make sure Unity is ready before the next run starts")
            time.sleep(10)

    print(f"\n[AUTO-RUN] Finished. Completed {run_count} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
