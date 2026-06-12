#!/usr/bin/env python3
"""
File-based Curriculum EC orchestrator for PushT / PushBlock.

Version 3.2: full multi-stage training loop.

Workflow per stage:
1. Run `mlagents-learn` with a run-id and config.
2. Wait for the training process to finish.
3. Locate the latest .onnx checkpoint.
4. Safely copy the checkpoint into Unity's FrozenEvaluator folder.
5. Write checkpoint_ready.json (so V2 loader imports the model).
6. Write ec_request.json (so V3 processor runs Curriculum EC).
7. Wait for Unity to write ec_done.json or ec_error.json.
8. Proceed to the next stage.

This script intentionally does NOT modify the ML-Agents trainer and does NOT use
sockets, REST, HTTP, gRPC, or WebSocket. Communication is file-based only.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
UNITY_PROJECT_DIR = PROJECT_ROOT / "Project"
FROZEN_EVALUATOR_DIR = UNITY_PROJECT_DIR / "Assets" / "Models" / "FrozenEvaluator"

REQUEST_PATH = FROZEN_EVALUATOR_DIR / "ec_request.json"
DONE_PATH = FROZEN_EVALUATOR_DIR / "ec_done.json"
ERROR_PATH = FROZEN_EVALUATOR_DIR / "ec_error.json"
CHECKPOINT_READY_PATH = FROZEN_EVALUATOR_DIR / "checkpoint_ready.json"
FROZEN_EVALUATOR_ONNX = FROZEN_EVALUATOR_DIR / "FrozenEvaluator_latest.onnx"

DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_EC_TIMEOUT = 600.0
DEFAULT_POST_CHECKPOINT_DELAY = 5.0


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------

def clear_ec_manifests():
    """Remove old done/error manifests before starting a new EC request."""
    for path in (DONE_PATH, ERROR_PATH):
        if path.exists():
            path.unlink()
            print(f"[Orchestrator] Cleared old manifest: {path}")


def write_checkpoint_ready(checkpoint_step: int, source_path: Path):
    """Write checkpoint_ready.json so Unity's V2 loader imports the model."""
    manifest = {
        "run_id": "curriculum_orchestration",
        "checkpoint_step": checkpoint_step,
        "source_onnx_path": str(source_path),
        "unity_onnx_path": str(FROZEN_EVALUATOR_ONNX),
        "copied_at": now_iso(),
        "status": "ready",
    }
    FROZEN_EVALUATOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_READY_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Orchestrator] Wrote {CHECKPOINT_READY_PATH}")


def write_ec_request(stage_cfg: dict, checkpoint_step: int):
    """Write ec_request.json to trigger Curriculum EC in Unity."""
    request_id = (
        f"stage_{stage_cfg['stage']:03d}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    request = {
        "request_id": request_id,
        "stage": stage_cfg["stage"],
        "run_id": stage_cfg["run_id"],
        "checkpoint_step": checkpoint_step,
        "onnx_path": "Assets/Models/FrozenEvaluator/FrozenEvaluator_latest.onnx",
        "requested_at": now_iso(),
        "action": "run_curriculum_ec",
        "status": "pending",
        "expected_behavior_name": stage_cfg.get("expected_behavior_name", "PushBlock"),
    }

    FROZEN_EVALUATOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(REQUEST_PATH, "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    print(f"[Orchestrator] Wrote {REQUEST_PATH}")
    print(f"[Orchestrator] request_id = {request_id}")
    return request_id


def wait_for_ec_result(
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_EC_TIMEOUT,
):
    """Poll for ec_done.json or ec_error.json. Returns (success, data)."""
    start = time.time()
    print(f"[Orchestrator] Waiting for EC result (timeout={timeout}s)...")

    while time.time() - start < timeout:
        if ERROR_PATH.exists():
            with open(ERROR_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("[Orchestrator] ERROR manifest detected:")
            print(json.dumps(data, indent=2))
            return False, data

        if DONE_PATH.exists():
            with open(DONE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("[Orchestrator] DONE manifest detected:")
            print(json.dumps(data, indent=2))
            return True, data

        time.sleep(poll_interval)

    print("[Orchestrator] TIMEOUT waiting for EC result.")
    return False, {"error": "timeout", "message": "Unity did not write ec_done.json or ec_error.json in time."}


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def find_latest_checkpoint(results_dir: Path, run_id: str, behavior_name: str) -> Optional[Path]:
    """Return the path to the latest .onnx model for a run."""
    run_dir = results_dir / run_id
    latest = run_dir / f"{behavior_name}.onnx"
    if latest.exists() and latest.stat().st_size > 0:
        return latest

    # Fallback: search recursively for any non-empty .onnx in the run folder.
    candidates = sorted(
        run_dir.rglob("*.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.stat().st_size > 0:
            return candidate

    return None


def copy_checkpoint_safely(source: Path, destination: Path):
    """Copy .onnx to destination via a temp file to avoid partial reads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".onnx.tmp")
    shutil.copy2(source, temp)

    if not temp.exists() or temp.stat().st_size != source.stat().st_size:
        if temp.exists():
            temp.unlink()
        raise IOError("Checkpoint temp copy verification failed.")

    temp.replace(destination)
    print(f"[Orchestrator] Copied checkpoint {source} -> {destination}")


def extract_checkpoint_step(file_name: str) -> int:
    """Try to extract a step number from a checkpoint file name."""
    stem = Path(file_name).stem
    for part in stem.replace("-", "_").split("_"):
        if part.isdigit():
            return int(part)
    return -1


# -----------------------------------------------------------------------------
# ML-Agents training
# -----------------------------------------------------------------------------

def build_mlagents_command(mlagents_learn: str, stage_cfg: dict, global_cfg: dict) -> list:
    """Build the mlagents-learn command for a stage."""
    cmd = [mlagents_learn, stage_cfg["config_path"], "--run-id", stage_cfg["run_id"]]

    base_port = stage_cfg.get("base_port") or global_cfg.get("base_port", 5005)
    cmd.extend(["--base-port", str(base_port)])

    env_path = stage_cfg.get("env_path") or global_cfg.get("env_path")
    if env_path:
        cmd.extend(["--env", env_path])

    num_envs = stage_cfg.get("num_envs") or global_cfg.get("num_envs")
    if num_envs:
        cmd.extend(["--num-envs", str(num_envs)])

    initialize_from = stage_cfg.get("initialize_from") or global_cfg.get("initialize_from")
    if initialize_from:
        cmd.extend(["--initialize-from", initialize_from])

    return cmd


def resolve_executable(name: str) -> Optional[str]:
    """Find an executable in PATH. Returns None if not found."""
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path_dir) / name
        if candidate.exists():
            return str(candidate)
        candidate_with_exe = Path(path_dir) / (name + ".exe")
        if candidate_with_exe.exists():
            return str(candidate_with_exe)
    return None


def run_training(cmd: list, timeout: Optional[float] = None, cwd: Optional[Path] = None) -> int:
    """Run mlagents-learn and return its exit code."""
    print(f"[Orchestrator] Running: {' '.join(cmd)}")
    print(f"[Orchestrator] cwd: {cwd or Path.cwd()}")

    executable = cmd[0]
    resolved = resolve_executable(executable) if not Path(executable).exists() else executable
    if resolved is None:
        print(f"[Orchestrator] ERROR: Cannot find executable '{executable}'.")
        print("[Orchestrator] Options:")
        print("  1. Activate the conda/venv environment where ML-Agents is installed.")
        print("  2. Set 'mlagents_learn' in the config to the full path of mlagents-learn.exe")
        print("     Example: mlagents_learn: \"C:/Users/user/miniconda3/Scripts/mlagents-learn.exe\"")
        return -1

    cmd[0] = resolved

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"[Orchestrator] Training timed out after {timeout}s.")
        return -1


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load orchestration config from YAML or JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def resolve_path(value: str, project_root: Path) -> Path:
    """Resolve a config path relative to project_root if it is not absolute."""
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


# -----------------------------------------------------------------------------
# Stage execution
# -----------------------------------------------------------------------------

def run_stage(stage_cfg: dict, global_cfg: dict) -> bool:
    """Run one full curriculum stage. Returns True on success."""
    stage = stage_cfg["stage"]
    run_id = stage_cfg["run_id"]
    behavior_name = global_cfg.get("behavior_name", "PushBlock")
    project_root = resolve_path(global_cfg.get("project_root", "."), PROJECT_ROOT)
    results_dir = resolve_path(global_cfg.get("results_dir", "results"), project_root)
    mlagents_learn = global_cfg.get("mlagents_learn") or "mlagents-learn"
    timeout = stage_cfg.get("timeout_seconds") or global_cfg.get("training_timeout_seconds") or None

    print(f"\n[Orchestrator] ===== Stage {stage}: {run_id} =====")

    # 1. Train
    cmd = build_mlagents_command(mlagents_learn, stage_cfg, global_cfg)
    exit_code = run_training(cmd, timeout=timeout, cwd=project_root)
    if exit_code != 0:
        print(f"[Orchestrator] Training failed with exit code {exit_code}. Stopping.")
        return False

    # 2. Locate latest checkpoint
    checkpoint = find_latest_checkpoint(results_dir, run_id, behavior_name)
    if checkpoint is None:
        print(f"[Orchestrator] No checkpoint found in {results_dir / run_id}. Stopping.")
        return False

    checkpoint_step = extract_checkpoint_step(checkpoint.name)
    print(f"[Orchestrator] Latest checkpoint: {checkpoint} (step={checkpoint_step})")

    # 3. Copy checkpoint into Unity
    copy_checkpoint_safely(checkpoint, FROZEN_EVALUATOR_ONNX)

    # 4. Write checkpoint_ready.json for V2 loader
    write_checkpoint_ready(checkpoint_step, checkpoint)

    # 5. Give Unity a moment to import/load the model
    delay = global_cfg.get("post_checkpoint_delay", DEFAULT_POST_CHECKPOINT_DELAY)
    if delay > 0:
        print(f"[Orchestrator] Waiting {delay}s for Unity to import checkpoint...")
        time.sleep(delay)

    # 6. Trigger Curriculum EC
    clear_ec_manifests()
    write_ec_request(stage_cfg, checkpoint_step)

    # 7. Wait for EC result
    success, data = wait_for_ec_result(
        poll_interval=global_cfg.get("ec_poll_interval", DEFAULT_POLL_INTERVAL),
        timeout=global_cfg.get("ec_timeout_seconds", DEFAULT_EC_TIMEOUT),
    )

    if not success:
        print(f"[Orchestrator] Stage {stage} EC failed.")
        return False

    print(f"[Orchestrator] Stage {stage} succeeded. selected_count={data.get('selected_count')}, generation_id={data.get('generation_id')}")
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        config_path = PROJECT_ROOT / "config" / "curriculum_orchestration.yaml"
    else:
        config_path = Path(sys.argv[1])

    print(f"[Orchestrator] Loading config: {config_path}")
    cfg = load_config(config_path)

    stages = cfg.get("stages", [])
    if not stages:
        print("[Orchestrator] No stages defined in config.")
        return

    for stage_cfg in stages:
        success = run_stage(stage_cfg, cfg)
        if not success:
            print("[Orchestrator] Stopping pipeline due to stage failure.")
            break

    print("[Orchestrator] Pipeline complete.")


if __name__ == "__main__":
    main()
