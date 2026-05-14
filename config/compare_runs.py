#!/usr/bin/env python3
"""
Compare multiple training runs to diagnose divergence.

Checks:
1. Seed consistency
2. Config consistency (from saved YAML)
3. Entropy / policy loss / value loss comparison
4. Early-exploration divergence
5. Final model quality indicators
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path(r"C:\soqqle\ml-agents\config\results")


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            try:
                rows.append({
                    "step": int(float(cleaned.get("Step", 0))),
                    "value": float(cleaned.get("Value", 0)),
                })
            except (ValueError, TypeError):
                continue
    return rows


def compare_runs():
    runs = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("train_")])
    print(f"\n{'═'*70}")
    print(f"TRAIN RUN DIAGNOSTIC COMPARISON")
    print(f"Runs: {', '.join(runs)}")
    print(f"{'═'*70}")

    # ── 1. Check logs for seed info ──
    print(f"\n── Check 1: Seed Information ──")
    for run in runs:
        log_file = RESULTS_DIR / run / "train_log" / f"{run}.log"
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            seed_line = [l for l in content.split("\n") if "seed" in l.lower()]
            if seed_line:
                print(f"  {run}: {seed_line[0].strip()}")
            else:
                print(f"  {run}: [NO SEED FOUND IN LOG — using default/random]")

    # ── 2. Config consistency ──
    print(f"\n── Check 2: Hyperparameter Consistency ──")
    for run in runs:
        log_file = RESULTS_DIR / run / "train_log" / f"{run}.log"
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            # Check for resume/initialize_from
            for keyword in ["resume", "initialize_from", "load_model", "force"]:
                if keyword.lower() in content.lower():
                    lines = [l.strip() for l in content.split("\n") if keyword.lower() in l.lower()]
                    print(f"  {run}: contains '{keyword}' → {lines[0][:80]}")

    # ── 3. Final metrics comparison ──
    print(f"\n── Check 3: Final Metrics (at 2M steps) ──")
    print(f"  {'run':<10} {'CumReward':>10} {'StdReward':>10} {'EpLength':>10} {'PolicyLoss':>12} {'ValueLoss':>12}")
    print(f"  {'─'*74}")

    metrics = {}
    for run in runs:
        tensor_dir = RESULTS_DIR / run / "training_outputs" / "tensor_export"
        rw = load_csv(tensor_dir / "Cumulative Reward.csv")
        pl = load_csv(tensor_dir / "Policy Loss.csv")
        vl = load_csv(tensor_dir / "Value Loss.csv")
        el = load_csv(tensor_dir / "Episode Length.csv")
        ex = load_csv(tensor_dir / "Extrinsic Reward.csv")

        final_rw = rw[-1]["value"] if rw else 0
        final_pl = pl[-1]["value"] if pl else 0
        final_vl = vl[-1]["value"] if vl else 0
        final_el = el[-1]["value"] if el else 0

        # std reward from last 5 data points
        std_rw = stdev([r["value"] for r in rw[-5:]]) if len(rw) >= 5 else 0

        metrics[run] = {
            "reward": final_rw,
            "std_reward": std_rw,
            "ep_length": final_el,
            "policy_loss": final_pl,
            "value_loss": final_vl,
        }
        print(f"  {run:<10} {final_rw:>10.2f} {std_rw:>10.3f} {final_el:>10.1f} {final_pl:>12.4f} {final_vl:>12.4f}")

    # ── 4. Early exploration divergence ──
    print(f"\n── Check 4: Early Exploration Divergence ──")
    steps_check = [20000, 100000, 200000, 400000, 600000]
    for run in runs:
        tensor_dir = RESULTS_DIR / run / "training_outputs" / "tensor_export"
        rw = load_csv(tensor_dir / "Cumulative Reward.csv")
        el = load_csv(tensor_dir / "Episode Length.csv")
        rw_map = {r["step"]: r["value"] for r in rw}
        el_map = {e["step"]: e["value"] for e in el}

        print(f"\n  {run}:")
        for s in steps_check:
            r_val = next((v for k, v in rw_map.items() if k == s), None)
            e_val = next((v for k, v in el_map.items() if k == s), None)
            if r_val is not None:
                print(f"    Step {s:>7}: reward={r_val:>7.2f}  ep_len={e_val:.0f}")
            else:
                print(f"    Step {s:>7}: [no data]")

    # ── 5. Value Loss comparison (diagnostic) ──
    print(f"\n── Check 5: Value Loss Trend (indicator of training stability) ──")
    print(f"  {'run':<10} {'VL@200k':>10} {'VL@600k':>10} {'VL@1M':>10} {'VL@2M':>10} {'trend':<10}")
    print(f"  {'─'*64}")

    for run in runs:
        tensor_dir = RESULTS_DIR / run / "training_outputs" / "tensor_export"
        vl = load_csv(tensor_dir / "Value Loss.csv")
        vl_map = {v["step"]: v["value"] for v in vl}

        def get_val(step):
            if step in vl_map:
                return vl_map[step]
            # nearest
            nearest = min(vl_map.keys(), key=lambda x: abs(x - step))
            if abs(nearest - step) < 20000:
                return vl_map[nearest]
            return None

        v200 = get_val(200000)
        v600 = get_val(600000)
        v1m = get_val(1000000)
        v2m = get_val(2000000)

        if v200 and v2m:
            trend = "↑ worsening" if v2m > v200 * 1.5 else "↓ improving" if v2m < v200 * 0.5 else "→ stable"
        else:
            trend = "?"

        print(f"  {run:<10} {v200:>10.3f} {v600:>10.3f} {v1m:>10.3f} {v2m:>10.3f} {trend:<10}")

    # ── Diagnosis ──
    print(f"\n{'═'*70}")
    print(f"DIAGNOSIS")
    print(f"{'═'*70}")

    rewards = [m["reward"] for m in metrics.values()]
    ep_lengths = [m["ep_length"] for m in metrics.values()]

    best_reward = max(rewards)
    worst_reward = min(rewards)
    gap = best_reward - worst_reward

    if gap > 1.0:
        print(f"\n⚠ Significant performance gap detected: {gap:.2f} reward points")
        print(f"  Best:  {max(metrics, key=lambda k: metrics[k]['reward'])} (reward={best_reward:.2f})")
        print(f"  Worst: {min(metrics, key=lambda k: metrics[k]['reward'])} (reward={worst_reward:.2f})")

    worst_run = min(metrics, key=lambda k: metrics[k]["reward"])
    worst_ep = metrics[worst_run]["ep_length"]

    if worst_ep > 100:
        print(f"\n⚠ {worst_run} has abnormally high episode length ({worst_ep:.0f} vs ~25-30 for others)")
        print(f"  This indicates the agent is NOT completing the task efficiently.")
        print(f"  Likely causes:")
        print(f"    1. Different random seed → different early exploration trajectory")
        print(f"    2. Environment stochasticity (block/goal positions vary per episode)")
        print(f"    3. Agent converged to a suboptimal local minimum")

    # Check if policy loss is still active
    worst_pl = metrics[worst_run]["policy_loss"]
    best_pl = max(m["policy_loss"] for m in metrics.values())

    if worst_pl > 0.05:
        print(f"\n  Policy loss for {worst_run}: {worst_pl:.4f} (still active)")
        print(f"  Agent is still learning, just learning a worse policy.")
        print(f"  Suggests: early-exploration divergence → locked into suboptimal behavior")
    else:
        print(f"\n  Policy loss for {worst_run}: {worst_pl:.4f} (very low)")
        print(f"  Agent may have prematurely converged to a poor strategy.")

    print(f"\n{'─'*70}")
    print(f"RECOMMENDATION:")
    print(f"  - Fix --seed explicitly (e.g. --seed 1, --seed 2, --seed 3)")
    print(f"  - Run 10+ seeds for reliable statistics (Colas et al. guideline)")
    print(f"  - Compare entropy curves to detect premature convergence")
    print(f"  - Consider reward shaping or curriculum for more stable early learning")
    print(f"{'─'*70}")


if __name__ == "__main__":
    compare_runs()
