#!/usr/bin/env python3
"""
Plateau detection for ML-Agents training runs.

Reads exported CSVs from Training Outputs directory and detects when
each run enters a performance plateau based on reward, episode length,
and reward variance stability.

Based on:
- Potential-Based Reward Shaping (IFAAMAS 2024) — episode length convergence
- Parametrized DQN (arxiv 1810.06394) — moving average smoothing
- Colas et al. — multi-run empirical mean/std with caution for small n
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

TRAINING_OUTPUTS = Path(r"C:\soqqle\ml-agents\config\Training Outputs")


def load_csv(path: Path) -> list[dict]:
    """Load a CSV file into list of dicts with numeric conversion."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                cleaned[k.strip()] = v.strip()
            try:
                row_dict = {
                    "wall_time": float(cleaned.get("Wall time", 0)),
                    "step": int(float(cleaned.get("Step", 0))),
                    "value": float(cleaned.get("Value", 0)),
                }
            except (ValueError, TypeError):
                continue
            rows.append(row_dict)
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    """Compute moving average. Edge values use smaller window."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(mean(values[start:end]))
    return result


def detect_plateau(run_name: str, reward_csv: Path, length_csv: Path):
    """Detect plateau step for a single run."""
    print(f"\n{'─'*60}")
    print(f"Analyzing {run_name}")
    print(f"{'─'*60}")

    reward_rows = load_csv(reward_csv)
    length_rows = load_csv(length_csv)

    if not reward_rows or not length_rows:
        print(f"  [SKIP] Insufficient data")
        return None

    # Align by step — use reward steps as reference
    steps = [r["step"] for r in reward_rows]
    rewards = [r["value"] for r in reward_rows]

    # Build length lookup
    length_map = {r["step"]: r["value"] for r in length_rows}
    lengths = []
    for s in steps:
        # Find nearest step
        nearest = min(length_map.keys(), key=lambda x: abs(x - s))
        if abs(nearest - s) < 5000:  # within 5k steps
            lengths.append(length_map[nearest])
        else:
            lengths.append(lengths[-1] if lengths else 1000)

    n = len(rewards)
    if n < 20:
        print(f"  [SKIP] Only {n} data points")
        return None

    # Moving average — adapt window to data size
    window = min(10, n // 5) if n < 100 else 50
    reward_ma = moving_average(rewards, window)
    length_ma = moving_average(lengths, window)

    # Step 3: Define late-phase stable level (last 10% of data)
    late_start = int(n * 0.9)
    late_rewards = reward_ma[late_start:]
    late_lengths = length_ma[late_start:]

    r_final = mean(late_rewards)
    r_final_sd = stdev(late_rewards) if len(late_rewards) > 1 else 0
    l_final = mean(late_lengths)
    l_final_sd = stdev(late_lengths) if len(late_lengths) > 1 else 0

    # Step 4: epsilon for reward
    r_start = reward_ma[0]
    total_gain = abs(r_final - r_start)
    epsilon_reward = max(0.05 * total_gain, 0.5 * r_final_sd)

    # Step 5: epsilon for episode length
    l_start = length_ma[0]
    length_change = abs(l_final - l_start)
    epsilon_length = max(0.05 * length_change, 0.5 * l_final_sd)

    # Step 6: Window-based detection
    # Use fixed step windows based on total steps
    total_steps = steps[-1]
    if total_steps > 1_000_000:
        window_steps = 100_000
    elif total_steps > 200_000:
        window_steps = 50_000
    else:
        window_steps = 20_000

    # Build window boundaries
    boundaries = list(range(0, total_steps + window_steps, window_steps))
    # Only check from 20% onwards (skip warmup)
    start_check = int(total_steps * 0.2)
    boundaries = [b for b in boundaries if b >= start_check]

    if len(boundaries) < 4:
        # Not enough windows, use data-index windows instead
        idx_boundaries = list(range(0, n, max(n // 20, 1)))
        idx_boundaries = [i for i in idx_boundaries if i >= int(n * 0.2)]
        boundaries = [steps[i] for i in idx_boundaries]

    def get_window_stats(step_from, step_to):
        """Get mean reward and length for a step range."""
        vals_r = [reward_ma[i] for i in range(n) if steps[i] >= step_from and steps[i] < step_to]
        vals_l = [length_ma[i] for i in range(n) if steps[i] >= step_from and steps[i] < step_to]
        if not vals_r:
            return None, None
        return mean(vals_r), mean(vals_l) if vals_l else None

    plateau_step = None
    plateau_details = {}

    # Check consecutive windows for stability
    consecutive_stable = 0
    for i in range(1, len(boundaries)):
        prev_w = get_window_stats(boundaries[i - 1], boundaries[i])
        curr_w = get_window_stats(boundaries[i], boundaries[i + 1] if i + 1 < len(boundaries) else total_steps)

        if prev_w[0] is None or curr_w[0] is None:
            continue

        reward_improve = curr_w[0] - prev_w[0]
        length_diff = abs(curr_w[1] - l_final) if curr_w[1] is not None else 0

        reward_stable = reward_improve < epsilon_reward
        length_stable = length_diff < epsilon_length

        if reward_stable and length_stable:
            consecutive_stable += 1
        else:
            consecutive_stable = 0

        if consecutive_stable >= 2:
            plateau_step = boundaries[i - 1]
            plateau_details = {
                "reward_improve": reward_improve,
                "epsilon_reward": epsilon_reward,
                "length_diff": length_diff,
                "epsilon_length": epsilon_length,
                "window": f"{boundaries[i-1]:,}–{boundaries[i]:,}",
            }
            break

    # Summary
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Data points:       {n}")
    print(f"  Moving avg window: {window}")
    print(f"  Window size:       {window_steps:,}")
    print(f"  R_start:           {r_start:.3f}")
    print(f"  R_final:           {r_final:.3f} (±{r_final_sd:.3f})")
    print(f"  L_final:           {l_final:.1f} (±{l_final_sd:.1f})")
    print(f"  epsilon_reward:    {epsilon_reward:.4f}")
    print(f"  epsilon_length:    {epsilon_length:.2f}")

    if plateau_step is not None:
        print(f"  ➜ Plateau at step: {plateau_step:,}")
        print(f"     Reward improvement: {plateau_details['reward_improve']:.4f} < {epsilon_reward:.4f}")
        print(f"     Length diff:        {plateau_details['length_diff']:.2f} < {epsilon_length:.2f}")
        print(f"     Check window:       {plateau_details['window']}")
    else:
        print(f"  ➜ No plateau detected (still improving at {total_steps:,})")

    return plateau_step


def main():
    runs = sorted([d for d in TRAINING_OUTPUTS.iterdir() if d.is_dir()])
    plateau_steps = []

    for run_dir in runs:
        reward_csv = run_dir / "Cumulative Reward.csv"
        length_csv = run_dir / "Episode Length.csv"

        if not reward_csv.exists() or not length_csv.exists():
            print(f"[SKIP] {run_dir.name}: missing CSV files")
            continue

        step = detect_plateau(run_dir.name, reward_csv, length_csv)
        if step is not None:
            plateau_steps.append(step)

    # Final summary
    if plateau_steps:
        sorted_steps = sorted(plateau_steps)
        median_step = sorted_steps[len(sorted_steps) // 2]
        conservative_step = sorted_steps[-1]

        print(f"\n{'═'*60}")
        print(f"PLATEAU DETECTION SUMMARY")
        print(f"{'═'*60}")
        for name, step in zip([d.name for d in runs if d.name in [r.name for r in runs]], plateau_steps):
            print(f"  {name}: {step:,} steps")
        print(f"  {'─'*40}")
        print(f"  Typical plateau step:      {median_step:,}")
        print(f"  Conservative plateau step: {conservative_step:,}")
        print(f"  Range:                     {sorted_steps[0]:,} – {sorted_steps[-1]:,}")
        print(f"{'═'*60}")

    else:
        print("\n[NO DATA] No runs with detected plateaus.")


if __name__ == "__main__":
    main()
