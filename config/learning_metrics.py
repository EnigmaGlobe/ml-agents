#!/usr/bin/env python3
"""
Compute four learning-improvement metrics from a PushBlock episode CSV.

Produces a detailed CSV with per-window/per-episode computation data
(not just a one-row summary).

Metrics:
  1. AUC — trapezoidal area under rolling normalized_task_progress
  2. Final Performance — last 10% window: IQM, success rate, bootstrap CI
  3. Stability — SD, IQR, CV of normalized_task_progress in final window
  4. Learning Slope — OLS slope of rolling progress vs training_step

Computation logic is aligned with TrainingCheckerWindow.cs (Unity Editor).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean


def load_episodes(csv_path: Path) -> list[dict]:
    """Load episode rows from learning_improvement CSV."""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            try:
                rows.append({
                    "source_row": int(cleaned.get("episode_id", len(rows) + 1)),
                    "training_step": int(float(cleaned.get("training_step", 0))),
                    "success": int(cleaned.get("success", 0)) == 1,
                    "end_reason": cleaned.get("end_reason", "unknown"),
                    "episode_reward": float(cleaned.get("episode_reward", 0)),
                    "episode_length": int(float(cleaned.get("episode_length", 0))),
                    "time_to_goal": float(cleaned.get("time_to_goal", "-1")),
                    "normalized_task_progress": float(cleaned.get("normalized_task_progress", 0)),
                    "final_goal_zone_error_xz": float(cleaned.get("final_goal_zone_error_xz", 0)),
                    "start_block_goal_distance": float(cleaned.get("start_block_goal_distance", 0)),
                    "final_block_goal_distance": float(cleaned.get("final_block_goal_distance", 0)),
                    "normalized_block_progress": float(cleaned.get("normalized_block_progress", 0)),
                })
            except (ValueError, TypeError):
                continue
    return sorted(rows, key=lambda r: r["source_row"])


def compute_rolling_window_size(episode_count: int) -> int:
    """Aligned with C# ComputeRollingWindowSize."""
    if episode_count <= 0:
        return 0
    if episode_count < 50:
        return max(5, episode_count // 10)
    return 50


def compute_rolling(episodes: list[dict], window_size: int) -> list[dict]:
    """Compute rolling normalized_task_progress using a sliding window.

    Aligned with C# BuildRollingProgressSeries: each episode gets one point,
    with a trailing window of size `window_size`.
    """
    rolling = []
    for i in range(len(episodes)):
        start = max(0, i - window_size + 1)
        window_eps = episodes[start:i + 1]
        rolling.append({
            "training_step": episodes[i]["training_step"],
            "episode_count": len(window_eps),
            "mean_reward": mean(e["episode_reward"] for e in window_eps),
            "success_rate": sum(1 for e in window_eps if e["success"]) / len(window_eps),
            "mean_normalized_progress": mean(e["normalized_task_progress"] for e in window_eps),
            "std_normalized_progress": compute_population_stdev(
                [e["normalized_task_progress"] for e in window_eps]
            ),
            "mean_goal_error": mean(e["final_goal_zone_error_xz"] for e in window_eps),
            "mean_episode_length": mean(e["episode_length"] for e in window_eps),
        })
    return rolling


def compute_population_stdev(values: list[float]) -> float:
    """Population standard deviation (divides by n), aligned with C# ComputeStandardDeviation."""
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def compute_auc(rolling: list[dict]) -> dict:
    """Trapezoidal AUC of rolling mean_normalized_progress vs training_step."""
    if len(rolling) < 2:
        return {"raw_auc": 0.0, "normalized_auc": 0.0, "span": 0.0, "points": []}

    xs = [r["training_step"] for r in rolling]
    ys = [r["mean_normalized_progress"] for r in rolling]

    raw_auc = 0.0
    points = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        avg_y = (ys[i] + ys[i - 1]) / 2.0
        segment = dx * avg_y
        raw_auc += segment
        points.append({
            "step_from": xs[i - 1],
            "step_to": xs[i],
            "segment_auc": segment,
            "cumulative_auc": raw_auc,
        })

    span = xs[-1] - xs[0]
    normalized_auc = raw_auc / abs(span) if abs(span) > 0 else 0.0

    return {"raw_auc": raw_auc, "normalized_auc": normalized_auc, "span": span, "points": points}


def erf(x: float) -> float:
    """Approximation of erf(x), aligned with C# Erf."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def normal_cdf(x: float) -> float:
    """Aligned with C# NormalCDF."""
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def compute_slope(rolling: list[dict]) -> dict:
    """OLS linear regression slope of rolling progress vs training_step.

    Aligned with C# ComputeSlopeStats.
    """
    if len(rolling) < 3:
        return {
            "slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
            "p_value": 1.0, "per_100k": 0.0, "n_points": len(rolling),
        }

    xs = [float(r["training_step"]) for r in rolling]
    ys = [r["mean_normalized_progress"] for r in rolling]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)

    if abs(ss_xx) < 1e-15:
        return {
            "slope": 0.0, "intercept": mean_y, "r_squared": 0.0,
            "p_value": 1.0, "per_100k": 0.0, "n_points": n,
        }

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-15 else 0.0

    # Residual sum of squares
    rss = sum((ys[i] - (intercept + slope * xs[i])) ** 2 for i in range(n))
    standard_error = math.sqrt(rss / (n - 2) / ss_xx) if n > 2 and ss_xx != 0 else float("nan")

    if not math.isnan(standard_error) and standard_error > 1e-15:
        t_stat = slope / standard_error
        abs_t = abs(t_stat)
        p_value = 2.0 * (1.0 - normal_cdf(abs_t))
        p_value = max(0.0, min(1.0, p_value))
    elif math.isnan(standard_error) or standard_error <= 1e-15:
        p_value = 1.0 if abs(slope) <= 1e-15 else 0.0
    else:
        p_value = 1.0

    per_100k = slope * 100_000

    return {
        "slope": slope, "intercept": intercept, "r_squared": r_squared,
        "p_value": p_value, "per_100k": per_100k, "n_points": n,
    }


def compute_final_window_size(episode_count: int) -> int:
    """Aligned with C# ComputeFinalWindowSize: max(25, ceil(count * 0.10))."""
    if episode_count <= 0:
        return 0
    return max(25, math.ceil(episode_count * 0.10))


def compute_percentile(values: list[float], percentile: float) -> float:
    """Aligned with C# ComputePercentile: linear interpolation."""
    if not values:
        return float("nan")
    sorted_vals = sorted(v for v in values if not math.isnan(v))
    if not sorted_vals:
        return float("nan")
    clamped = max(0.0, min(1.0, percentile))
    position = (len(sorted_vals) - 1) * clamped
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_vals[lower]
    weight = position - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def compute_median(values: list[float]) -> float:
    """Aligned with C# ComputeMedian."""
    if not values:
        return float("nan")
    sorted_vals = sorted(v for v in values if not math.isnan(v))
    if not sorted_vals:
        return float("nan")
    middle = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[middle]
    return (sorted_vals[middle - 1] + sorted_vals[middle]) / 2.0


def compute_iqm(values: list[float]) -> float:
    """Interquartile mean, aligned with C# ComputeIQM."""
    if not values:
        return float("nan")
    sorted_vals = sorted(v for v in values if not math.isnan(v))
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) < 4:
        return mean(sorted_vals)

    q1_index = int(math.floor(len(sorted_vals) * 0.25))
    q3_index = int(math.ceil(len(sorted_vals) * 0.75))
    q1_index = max(0, min(q1_index, len(sorted_vals) - 1))
    q3_index = max(q1_index + 1, min(q3_index, len(sorted_vals)))

    middle = sorted_vals[q1_index:q3_index]
    return mean(middle) if middle else mean(sorted_vals)


def compute_bootstrap_ci(values: list[float], iterations: int = 10000, confidence_level: float = 0.95) -> dict:
    """Bootstrap CI for IQM, aligned with C# ComputeBootstrapCI."""
    cleaned = [v for v in values if not math.isnan(v)]
    if not cleaned:
        return {"lower": float("nan"), "upper": float("nan")}
    if len(cleaned) < 3:
        iqm = compute_iqm(cleaned)
        return {"lower": iqm, "upper": iqm}

    rng = random.Random(42)
    bootstrap_means = []
    for _ in range(iterations):
        sample = [cleaned[rng.randint(0, len(cleaned) - 1)] for _ in range(len(cleaned))]
        bootstrap_means.append(compute_iqm(sample))

    bootstrap_means.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, int(math.floor(alpha / 2.0 * (len(bootstrap_means) - 1))))
    upper_index = min(len(bootstrap_means) - 1, int(math.ceil((1.0 - alpha / 2.0) * (len(bootstrap_means) - 1))))

    return {"lower": bootstrap_means[lower_index], "upper": bootstrap_means[upper_index]}


def compute_final_window(episodes: list[dict]) -> dict:
    """Compute metrics for the final window (last 10% of episodes)."""
    window_size = compute_final_window_size(len(episodes))
    window = episodes[-window_size:]

    progress = [e["normalized_task_progress"] for e in window]
    rewards = [e["episode_reward"] for e in window]
    errors = [e["final_goal_zone_error_xz"] for e in window]
    successes = [1 if e["success"] else 0 for e in window]

    iqm = compute_iqm(progress)
    ci = compute_bootstrap_ci(progress)

    sd = compute_population_stdev(progress)
    q1 = compute_percentile(progress, 0.25)
    q3 = compute_percentile(progress, 0.75)
    median_val = compute_median(progress)
    cv = sd / abs(mean(progress)) if abs(mean(progress)) > 1e-15 else float("nan")

    return {
        "window_size": len(window),
        "start_episode": window[0]["source_row"],
        "end_episode": window[-1]["source_row"],
        "start_step": window[0]["training_step"],
        "end_step": window[-1]["training_step"],
        "mean_progress": mean(progress),
        "median_progress": median_val,
        "iqm_progress": iqm,
        "min_progress": min(progress),
        "max_progress": max(progress),
        "q1_progress": q1,
        "q3_progress": q3,
        "iqr_progress": q3 - q1,
        "sd_progress": sd,
        "cv_progress": cv,
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
        "mean_reward": mean(rewards),
        "mean_goal_error": mean(errors),
        "success_rate": mean(successes),
    }


def compute_stability(final_window: dict) -> dict:
    """Stability verdict based on final window statistics."""
    iqr = final_window["iqr_progress"]
    sd = final_window["sd_progress"]
    cv = final_window["cv_progress"]

    if iqr <= 0.10 and (sd > 0.50 or cv > 1.00):
        verdict, confidence = "OUTLIER_WARNING", "MEDIUM"
    elif iqr <= 0.10 and sd <= 0.25:
        verdict, confidence = "HIGH_CONFIDENCE", "HIGH"
    elif iqr <= 0.25:
        verdict, confidence = "MEDIUM_CONFIDENCE", "MEDIUM"
    elif iqr > 0.50 or cv > 1.00:
        verdict, confidence = "LOW_CONFIDENCE", "LOW"
    else:
        verdict, confidence = "UNSTABLE_FAIL", "LOW"

    return {"verdict": verdict, "confidence": confidence, **final_window}


def compute_verdicts(auc_norm: float, final_success: float, final_iqm: float,
                     stability_verdict: str, slope_per_100k: float) -> dict:
    """Compute all verdict strings."""
    # AUC
    if auc_norm >= 0.60: auc_v = "EXCELLENT"
    elif auc_norm >= 0.45: auc_v = "STRONG"
    elif auc_norm >= 0.30: auc_v = "PASS"
    elif auc_norm >= 0.15: auc_v = "WARN"
    else: auc_v = "FAIL"

    # Final performance
    if final_success >= 0.95 and final_iqm >= 0.90: fp_v = "EXCELLENT"
    elif final_success >= 0.90 and final_iqm >= 0.85: fp_v = "STRONG"
    elif final_success >= 0.80 and final_iqm >= 0.70: fp_v = "PASS"
    elif final_success >= 0.60 or final_iqm >= 0.50: fp_v = "WARN"
    else: fp_v = "FAIL"

    # Slope
    if slope_per_100k >= 0.05: sl_v = "STRONG"
    elif slope_per_100k > 0: sl_v = "WEAK"
    elif abs(slope_per_100k) <= 0.0001: sl_v = "FLAT"
    else: sl_v = "NEGATIVE"

    # Overall
    if auc_v == "FAIL" or fp_v == "FAIL": overall = "FAIL"
    elif auc_v == "WARN" or fp_v == "WARN": overall = "WARN"
    elif stability_verdict == "HIGH_CONFIDENCE": overall = "PASS"
    else: overall = "PASS_WITH_WARNING"

    return {
        "auc_verdict": auc_v, "final_performance_verdict": fp_v,
        "learning_slope_verdict": sl_v, "overall_benchmark_verdict": overall,
    }


def compute_all(episodes_csv: Path, output_csv: Path):
    """Main computation pipeline."""
    episodes = load_episodes(episodes_csv)
    if len(episodes) < 3:
        print(f"[WARN] Only {len(episodes)} episodes, not enough for metrics")
        return

    # Rolling window — sliding window, aligned with C#
    rolling_size = compute_rolling_window_size(len(episodes))
    rolling = compute_rolling(episodes, rolling_size)

    # AUC
    auc = compute_auc(rolling)

    # Slope
    slope = compute_slope(rolling)

    # Final window
    final = compute_final_window(episodes)

    # Stability
    stability = compute_stability(final)

    # Verdicts
    verdicts = compute_verdicts(
        auc["normalized_auc"], final["success_rate"],
        final["iqm_progress"], stability["verdict"], slope["per_100k"]
    )

    # Write detailed CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Section 0: Summary at the top
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["metric", "value", "verdict"])
        writer.writerow(["AUC (raw)", f"{auc['raw_auc']:.4f}", verdicts["auc_verdict"]])
        writer.writerow(["AUC (normalized)", f"{auc['normalized_auc']:.4f}", ""])
        writer.writerow(["Final IQM progress", f"{final['iqm_progress']:.4f}", verdicts["final_performance_verdict"]])
        writer.writerow(["Final success rate", f"{final['success_rate']:.4f}", ""])
        writer.writerow(["Stability IQR", f"{final['iqr_progress']:.4f}", stability["verdict"]])
        writer.writerow(["Stability SD", f"{final['sd_progress']:.4f}", ""])
        writer.writerow(["Stability CV", f"{final['cv_progress']:.4f}", ""])
        writer.writerow(["Learning slope (per 100k)", f"{slope['per_100k']:.6f}", verdicts["learning_slope_verdict"]])
        writer.writerow(["Overall benchmark", "", verdicts["overall_benchmark_verdict"]])
        writer.writerow(["Confidence", "", stability["confidence"]])
        writer.writerow(["Episodes", len(episodes), ""])
        writer.writerow(["Rolling window size", rolling_size, ""])
        writer.writerow(["Final window size", final["window_size"], ""])
        writer.writerow([])

        # Section 1: AUC detail
        writer.writerow(["=== AUC COMPUTATION ==="])
        writer.writerow(["raw_auc", "normalized_auc", "span"])
        writer.writerow([f"{auc['raw_auc']:.6f}", f"{auc['normalized_auc']:.6f}", f"{auc['span']:.6f}"])
        writer.writerow([])

        if auc["points"]:
            writer.writerow(["step_from", "step_to", "segment_auc", "cumulative_auc"])
            for pt in auc["points"]:
                writer.writerow([pt["step_from"], pt["step_to"],
                                 f"{pt['segment_auc']:.6f}",
                                 f"{pt['cumulative_auc']:.6f}"])
            writer.writerow([])

        # Section 2: Rolling window detail
        writer.writerow(["=== ROLLING WINDOW DETAIL ==="])
        writer.writerow(["window_start_episode", "window_end_episode", "training_step",
                          "episode_count", "mean_reward", "success_rate",
                          "mean_normalized_progress", "std_normalized_progress",
                          "mean_goal_error", "mean_episode_length"])
        for i, r in enumerate(rolling):
            start_idx = max(0, i - rolling_size + 1)
            writer.writerow([
                episodes[start_idx]["source_row"],
                episodes[i]["source_row"],
                r["training_step"], r["episode_count"],
                f"{r['mean_reward']:.4f}", f"{r['success_rate']:.4f}",
                f"{r['mean_normalized_progress']:.4f}",
                f"{r['std_normalized_progress']:.4f}",
                f"{r['mean_goal_error']:.4f}",
                f"{r['mean_episode_length']:.1f}"
            ])
        writer.writerow([])

        # Section 3: Final Window & Stability
        writer.writerow(["=== FINAL WINDOW & STABILITY ==="])
        writer.writerow(["window_size", "start_episode", "end_episode",
                          "start_step", "end_step"])
        writer.writerow([final["window_size"], final["start_episode"], final["end_episode"],
                          final["start_step"], final["end_step"]])
        writer.writerow([])
        writer.writerow(["mean_progress", "median_progress", "iqm_progress",
                          "min_progress", "max_progress", "q1_progress", "q3_progress",
                          "iqr_progress", "sd_progress", "cv_progress",
                          "ci_lower", "ci_upper",
                          "mean_reward", "mean_goal_error", "success_rate"])
        writer.writerow([f"{final['mean_progress']:.4f}", f"{final['median_progress']:.4f}",
                          f"{final['iqm_progress']:.4f}", f"{final['min_progress']:.4f}",
                          f"{final['max_progress']:.4f}", f"{final['q1_progress']:.4f}",
                          f"{final['q3_progress']:.4f}", f"{final['iqr_progress']:.4f}",
                          f"{final['sd_progress']:.4f}", f"{final['cv_progress']:.4f}",
                          f"{final['ci_lower']:.4f}", f"{final['ci_upper']:.4f}",
                          f"{final['mean_reward']:.4f}", f"{final['mean_goal_error']:.4f}",
                          f"{final['success_rate']:.4f}"])
        writer.writerow([])
        writer.writerow(["stability_verdict", "confidence"])
        writer.writerow([stability["verdict"], stability["confidence"]])
        writer.writerow([])

        # Section 4: Learning Slope
        writer.writerow(["=== LEARNING SLOPE (OLS) ==="])
        writer.writerow(["slope", "intercept", "r_squared", "p_value",
                          "slope_per_100k_steps", "n_points"])
        writer.writerow([f"{slope['slope']:.8e}", f"{slope['intercept']:.6f}",
                          f"{slope['r_squared']:.6f}", f"{slope['p_value']:.6f}",
                          f"{slope['per_100k']:.8e}", slope["n_points"]])
        writer.writerow([])

        # Section 5: Verdicts Summary
        writer.writerow(["=== VERDICTS SUMMARY ==="])
        writer.writerow(["auc_verdict", "final_performance_verdict",
                          "stability_verdict", "learning_slope_verdict",
                          "overall_benchmark_verdict"])
        writer.writerow([verdicts["auc_verdict"], verdicts["final_performance_verdict"],
                          stability["verdict"], verdicts["learning_slope_verdict"],
                          verdicts["overall_benchmark_verdict"]])

    print(f"[OK] Detailed learning-improvement metrics written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: learning_metrics.py <input_csv> <output_csv>")
        sys.exit(1)
    compute_all(Path(sys.argv[1]), Path(sys.argv[2]))
