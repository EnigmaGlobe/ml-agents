#!/usr/bin/env python3
"""
Compute v3 learning-improvement metrics from a PushBlock episode CSV.

This is a progressive, low-collinearity revision of the v2 pipeline.

v3 Core Metrics (headline):
  1. Learning Exposure Score (LES)        — trapezoidal AUC of rolling normalized_task_progress
  2. Competence Arrival Step (CAS)        — first sustained crossing of rolling success threshold
  3. Learning Retention Score (LRS)       — peak-to-end drop in rolling progress
  4. Rolling Success Attainment (RSA)     — trapezoidal AUC of rolling success_rate
  5. Rolling Goal-Error Consistency (RGEC) — mean rolling SD of final_goal_zone_error_xz
  6. Rolling Time-to-Goal Efficiency (RTGE) — mean efficiency of successful episodes across training

v2 Diagnostic Metrics (retained for backward compatibility):
  - Final window IQM, success rate, stability verdict
  - Learning slope (OLS)

Design goals:
  - All headline metrics are progressive (computed over the full training trajectory).
  - Spread indicator families across success, goal-error, time-to-goal, and progress
    to reduce collinearity and Heywood-case risk.
  - Output is structured for a 2-dimension CFA (Learning Dynamics + Attainment Consistency).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean


_REQUIRED_COLUMNS = [
    "training_step",
    "success",
    "normalized_task_progress",
    "final_goal_zone_error_xz",
    "time_to_goal",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes(csv_path: Path) -> list[dict]:
    """Load episode rows from learning_improvement CSV.

    Validates required columns, skips rows with empty required fields,
    sorts by (training_step, source_row), and checks training_step monotonicity.
    """
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {csv_path} has no header row")
        headers = [h.strip().lower() for h in reader.fieldnames]
        missing = [c for c in _REQUIRED_COLUMNS if c.lower() not in headers]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {missing}"
            )

        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            if any(cleaned.get(c, "").strip() == "" for c in _REQUIRED_COLUMNS):
                continue
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

    if not rows:
        raise ValueError(f"No valid episode rows loaded from {csv_path}")

    # Sort by training_step, then source_row
    rows.sort(key=lambda r: (r["training_step"], r["source_row"]))
    for i in range(1, len(rows)):
        if rows[i]["training_step"] < rows[i - 1]["training_step"]:
            raise ValueError(
                f"training_step is not monotonic at row {rows[i]['source_row']}: "
                f"{rows[i]['training_step']} < {rows[i - 1]['training_step']}"
            )
    return rows


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------

def compute_rolling_window_size(episode_count: int, override: int | None = None) -> int:
    """Default aligned with FRD recommendation: max(25, ceil(count * 0.01)).

    Override is accepted for sensitivity analysis.
    """
    if override is not None:
        return max(1, override)
    if episode_count <= 0:
        return 0
    return max(25, math.ceil(episode_count * 0.01))


def compute_rolling(episodes: list[dict], window_size: int) -> list[dict]:
    """Compute rolling statistics using a sliding window.

    Aligned with C# BuildRollingProgressSeries: each episode gets one point,
    with a trailing window of size `window_size`.
    """
    rolling = []
    for i in range(len(episodes)):
        start = max(0, i - window_size + 1)
        window_eps = episodes[start:i + 1]

        # Time-to-goal: successful episodes only
        successful_times = [
            e["time_to_goal"] for e in window_eps
            if e["success"] and e["time_to_goal"] >= 0
        ]
        mean_time_to_goal = mean(successful_times) if successful_times else float("nan")

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
            "sd_goal_error": compute_population_stdev(
                [e["final_goal_zone_error_xz"] for e in window_eps]
            ),
            "mean_episode_length": mean(e["episode_length"] for e in window_eps),
            "mean_time_to_goal": mean_time_to_goal,
        })
    return rolling


def compute_population_stdev(values: list[float]) -> float:
    """Population standard deviation (divides by n), aligned with C# ComputeStandardDeviation."""
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# AUC (generic + backward-compat wrapper)
# ---------------------------------------------------------------------------

def compute_auc_generic(xs: list[float], ys: list[float]) -> dict:
    """Trapezoidal AUC of generic y-series vs x-series."""
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return {"raw_auc": 0.0, "normalized_auc": 0.0, "span": 0.0, "points": []}

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


def compute_auc(rolling: list[dict]) -> dict:
    """Trapezoidal AUC of rolling mean_normalized_progress vs training_step."""
    xs = [r["training_step"] for r in rolling]
    ys = [r["mean_normalized_progress"] for r in rolling]
    return compute_auc_generic(xs, ys)


# ---------------------------------------------------------------------------
# v3 Core metric computations
# ---------------------------------------------------------------------------

def compute_competence_arrival(
    rolling: list[dict],
    threshold: float = 0.80,
    sustain: int = 3,
    min_window_size: int | None = None,
) -> dict:
    """First training_step where rolling success_rate >= threshold for `sustain` consecutive windows.

    Returns both the *start* of the sustained period (first crossing)
    and the *confirm* point (after sustain consecutive windows).
    """
    consecutive = 0
    confirm_step = -1
    confirm_episode = -1
    start_step = -1
    start_episode = -1
    min_ws = min_window_size if min_window_size is not None else sustain

    for idx, r in enumerate(rolling):
        # Require full window size to avoid early false positives
        if r.get("episode_count", 0) < min_ws:
            consecutive = 0
            continue

        if r["success_rate"] >= threshold:
            if consecutive == 0:
                start_step = r["training_step"]
                start_episode = idx + 1  # 1-based
            consecutive += 1
            if consecutive >= sustain and confirm_step == -1:
                confirm_step = r["training_step"]
                confirm_episode = idx + 1
        else:
            consecutive = 0
            start_step = -1
            start_episode = -1

    return {
        "competence_arrival_start_step": start_step,
        "competence_arrival_start_episode": start_episode,
        "competence_arrival_confirm_step": confirm_step,
        "competence_arrival_confirm_episode": confirm_episode,
        "competence_arrival_reached": confirm_step != -1,
    }


def compute_retention_drop(rolling: list[dict]) -> dict:
    """Peak-to-end drop in rolling mean_normalized_progress."""
    if not rolling:
        return {
            "best_rolling_progress": 0.0,
            "end_rolling_progress": 0.0,
            "learning_retention_drop": 0.0,
            "learning_retention_score": 1.0,
        }

    rp = [r["mean_normalized_progress"] for r in rolling]
    best = max(rp)
    end = rp[-1]
    drop = best - end
    score = 1.0 - max(0.0, drop)

    return {
        "best_rolling_progress": best,
        "end_rolling_progress": end,
        "learning_retention_drop": drop,
        "learning_retention_score": score,
    }


def compute_goal_error_consistency(rolling: list[dict]) -> dict:
    """Mean rolling SD of final_goal_zone_error_xz.

    Consistency score uses soft normalization so it is unit-invariant:
        score = 1 / (1 + mean_rolling_goal_error_sd)
    """
    if not rolling:
        return {
            "mean_rolling_goal_error_sd": float("nan"),
            "rolling_goal_error_consistency": float("nan"),
        }

    gs = [r["sd_goal_error"] for r in rolling]
    mean_gs = mean(gs)
    # Soft-normalized consistency: 0 -> 0 (maximally inconsistent),
    # inf -> 0, 0 -> 1 (perfectly consistent)
    rgec = 1.0 / (1.0 + mean_gs)

    return {
        "mean_rolling_goal_error_sd": mean_gs,
        "rolling_goal_error_consistency": rgec,
    }


def compute_rolling_time_to_goal_efficiency(
    rolling: list[dict],
    max_episode_step: int | None,
) -> dict:
    """Mean efficiency of successful episodes across the rolling trajectory.

    efficiency_i = 1 - (rolling_mean_time_to_goal_i / max_episode_step)

    max_episode_step must come from training config (CLI --max-episode-step),
    not inferred from observed episode lengths.
    """
    if max_episode_step is None or max_episode_step <= 0:
        return {
            "rolling_time_to_goal_efficiency_score": float("nan"),
            "rolling_mean_time_to_goal": float("nan"),
        }

    effs = []
    valid_rtgs = []
    for r in rolling:
        rtg = r.get("mean_time_to_goal", float("nan"))
        if not math.isnan(rtg):
            eff = 1.0 - (rtg / max_episode_step)
            effs.append(eff)
            valid_rtgs.append(rtg)
        else:
            effs.append(float("nan"))
            valid_rtgs.append(float("nan"))

    clean_effs = [e for e in effs if not math.isnan(e)]
    score = mean(clean_effs) if clean_effs else float("nan")
    clean_rtgs = [r for r in valid_rtgs if not math.isnan(r)]
    mean_rtg = mean(clean_rtgs) if clean_rtgs else float("nan")

    return {
        "rolling_time_to_goal_efficiency_score": score,
        "rolling_mean_time_to_goal": mean_rtg,
    }


# ---------------------------------------------------------------------------
# v3 Verdicts
# ---------------------------------------------------------------------------

def compute_v3_verdicts(
    les: float,
    cas_ratio: float,
    cas_reached: bool,
    lrs: float,
    rsa: float,
    rgec: float,
    rtge: float,
) -> dict:
    """Compute verdict strings for the six v3 headline metrics.

    RGEC thresholds are calibrated against an initial batch of 32 training runs.
    CAS thresholds are provisional and will be calibrated after observing the
    ratio distribution.
    """
    # Learning Exposure Score
    if les >= 0.60:
        les_v = "EXCELLENT"
    elif les >= 0.45:
        les_v = "STRONG"
    elif les >= 0.30:
        les_v = "PASS"
    elif les >= 0.15:
        les_v = "WARN"
    else:
        les_v = "FAIL"

    # Competence Arrival Ratio (lower is better, calibrated on 32 runs)
    # Distribution of reached runs: p25=0.21, p50=0.35, p75=0.39, p90=0.67
    if not cas_reached or math.isnan(cas_ratio):
        cas_v = "FAIL"
    elif cas_ratio <= 0.25:
        cas_v = "EXCELLENT"
    elif cas_ratio <= 0.35:
        cas_v = "STRONG"
    elif cas_ratio <= 0.50:
        cas_v = "PASS"
    elif cas_ratio <= 0.70:
        cas_v = "WARN"
    else:
        cas_v = "FAIL"

    # Learning Retention Score (higher is better)
    if lrs >= 0.95:
        lrs_v = "EXCELLENT"
    elif lrs >= 0.85:
        lrs_v = "STRONG"
    elif lrs >= 0.70:
        lrs_v = "PASS"
    elif lrs >= 0.50:
        lrs_v = "WARN"
    else:
        lrs_v = "FAIL"

    # Rolling Success Attainment
    if rsa >= 0.60:
        rsa_v = "EXCELLENT"
    elif rsa >= 0.45:
        rsa_v = "STRONG"
    elif rsa >= 0.30:
        rsa_v = "PASS"
    elif rsa >= 0.15:
        rsa_v = "WARN"
    else:
        rsa_v = "FAIL"

    # Rolling Goal-Error Consistency (calibrated on 32 runs)
    # raw mean_gs distribution: p10=1.68, p25=2.49, p50=3.65, p75=5.66
    # mapped to soft-normalized score 1/(1+mean_gs)
    if rgec >= 0.35:
        rgec_v = "EXCELLENT"
    elif rgec >= 0.28:
        rgec_v = "STRONG"
    elif rgec >= 0.20:
        rgec_v = "PASS"
    elif rgec >= 0.15:
        rgec_v = "WARN"
    else:
        rgec_v = "FAIL"

    # Rolling Time-to-Goal Efficiency
    if math.isnan(rtge):
        rtge_v = "FAIL"
    elif rtge >= 0.80:
        rtge_v = "EXCELLENT"
    elif rtge >= 0.65:
        rtge_v = "STRONG"
    elif rtge >= 0.50:
        rtge_v = "PASS"
    elif rtge >= 0.30:
        rtge_v = "WARN"
    else:
        rtge_v = "FAIL"

    return {
        "learning_exposure_verdict": les_v,
        "competence_arrival_verdict": cas_v,
        "learning_retention_verdict": lrs_v,
        "rolling_success_attainment_verdict": rsa_v,
        "rolling_goal_error_consistency_verdict": rgec_v,
        "rolling_time_to_goal_efficiency_verdict": rtge_v,
    }


# ---------------------------------------------------------------------------
# v2 helpers (retained for diagnostics)
# ---------------------------------------------------------------------------

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
    """OLS linear regression slope of rolling progress vs training_step."""
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
    """Compute all v2 verdict strings (retained for diagnostic output)."""
    if auc_norm >= 0.60:
        auc_v = "EXCELLENT"
    elif auc_norm >= 0.45:
        auc_v = "STRONG"
    elif auc_norm >= 0.30:
        auc_v = "PASS"
    elif auc_norm >= 0.15:
        auc_v = "WARN"
    else:
        auc_v = "FAIL"

    if final_success >= 0.95 and final_iqm >= 0.90:
        fp_v = "EXCELLENT"
    elif final_success >= 0.90 and final_iqm >= 0.85:
        fp_v = "STRONG"
    elif final_success >= 0.80 and final_iqm >= 0.70:
        fp_v = "PASS"
    elif final_success >= 0.60 or final_iqm >= 0.50:
        fp_v = "WARN"
    else:
        fp_v = "FAIL"

    if slope_per_100k >= 0.05:
        sl_v = "STRONG"
    elif slope_per_100k > 0:
        sl_v = "WEAK"
    elif abs(slope_per_100k) <= 0.0001:
        sl_v = "FLAT"
    else:
        sl_v = "NEGATIVE"

    if auc_v == "FAIL" or fp_v == "FAIL":
        overall = "FAIL"
    elif auc_v == "WARN" or fp_v == "WARN":
        overall = "WARN"
    elif stability_verdict == "HIGH_CONFIDENCE":
        overall = "PASS"
    else:
        overall = "PASS_WITH_WARNING"

    return {
        "auc_verdict": auc_v, "final_performance_verdict": fp_v,
        "learning_slope_verdict": sl_v, "overall_benchmark_verdict": overall,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_all(
    episodes_csv: Path,
    output_csv: Path,
    max_episode_step: int | None = None,
    rolling_window_override: int | None = None,
):
    """Main v3 computation pipeline."""
    episodes = load_episodes(episodes_csv)
    if len(episodes) < 3:
        print(f"[WARN] Only {len(episodes)} episodes, not enough for metrics")
        return

    rolling_size = compute_rolling_window_size(len(episodes), override=rolling_window_override)
    rolling = compute_rolling(episodes, rolling_size)

    # ------------------------------------------------------------------
    # v3 Core Metrics
    # ------------------------------------------------------------------

    # 1. Learning Exposure Score (LES) — AUC on rolling progress
    les_result = compute_auc(rolling)
    les = les_result["normalized_auc"]

    # 2. Competence Arrival Step (CAS) — raw step + ratio
    cas_result = compute_competence_arrival(
        rolling, threshold=0.80, sustain=3, min_window_size=rolling_size
    )
    cas_reached = cas_result["competence_arrival_reached"]
    total_training_steps = rolling[-1]["training_step"] if rolling else 0
    cas_ratio = (
        cas_result["competence_arrival_confirm_step"] / total_training_steps
        if cas_reached and total_training_steps > 0
        else float("nan")
    )

    # 3. Learning Retention Score (LRS)
    lrs_result = compute_retention_drop(rolling)
    lrs = lrs_result["learning_retention_score"]

    # 4. Rolling Success Attainment (RSA) — AUC on rolling success rate
    rsa_result = compute_auc_generic(
        [r["training_step"] for r in rolling],
        [r["success_rate"] for r in rolling],
    )
    rsa = rsa_result["normalized_auc"]

    # 5. Rolling Goal-Error Consistency (RGEC)
    rgec_result = compute_goal_error_consistency(rolling)
    rgec = rgec_result["rolling_goal_error_consistency"]

    # 6. Rolling Time-to-Goal Efficiency (RTGE)
    if max_episode_step is None:
        print("[WARN] --max-episode-step not provided; RTGE will be NaN")
        rtge = float("nan")
        rtge_result = {
            "rolling_time_to_goal_efficiency_score": float("nan"),
            "rolling_mean_time_to_goal": float("nan"),
        }
    else:
        rtge_result = compute_rolling_time_to_goal_efficiency(rolling, max_episode_step)
        rtge = rtge_result["rolling_time_to_goal_efficiency_score"]

    # v3 Verdicts
    v3_verdicts = compute_v3_verdicts(les, cas_ratio, cas_reached, lrs, rsa, rgec, rtge)

    # ------------------------------------------------------------------
    # v2 Diagnostic Metrics (retained for backward compatibility)
    # ------------------------------------------------------------------
    slope = compute_slope(rolling)
    final = compute_final_window(episodes)
    stability = compute_stability(final)
    v2_verdicts = compute_verdicts(
        les, final["success_rate"], final["iqm_progress"],
        stability["verdict"], slope["per_100k"],
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Section 0: v3 Summary
        writer.writerow(["=== V3 SUMMARY ==="])
        writer.writerow(["metric", "value", "verdict"])
        writer.writerow(["Learning Exposure Score", f"{les:.4f}", v3_verdicts["learning_exposure_verdict"]])
        writer.writerow(["Competence Arrival Ratio", f"{cas_ratio:.4f}" if cas_reached else "NEVER", v3_verdicts["competence_arrival_verdict"]])
        writer.writerow(["Learning Retention Score", f"{lrs:.4f}", v3_verdicts["learning_retention_verdict"]])
        writer.writerow(["Rolling Success Attainment", f"{rsa:.4f}", v3_verdicts["rolling_success_attainment_verdict"]])
        writer.writerow(["Rolling Goal-Error Consistency", f"{rgec:.4f}", v3_verdicts["rolling_goal_error_consistency_verdict"]])
        writer.writerow(["Rolling Time-to-Goal Efficiency",
                         f"{rtge:.4f}" if not math.isnan(rtge) else "N/A",
                         v3_verdicts["rolling_time_to_goal_efficiency_verdict"]])
        writer.writerow(["Episodes", len(episodes), ""])
        writer.writerow(["Rolling window size", rolling_size, ""])
        writer.writerow([])

        # Section 1: Learning Exposure Detail
        writer.writerow(["=== LEARNING EXPOSURE (AUC) ==="])
        writer.writerow(["raw_auc", "normalized_auc", "span"])
        writer.writerow([f"{les_result['raw_auc']:.6f}", f"{les_result['normalized_auc']:.6f}", f"{les_result['span']:.6f}"])
        writer.writerow([])
        if les_result["points"]:
            writer.writerow(["step_from", "step_to", "segment_auc", "cumulative_auc"])
            for pt in les_result["points"]:
                writer.writerow([pt["step_from"], pt["step_to"],
                                 f"{pt['segment_auc']:.6f}", f"{pt['cumulative_auc']:.6f}"])
            writer.writerow([])

        # Section 2: Competence Arrival Detail
        writer.writerow(["=== COMPETENCE ARRIVAL ==="])
        writer.writerow(["competence_arrival_start_step", "competence_arrival_start_episode",
                         "competence_arrival_confirm_step", "competence_arrival_confirm_episode",
                         "competence_arrival_reached"])
        writer.writerow([cas_result["competence_arrival_start_step"],
                         cas_result["competence_arrival_start_episode"],
                         cas_result["competence_arrival_confirm_step"],
                         cas_result["competence_arrival_confirm_episode"],
                         cas_result["competence_arrival_reached"]])
        writer.writerow([])

        # Section 3: Learning Retention Detail
        writer.writerow(["=== LEARNING RETENTION ==="])
        writer.writerow(["best_rolling_progress", "end_rolling_progress", "retention_drop", "retention_score"])
        writer.writerow([f"{lrs_result['best_rolling_progress']:.6f}",
                         f"{lrs_result['end_rolling_progress']:.6f}",
                         f"{lrs_result['learning_retention_drop']:.6f}",
                         f"{lrs_result['learning_retention_score']:.6f}"])
        writer.writerow([])

        # Section 4: Rolling Success Attainment Detail
        writer.writerow(["=== ROLLING SUCCESS ATTAINMENT (AUC) ==="])
        writer.writerow(["raw_auc", "normalized_auc", "span"])
        writer.writerow([f"{rsa_result['raw_auc']:.6f}", f"{rsa_result['normalized_auc']:.6f}", f"{rsa_result['span']:.6f}"])
        writer.writerow([])
        if rsa_result["points"]:
            writer.writerow(["step_from", "step_to", "segment_auc", "cumulative_auc"])
            for pt in rsa_result["points"]:
                writer.writerow([pt["step_from"], pt["step_to"],
                                 f"{pt['segment_auc']:.6f}", f"{pt['cumulative_auc']:.6f}"])
            writer.writerow([])

        # Section 5: Rolling Goal-Error Consistency Detail
        writer.writerow(["=== ROLLING GOAL-ERROR CONSISTENCY ==="])
        writer.writerow(["mean_rolling_goal_error_sd", "rolling_goal_error_consistency"])
        writer.writerow([f"{rgec_result['mean_rolling_goal_error_sd']:.6f}",
                         f"{rgec_result['rolling_goal_error_consistency']:.6f}"])
        writer.writerow([])

        # Section 6: Rolling Time-to-Goal Efficiency Detail
        writer.writerow(["=== ROLLING TIME-TO-GOAL EFFICIENCY ==="])
        writer.writerow(["rolling_mean_time_to_goal", "efficiency_score", "max_episode_step"])
        writer.writerow([f"{rtge_result['rolling_mean_time_to_goal']:.4f}" if not math.isnan(rtge_result["rolling_mean_time_to_goal"]) else "N/A",
                         f"{rtge_result['rolling_time_to_goal_efficiency_score']:.4f}" if not math.isnan(rtge_result["rolling_time_to_goal_efficiency_score"]) else "N/A",
                         max_episode_step if max_episode_step is not None else "N/A"])
        writer.writerow([])

        # Section 7: v3 Rolling Window Detail
        writer.writerow(["=== ROLLING WINDOW DETAIL ==="])
        writer.writerow(["window_start_episode", "window_end_episode", "training_step",
                         "episode_count", "mean_reward", "success_rate",
                         "mean_normalized_progress", "std_normalized_progress",
                         "mean_goal_error", "sd_goal_error", "mean_time_to_goal", "mean_episode_length"])
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
                f"{r['sd_goal_error']:.4f}",
                f"{r['mean_time_to_goal']:.4f}" if not math.isnan(r["mean_time_to_goal"]) else "N/A",
                f"{r['mean_episode_length']:.1f}",
            ])
        writer.writerow([])

        # Section 8: v3 CFA-ready Key-Value Summary
        writer.writerow(["=== V3 CFA SUMMARY (KEY-VALUE) ==="])
        cfa_rows = [
            ["learning_exposure_score", f"{les:.6f}"],
            ["learning_exposure_verdict", v3_verdicts["learning_exposure_verdict"]],
            ["competence_arrival_start_step", str(cas_result["competence_arrival_start_step"])],
            ["competence_arrival_confirm_step", str(cas_result["competence_arrival_confirm_step"])],
            ["competence_arrival_ratio", f"{cas_ratio:.6f}" if not math.isnan(cas_ratio) else "NaN"],
            ["competence_arrival_reached", str(cas_reached)],
            ["competence_arrival_verdict", v3_verdicts["competence_arrival_verdict"]],
            ["learning_retention_score", f"{lrs:.6f}"],
            ["learning_retention_verdict", v3_verdicts["learning_retention_verdict"]],
            ["rolling_success_attainment", f"{rsa:.6f}"],
            ["rolling_success_attainment_verdict", v3_verdicts["rolling_success_attainment_verdict"]],
            ["mean_rolling_goal_error_sd", f"{rgec_result['mean_rolling_goal_error_sd']:.6f}"],
            ["rolling_goal_error_consistency", f"{rgec_result['rolling_goal_error_consistency']:.6f}"],
            ["rolling_goal_error_consistency_verdict", v3_verdicts["rolling_goal_error_consistency_verdict"]],
            ["rolling_time_to_goal_efficiency_score", f"{rtge:.6f}" if not math.isnan(rtge) else "NaN"],
            ["rolling_time_to_goal_efficiency_verdict", v3_verdicts["rolling_time_to_goal_efficiency_verdict"]],
            ["rolling_window_size", str(rolling_size)],
            ["episodes", str(len(episodes))],
        ]
        for kv in cfa_rows:
            writer.writerow(kv)
        writer.writerow([])

        # Section 9: v2 Diagnostics (Final Window & Stability)
        writer.writerow(["=== DIAGNOSTIC: FINAL WINDOW & STABILITY (v2) ==="])
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

        # Section 10: v2 Diagnostics (Learning Slope)
        writer.writerow(["=== DIAGNOSTIC: LEARNING SLOPE (v2) ==="])
        writer.writerow(["slope", "intercept", "r_squared", "p_value",
                         "slope_per_100k_steps", "n_points"])
        writer.writerow([f"{slope['slope']:.8e}", f"{slope['intercept']:.6f}",
                         f"{slope['r_squared']:.6f}", f"{slope['p_value']:.6f}",
                         f"{slope['per_100k']:.8e}", slope["n_points"]])
        writer.writerow([])

        # Section 11: v2 Diagnostics (Verdicts Summary)
        writer.writerow(["=== DIAGNOSTIC: V2 VERDICTS SUMMARY ==="])
        writer.writerow(["auc_verdict", "final_performance_verdict",
                         "stability_verdict", "learning_slope_verdict",
                         "overall_benchmark_verdict"])
        writer.writerow([v2_verdicts["auc_verdict"], v2_verdicts["final_performance_verdict"],
                         stability["verdict"], v2_verdicts["learning_slope_verdict"],
                         v2_verdicts["overall_benchmark_verdict"]])

    print(f"[OK] v3 learning-improvement metrics written to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute v3 learning-improvement metrics from a PushBlock episode CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Input episode CSV")
    parser.add_argument("output_csv", type=Path, help="Output metrics CSV")
    parser.add_argument(
        "--max-episode-step", type=int, default=None,
        help="Episode MaxStep from training config (required for RTGE)"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=None,
        help="Override rolling window size (default: max(25, ceil(1% of episodes)))"
    )
    args = parser.parse_args()
    compute_all(
        args.input_csv, args.output_csv,
        max_episode_step=args.max_episode_step,
        rolling_window_override=args.rolling_window,
    )
