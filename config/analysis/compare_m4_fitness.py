#!/usr/bin/env python3
"""
Compare episode utility between two Curriculum EC runs (e.g., M4 fitness off vs. on).

Computes the CFA-adjusted training utility bundle:
    final_20pct_progress, retained_progress_ratio,
    progress_weighted_path_efficiency, progress_per_action

training_utility_score = mean(z_score of each indicator)

Usage:
    python compare_m4_fitness.py \
        --before path/to/before/curriculum_outputs/episodes_*.csv \
        --after  path/to/after/curriculum_outputs/episodes_*.csv \
        --output comparison_result.csv

The script can also take --before-dir and --after-dir and will auto-pick the latest episodes_*.csv.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


UTILITY_COLS = [
    "final_20pct_progress",
    "retained_progress_ratio",
    "progress_weighted_path_efficiency",
    "progress_per_action",
]


def find_latest_episodes_csv(directory):
    """Find the most recent episodes_*.csv in a curriculum_outputs directory."""
    pattern = os.path.join(directory, "episodes_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No episodes_*.csv found in {directory}")
    return max(files, key=os.path.getmtime)


def load_episodes(path):
    """Load an episodes CSV and ensure utility columns exist."""
    df = pd.read_csv(path)
    for col in UTILITY_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def compute_utility_score(df, reference_df=None):
    """
    Compute training_utility_score as the mean of z-scored indicators.

    If reference_df is provided, z-scores are computed on the reference
    and applied to df. Otherwise z-scores are computed within df.
    """
    ref = reference_df if reference_df is not None else df
    z_scores = []
    for col in UTILITY_COLS:
        mean = ref[col].mean()
        std = ref[col].std(ddof=0)
        if std < 1e-9:
            z = pd.Series(0.0, index=df.index)
        else:
            z = (df[col] - mean) / std
        z_scores.append(z)
    df = df.copy()
    df["training_utility_score"] = pd.concat(z_scores, axis=1).mean(axis=1)
    return df


def add_top50_split(df):
    """Split episodes into top_50 / bottom_50 by median utility score."""
    median = df["training_utility_score"].median()
    df = df.copy()
    df["utility_split"] = df["training_utility_score"].apply(
        lambda x: "top_50" if x >= median else "bottom_50"
    )
    return df


def iqm(series):
    """Interquartile mean (mean of values between Q1 and Q3)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    mask = (series >= q1) & (series <= q3)
    if mask.sum() == 0:
        return series.mean()
    return series[mask].mean()


def summarize(df, label):
    """Compute summary statistics for one condition."""
    n_selected = df[["generation", "genome_index"]].drop_duplicates().shape[0]

    return {
        "condition": label,
        "selected_genomes": n_selected,
        "episodes": len(df),
        "success_rate": df["success"].mean(),
        "mean_utility": df["training_utility_score"].mean(),
        "iqm_utility": iqm(df["training_utility_score"]),
        "sd_utility": df["training_utility_score"].std(ddof=0),
        "progress_efficiency": df["progress_weighted_path_efficiency"].mean(),
        "m4_spatial_score": df["task_centroid_centrality"].mean(),
        "stability": 1.0 / (1.0 + df["training_utility_score"].std(ddof=0)),
        "top_50_pct": (df["utility_split"] == "top_50").mean(),
    }


def compare(before_df, after_df):
    """Run statistical comparisons and build the comparison table."""
    # Z-score both against the combined distribution so scores are comparable
    combined = pd.concat([before_df, after_df], ignore_index=True)
    before_df = compute_utility_score(before_df, reference_df=combined)
    after_df = compute_utility_score(after_df, reference_df=combined)

    before_df = add_top50_split(before_df)
    after_df = add_top50_split(after_df)

    before_summary = summarize(before_df, "before M4 fitness")
    after_summary = summarize(after_df, "after M4 fitness")

    # Welch t-test on utility scores
    t_stat, p_value = stats.ttest_ind(
        after_df["training_utility_score"],
        before_df["training_utility_score"],
        equal_var=False,
        nan_policy="omit",
    )

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(
        after_df["training_utility_score"],
        before_df["training_utility_score"],
        alternative="two-sided",
    )

    comparison = pd.DataFrame([before_summary, after_summary]).set_index("condition").T
    comparison["delta"] = comparison["after M4 fitness"] - comparison["before M4 fitness"]
    comparison["pct_change"] = (
        comparison["delta"] / comparison["before M4 fitness"].replace(0, np.nan)
    ) * 100.0

    stats_text = f"""
Statistical tests on training_utility_score:
  Welch t-test:     t = {t_stat:.4f}, p = {p_value:.4f}
  Mann-Whitney U:   U = {u_stat:.1f}, p = {u_pvalue:.4f}
"""

    return comparison, before_df, after_df, stats_text


def main():
    parser = argparse.ArgumentParser(
        description="Compare episode utility before/after M4 fitness integration."
    )
    parser.add_argument(
        "--before",
        type=str,
        help="Path to episodes CSV for the baseline (M4 fitness off) run.",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Path to episodes CSV for the M4 fitness on run.",
    )
    parser.add_argument(
        "--before-dir",
        type=str,
        help="Directory containing baseline curriculum_outputs (auto-picks latest episodes CSV).",
    )
    parser.add_argument(
        "--after-dir",
        type=str,
        help="Directory containing M4 curriculum_outputs (auto-picks latest episodes CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="m4_fitness_comparison.csv",
        help="Output CSV path for the comparison table.",
    )
    parser.add_argument(
        "--detail-output",
        type=str,
        default=None,
        help="Optional output CSV path for per-episode utility scores.",
    )
    args = parser.parse_args()

    before_path = args.before
    after_path = args.after
    if args.before_dir:
        before_path = find_latest_episodes_csv(args.before_dir)
    if args.after_dir:
        after_path = find_latest_episodes_csv(args.after_dir)

    if not before_path or not after_path:
        raise ValueError(
            "Please provide both --before/--after (or --before-dir/--after-dir)."
        )

    print(f"Before: {before_path}")
    print(f"After:  {after_path}")

    before_df = load_episodes(before_path)
    after_df = load_episodes(after_path)

    comparison, before_utility, after_utility, stats_text = compare(before_df, after_df)

    print("\n=== Comparison Table ===")
    print(comparison.to_string())
    print(stats_text)

    comparison.to_csv(args.output)
    print(f"Comparison table saved to: {args.output}")

    if args.detail_output:
        before_utility["condition"] = "before"
        after_utility["condition"] = "after"
        detail = pd.concat([before_utility, after_utility], ignore_index=True)
        detail.to_csv(args.detail_output, index=False)
        print(f"Per-episode detail saved to: {args.detail_output}")


if __name__ == "__main__":
    main()
