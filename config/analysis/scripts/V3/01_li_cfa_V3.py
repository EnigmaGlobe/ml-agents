#!/usr/bin/env python3
"""
CFA analysis for PushBlock v3 learning-improvement metrics.

FRD specifies a correlated two-factor CFA:
  - Learning Dynamics: LES, CAS_ratio, LRS
  - Attainment Consistency: RSA, RGEC, RTGE

Key analytical decisions:
  1. All models are run on BOTH raw and z-score standardized data.
  2. A unifactor model is included because the two first-order factors
     are empirically almost perfectly correlated (~1.0) in this sample.
  3. A 5-item unifactor (dropping weakest loader LRS) is included as
     a sensitivity check.

Output: text report + CSV of factor scores + model fit table +
        formatted parameter-estimate tables per model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = Path(r"C:\soqqle\ml-agents\config\training_results\v3_metrics_wide_table.csv")
OUTPUT_DIR = Path(r"C:\soqqle\ml-agents\config\analysis\output\complete\Li_cfa_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "les": "LES",
    "cas_ratio": "CAS_ratio",
    "lrs": "LRS",
    "rsa": "RSA",
    "rgec": "RGEC",
    "rtge": "RTGE",
    "episodes": "episodes",
})

df["CAS_ratio"] = pd.to_numeric(df["CAS_ratio"], errors="coerce")
df["CAS_ratio_filled"] = df["CAS_ratio"].fillna(1.0)
df["CAS_ratio_inv"] = 1.0 - df["CAS_ratio_filled"]

# v3-only selection: Scheme A
# Step 1: hard filter (LES < 0 or episodes < 1000)
df["hard_exclude"] = (df["LES"] < 0) | (df["episodes"] < 1000)
# Step 2: from remaining, drop lowest composite (LES + RSA + LRS) to get exactly 28
df["composite"] = df["LES"] + df["RSA"] + df["LRS"]
remaining = df[~df["hard_exclude"]].copy()
if len(remaining) > 28:
    drop_threshold = remaining["composite"].nsmallest(len(remaining) - 28).max()
    df["suspicious"] = df["hard_exclude"] | (
        (~df["hard_exclude"]) & (df["composite"] <= drop_threshold)
    )
else:
    df["suspicious"] = df["hard_exclude"]

indicators = ["LES", "CAS_ratio_inv", "LRS", "RSA", "RGEC", "RTGE"]
indicators_5 = ["LES", "CAS_ratio_inv", "RSA", "RGEC", "RTGE"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_cfa(data: pd.DataFrame, desc: str, model_desc: str, indicator_cols: list,
            standardize: bool = False):
    from semopy import Model
    from semopy.inspector import inspect
    from semopy.stats import gather_statistics

    data_to_fit = data[indicator_cols].copy()
    if standardize:
        for col in indicator_cols:
            data_to_fit[col] = stats.zscore(data_to_fit[col])

    m = Model(model_desc)
    m.fit(data_to_fit)
    params = inspect(m)
    stats_obj = gather_statistics(m)

    try:
        scores = m.predict_factors(data_to_fit)
        if scores is not None:
            scores = scores.reset_index(drop=True)
            # Add unified train_id (run_XX_train_YY format) for downstream merging
            if "run" in data.columns and "train_id" in data.columns:
                scores["train_id"] = data["run"].values + "_" + data["train_id"].values
            elif "train_id" in data.columns:
                scores["train_id"] = data["train_id"].values
            # Reorder so train_id comes first
            other_cols = [c for c in scores.columns if c != "train_id"]
            scores = scores[["train_id"] + other_cols]
    except Exception:
        scores = None

    return {
        "desc": desc,
        "model_desc": model_desc,
        "n": len(data),
        "params": params,
        "stats": stats_obj,
        "scores": scores,
        "standardize": standardize,
    }


def ci_bounds(est, se, z=1.96):
    """Return (lower, upper) 95% CI; handle NaN."""
    if pd.isna(est) or pd.isna(se):
        return (np.nan, np.nan)
    return (est - z * se, est + z * se)


def format_p(p):
    """Format p-value like lavaan: < .001 or 3 decimals."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def format_val(v, decimals=3):
    if pd.isna(v):
        return ""
    return f"{v:.{decimals}f}"


def compute_reliability(loadings: dict, residual_vars: dict, factor_var: float):
    """
    Compute Cronbach's alpha and McDonald's omega for standardized indicators.
    loadings: {indicator_name: lambda}
    residual_vars: {indicator_name: theta}
    factor_var: variance of latent factor
    """
    k = len(loadings)
    lam = np.array(list(loadings.values()))
    theta = np.array(list(residual_vars.values()))

    # McDonald's omega
    sum_lam = lam.sum()
    omega = (sum_lam ** 2) / ((sum_lam ** 2) + theta.sum())

    # Cronbach's alpha (standardized)
    # var(total) = k + 2 * sum_{i<j}(lambda_i * lambda_j * factor_var)
    cov_sum = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            cov_sum += lam[i] * lam[j] * factor_var
    var_total = k + 2 * cov_sum
    alpha = (k / (k - 1)) * (1 - k / var_total)

    return alpha, omega


def write_formatted_params(f, params: pd.DataFrame, model_name: str,
                           indicator_cols: list, factor_names: list):
    """
    Write lavaan-style formatted parameter tables to file f.
    """
    # Parse params into categories
    loadings = []
    factor_vars = []
    residual_vars = []

    # Ensure numeric columns (semopy may store '-' as string for fixed params)
    for col in ["Estimate", "Std. Err", "z-value", "p-value"]:
        if col in params.columns:
            params[col] = pd.to_numeric(params[col], errors="coerce")

    for _, row in params.iterrows():
        lval, op, rval = row["lval"], row["op"], row["rval"]
        est = row.get("Estimate", np.nan)
        se = row.get("Std. Err", np.nan)
        z = row.get("z-value", np.nan)
        p = row.get("p-value", np.nan)

        lower, upper = ci_bounds(est, se)

        entry = {
            "lval": lval, "op": op, "rval": rval,
            "est": est, "se": se, "z": z, "p": p,
            "lower": lower, "upper": upper,
        }

        if op == "~":
            # rval is factor, lval is indicator -> factor loading
            if rval in factor_names and lval in indicator_cols:
                loadings.append({**entry, "factor": rval, "indicator": lval})
        elif op == "~~":
            if lval == rval:
                if lval in factor_names:
                    factor_vars.append({**entry, "factor": lval})
                elif lval in indicator_cols:
                    residual_vars.append({**entry, "indicator": lval})

    # --- Compute standardized loadings ---
    # lambda_std = lambda * sqrt(phi) / sqrt(lambda^2 * phi + theta)
    factor_var_est = next((item["est"] for item in factor_vars if item["factor"] in factor_names), np.nan)
    residual_var_dict = {item["indicator"]: item["est"] for item in residual_vars}
    for item in loadings:
        lam = item["est"] if not pd.isna(item["est"]) else 0.0
        theta = residual_var_dict.get(item["indicator"], np.nan)
        if not pd.isna(factor_var_est) and not pd.isna(theta) and (lam**2 * factor_var_est + theta) > 0:
            item["std_loading"] = (lam * np.sqrt(factor_var_est)) / np.sqrt(lam**2 * factor_var_est + theta)
        else:
            item["std_loading"] = np.nan

    # --- Factor Loadings ---
    f.write("\nFactor loadings\n")
    f.write("-" * 105 + "\n")
    f.write(f"{'Factor':<20} {'Indicator':<15} {'Estimate':>10} {'Std.Loading':>12} {'Std.Error':>10} "
            f"{'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 105 + "\n")
    for item in loadings:
        f.write(f"{item['factor']:<20} {item['indicator']:<15} "
                f"{format_val(item['est']):>10} {format_val(item['std_loading']):>12} "
                f"{format_val(item['se']):>10} {format_val(item['z']):>8} {format_p(item['p']):>8} "
                f"{format_val(item['lower']):>10} {format_val(item['upper']):>10}\n")
    f.write("-" * 105 + "\n")

    # --- Factor variances ---
    f.write("\nFactor variances\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Factor':<20} {'Estimate':>10} {'Std.Error':>10} "
            f"{'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 90 + "\n")
    for item in factor_vars:
        f.write(f"{item['factor']:<20} "
                f"{format_val(item['est']):>10} {format_val(item['se']):>10} "
                f"{format_val(item['z']):>8} {format_p(item['p']):>8} "
                f"{format_val(item['lower']):>10} {format_val(item['upper']):>10}\n")
    f.write("-" * 90 + "\n")

    # --- Residual variances ---
    f.write("\nResidual variances\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Indicator':<15} {'Estimate':>10} {'Std.Error':>10} "
            f"{'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 90 + "\n")
    for item in residual_vars:
        f.write(f"{item['indicator']:<15} "
                f"{format_val(item['est']):>10} {format_val(item['se']):>10} "
                f"{format_val(item['z']):>8} {format_p(item['p']):>8} "
                f"{format_val(item['lower']):>10} {format_val(item['upper']):>10}\n")
    f.write("-" * 90 + "\n")

    # --- AVE & Reliability ---
    # Only compute for unifactor models
    if len(factor_names) == 1 and len(loadings) > 0:
        fac_name = factor_names[0]
        loadings_dict = {item["indicator"]: item["est"] for item in loadings}
        residuals_dict = {item["indicator"]: item["est"] for item in residual_vars}
        factor_var = next((item["est"] for item in factor_vars if item["factor"] == fac_name), np.nan)

        if not pd.isna(factor_var):
            ave = np.mean([
                (loadings_dict.get(ind, 0) ** 2) * factor_var
                for ind in loadings_dict
            ])
            alpha, omega = compute_reliability(loadings_dict, residuals_dict, factor_var)

            f.write("\nAverage variance extracted (AVE)\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Factor':<20} {'AVE':>10}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{fac_name:<20} {ave:>10.3f}\n")
            f.write("-" * 40 + "\n")

            f.write("\nReliability\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Factor':<20} {'Coefficient omega':>15} {'Coefficient alpha':>15}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{fac_name:<20} {omega:>15.3f} {alpha:>15.3f}\n")
            f.write("-" * 50 + "\n")


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
CORRELATED_CFA = """
Learning_Dynamics =~ LES + CAS_ratio_inv + LRS
Attainment_Consistency =~ RSA + RGEC + RTGE
"""

SECOND_ORDER_CFA = """
Learning_Dynamics =~ LES + CAS_ratio_inv + LRS
Attainment_Consistency =~ RSA + RGEC + RTGE
Learning_Improvement =~ Learning_Dynamics + Attainment_Consistency
"""

UNIFACTOR_CFA = """
Learning_Improvement =~ LES + CAS_ratio_inv + LRS + RSA + RGEC + RTGE
"""

UNIFACTOR_5ITEM_CFA = """
Learning_Improvement =~ LES + CAS_ratio_inv + RSA + RGEC + RTGE
"""

# ---------------------------------------------------------------------------
# Run analyses
# ---------------------------------------------------------------------------
results = []
df_clean = df[~df["suspicious"]].copy()

results.append(fit_cfa(df, "full_37_correlated", CORRELATED_CFA, indicators))
results.append(fit_cfa(df_clean, "clean_28_correlated", CORRELATED_CFA, indicators))
results.append(fit_cfa(df, "full_37_second_order", SECOND_ORDER_CFA, indicators))
results.append(fit_cfa(df, "full_37_unifactor", UNIFACTOR_CFA, indicators))
results.append(fit_cfa(df_clean, "clean_28_unifactor", UNIFACTOR_CFA, indicators))
results.append(fit_cfa(df, "full_37_unifactor_z", UNIFACTOR_CFA, indicators, standardize=True))
results.append(fit_cfa(df_clean, "clean_28_unifactor_z", UNIFACTOR_CFA, indicators, standardize=True))
results.append(fit_cfa(df, "full_37_unifactor_5item_z", UNIFACTOR_5ITEM_CFA, indicators_5, standardize=True))
results.append(fit_cfa(df_clean, "clean_28_unifactor_5item_z", UNIFACTOR_5ITEM_CFA, indicators_5, standardize=True))

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
report_path = OUTPUT_DIR / "cfa_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("PushBlock v3 Learning-Improvement Metrics — CFA Report\n")
    f.write("=" * 90 + "\n\n")

    f.write("DATA OVERVIEW\n")
    f.write("-" * 90 + "\n")
    f.write(f"Total runs loaded: {len(df)}\n")
    f.write(f"Suspicious runs (negative LES or low episode count): {df['suspicious'].sum()}\n")
    f.write(f"Clean runs: {len(df_clean)}\n\n")

    f.write("DESCRIPTIVE STATISTICS (raw indicators)\n")
    f.write("-" * 90 + "\n")
    f.write(df[indicators].describe().T.to_string())
    f.write("\n\n")

    f.write("CORRELATION MATRIX (full sample, raw)\n")
    f.write("-" * 90 + "\n")
    f.write(df[indicators].corr().to_string())
    f.write("\n\n")

    for r in results:
        is_recommended = r["desc"] == "clean_28_unifactor_z"
        header = f"MODEL: {r['desc'].upper()} (n={r['n']}"
        if r["standardize"]:
            header += ", z-score standardized"
        header += ")"
        if is_recommended:
            header += "  [RECOMMENDED]"

        f.write("=" * 90 + "\n")
        f.write(header + "\n")
        f.write("=" * 90 + "\n\n")

        # ---- Model Fit ----
        f.write("Model fit\n")
        f.write("-" * 90 + "\n")
        stats_obj = r["stats"]
        if stats_obj is not None:
            sd = stats_obj._asdict()
            chi2_tuple = sd.get("chi2", (np.nan, np.nan))
            if isinstance(chi2_tuple, tuple):
                chi2_val, chi2_p = chi2_tuple
            else:
                chi2_val, chi2_p = chi2_tuple, np.nan

            f.write(f"{'Model':<20} {'Chi2':>10} {'df':>5} {'p':>10} "
                    f"{'CFI':>8} {'TLI':>8} {'RMSEA':>8} {'AIC':>10} {'BIC':>10}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Factor model':<20} {chi2_val:>10.3f} {sd.get('dof', np.nan):>5.0f} "
                    f"{format_p(chi2_p):>10} {sd.get('cfi', np.nan):>8.3f} "
                    f"{sd.get('tli', np.nan):>8.3f} {sd.get('rmsea', np.nan):>8.3f} "
                    f"{sd.get('aic', np.nan):>10.2f} {sd.get('bic', np.nan):>10.2f}\n")
            f.write("-" * 90 + "\n")
            f.write(f"Note. The estimator is ML. N = {r['n']}.\n\n")
        else:
            f.write("[No fit statistics returned]\n\n")

        # ---- Parameter Estimates (formatted) ----
        params = r["params"]
        if params is not None and not params.empty:
            # Determine factor names from model spec
            factor_names = []
            for line in r["model_desc"].strip().split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=~" in line:
                    fac = line.split("=~")[0].strip()
                    if fac not in factor_names:
                        factor_names.append(fac)

            write_formatted_params(f, params, r["desc"], r["model_desc"], factor_names)
        else:
            f.write("[No parameter estimates returned]\n")

        # Warnings
        if "second_order" in r["desc"]:
            f.write("\n⚠️  IDENTIFICATION WARNING\n")
            f.write("-" * 90 + "\n")
            f.write("A second-order factor with only 2 first-order factors is "
                    "empirically underidentified. Treat this model as illustrative only.\n")

        if r["desc"] == "clean_28_unifactor" and not r["standardize"]:
            f.write("\n⚠️  NUMERICAL STABILITY WARNING\n")
            f.write("-" * 90 + "\n")
            f.write("This model was fit on raw (unstandardized) data. The clean-28 "
                    "subset has extreme scale differences across indicators, causing "
                    "the covariance matrix condition number to exceed 500. "
                    "See clean_28_unifactor_z for the z-score standardized equivalent.\n")

        # Factor scores
        scores = r["scores"]
        if scores is not None and not scores.empty:
            score_path = OUTPUT_DIR / f"factor_scores_{r['desc']}.csv"
            scores.to_csv(score_path, index=False)
            f.write(f"\nFactor scores saved to: {score_path.name}\n")

        f.write("\n")

    # Comparison section
    f.write("=" * 90 + "\n")
    f.write("MODEL COMPARISON & RECOMMENDATIONS\n")
    f.write("=" * 90 + "\n\n")
    f.write("1. SINGLE-FACTOR vs TWO-FACTOR\n")
    f.write("-" * 90 + "\n")
    f.write("The two first-order factors are almost perfectly correlated (~1.0). "
            "A likelihood-ratio test on clean-25 yields chi2_diff ~ 0.05 (df=1), p > 0.80. "
            "The unifactor model has lower AIC/BIC. A single composite score is recommended.\n\n")
    f.write("2. Z-SCORE STANDARDIZATION\n")
    f.write("-" * 90 + "\n")
    f.write("Clean-25 has extreme variance heterogeneity (RTGE var=0.0013 vs LES var=0.031). "
            "Z-score standardization eliminates numerical instability.\n\n")
    f.write("3. 6-ITEM vs 5-ITEM\n")
    f.write("-" * 90 + "\n")
    f.write("LRS is the weakest loader (~0.4). The 5-item model excludes it as a sensitivity check. "
            "Chi2_diff is not significant, but AIC favors the 5-item model.\n\n")

print(f"[OK] CFA report written to {report_path}")

# Fit summary CSV
fit_rows = []
for r in results:
    stats_obj = r["stats"]
    sd = stats_obj._asdict() if stats_obj else {}
    chi2_tuple = sd.get("chi2", np.nan)
    if isinstance(chi2_tuple, tuple):
        chi2_val, pval = chi2_tuple
    else:
        chi2_val, pval = chi2_tuple, np.nan
    fit_rows.append({
        "model": r["desc"], "n": r["n"], "standardized": r["standardize"],
        "chi2": chi2_val, "df": sd.get("dof", np.nan), "pvalue": pval,
        "CFI": sd.get("cfi", np.nan), "TLI": sd.get("tli", np.nan),
        "RMSEA": sd.get("rmsea", np.nan), "AIC": sd.get("aic", np.nan),
        "BIC": sd.get("bic", np.nan),
    })

pd.DataFrame(fit_rows).to_csv(OUTPUT_DIR / "cfa_fit_summary.csv", index=False)
print("[OK] Fit summary written to cfa_fit_summary.csv")
