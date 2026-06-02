#!/usr/bin/env python3
"""
CFA analysis for PushBlock v3 metrics — FINAL clean-28 version.

Model: clean_28_unifactor_5_z (5 items, z-score standardized)
  Learning_Improvement =~ LES + CAS_ratio_inv + LRS + RSA + RGEC

Uses lavaan (via R subprocess) for:
  - Model fitting with standardized output
  - Factor score extraction via lavPredict(method = "regression")

Output:
  - cfa_report_final.txt
  - cfa_fit_summary_final.csv
  - factor_scores_clean_28_final.csv
"""

from pathlib import Path
import subprocess
import tempfile
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
df["hard_exclude"] = (df["LES"] < 0) | (df["episodes"] < 1000)
df["composite"] = df["LES"] + df["RSA"] + df["LRS"]
remaining = df[~df["hard_exclude"]].copy()
if len(remaining) > 28:
    drop_threshold = remaining["composite"].nsmallest(len(remaining) - 28).max()
    df["suspicious"] = df["hard_exclude"] | (
        (~df["hard_exclude"]) & (df["composite"] <= drop_threshold)
    )
else:
    df["suspicious"] = df["hard_exclude"]

indicators = ["LES", "CAS_ratio_inv", "LRS", "RSA", "RGEC"]
df_clean = df[~df["suspicious"]].copy()

# Build unified train_id
df_clean["train_id_unified"] = df_clean["run"] + "_" + df_clean["train_id"]

# Rename columns to match model syntax
df_z = df_clean.rename(columns={
    "les": "LES",
    "cas_ratio": "CAS_ratio",
    "lrs": "LRS",
    "rsa": "RSA",
    "rgec": "RGEC",
})
df_z["CAS_ratio_inv"] = 1.0 - df_z["CAS_ratio"].fillna(1.0)

# Z-score standardize for CFA
for col in indicators:
    df_z[col] = stats.zscore(df_z[col])

# ---------------------------------------------------------------------------
# Run lavaan via R subprocess
# ---------------------------------------------------------------------------
def run_lavaan_cfa(df_z: pd.DataFrame, indicators: list):
    """Call R/lavaan via subprocess. Return parsed stdout + factor scores CSV."""

    # Save temp data CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        tmp_data = Path(f.name)
    df_z[["train_id_unified"] + indicators].to_csv(tmp_data, index=False)

    # Inline R script
    r_script = f'''
library(readr)
library(dplyr)
library(lavaan)

# 1. Load data
df <- read_csv("{tmp_data.as_posix()}", show_col_types = FALSE)
train_ids <- df$train_id_unified
df <- df %>% select(-train_id_unified)

# 2. Fit CFA
model <- "
  Learning_Improvement =~ LES + CAS_ratio_inv + LRS + RSA + RGEC
"
fit <- cfa(model, data = df, std.lv = TRUE, estimator = "ML")

# 3. Extract fit measures
fm <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic"))

# 4. Extract standardized parameters
pe <- parameterEstimates(fit, standardized = TRUE)
loadings <- pe[pe$op == "=~", c("lhs", "rhs", "est", "std.lv", "std.all", "se", "z", "pvalue", "ci.lower", "ci.upper")]
residuals <- pe[pe$op == "~~" & pe$lhs == pe$rhs & pe$lhs != "Learning_Improvement", c("lhs", "est", "se", "z", "pvalue", "ci.lower", "ci.upper")]
factor_var <- pe[pe$op == "~~" & pe$lhs == "Learning_Improvement" & pe$rhs == "Learning_Improvement", c("est", "se", "z", "pvalue")]

# 5. Extract factor scores
scores <- lavPredict(fit, method = "regression")
score_df <- data.frame(train_id = train_ids, lavaan_score = as.numeric(scores[, 1]))

# 6. Compute AVE and reliability (omega, alpha)
std_loadings <- loadings$std.all
ave <- mean(std_loadings^2)
k <- length(std_loadings)
alpha <- (k / (k - 1)) * (1 - k / (k + 2 * sum(cor(df, use = "pairwise.complete.obs")[upper.tri(cor(df, use = "pairwise.complete.obs"))])))
# Simplified omega from standardized loadings
omega_num <- sum(std_loadings)^2
omega_den <- sum(std_loadings)^2 + sum(residuals$est)
omega <- omega_num / omega_den

# 7. Output JSON-like summary
cat("===FIT===\\n")
cat(paste(names(fm), collapse = "\\t"), "\\n")
cat(paste(round(fm, 6), collapse = "\\t"), "\\n")

cat("\\n===LOADINGS===\\n")
for (i in seq_len(nrow(loadings))) {{
  cat(paste(loadings[i, ], collapse = "\\t"), "\\n")
}}

cat("\\n===RESIDUALS===\\n")
for (i in seq_len(nrow(residuals))) {{
  cat(paste(residuals[i, ], collapse = "\\t"), "\\n")
}}

cat("\\n===FACTOR_VAR===\\n")
cat(paste(factor_var[1, ], collapse = "\\t"), "\\n")

cat("\\n===AVE_RELIABILITY===\\n")
cat(sprintf("AVE\\t%.6f\\n", ave))
cat(sprintf("Omega\\t%.6f\\n", omega))

# 8. Save factor scores
score_out <- "{OUTPUT_DIR.as_posix()}/factor_scores_clean_28_final.csv"
write.csv(score_df, score_out, row.names = FALSE)
cat(sprintf("\\n===SCORES_SAVED===\\t%s\\n", score_out))
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as f:
        tmp_r = Path(f.name)
    tmp_r.write_text(r_script, encoding="utf-8")

    result = subprocess.run(
        ["Rscript", str(tmp_r)],
        capture_output=True,
        text=True,
    )

    tmp_data.unlink(missing_ok=True)
    tmp_r.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"R/lavaan failed:\n{result.stderr}")

    return result.stdout


# ---------------------------------------------------------------------------
# Parse R stdout
# ---------------------------------------------------------------------------
def parse_lavaan_output(stdout: str):
    lines = stdout.strip().split("\n")
    sections = {}
    current = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            current = line.strip("=")
            sections[current] = []
        elif current:
            sections[current].append(line)

    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("[CFA] Running lavaan CFA on clean-28 (5 items, z-score)...")
stdout = run_lavaan_cfa(df_z, indicators)
sections = parse_lavaan_output(stdout)

# Parse fit measures
fit_line = sections["FIT"][1].split("\t")
fit_names = sections["FIT"][0].split("\t")
fit_dict = {k: float(v) for k, v in zip(fit_names, fit_line)}

# Parse loadings
loadings = []
for line in sections["LOADINGS"]:
    parts = line.split("\t")
    loadings.append({
        "factor": parts[0], "indicator": parts[1],
        "est": float(parts[2]), "std_lv": float(parts[3]), "std_all": float(parts[4]),
        "se": float(parts[5]), "z": float(parts[6]), "pvalue": float(parts[7]),
        "ci_lower": float(parts[8]), "ci_upper": float(parts[9]),
    })

# Parse residuals
residuals = []
for line in sections["RESIDUALS"]:
    parts = line.split("\t")
    residuals.append({
        "indicator": parts[0], "est": float(parts[1]),
        "se": float(parts[2]), "z": float(parts[3]), "pvalue": float(parts[4]),
        "ci_lower": float(parts[5]), "ci_upper": float(parts[6]),
    })

# Parse factor variance
fv_parts = sections["FACTOR_VAR"][0].split("\t")
def safe_float(s):
    try:
        return float(s)
    except ValueError:
        return np.nan
factor_var = {"est": safe_float(fv_parts[0]), "se": safe_float(fv_parts[1]), "z": safe_float(fv_parts[2]), "pvalue": safe_float(fv_parts[3])}

# Parse AVE & reliability
ave = float(sections["AVE_RELIABILITY"][0].split("\t")[1])
omega = float(sections["AVE_RELIABILITY"][1].split("\t")[1])

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
report_path = OUTPUT_DIR / "cfa_report_final.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write("PushBlock v3 Learning-Improvement Metrics — CFA Final Report (clean-28, 5-item, z-score)\n")
    f.write("=" * 100 + "\n\n")

    f.write("DATA OVERVIEW\n")
    f.write("-" * 100 + "\n")
    f.write(f"Total runs loaded: {len(df)}\n")
    f.write(f"Excluded runs: {df['suspicious'].sum()}\n")
    f.write(f"Clean runs (CFA sample): {len(df_clean)}\n\n")

    f.write("DESCRIPTIVE STATISTICS (clean-28, z-standardized)\n")
    f.write("-" * 100 + "\n")
    f.write(df_clean[indicators].describe().T.to_string())
    f.write("\n\n")

    f.write("CORRELATION MATRIX (clean-28, z-standardized)\n")
    f.write("-" * 100 + "\n")
    f.write(df_z[indicators].corr().round(3).to_string())
    f.write("\n\n")

    f.write("=" * 100 + "\n")
    f.write("MODEL: CLEAN_28_UNIFACTOR_5_Z (n=28, z-score standardized)\n")
    f.write("=" * 100 + "\n\n")

    f.write("Model specification\n")
    f.write("-" * 100 + "\n")
    f.write("  Learning_Improvement =~ LES + CAS_ratio_inv + LRS + RSA + RGEC\n\n")

    f.write("Model fit\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Metric':<20} {'Value':>12}\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Chi-square':<20} {fit_dict['chisq']:>12.3f}\n")
    f.write(f"{'df':<20} {fit_dict['df']:>12.0f}\n")
    f.write(f"{'p-value':<20} {fit_dict['pvalue']:>12.6f}\n")
    f.write(f"{'CFI':<20} {fit_dict['cfi']:>12.3f}\n")
    f.write(f"{'TLI':<20} {fit_dict['tli']:>12.3f}\n")
    f.write(f"{'RMSEA':<20} {fit_dict['rmsea']:>12.3f}\n")
    f.write(f"{'SRMR':<20} {fit_dict['srmr']:>12.3f}\n")
    f.write(f"{'AIC':<20} {fit_dict['aic']:>12.2f}\n")
    f.write(f"{'BIC':<20} {fit_dict['bic']:>12.2f}\n")
    f.write("-" * 100 + "\n")
    f.write(f"Note. Estimator = ML. N = 28.\n\n")

    f.write("Factor loadings\n")
    f.write("-" * 105 + "\n")
    f.write(f"{'Factor':<20} {'Indicator':<15} {'Estimate':>10} {'Std.Loading':>12} {'Std.Error':>10} {'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 105 + "\n")
    for item in loadings:
        p_str = "< .001" if item["pvalue"] < 0.001 else f"{item['pvalue']:.3f}"
        f.write(f"{item['factor']:<20} {item['indicator']:<15} {item['est']:>10.3f} {item['std_all']:>12.3f} {item['se']:>10.3f} {item['z']:>8.3f} {p_str:>8} {item['ci_lower']:>10.3f} {item['ci_upper']:>10.3f}\n")
    f.write("-" * 105 + "\n\n")

    f.write("Factor variance\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Factor':<20} {'Estimate':>10} {'Std.Error':>10} {'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 90 + "\n")
    p_str = "< .001" if factor_var["pvalue"] < 0.001 else f"{factor_var['pvalue']:.3f}"
    f.write(f"{'Learning_Improvement':<20} {factor_var['est']:>10.3f} {factor_var['se']:>10.3f} {factor_var['z']:>8.3f} {p_str:>8} {factor_var['est']-1.96*factor_var['se']:>10.3f} {factor_var['est']+1.96*factor_var['se']:>10.3f}\n")
    f.write("-" * 90 + "\n\n")

    f.write("Residual variances\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Indicator':<15} {'Estimate':>10} {'Std.Error':>10} {'z-value':>8} {'p':>8} {'Lower':>10} {'Upper':>10}\n")
    f.write("-" * 90 + "\n")
    for item in residuals:
        p_str = "< .001" if item["pvalue"] < 0.001 else f"{item['pvalue']:.3f}"
        f.write(f"{item['indicator']:<15} {item['est']:>10.3f} {item['se']:>10.3f} {item['z']:>8.3f} {p_str:>8} {item['ci_lower']:>10.3f} {item['ci_upper']:>10.3f}\n")
    f.write("-" * 90 + "\n\n")

    f.write("Average variance extracted (AVE)\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Factor':<20} {'AVE':>10}\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Learning_Improvement':<20} {ave:>10.3f}\n")
    f.write("-" * 40 + "\n\n")

    f.write("Reliability\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Factor':<20} {'Coefficient omega':>15} {'Coefficient alpha':>15}\n")
    f.write("-" * 50 + "\n")
    # Approximate alpha from standardized loadings
    k = len(loadings)
    std_lam = np.array([item["std_all"] for item in loadings])
    cov_sum = sum(std_lam[i] * std_lam[j] for i in range(k) for j in range(i+1, k))
    var_total = k + 2 * cov_sum
    alpha = (k / (k - 1)) * (1 - k / var_total)
    f.write(f"{'Learning_Improvement':<20} {omega:>15.3f} {alpha:>15.3f}\n")
    f.write("-" * 50 + "\n\n")

    f.write("Factor scores\n")
    f.write("-" * 60 + "\n")
    f.write(f"Extracted via lavPredict(method = 'regression') from lavaan 0.6-21.\n")
    f.write(f"Saved to: factor_scores_clean_28_final.csv\n")
    f.write("-" * 60 + "\n")

print(f"[OK] CFA report written to {report_path}")

# ---------------------------------------------------------------------------
# Fit summary CSV
# ---------------------------------------------------------------------------
pd.DataFrame([{
    "model": "clean_28_unifactor_5_z",
    "n": 28,
    "standardized": True,
    "chi2": fit_dict["chisq"],
    "df": fit_dict["df"],
    "pvalue": fit_dict["pvalue"],
    "CFI": fit_dict["cfi"],
    "TLI": fit_dict["tli"],
    "RMSEA": fit_dict["rmsea"],
    "SRMR": fit_dict["srmr"],
    "AIC": fit_dict["aic"],
    "BIC": fit_dict["bic"],
    "AVE": ave,
    "omega": omega,
    "alpha": alpha,
}]).to_csv(OUTPUT_DIR / "cfa_fit_summary_final.csv", index=False)
print("[OK] Fit summary written to cfa_fit_summary_final.csv")

# ---------------------------------------------------------------------------
# Verify factor scores
# ---------------------------------------------------------------------------
score_file = OUTPUT_DIR / "factor_scores_clean_28_final.csv"
if score_file.exists():
    scores_df = pd.read_csv(score_file)
    print(f"[OK] Factor scores: N={len(scores_df)}, Mean={scores_df['lavaan_score'].mean():.4f}, SD={scores_df['lavaan_score'].std():.4f}")
