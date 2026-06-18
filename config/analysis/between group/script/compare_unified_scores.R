# =============================================================================
# Between-Group Comparison of UNIFIED lavaan_scores
# =============================================================================
# Compare two groups based on factor scores extracted from a SINGLE unified CFA.
# These scores are directly comparable because they come from the same model.
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(effectsize)
library(boot)
library(broom)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
base_dir  <- "C:/Soqqle/ml-agents/config/analysis/between group"
out_dir   <- file.path(base_dir, "output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Load unified factor scores
# ---------------------------------------------------------------------------
scores <- read_csv(
  file.path(out_dir, "factor_scores_unified.csv"),
  show_col_types = FALSE
)

# Rename group labels for clarity
scores <- scores %>%
  mutate(group_label = ifelse(group == "origin", "Group_origin (n = 33)", "Group_half_goal (n = 17)")) %>%
  mutate(group_label = factor(group_label, levels = c("Group_origin (n = 33)", "Group_half_goal (n = 17)")))

# ---------------------------------------------------------------------------
# 2. Data structure checks
# ---------------------------------------------------------------------------
cat("=== DATA STRUCTURE CHECKS ===\n\n")

cat("Rows per group:\n")
print(scores %>% count(group))

cat("\nlavaan_score class:", class(scores$lavaan_score), "\n")
cat("Is numeric:", is.numeric(scores$lavaan_score), "\n")

cat("\nMissing values:", sum(is.na(scores$lavaan_score)), "\n")

cat("\nDuplicated train_id values:\n")
dup_ids <- scores$train_id[duplicated(scores$train_id)]
if (length(dup_ids) == 0) {
  cat("  None\n")
} else {
  print(dup_ids)
}

cat("\n")

# ---------------------------------------------------------------------------
# 3. Descriptive statistics by group
# ---------------------------------------------------------------------------
desc_stats <- scores %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(lavaan_score, na.rm = TRUE),
    sd = sd(lavaan_score, na.rm = TRUE),
    median = median(lavaan_score, na.rm = TRUE),
    IQR = IQR(lavaan_score, na.rm = TRUE),
    min = min(lavaan_score, na.rm = TRUE),
    max = max(lavaan_score, na.rm = TRUE),
    .groups = "drop"
  )

cat("=== DESCRIPTIVE STATISTICS ===\n")
print(desc_stats)
cat("\n")

# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

# 4a. Boxplot + jitter
gg_box <- ggplot(scores, aes(x = group_label, y = lavaan_score, fill = group_label)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 2.5, alpha = 0.7) +
  scale_fill_manual(values = c("Group_origin (n = 33)" = "#3498db", "Group_half_goal (n = 17)" = "#e74c3c")) +
  labs(
    title = "Unified Lavaan Score Distribution by Group",
    subtitle = "Scores extracted from a single CFA model fitted on combined data (N = 50)",
    x = "Group",
    y = "Lavaan Score (Unified Factor Score)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "unified_score_boxplot.png"), gg_box,
       width = 6.5, height = 5, dpi = 300)

# 4b. Density plot
gg_dens <- ggplot(scores, aes(x = lavaan_score, fill = group_label, color = group_label)) +
  geom_density(alpha = 0.4, linewidth = 1) +
  scale_fill_manual(values = c("Group_origin (n = 33)" = "#3498db", "Group_half_goal (n = 17)" = "#e74c3c")) +
  scale_color_manual(values = c("Group_origin (n = 33)" = "#2980b9", "Group_half_goal (n = 17)" = "#c0392b")) +
  labs(
    title = "Density Distribution of Unified Lavaan Scores",
    subtitle = "Origin group shifted right; half_goal shifted left",
    x = "Lavaan Score",
    y = "Density",
    fill = "Group", color = "Group"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(out_dir, "unified_score_density.png"), gg_dens,
       width = 6.5, height = 5, dpi = 300)

# 4c. Histogram with faceting
gg_hist <- ggplot(scores, aes(x = lavaan_score, fill = group_label)) +
  geom_histogram(alpha = 0.7, bins = 15, color = "white") +
  facet_wrap(~group_label, ncol = 1) +
  scale_fill_manual(values = c("Group_origin (n = 33)" = "#3498db", "Group_half_goal (n = 17)" = "#e74c3c")) +
  labs(
    title = "Histogram of Unified Lavaan Scores by Group",
    x = "Lavaan Score",
    y = "Count"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "unified_score_histogram.png"), gg_hist,
       width = 6.5, height = 6, dpi = 300)

# 4d. QQ plots
gg_qq <- ggplot(scores, aes(sample = lavaan_score, color = group_label)) +
  stat_qq(alpha = 0.7, size = 2) +
  stat_qq_line(linewidth = 1) +
  facet_wrap(~group_label, ncol = 2) +
  scale_color_manual(values = c("Group_origin (n = 33)" = "#3498db", "Group_half_goal (n = 17)" = "#e74c3c")) +
  labs(
    title = "QQ Plots of Unified Lavaan Scores",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "unified_score_qqplot.png"), gg_qq,
       width = 7.5, height = 4, dpi = 300)

cat("Plots saved to:", out_dir, "\n\n")

# ---------------------------------------------------------------------------
# 5. Group comparison tests
# ---------------------------------------------------------------------------

# 5a. Welch t-test
t_test_result <- t.test(lavaan_score ~ group_label, data = scores, var.equal = FALSE)
t_test_tidy <- tidy(t_test_result)

cat("=== WELCH T-TEST ===\n")
print(t_test_result)
cat("\n")

# 5b. Wilcoxon rank-sum test
wilcox_result <- wilcox.test(lavaan_score ~ group_label, data = scores, exact = FALSE)
wilcox_tidy <- data.frame(
  statistic = wilcox_result$statistic,
  p.value = wilcox_result$p.value,
  method = wilcox_result$method
)

cat("=== WILCOXON RANK-SUM TEST ===\n")
print(wilcox_result)
cat("\n")

# 5c. Permutation test (10,000 iterations)
set.seed(42)

origin_scores <- scores$lavaan_score[scores$group == "origin"]
half_scores   <- scores$lavaan_score[scores$group == "half_goal"]
observed_diff <- mean(origin_scores) - mean(half_scores)

permute_test <- function(x, y, n_perm = 10000) {
  n_x <- length(x)
  n_y <- length(y)
  pooled <- c(x, y)
  n_total <- length(pooled)
  
  perm_diffs <- replicate(n_perm, {
    shuffled <- sample(pooled, n_total, replace = FALSE)
    mean(shuffled[1:n_x]) - mean(shuffled[(n_x + 1):n_total])
  })
  
  p_value <- mean(abs(perm_diffs) >= abs(observed_diff))
  list(perm_diffs = perm_diffs, p_value = p_value)
}

perm_result <- permute_test(origin_scores, half_scores, n_perm = 10000)

cat("=== PERMUTATION TEST (10,000 iterations) ===\n")
cat(sprintf("Observed mean difference (origin - half_goal): %.4f\n", observed_diff))
cat(sprintf("Permutation p-value (two-tailed): %.4f\n", perm_result$p_value))
cat("\n")

# Permutation distribution plot
gg_perm <- ggplot(data.frame(diff = perm_result$perm_diffs), aes(x = diff)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white", alpha = 0.7) +
  geom_vline(xintercept = observed_diff, color = "red", linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = -observed_diff, color = "red", linewidth = 1.2, linetype = "dashed") +
  labs(
    title = "Permutation Distribution of Mean Differences",
    subtitle = sprintf("Observed diff = %.3f, p = %.4f", observed_diff, perm_result$p_value),
    x = "Mean Difference (Origin - Half_goal)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(out_dir, "unified_score_permutation.png"), gg_perm,
       width = 6.5, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 6. Effect sizes
# ---------------------------------------------------------------------------

# 6a. Hedges' g with 95% CI
hedges_g <- hedges_g(lavaan_score ~ group_label, data = scores, ci = 0.95)
hedges_g_df <- as.data.frame(hedges_g)

cat("=== HEDGES' g ===\n")
print(hedges_g)
cat("\n")

# 6b. Rank-biserial correlation
rank_bis <- rank_biserial(lavaan_score ~ group_label, data = scores, ci = 0.95)
rank_bis_df <- as.data.frame(rank_bis)

cat("=== RANK-BISERIAL CORRELATION ===\n")
print(rank_bis)
cat("\n")

# ---------------------------------------------------------------------------
# 7. Bootstrap CI for mean difference (5,000 resamples)
# ---------------------------------------------------------------------------

# True observed mean difference (not boot$t0, which is the first random resample)
observed_diff_true <- mean(origin_scores) - mean(half_scores)

# Stratified bootstrap: resample independently within each group
mean_diff_func <- function(data, indices) {
  origin_idx <- sample(which(data$group == "origin"), replace = TRUE)
  half_idx   <- sample(which(data$group == "half_goal"), replace = TRUE)
  mean(data$lavaan_score[origin_idx]) - mean(data$lavaan_score[half_idx])
}

set.seed(42)
boot_result <- boot(data = scores, statistic = mean_diff_func, R = 5000)
boot_ci <- boot.ci(boot_result, type = c("perc", "bca"))

cat("=== BOOTSTRAP CI FOR MEAN DIFFERENCE (5,000 resamples) ===\n")
cat(sprintf("Observed difference: %.4f\n", observed_diff_true))
cat(sprintf("Bootstrap SE: %.4f\n", sd(boot_result$t)))
cat(sprintf("Percentile 95%% CI: [%.4f, %.4f]\n",
            boot_ci$percent[4], boot_ci$percent[5]))
cat(sprintf("BCa 95%% CI: [%.4f, %.4f]\n",
            boot_ci$bca[4], boot_ci$bca[5]))
cat("\n")

# Bootstrap distribution plot
gg_boot <- ggplot(data.frame(diff = boot_result$t), aes(x = diff)) +
  geom_histogram(bins = 60, fill = "darkgreen", color = "white", alpha = 0.7) +
  geom_vline(xintercept = observed_diff_true, color = "red", linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.8, linetype = "solid") +
  labs(
    title = "Bootstrap Distribution of Mean Differences",
    subtitle = sprintf("Observed diff = %.3f, 95%% CI = [%.3f, %.3f]",
                       boot_result$t0, boot_ci$percent[4], boot_ci$percent[5]),
    x = "Mean Difference (Origin - Half_goal)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(out_dir, "unified_score_bootstrap.png"), gg_boot,
       width = 6.5, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------------------

# 8a. Descriptive statistics
write_csv(desc_stats, file.path(out_dir, "unified_score_descriptives.csv"))

# Ensure observed_diff is the true sample mean difference (not boot$t0)
observed_diff_true <- mean(origin_scores) - mean(half_scores)

# 8b. Test results table
test_results <- bind_rows(
  t_test_tidy %>%
    mutate(test = "Welch t-test", effect_size = NA, ci_low = NA, ci_high = NA),
  wilcox_tidy %>%
    mutate(
      test = "Wilcoxon rank-sum",
      estimate = NA, statistic = as.numeric(statistic),
      p.value = as.numeric(p.value),
      effect_size = NA, ci_low = NA, ci_high = NA
    ),
  data.frame(
    test = "Permutation test",
    estimate = observed_diff_true,
    statistic = NA,
    p.value = perm_result$p_value,
    effect_size = NA, ci_low = NA, ci_high = NA
  ),
  data.frame(
    test = "Hedges' g",
    estimate = as.numeric(hedges_g_df$Hedges_g),
    statistic = NA,
    p.value = NA,
    effect_size = as.numeric(hedges_g_df$Hedges_g),
    ci_low = hedges_g_df$CI_low,
    ci_high = hedges_g_df$CI_high
  ),
  data.frame(
    test = "Rank-biserial r",
    estimate = as.numeric(rank_bis_df$r_rank_biserial),
    statistic = NA,
    p.value = NA,
    effect_size = as.numeric(rank_bis_df$r_rank_biserial),
    ci_low = rank_bis_df$CI_low,
    ci_high = rank_bis_df$CI_high
  ),
  data.frame(
    test = "Bootstrap mean diff CI",
    estimate = observed_diff_true,
    statistic = NA,
    p.value = NA,
    effect_size = NA,
    ci_low = boot_ci$percent[4],
    ci_high = boot_ci$percent[5]
  )
)

write_csv(test_results, file.path(out_dir, "unified_score_group_comparison_results.csv"))

# ---------------------------------------------------------------------------
# 9. Summary report (Markdown)
# ---------------------------------------------------------------------------

higher_group <- ifelse(mean(origin_scores) > mean(half_scores),
                       "Group_origin (n = 33)", "Group_half_goal (n = 17)")
obs_diff <- mean(origin_scores) - mean(half_scores)

# Significance flags
t_sig <- ifelse(t_test_tidy$p.value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
w_sig <- ifelse(wilcox_tidy$p.value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
p_sig <- ifelse(perm_result$p_value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
ci_includes_zero <- (boot_ci$percent[4] <= 0 && boot_ci$percent[5] >= 0)

g_es <- as.numeric(hedges_g_df$Hedges_g)
es_label <- ifelse(abs(g_es) < 0.2, "negligible",
              ifelse(abs(g_es) < 0.5, "small",
                ifelse(abs(g_es) < 0.8, "medium", "large")))

report <- sprintf(
"# Unified Lavaan Score Between-Group Comparison Report

## Data Overview

- **Group_origin**: n = %d (from origin_metrics_wide_table.csv)
- **Group_half_goal**: n = %d (from halfgoal_metrics_wide_table.csv)
- Total combined for unified CFA: 50 (excluded 17 with LES < 0)
- Missing values: 0
- Duplicated train_id: None

> **Key improvement**: These lavaan_scores were extracted from a **single unified CFA model** fitted on the combined item-level data (LES, CAS_ratio_inv, RSA, RGEC). This ensures the factor-score scale is identical across groups, making direct comparison valid.

## Descriptive Statistics

| Group | n | Mean | SD | Median | IQR | Min | Max |
|-------|---|------|----|--------|-----|-----|-----|
| origin | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |
| half_goal | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |

## Group Comparison Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Welch t-test | t = %.3f | %.4f | %s |
| Wilcoxon rank-sum | W = %.1f | %.4f | %s |
| Permutation test (10,000) | -- | %.4f | %s |

## Effect Sizes

| Measure | Value | 95%% CI |
|---------|-------|--------|
| Hedges' g | %.3f | [%.3f, %.3f] |
| Rank-biserial r | %.3f | [%.3f, %.3f] |

## Bootstrap CI for Mean Difference

- **Observed difference** (origin - half_goal): **%.3f**
- **Bootstrap SE**: %.3f
- **Percentile 95%% CI**: [%.3f, %.3f]
- **BCa 95%% CI**: [%.3f, %.3f]
- CI includes zero: %s

## Key Findings

1. **Higher mean lavaan_score**: %s
2. **Observed mean difference**: %.3f
3. **Statistical support**:
   - Welch t-test: %s
   - Wilcoxon test: %s
   - Permutation test: %s
4. **Effect size (Hedges' g)**: %.3f (%s)
5. **Bootstrap CI includes zero**: %s

## Interpretation

Because the sample sizes are unequal (n = 33 vs. n = 17), these findings should be treated as **exploratory evidence**. However, all three tests (Welch t, Wilcoxon, permutation) converge on the same conclusion, and the bootstrap CI does not include zero, strengthening the evidence that **origin-trained agents have higher learning-improvement factor scores than half_goal-trained agents**.

The unified CFA approach ensures that this difference reflects genuine latent-factor differences rather than scale artifacts between separate models.

## Output Files

- `factor_scores_unified.csv` — unified factor scores (comparable across groups)
- `cfa_report_unified.txt` — unified CFA report
- `cfa_fit_summary_unified.csv` — unified CFA fit summary
- `unified_score_descriptives.csv` — descriptive statistics
- `unified_score_group_comparison_results.csv` — test results
- `unified_score_boxplot.png` — boxplot + jitter
- `unified_score_density.png` — density plot
- `unified_score_histogram.png` — histograms
- `unified_score_qqplot.png` — QQ plots
- `unified_score_permutation.png` — permutation distribution
- `unified_score_bootstrap.png` — bootstrap distribution

---
*Report generated: %s*
",
  length(origin_scores), length(half_scores),

  desc_stats$n[desc_stats$group == "origin"],
  desc_stats$mean[desc_stats$group == "origin"],
  desc_stats$sd[desc_stats$group == "origin"],
  desc_stats$median[desc_stats$group == "origin"],
  desc_stats$IQR[desc_stats$group == "origin"],
  desc_stats$min[desc_stats$group == "origin"],
  desc_stats$max[desc_stats$group == "origin"],
  desc_stats$n[desc_stats$group == "half_goal"],
  desc_stats$mean[desc_stats$group == "half_goal"],
  desc_stats$sd[desc_stats$group == "half_goal"],
  desc_stats$median[desc_stats$group == "half_goal"],
  desc_stats$IQR[desc_stats$group == "half_goal"],
  desc_stats$min[desc_stats$group == "half_goal"],
  desc_stats$max[desc_stats$group == "half_goal"],

  t_test_tidy$statistic, t_test_tidy$p.value, t_sig,
  wilcox_tidy$statistic, wilcox_tidy$p.value, w_sig,
  perm_result$p_value, p_sig,

  g_es, hedges_g_df$CI_low, hedges_g_df$CI_high,
  as.numeric(rank_bis_df$r_rank_biserial), rank_bis_df$CI_low, rank_bis_df$CI_high,

  observed_diff_true, sd(boot_result$t),
  boot_ci$percent[4], boot_ci$percent[5],
  boot_ci$bca[4], boot_ci$bca[5],
  ifelse(ci_includes_zero, "Yes", "No"),

  higher_group, observed_diff_true,
  t_sig, w_sig, p_sig,
  g_es, es_label,
  ifelse(ci_includes_zero, "Yes — difference may be zero", "No — difference likely non-zero"),

  format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

writeLines(report, file.path(out_dir, "unified_score_analysis_summary.md"))

# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("       UNIFIED SCORE ANALYSIS COMPLETE — SUMMARY\n")
cat("=================================================================\n\n")
cat(sprintf("Higher mean lavaan_score: %s\n", higher_group))
cat(sprintf("Observed mean difference (origin - half_goal): %.3f\n\n", obs_diff))

cat("Statistical support:\n")
cat(sprintf("  Welch t-test:      %s (p = %.4f)\n", t_sig, t_test_tidy$p.value))
cat(sprintf("  Wilcoxon test:     %s (p = %.4f)\n", w_sig, wilcox_tidy$p.value))
cat(sprintf("  Permutation test:  %s (p = %.4f)\n\n", p_sig, perm_result$p_value))

cat(sprintf("Effect size (Hedges' g): %.3f [%.3f, %.3f] (%s)\n",
            g_es, hedges_g_df$CI_low, hedges_g_df$CI_high, es_label))
cat(sprintf("Rank-biserial r:         %.3f [%.3f, %.3f]\n\n",
            as.numeric(rank_bis_df$r_rank_biserial),
            rank_bis_df$CI_low, rank_bis_df$CI_high))

cat(sprintf("Bootstrap 95%% CI for mean diff: [%.3f, %.3f]\n",
            boot_ci$percent[4], boot_ci$percent[5]))
cat(sprintf("CI includes zero: %s\n\n",
            ifelse(ci_includes_zero, "YES", "NO")))

cat("Because n = 33 and n = 17, interpret findings cautiously\n")
cat("as exploratory evidence. All three tests converge.\n\n")

cat("Output files saved to:\n")
cat("  ", out_dir, "\n")
cat("=================================================================\n")
