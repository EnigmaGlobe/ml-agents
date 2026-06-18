# =============================================================================
# Between-Group Comparison of lavaan_scores
# =============================================================================
# Compare two groups of CFA factor scores (exploratory, small samples).
# Do NOT rerun CFA. Treat lavaan_score as dependent variable.
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
data_dir  <- file.path(base_dir, "data")
out_dir   <- file.path(base_dir, "output")
script_dir <- file.path(base_dir, "script")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
group_28 <- read_csv(
  file.path(data_dir, "factor_scores_origin.csv"),
  show_col_types = FALSE
)
group_17 <- read_csv(
  file.path(data_dir, "factor_scores_half_goal.csv"),
  show_col_types = FALSE
)

# ---------------------------------------------------------------------------
# 2. Add group column & combine
# ---------------------------------------------------------------------------
group_28 <- group_28 %>% mutate(group = "Group_28")
group_17 <- group_17 %>% mutate(group = "Group_17")

combined <- bind_rows(group_28, group_17) %>%
  mutate(group = factor(group, levels = c("Group_28", "Group_17")))

# ---------------------------------------------------------------------------
# 3. Data structure checks
# ---------------------------------------------------------------------------
cat("=== DATA STRUCTURE CHECKS ===\n\n")

cat("Rows per group:\n")
print(combined %>% count(group))

cat("\nlavaan_score class:", class(combined$lavaan_score), "\n")
cat("Is numeric:", is.numeric(combined$lavaan_score), "\n")

cat("\nMissing values:\n")
print(combined %>% summarise(
  total_missing = sum(is.na(lavaan_score)),
  missing_by_group = list(tapply(is.na(lavaan_score), group, sum))
))

cat("\nDuplicated train_id values:\n")
dup_ids <- combined$train_id[duplicated(combined$train_id)]
if (length(dup_ids) == 0) {
  cat("  None\n")
} else {
  print(dup_ids)
}

cat("\n")

# ---------------------------------------------------------------------------
# 4. Descriptive statistics by group
# ---------------------------------------------------------------------------
desc_stats <- combined %>%
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
# 5. Visualizations
# ---------------------------------------------------------------------------

# 5a. Boxplot + jitter
gg_box <- ggplot(combined, aes(x = group, y = lavaan_score, fill = group)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.7) +
  scale_fill_manual(values = c("Group_28" = "#3498db", "Group_17" = "#e74c3c")) +
  labs(
    title = "Lavaan Score Distribution by Group",
    subtitle = "Boxplot with jittered individual points",
    x = "Group",
    y = "Lavaan Score (Factor Score)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "lavaan_score_boxplot.png"), gg_box,
       width = 6, height = 5, dpi = 300)

# 5b. Density plot
gg_dens <- ggplot(combined, aes(x = lavaan_score, fill = group, color = group)) +
  geom_density(alpha = 0.4, linewidth = 1) +
  scale_fill_manual(values = c("Group_28" = "#3498db", "Group_17" = "#e74c3c")) +
  scale_color_manual(values = c("Group_28" = "#2980b9", "Group_17" = "#c0392b")) +
  labs(
    title = "Density Distribution of Lavaan Scores",
    x = "Lavaan Score",
    y = "Density",
    fill = "Group", color = "Group"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lavaan_score_density.png"), gg_dens,
       width = 6, height = 5, dpi = 300)

# 5c. Histogram with faceting
gg_hist <- ggplot(combined, aes(x = lavaan_score, fill = group)) +
  geom_histogram(alpha = 0.7, bins = 12, color = "white") +
  facet_wrap(~group, ncol = 1) +
  scale_fill_manual(values = c("Group_28" = "#3498db", "Group_17" = "#e74c3c")) +
  labs(
    title = "Histogram of Lavaan Scores by Group",
    x = "Lavaan Score",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "lavaan_score_histogram.png"), gg_hist,
       width = 6, height = 6, dpi = 300)

# 5d. QQ plots
gg_qq <- ggplot(combined, aes(sample = lavaan_score, color = group)) +
  stat_qq(alpha = 0.7, size = 2) +
  stat_qq_line(linewidth = 1) +
  facet_wrap(~group, ncol = 2) +
  scale_color_manual(values = c("Group_28" = "#3498db", "Group_17" = "#e74c3c")) +
  labs(
    title = "QQ Plots of Lavaan Scores",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "lavaan_score_qqplot.png"), gg_qq,
       width = 7, height = 4, dpi = 300)

cat("Plots saved to:", out_dir, "\n\n")

# ---------------------------------------------------------------------------
# 6. Group comparison tests
# ---------------------------------------------------------------------------

# 6a. Welch t-test
t_test_result <- t.test(lavaan_score ~ group, data = combined, var.equal = FALSE)
t_test_tidy <- tidy(t_test_result)

cat("=== WELCH T-TEST ===\n")
print(t_test_result)
cat("\n")

# 6b. Wilcoxon rank-sum test
wilcox_result <- wilcox.test(lavaan_score ~ group, data = combined, exact = FALSE)
wilcox_tidy <- data.frame(
  statistic = wilcox_result$statistic,
  p.value = wilcox_result$p.value,
  method = wilcox_result$method
)

cat("=== WILCOXON RANK-SUM TEST ===\n")
print(wilcox_result)
cat("\n")

# 6c. Permutation test (10,000 iterations)
set.seed(42)

observed_diff <- mean(group_28$lavaan_score) - mean(group_17$lavaan_score)

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

perm_result <- permute_test(group_28$lavaan_score, group_17$lavaan_score, n_perm = 10000)

cat("=== PERMUTATION TEST (10,000 iterations) ===\n")
cat(sprintf("Observed mean difference (28 - 17): %.4f\n", observed_diff))
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
    x = "Mean Difference (Group_28 - Group_17)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lavaan_score_permutation.png"), gg_perm,
       width = 6, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 7. Effect sizes
# ---------------------------------------------------------------------------

# 7a. Hedges' g with 95% CI
hedges_g <- hedges_g(lavaan_score ~ group, data = combined, ci = 0.95)
hedges_g_df <- as.data.frame(hedges_g)

cat("=== HEDGES' g ===\n")
print(hedges_g)
cat("\n")

# 7b. Rank-biserial correlation (from Wilcoxon test)
# Using effectsize::rank_biserial
rank_bis <- rank_biserial(lavaan_score ~ group, data = combined, ci = 0.95)
rank_bis_df <- as.data.frame(rank_bis)

cat("=== RANK-BISERIAL CORRELATION ===\n")
print(rank_bis)
cat("\n")

# ---------------------------------------------------------------------------
# 8. Bootstrap CI for mean difference (5,000 resamples)
# ---------------------------------------------------------------------------

mean_diff_func <- function(data, indices) {
  d <- data[indices, ]
  g28 <- d$lavaan_score[d$group == "Group_28"]
  g17 <- d$lavaan_score[d$group == "Group_17"]
  mean(g28, na.rm = TRUE) - mean(g17, na.rm = TRUE)
}

set.seed(42)
boot_result <- boot(data = combined, statistic = mean_diff_func, R = 5000, strata = combined$group)
boot_ci <- boot.ci(boot_result, type = c("perc", "bca"))

cat("=== BOOTSTRAP CI FOR MEAN DIFFERENCE (5,000 resamples) ===\n")
cat(sprintf("Observed difference: %.4f\n", boot_result$t0))
cat(sprintf("Bootstrap SE: %.4f\n", sd(boot_result$t)))
cat(sprintf("Percentile 95%% CI: [%.4f, %.4f]\n",
            boot_ci$percent[4], boot_ci$percent[5]))
cat(sprintf("BCa 95%% CI: [%.4f, %.4f]\n",
            boot_ci$bca[4], boot_ci$bca[5]))
cat("\n")

# Bootstrap distribution plot
gg_boot <- ggplot(data.frame(diff = boot_result$t), aes(x = diff)) +
  geom_histogram(bins = 60, fill = "darkgreen", color = "white", alpha = 0.7) +
  geom_vline(xintercept = boot_result$t0, color = "red", linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.8, linetype = "solid") +
  labs(
    title = "Bootstrap Distribution of Mean Differences",
    subtitle = sprintf("Observed diff = %.3f, 95%% CI = [%.3f, %.3f]",
                       boot_result$t0, boot_ci$percent[4], boot_ci$percent[5]),
    x = "Mean Difference (Group_28 - Group_17)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lavaan_score_bootstrap.png"), gg_boot,
       width = 6, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------

# 9a. Combined dataset
write_csv(combined, file.path(out_dir, "combined_lavaan_scores.csv"))

# 9b. Descriptive statistics
write_csv(desc_stats, file.path(out_dir, "lavaan_score_descriptives.csv"))

# 9c. Test results table
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
    estimate = observed_diff,
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
    estimate = boot_result$t0,
    statistic = NA,
    p.value = NA,
    effect_size = NA,
    ci_low = boot_ci$percent[4],
    ci_high = boot_ci$percent[5]
  )
)

write_csv(test_results, file.path(out_dir, "lavaan_score_group_comparison_results.csv"))

# ---------------------------------------------------------------------------
# 10. Summary report (Markdown)
# ---------------------------------------------------------------------------

higher_group <- ifelse(mean(group_28$lavaan_score) > mean(group_17$lavaan_score),
                       "Group_28 (origin, n = 28)", "Group_17 (half_goal, n = 17)")
obs_diff <- mean(group_28$lavaan_score) - mean(group_17$lavaan_score)

# Significance flags
t_sig <- ifelse(t_test_tidy$p.value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
w_sig <- ifelse(wilcox_tidy$p.value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
p_sig <- ifelse(perm_result$p_value < 0.05, "significant (p < .05)", "not significant (p >= .05)")
ci_includes_zero <- (boot_ci$percent[4] <= 0 && boot_ci$percent[5] >= 0)

report <- sprintf(
"# Lavaan Score Between-Group Comparison Report

## Data Overview

- **Group_28** (origin): n = %d
- **Group_17** (half_goal): n = %d
- Total observations: %d
- Missing values: %d
- Duplicated train_id: %s

## Descriptive Statistics

| Group | n | Mean | SD | Median | IQR | Min | Max |
|-------|---|------|----|--------|-----|-----|-----|
| Group_28 | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |
| Group_17 | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |

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

- **Observed difference** (Group_28 - Group_17): **%.3f**
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

## Cautious Interpretation

Because the sample sizes are small and unequal (n = 28 vs. n = 17), these findings should be treated as **exploratory evidence only**. The permutation and bootstrap analyses help gauge the robustness of the observed difference under resampling, but they do not overcome the limitations of small-sample inference.

## ⚠️ Important Warning

> If these lavaan_scores were extracted from two separate CFA models, the scores may **not be directly comparable** because the factor-score scale may differ between models. The preferred approach is to estimate **one CFA model on the combined item-level data** and then extract scores for all cases from the same model.

## Output Files

- `combined_lavaan_scores.csv` — combined dataset
- `lavaan_score_descriptives.csv` — descriptive statistics
- `lavaan_score_group_comparison_results.csv` — test results
- `lavaan_score_boxplot.png` — boxplot + jitter
- `lavaan_score_density.png` — density plot
- `lavaan_score_histogram.png` — histograms
- `lavaan_score_qqplot.png` — QQ plots
- `lavaan_score_permutation.png` — permutation distribution
- `lavaan_score_bootstrap.png` — bootstrap distribution

---
*Report generated: %s*
",
  nrow(group_28), nrow(group_17), nrow(combined),
  sum(is.na(combined$lavaan_score)),
  ifelse(length(dup_ids) == 0, "None", paste(dup_ids, collapse = ", ")),

  desc_stats$n[1], desc_stats$mean[1], desc_stats$sd[1],
  desc_stats$median[1], desc_stats$IQR[1], desc_stats$min[1], desc_stats$max[1],
  desc_stats$n[2], desc_stats$mean[2], desc_stats$sd[2],
  desc_stats$median[2], desc_stats$IQR[2], desc_stats$min[2], desc_stats$max[2],

  t_test_tidy$statistic, t_test_tidy$p.value, t_sig,
  wilcox_tidy$statistic, wilcox_tidy$p.value, w_sig,
  perm_result$p_value, p_sig,

  as.numeric(hedges_g_df$Hedges_g), hedges_g_df$CI_low, hedges_g_df$CI_high,
  as.numeric(rank_bis_df$r_rank_biserial), rank_bis_df$CI_low, rank_bis_df$CI_high,

  boot_result$t0, sd(boot_result$t),
  boot_ci$percent[4], boot_ci$percent[5],
  boot_ci$bca[4], boot_ci$bca[5],
  ifelse(ci_includes_zero, "Yes", "No"),

  higher_group, obs_diff,
  t_sig, w_sig, p_sig,
  as.numeric(hedges_g_df$Hedges_g),
  ifelse(abs(as.numeric(hedges_g_df$Hedges_g)) < 0.2, "negligible",
         ifelse(abs(as.numeric(hedges_g_df$Hedges_g)) < 0.5, "small",
                ifelse(abs(as.numeric(hedges_g_df$Hedges_g)) < 0.8, "medium", "large"))),
  ifelse(ci_includes_zero, "Yes — difference may be zero", "No — difference likely non-zero"),

  format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

writeLines(report, file.path(out_dir, "lavaan_score_analysis_summary.md"))

# ---------------------------------------------------------------------------
# 11. Console summary
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("              ANALYSIS COMPLETE — SUMMARY\n")
cat("=================================================================\n\n")
cat(sprintf("Higher mean lavaan_score: %s\n", higher_group))
cat(sprintf("Observed mean difference (28 - 17): %.3f\n\n", obs_diff))

cat("Statistical support:\n")
cat(sprintf("  Welch t-test:      %s (p = %.4f)\n", t_sig, t_test_tidy$p.value))
cat(sprintf("  Wilcoxon test:     %s (p = %.4f)\n", w_sig, wilcox_tidy$p.value))
cat(sprintf("  Permutation test:  %s (p = %.4f)\n\n", p_sig, perm_result$p_value))

cat(sprintf("Effect size (Hedges' g): %.3f [%.3f, %.3f]\n",
            as.numeric(hedges_g_df$Hedges_g),
            hedges_g_df$CI_low, hedges_g_df$CI_high))
cat(sprintf("Rank-biserial r:         %.3f [%.3f, %.3f]\n\n",
            as.numeric(rank_bis_df$r_rank_biserial),
            rank_bis_df$CI_low, rank_bis_df$CI_high))

cat(sprintf("Bootstrap 95%% CI for mean diff: [%.3f, %.3f]\n",
            boot_ci$percent[4], boot_ci$percent[5]))
cat(sprintf("CI includes zero: %s\n\n",
            ifelse(ci_includes_zero, "YES", "NO")))

cat("Because n = 28 and n = 17, interpret findings cautiously\n")
cat("as exploratory evidence.\n\n")

cat("Output files saved to:\n")
cat("  ", out_dir, "\n")
cat("=================================================================\n")
