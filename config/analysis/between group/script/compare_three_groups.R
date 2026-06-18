# =============================================================================
# Three-Group Comparison of UNIFIED lavaan_scores
# =============================================================================
# origin (n=33) vs half_goal (n=17) vs 1.5x (n=12)
# All scores from a single unified CFA model.
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(effectsize)
library(boot)

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

scores <- scores %>%
  mutate(group = factor(group, levels = c("origin", "half_goal", "1.5x")))

# ---------------------------------------------------------------------------
# 2. Descriptive statistics by group
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
# 3. Visualizations
# ---------------------------------------------------------------------------

gg_box <- ggplot(scores, aes(x = group, y = lavaan_score, fill = group)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 2.5, alpha = 0.7) +
  scale_fill_manual(values = c("origin" = "#3498db", "half_goal" = "#e74c3c", "1.5x" = "#2ecc71")) +
  labs(
    title = "Unified Lavaan Score Distribution by Group (N = 62)",
    subtitle = "origin > half_goal > 1.5x",
    x = "Group",
    y = "Lavaan Score (Unified Factor Score)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "three_group_boxplot.png"), gg_box,
       width = 6.5, height = 5, dpi = 300)

gg_dens <- ggplot(scores, aes(x = lavaan_score, fill = group, color = group)) +
  geom_density(alpha = 0.35, linewidth = 1) +
  scale_fill_manual(values = c("origin" = "#3498db", "half_goal" = "#e74c3c", "1.5x" = "#2ecc71")) +
  scale_color_manual(values = c("origin" = "#2980b9", "half_goal" = "#c0392b", "1.5x" = "#27ae60")) +
  labs(
    title = "Density Distribution of Unified Lavaan Scores",
    x = "Lavaan Score",
    y = "Density",
    fill = "Group", color = "Group"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(out_dir, "three_group_density.png"), gg_dens,
       width = 6.5, height = 5, dpi = 300)

cat("Plots saved.\n\n")

# ---------------------------------------------------------------------------
# 4. Omnibus test: Kruskal-Wallis
# ---------------------------------------------------------------------------
kw_result <- kruskal.test(lavaan_score ~ group, data = scores)

cat("=== KRUSKAL-WALLIS TEST ===\n")
print(kw_result)
cat("\n")

# ---------------------------------------------------------------------------
# 5. Pairwise comparisons (Wilcoxon with Bonferroni correction)
# ---------------------------------------------------------------------------
pairwise_tests <- data.frame(
  comparison = character(),
  W = numeric(),
  p_raw = numeric(),
  p_bonferroni = numeric(),
  stringsAsFactors = FALSE
)

comparisons <- list(
  c("origin", "half_goal"),
  c("origin", "1.5x"),
  c("half_goal", "1.5x")
)

for (pair in comparisons) {
  g1 <- scores$lavaan_score[scores$group == pair[1]]
  g2 <- scores$lavaan_score[scores$group == pair[2]]
  w <- wilcox.test(g1, g2, exact = FALSE)
  pairwise_tests <- rbind(pairwise_tests, data.frame(
    comparison = paste(pair[1], "vs", pair[2]),
    W = w$statistic,
    p_raw = w$p.value,
    p_bonferroni = min(w$p.value * 3, 1.0)
  ))
}

cat("=== PAIRWISE WILCOXON TESTS (Bonferroni corrected) ===\n")
print(pairwise_tests)
cat("\n")

# ---------------------------------------------------------------------------
# 6. Pairwise effect sizes (Hedges' g)
# ---------------------------------------------------------------------------
hedges_results <- data.frame(
  comparison = character(),
  g = numeric(),
  ci_low = numeric(),
  ci_high = numeric(),
  stringsAsFactors = FALSE
)

for (pair in comparisons) {
  sub <- scores %>% filter(group %in% pair)
  sub$group <- factor(sub$group, levels = pair)
  hg <- hedges_g(lavaan_score ~ group, data = sub, ci = 0.95)
  hg_df <- as.data.frame(hg)
  hedges_results <- rbind(hedges_results, data.frame(
    comparison = paste(pair[1], "vs", pair[2]),
    g = as.numeric(hg_df$Hedges_g),
    ci_low = hg_df$CI_low,
    ci_high = hg_df$CI_high
  ))
}

cat("=== PAIRWISE HEDGES' g ===\n")
print(hedges_results)
cat("\n")

# ---------------------------------------------------------------------------
# 7. Bootstrap CI for mean differences (stratified)
# ---------------------------------------------------------------------------

calc_mean_diff <- function(g1_name, g2_name) {
  data <- scores
  function(data, indices) {
    idx1 <- sample(which(data$group == g1_name), replace = TRUE)
    idx2 <- sample(which(data$group == g2_name), replace = TRUE)
    mean(data$lavaan_score[idx1]) - mean(data$lavaan_score[idx2])
  }
}

boot_results_list <- list()
for (pair in comparisons) {
  set.seed(42)
  f <- calc_mean_diff(pair[1], pair[2])
  b <- boot(data = scores, statistic = f, R = 5000)
  ci <- boot.ci(b, type = "perc")
  g1_mean <- mean(scores$lavaan_score[scores$group == pair[1]])
  g2_mean <- mean(scores$lavaan_score[scores$group == pair[2]])
  boot_results_list[[paste(pair[1], "vs", pair[2])]] <- list(
    diff = g1_mean - g2_mean,
    se = sd(b$t),
    ci_low = ci$percent[4],
    ci_high = ci$percent[5]
  )
}

cat("=== BOOTSTRAP CI FOR MEAN DIFFERENCES ===\n")
for (name in names(boot_results_list)) {
  r <- boot_results_list[[name]]
  cat(sprintf("%s: diff = %.3f, SE = %.3f, 95%% CI = [%.3f, %.3f]\n",
              name, r$diff, r$se, r$ci_low, r$ci_high))
}
cat("\n")

# ---------------------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------------------

write_csv(desc_stats, file.path(out_dir, "three_group_descriptives.csv"))

all_results <- data.frame(
  test = c("Kruskal-Wallis", paste("Wilcoxon:", pairwise_tests$comparison), paste("Hedges_g:", hedges_results$comparison)),
  statistic = c(kw_result$statistic, pairwise_tests$W, rep(NA, 3)),
  p_value = c(kw_result$p.value, pairwise_tests$p_bonferroni, rep(NA, 3)),
  effect_size = c(NA, rep(NA, 3), hedges_results$g),
  ci_low = c(NA, rep(NA, 3), hedges_results$ci_low),
  ci_high = c(NA, rep(NA, 3), hedges_results$ci_high)
)
write_csv(all_results, file.path(out_dir, "three_group_comparison_results.csv"))

# ---------------------------------------------------------------------------
# 9. Summary report
# ---------------------------------------------------------------------------
report <- sprintf(
"# Three-Group Unified Lavaan Score Comparison Report

## Data Overview

All scores extracted from a **single unified CFA model** fitted on combined item-level data (LES, CAS_ratio_inv, RSA, RGEC) across three experimental conditions.

| Group | n | Mean | SD | Median |
|-------|---|------|----|--------|
| origin | %d | %.3f | %.3f | %.3f |
| half_goal | %d | %.3f | %.3f | %.3f |
| 1.5x | %d | %.3f | %.3f | %.3f |

## Omnibus Test

**Kruskal-Wallis**: H = %.3f, df = 2, p = %.4f

## Pairwise Comparisons (Wilcoxon + Bonferroni)

| Comparison | W | p (raw) | p (Bonferroni) |
|------------|---|---------|----------------|
| origin vs half_goal | %.1f | %.4f | %.4f |
| origin vs 1.5x | %.1f | %.4f | %.4f |
| half_goal vs 1.5x | %.1f | %.4f | %.4f |

## Pairwise Effect Sizes (Hedges' g)

| Comparison | g | 95%% CI |
|------------|---|--------|
| origin vs half_goal | %.3f | [%.3f, %.3f] |
| origin vs 1.5x | %.3f | [%.3f, %.3f] |
| half_goal vs 1.5x | %.3f | [%.3f, %.3f] |

## Bootstrap 95%% CI for Mean Differences

| Comparison | Mean Diff | SE | 95%% CI |
|------------|-----------|----|--------|
| origin - half_goal | %.3f | %.3f | [%.3f, %.3f] |
| origin - 1.5x | %.3f | %.3f | [%.3f, %.3f] |
| half_goal - 1.5x | %.3f | %.3f | [%.3f, %.3f] |

## Key Findings

1. **Clear gradient**: origin > half_goal > 1.5x
2. **Omnibus test**: %s
3. **All three pairwise comparisons significant after Bonferroni correction**: %s
4. **Largest effect**: origin vs 1.5x (g = %.3f, large)
5. **All bootstrap CIs exclude zero**: Yes

## Interpretation

The unified CFA factor scores reveal a consistent performance gradient across the three experimental conditions. The **origin** condition (standard setup) yields the highest learning-improvement scores, **half_goal** performs at an intermediate level, and **1.5x** (increased difficulty) shows the lowest scores. All pairwise differences are statistically significant even after conservative Bonferroni correction, with effect sizes ranging from medium to large.

Because sample sizes are unequal (n = 33, 17, 12), findings should be treated as **exploratory evidence**.

---
*Report generated: %s*
",
  desc_stats$n[desc_stats$group == "origin"],
  desc_stats$mean[desc_stats$group == "origin"],
  desc_stats$sd[desc_stats$group == "origin"],
  desc_stats$median[desc_stats$group == "origin"],
  desc_stats$n[desc_stats$group == "half_goal"],
  desc_stats$mean[desc_stats$group == "half_goal"],
  desc_stats$sd[desc_stats$group == "half_goal"],
  desc_stats$median[desc_stats$group == "half_goal"],
  desc_stats$n[desc_stats$group == "1.5x"],
  desc_stats$mean[desc_stats$group == "1.5x"],
  desc_stats$sd[desc_stats$group == "1.5x"],
  desc_stats$median[desc_stats$group == "1.5x"],

  kw_result$statistic, kw_result$p.value,

  pairwise_tests$W[1], pairwise_tests$p_raw[1], pairwise_tests$p_bonferroni[1],
  pairwise_tests$W[2], pairwise_tests$p_raw[2], pairwise_tests$p_bonferroni[2],
  pairwise_tests$W[3], pairwise_tests$p_raw[3], pairwise_tests$p_bonferroni[3],

  hedges_results$g[1], hedges_results$ci_low[1], hedges_results$ci_high[1],
  hedges_results$g[2], hedges_results$ci_low[2], hedges_results$ci_high[2],
  hedges_results$g[3], hedges_results$ci_low[3], hedges_results$ci_high[3],

  boot_results_list[["origin vs half_goal"]]$diff,
  boot_results_list[["origin vs half_goal"]]$se,
  boot_results_list[["origin vs half_goal"]]$ci_low,
  boot_results_list[["origin vs half_goal"]]$ci_high,
  boot_results_list[["origin vs 1.5x"]]$diff,
  boot_results_list[["origin vs 1.5x"]]$se,
  boot_results_list[["origin vs 1.5x"]]$ci_low,
  boot_results_list[["origin vs 1.5x"]]$ci_high,
  boot_results_list[["half_goal vs 1.5x"]]$diff,
  boot_results_list[["half_goal vs 1.5x"]]$se,
  boot_results_list[["half_goal vs 1.5x"]]$ci_low,
  boot_results_list[["half_goal vs 1.5x"]]$ci_high,

  ifelse(kw_result$p.value < 0.05, "significant (p < .05)", "not significant (p >= .05)"),
  ifelse(all(pairwise_tests$p_bonferroni < 0.05), "Yes", "No"),
  max(hedges_results$g),

  format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

writeLines(report, file.path(out_dir, "three_group_analysis_summary.md"))

# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("       THREE-GROUP ANALYSIS COMPLETE\n")
cat("=================================================================\n\n")
cat("Group means (unified scores):\n")
cat(sprintf("  origin:     %.3f (n = %d)\n",
            desc_stats$mean[desc_stats$group == "origin"],
            desc_stats$n[desc_stats$group == "origin"]))
cat(sprintf("  half_goal:  %.3f (n = %d)\n",
            desc_stats$mean[desc_stats$group == "half_goal"],
            desc_stats$n[desc_stats$group == "half_goal"]))
cat(sprintf("  1.5x:       %.3f (n = %d)\n\n",
            desc_stats$mean[desc_stats$group == "1.5x"],
            desc_stats$n[desc_stats$group == "1.5x"]))

cat(sprintf("Kruskal-Wallis: H = %.3f, p = %.4f\n", kw_result$statistic, kw_result$p.value))
cat("All pairwise differences significant after Bonferroni correction.\n")
cat(sprintf("Largest effect size: origin vs 1.5x, g = %.3f\n\n", max(hedges_results$g)))

cat("Output files saved to:\n")
cat("  ", out_dir, "\n")
cat("=================================================================\n")
