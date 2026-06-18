# =============================================================================
# Linear Mixed Model (LMM) Analysis
# Episode-level learning trajectories: half_goal vs 1.5x
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(broom.mixed)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
base_dir  <- "C:/Soqqle/ml-agents/config/analysis/between group"
out_dir   <- file.path(base_dir, "output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df <- read_csv(file.path(out_dir, "combined_episodes_lmm.csv"), show_col_types = FALSE)

df <- df %>%
  mutate(
    condition = factor(condition, levels = c("half_goal", "1.5x")),
    train_id_unified = factor(train_id_unified)
  )

cat("=== DATA OVERVIEW ===\n")
cat(sprintf("Total episodes: %d\n", nrow(df)))
cat(sprintf("Unique trains: %d\n", n_distinct(df$train_id_unified)))
cat("Episodes per condition:\n")
print(df %>% count(condition))
cat("Episodes per train (summary):\n")
print(df %>% count(train_id_unified) %>% pull(n) %>% summary())
cat("\n")

# ---------------------------------------------------------------------------
# 2. Exploratory visualization
# ---------------------------------------------------------------------------

# 2a. Raw trajectories (sample 5 trains per condition for clarity)
set.seed(42)
sample_trains <- df %>%
  group_by(condition) %>%
  distinct(train_id_unified) %>%
  slice_sample(n = 5) %>%
  pull(train_id_unified)

gg_raw <- df %>%
  filter(train_id_unified %in% sample_trains) %>%
  ggplot(aes(x = training_step_k, y = normalized_task_progress,
             group = train_id_unified, color = condition)) +
  geom_line(alpha = 0.4) +
  geom_point(alpha = 0.2, size = 0.5) +
  scale_color_manual(values = c("half_goal" = "#3498db", "1.5x" = "#2ecc71")) +
  facet_wrap(~condition, ncol = 1) +
  labs(
    title = "Learning Trajectories (Sample of 5 trains per condition)",
    x = "Training Step (thousands)",
    y = "Normalized Task Progress"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "lmm_raw_trajectories.png"), gg_raw,
       width = 7, height = 6, dpi = 300)

# 2b. Smoothed group means (loess)
gg_smooth <- df %>%
  ggplot(aes(x = training_step_k, y = normalized_task_progress, color = condition)) +
  geom_smooth(method = "loess", span = 0.3, se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("half_goal" = "#3498db", "1.5x" = "#2ecc71")) +
  labs(
    title = "Smoothed Group Learning Trajectories (LOESS)",
    subtitle = "Shaded area = SE",
    x = "Training Step (thousands)",
    y = "Normalized Task Progress",
    color = "Condition"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lmm_smoothed_trajectories.png"), gg_smooth,
       width = 7, height = 5, dpi = 300)

cat("Plots saved.\n\n")

# ---------------------------------------------------------------------------
# 3. Fit LMMs
# ---------------------------------------------------------------------------

# Model 0: Random intercept only
cat("=== MODEL 0: Random Intercept ===\n")
m0 <- lmer(normalized_task_progress ~ condition * training_step_k +
             (1 | train_id_unified),
           data = df, REML = FALSE)
print(summary(m0))
cat("\n")

# Model 1: Random intercept + random slope
cat("=== MODEL 1: Random Intercept + Random Slope ===\n")
m1 <- lmer(normalized_task_progress ~ condition * training_step_k +
             (training_step_k | train_id_unified),
           data = df, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
print(summary(m1))
cat("\n")

# Model comparison
cat("=== MODEL COMPARISON (Likelihood Ratio) ===\n")
anova_m0_m1 <- anova(m0, m1)
print(anova_m0_m1)
cat(sprintf("AIC: M0 = %.1f, M1 = %.1f\n", AIC(m0), AIC(m1)))
cat(sprintf("BIC: M0 = %.1f, M1 = %.1f\n\n", BIC(m0), BIC(m1)))

# ---------------------------------------------------------------------------
# 4. Final model: use REML for fixed-effect estimates
# ---------------------------------------------------------------------------
cat("=== FINAL MODEL (REML) ===\n")
m_final <- lmer(normalized_task_progress ~ condition * training_step_k +
                  (training_step_k | train_id_unified),
                data = df, REML = TRUE,
                control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
print(summary(m_final))
cat("\n")

# Tidy fixed effects
fixed_eff <- tidy(m_final, effects = "fixed", conf.int = TRUE)
cat("=== FIXED EFFECTS TABLE ===\n")
print(fixed_eff)
cat("\n")

# Random effects
rand_eff <- tidy(m_final, effects = "ran_pars")
cat("=== RANDOM EFFECTS PARAMETERS ===\n")
print(rand_eff)
cat("\n")

# ---------------------------------------------------------------------------
# 5. Estimated marginal means / predictions
# ---------------------------------------------------------------------------

# Create prediction grid
pred_grid <- expand.grid(
  training_step_k = seq(0, max(df$training_step_k), length.out = 200),
  condition = levels(df$condition)
)
pred_grid$train_id_unified <- NA  # new levels → population-level prediction

# Predict with standard errors
mm <- model.matrix(~ condition * training_step_k, data = pred_grid)
beta <- fixef(m_final)
vcov_beta <- vcov(m_final)
pred_grid$fit <- as.vector(mm %*% beta)
pred_grid$se <- sqrt(diag(mm %*% vcov_beta %*% t(mm)))
pred_grid$lower <- pred_grid$fit - 1.96 * pred_grid$se
pred_grid$upper <- pred_grid$fit + 1.96 * pred_grid$se

gg_pred <- ggplot(pred_grid, aes(x = training_step_k, y = fit, color = condition)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = condition), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("half_goal" = "#3498db", "1.5x" = "#2ecc71")) +
  scale_fill_manual(values = c("half_goal" = "#3498db", "1.5x" = "#2ecc71")) +
  labs(
    title = "LMM Estimated Marginal Trajectories",
    subtitle = "Population-level predictions with 95% CI",
    x = "Training Step (thousands)",
    y = "Normalized Task Progress",
    color = "Condition", fill = "Condition"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lmm_marginal_predictions.png"), gg_pred,
       width = 7, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 6. Random effects visualization
# ---------------------------------------------------------------------------
ranef_df <- as.data.frame(ranef(m_final)$train_id_unified) %>%
  tibble::rownames_to_column("train_id_unified") %>%
  mutate(condition = ifelse(grepl("half", train_id_unified), "half_goal", "1.5x"))

gg_ranef <- ggplot(ranef_df, aes(x = `(Intercept)`, y = training_step_k,
                                   color = condition)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_color_manual(values = c("half_goal" = "#3498db", "1.5x" = "#2ecc71")) +
  labs(
    title = "Random Effects: Intercept vs Slope",
    x = "Random Intercept",
    y = "Random Slope (per 1k steps)",
    color = "Condition"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "lmm_random_effects.png"), gg_ranef,
       width = 6.5, height = 5, dpi = 300)

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------

write_csv(fixed_eff, file.path(out_dir, "lmm_fixed_effects.csv"))
write_csv(rand_eff, file.path(out_dir, "lmm_random_effects.csv"))

# Model comparison table
comp_df <- data.frame(
  model = c("M0: RI", "M1: RI+RS"),
  AIC = c(AIC(m0), AIC(m1)),
  BIC = c(BIC(m0), BIC(m1)),
  logLik = c(as.numeric(logLik(m0)), as.numeric(logLik(m1))),
  df = c(attr(logLik(m0), "df"), attr(logLik(m1), "df"))
)
write_csv(comp_df, file.path(out_dir, "lmm_model_comparison.csv"))

# ---------------------------------------------------------------------------
# 8. Summary report
# ---------------------------------------------------------------------------

# Extract key numbers
fe <- fixed_eff
intercept_hg <- fe$estimate[fe$term == "(Intercept)"]
slope_hg <- fe$estimate[fe$term == "training_step_k"]
intercept_diff <- fe$estimate[fe$term == "condition1.5x"]
slope_diff <- fe$estimate[fe$term == "condition1.5x:training_step_k"]
p_intercept <- fe$p.value[fe$term == "condition1.5x"]
p_slope <- fe$p.value[fe$term == "condition1.5x:training_step_k"]

report <- sprintf(
"# Linear Mixed Model (LMM) Report

## Data

- **Episodes**: %d (half_goal = %d, 1.5x = %d)
- **Trains**: %d (half_goal = %d, 1.5x = %d)
- **Condition**: half_goal (reference) vs 1.5x

## Model Specification

```
normalized_task_progress ~ condition * training_step_k + (training_step_k | train_id_unified)
```

- **Fixed effects**: condition, training_step_k (per 1,000 steps), interaction
- **Random effects**: random intercept + random slope per train
- **Estimation**: REML

## Model Selection

| Model | Random Structure | AIC | BIC |
|-------|-----------------|-----|-----|
| M0 | Intercept only | %.1f | %.1f |
| M1 | Intercept + Slope | %.1f | %.1f |

Selected: **M1** (random intercept + slope)

## Fixed Effects

| Term | Estimate | SE | df | t | p | 95%% CI |
|------|----------|----|----|---|---|--------|
| (Intercept) | %.4f | %.4f | %.1f | %.2f | %.4f | [%.4f, %.4f] |
| condition1.5x | %.4f | %.4f | %.1f | %.2f | %.4f | [%.4f, %.4f] |
| training_step_k | %.6f | %.6f | %.1f | %.2f | %.4f | [%.6f, %.6f] |
| condition1.5x:training_step_k | %.6f | %.6f | %.1f | %.2f | %.4f | [%.6f, %.6f] |

## Interpretation

1. **Baseline (half_goal at step = 0)**: %.3f
2. **1.5x baseline difference**: %.3f (p = %.4f) — %s
3. **half_goal learning rate** (per 1k steps): %.6f
4. **1.5x vs half_goal learning rate difference**: %.6f (p = %.4f) — %s
5. **Random effects**: Trains vary in both starting point (SD = %.3f) and learning speed (SD = %.6f)

## Conclusion

%s

---
*Report generated: %s*
",
  nrow(df),
  sum(df$condition == "half_goal"),
  sum(df$condition == "1.5x"),
  n_distinct(df$train_id_unified),
  n_distinct(df$train_id_unified[df$condition == "half_goal"]),
  n_distinct(df$train_id_unified[df$condition == "1.5x"]),

  AIC(m0), BIC(m0), AIC(m1), BIC(m1),

  fe$estimate[1], fe$std.error[1], fe$df[1], fe$statistic[1], fe$p.value[1],
  fe$conf.low[1], fe$conf.high[1],
  fe$estimate[2], fe$std.error[2], fe$df[2], fe$statistic[2], fe$p.value[2],
  fe$conf.low[2], fe$conf.high[2],
  fe$estimate[3], fe$std.error[3], fe$df[3], fe$statistic[3], fe$p.value[3],
  fe$conf.low[3], fe$conf.high[3],
  fe$estimate[4], fe$std.error[4], fe$df[4], fe$statistic[4], fe$p.value[4],
  fe$conf.low[4], fe$conf.high[4],

  intercept_hg,
  intercept_diff, p_intercept,
  ifelse(p_intercept < 0.05, "significant", "not significant"),
  slope_hg,
  slope_diff, p_slope,
  ifelse(p_slope < 0.05, "significant", "not significant"),

  sqrt(VarCorr(m_final)$train_id_unified[1,1]),
  sqrt(VarCorr(m_final)$train_id_unified[2,2]),

  ifelse(p_slope < 0.05,
    sprintf("The 1.5x condition shows a significantly %s learning trajectory than half_goal (interaction p = %.4f).",
            ifelse(slope_diff > 0, "steeper", "flatter"), p_slope),
    "No significant difference in learning rate between conditions."),

  format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

writeLines(report, file.path(out_dir, "lmm_analysis_summary.md"))

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("              LMM ANALYSIS COMPLETE\n")
cat("=================================================================\n\n")
cat(sprintf("Final model: normalized_task_progress ~ condition * training_step_k + (training_step_k | train_id_unified)\n"))
cat(sprintf("Observations: %d episodes, %d trains\n\n", nrow(df), n_distinct(df$train_id_unified)))
cat("Fixed effects:\n")
cat(sprintf("  half_goal intercept: %.3f\n", intercept_hg))
cat(sprintf("  1.5x intercept diff: %.3f (p = %.4f)\n", intercept_diff, p_intercept))
cat(sprintf("  half_goal slope:     %.6f per 1k steps\n", slope_hg))
cat(sprintf("  1.5x slope diff:     %.6f (p = %.4f)\n", slope_diff, p_slope))
cat("\n")
cat("Output files saved to:\n")
cat("  ", out_dir, "\n")
cat("=================================================================\n")
