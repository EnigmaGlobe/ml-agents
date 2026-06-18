# =============================================================================
# 03_advanced_analysis.R
# Advanced Analysis: LI vs Tensor Composite Scores
# =============================================================================
# Reads outputs from 01_li_analysis.R and 02_tensor_analysis.R,
# performs correlation analysis, linear regression, and
# leave-one-train-out (LOTO) cross-validation.
#
# Usage:
#   Rscript 03_advanced_analysis.R --run_name run01 --out_dir ../output/run01
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(corrplot)

# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

run_name <- ifelse("--run_name" %in% args, args[which(args == "--run_name") + 1], "run01")

# Default out_dir relative to THIS script's location, not the working directory
args_full <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_full[grep("^--file=", args_full)])
if (length(script_path) == 0 || script_path == "") {
  # Fallback for interactive sessions or environments where --file is not provided.
  script_dir <- tryCatch(
    normalizePath(dirname(sys.frames()[[1]]$ofile)),
    error = function(e) getwd()
  )
} else {
  script_dir <- dirname(normalizePath(script_path))
}
# Default out_dir relative to the script's directory.
default_out_dir <- file.path(script_dir, "..", "output", run_name)

out_dir  <- ifelse("--out_dir" %in% args, args[which(args == "--out_dir") + 1], default_out_dir)

out_dir <- normalizePath(out_dir, mustWork = FALSE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

li_file    <- file.path(out_dir, "li_scores.csv")
tensor_file <- file.path(out_dir, "tensor_scores.csv")

if (!file.exists(li_file))    stop(sprintf("Missing %s. Run 01_li_analysis.R first.", li_file))
if (!file.exists(tensor_file)) stop(sprintf("Missing %s. Run 02_tensor_analysis.R first.", tensor_file))

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
li_df    <- read_csv(li_file, show_col_types = FALSE)
tensor_df <- read_csv(tensor_file, show_col_types = FALSE)

merged <- li_df %>%
  select(train_id, li_composite, auc_norm, final_sr, learning_slope) %>%
  left_join(
    tensor_df %>% select(train_id, tensor_full_composite, tensor_final20_composite),
    by = "train_id"
  )

message("[Advanced] Merged dataset:")
print(merged, n = Inf)

# Drop rows with any missing composite scores
merged_complete <- merged %>%
  filter(!is.na(li_composite), !is.na(tensor_full_composite), !is.na(tensor_final20_composite))

n_complete <- nrow(merged_complete)
message(sprintf("[Advanced] Complete cases: %d / %d", n_complete, nrow(merged)))

if (n_complete < 3) {
  stop("[Advanced] Too few complete cases for regression. Need at least 3 trains.")
}

# ---------------------------------------------------------------------------
# 2. Correlation Analysis
# ---------------------------------------------------------------------------
corr_vars <- c("li_composite", "tensor_full_composite", "tensor_final20_composite",
               "auc_norm", "final_sr", "learning_slope")

corr_mat <- merged_complete %>%
  select(all_of(corr_vars)) %>%
  cor(use = "pairwise.complete.obs")

message("[Advanced] Correlation matrix:")
print(round(corr_mat, 3))

png(file.path(out_dir, "correlation_matrix.png"), width = 800, height = 800)
corrplot(corr_mat, method = "color", type = "upper", order = "original",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "LI vs Tensor Indicators Correlation", mar = c(0,0,1,0))
dev.off()
message(sprintf("[Advanced] Correlation plot saved to: %s", file.path(out_dir, "correlation_matrix.png")))

# Correlation table: lower triangle = r, upper triangle = p-value
corr_pvalues <- function(mat) {
  vars <- colnames(mat)
  n <- length(vars)
  pmat <- matrix(NA, nrow = n, ncol = n, dimnames = list(vars, vars))
  df <- as.data.frame(mat)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        test <- cor.test(df[[vars[i]]], df[[vars[j]]])
        pmat[i, j] <- test$p.value
      }
    }
  }
  pmat
}

pmat <- corr_pvalues(corr_mat)

# Build formatted table: lower triangle = r (3 decimals), upper triangle = p
corr_table <- corr_mat
n <- nrow(corr_table)
for (i in 1:n) {
  for (j in 1:n) {
    if (i == j) {
      corr_table[i, j] <- NA
    } else if (i < j) {
      # upper triangle = p-value
      corr_table[i, j] <- pmat[i, j]
    }
    # lower triangle already has r value
  }
}

write.csv(round(corr_table, 4), file.path(out_dir, "correlation_table.csv"))
message(sprintf("[Advanced] Correlation table saved to: %s", file.path(out_dir, "correlation_table.csv")))

# ---------------------------------------------------------------------------
# 3. Linear Regression
# ---------------------------------------------------------------------------
# Model A: Tensor (full) ~ LI
lm_full <- lm(tensor_full_composite ~ li_composite, data = merged_complete)
message("\n[Advanced] LM: tensor_full_composite ~ li_composite")
print(summary(lm_full))

# Model B: Tensor (final20) ~ LI
lm_final <- lm(tensor_final20_composite ~ li_composite, data = merged_complete)
message("\n[Advanced] LM: tensor_final20_composite ~ li_composite")
print(summary(lm_final))

# Model C: LI ~ Tensor (full)  (reverse direction)
lm_rev <- lm(li_composite ~ tensor_full_composite, data = merged_complete)
message("\n[Advanced] LM: li_composite ~ tensor_full_composite")
print(summary(lm_rev))

# ---------------------------------------------------------------------------
# 4. Scatter plots
# ---------------------------------------------------------------------------
p_full <- ggplot(merged_complete, aes(x = li_composite, y = tensor_full_composite)) +
  geom_point(size = 3, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(title = "LI Composite vs Tensor Full Composite",
       x = "LI Composite Score", y = "Tensor Full Composite Score") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "scatter_li_vs_tensor_full.png"), p_full, width = 6, height = 5, dpi = 150)

p_final <- ggplot(merged_complete, aes(x = li_composite, y = tensor_final20_composite)) +
  geom_point(size = 3, color = "forestgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(title = "LI Composite vs Tensor Final-20% Composite",
       x = "LI Composite Score", y = "Tensor Final-20% Composite Score") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "scatter_li_vs_tensor_final20.png"), p_final, width = 6, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# 5. Q-Q Plot (composite scores vs Normal)
# ---------------------------------------------------------------------------
qq_data <- function(x, name) {
  probs <- ppoints(length(x))
  theoretical <- qnorm(probs)
  sample_sorted <- sort(x)
  data.frame(theoretical = theoretical, sample = sample_sorted, variable = name)
}

qq_combined <- rbind(
  qq_data(merged_complete$li_composite, "LI Composite"),
  qq_data(merged_complete$tensor_full_composite, "Tensor Full Composite"),
  qq_data(merged_complete$tensor_final20_composite, "Tensor Final20 Composite")
)

p_qq <- ggplot(qq_combined, aes(x = theoretical, y = sample, color = variable)) +
  geom_point(size = 2, alpha = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40") +
  labs(title = "Q-Q Plot: Composite Scores vs Normal Distribution",
       x = "Theoretical Quantiles", y = "Sample Quantiles",
       color = "Variable") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "qq_plot_composites.png"), p_qq, width = 7, height = 6, dpi = 150)
message(sprintf("[Advanced] Q-Q plot saved to: %s", file.path(out_dir, "qq_plot_composites.png")))

# ---------------------------------------------------------------------------
# 6. Leave-One-Train-Out Cross-Validation (LOTO CV)
# ---------------------------------------------------------------------------
run_loto_cv <- function(df, response, predictor) {
  trains <- df$train_id
  results <- map_dfr(trains, function(left_out) {
    train_df <- df %>% filter(train_id != left_out)
    test_df  <- df %>% filter(train_id == left_out)
    
    formula_str <- paste(response, "~", predictor)
    fit <- lm(as.formula(formula_str), data = train_df)
    pred <- predict(fit, newdata = test_df)
    actual <- test_df[[response]]
    
    tibble(
      left_out = left_out,
      actual = actual,
      predicted = pred,
      residual = actual - pred,
      rmse = sqrt(mean((actual - pred)^2))  # single-obs RMSE = |residual|
    )
  })
  
  overall_rmse <- sqrt(mean(results$residual^2, na.rm = TRUE))
  overall_mae  <- mean(abs(results$residual), na.rm = TRUE)
  overall_r2   <- 1 - sum(results$residual^2) / sum((results$actual - mean(results$actual))^2)
  
  list(predictions = results, rmse = overall_rmse, mae = overall_mae, r2 = overall_r2)
}

message("\n[Advanced] Running LOTO CV ...")

cv_full  <- run_loto_cv(merged_complete, "tensor_full_composite", "li_composite")
cv_final <- run_loto_cv(merged_complete, "tensor_final20_composite", "li_composite")

message("\n[Advanced] LOTO CV: tensor_full_composite ~ li_composite")
print(cv_full$predictions)
message(sprintf("  Overall RMSE: %.4f | MAE: %.4f | R2: %.4f", cv_full$rmse, cv_full$mae, cv_full$r2))

message("\n[Advanced] LOTO CV: tensor_final20_composite ~ li_composite")
print(cv_final$predictions)
message(sprintf("  Overall RMSE: %.4f | MAE: %.4f | R2: %.4f", cv_final$rmse, cv_final$mae, cv_final$r2))

# Save CV results
write_csv(cv_full$predictions,  file.path(out_dir, "loto_cv_tensor_full.csv"))
write_csv(cv_final$predictions, file.path(out_dir, "loto_cv_tensor_final20.csv"))

# ---------------------------------------------------------------------------
# 7. Summary report
# ---------------------------------------------------------------------------
report <- tibble(
  metric = c("N_trains", "N_complete",
             "lm_full_r2", "lm_full_p",
             "lm_final20_r2", "lm_final20_p",
             "loto_full_rmse", "loto_full_mae", "loto_full_r2",
             "loto_final20_rmse", "loto_final20_mae", "loto_final20_r2"),
  value = c(
    nrow(merged), n_complete,
    summary(lm_full)$r.squared,  summary(lm_full)$coefficients[2,4],
    summary(lm_final)$r.squared, summary(lm_final)$coefficients[2,4],
    cv_full$rmse, cv_full$mae, cv_full$r2,
    cv_final$rmse, cv_final$mae, cv_final$r2
  )
)

write_csv(report, file.path(out_dir, "advanced_summary.csv"))
message(sprintf("[Advanced] Summary report saved to: %s", file.path(out_dir, "advanced_summary.csv")))

message("[Advanced] Done.")
