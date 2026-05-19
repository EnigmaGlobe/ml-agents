# =============================================================================
# 04_joint_factor.R
# Higher-Order Factor Analysis: LI vs Tensor Composites
# =============================================================================
# Treats LI composite and Tensor composite as indicators of a higher-order
# factor. Runs CFA (lavaan) and PCA fallback, outputs standardized weights.
#
# Usage:
#   Rscript 04_joint_factor.R --out_dir ../output/run01
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(lavaan)

# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# Default out_dir relative to THIS script's location, not the working directory
args_full <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_full[grep("^--file=", args_full)])
if (length(script_path) == 0 || script_path == "") {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}
default_out_dir <- file.path(script_dir, "..", "output", "run01")

out_dir  <- ifelse("--out_dir" %in% args, args[which(args == "--out_dir") + 1], default_out_dir)
out_dir <- normalizePath(out_dir, mustWork = FALSE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

li_file    <- file.path(out_dir, "li_scores.csv")
tensor_file <- file.path(out_dir, "tensor_scores.csv")

if (!file.exists(li_file))    stop(sprintf("Missing %s. Run 01_li_analysis.R first.", li_file))
if (!file.exists(tensor_file)) stop(sprintf("Missing %s. Run 02_tensor_analysis.R first.", tensor_file))

# ---------------------------------------------------------------------------
# 1. Load composites
# ---------------------------------------------------------------------------
li_df     <- read_csv(li_file, show_col_types = FALSE)
tensor_df <- read_csv(tensor_file, show_col_types = FALSE)

joint_df <- li_df %>%
  select(train_id, li_composite) %>%
  inner_join(
    tensor_df %>% select(train_id, tensor_full_composite, tensor_final20_composite),
    by = "train_id"
  )

message("[Joint] Merged composites:")
print(joint_df, n = Inf)

if (nrow(joint_df) < 3) {
  stop("[Joint] Need at least 3 trains for joint factor analysis.")
}

# ---------------------------------------------------------------------------
# 2. Helper: run joint CFA / PCA for a given tensor composite
# ---------------------------------------------------------------------------
run_joint_model <- function(df, tensor_col, suffix) {
  ind_cols <- c("li_composite", tensor_col)
  
  # Standardize
  df_std <- df %>%
    select(train_id, all_of(ind_cols)) %>%
    mutate(across(all_of(ind_cols), ~ scale(.)[,1]))
  
  # --- CFA (marker variable + std.lv: fix LI loading = 1, fix factor var = 1)
  # With 2 indicators this is JUST-IDENTIFIED (df=0) with a UNIQUE solution:
  #   loading_LI = 1 (fixed), loading_Tensor = r (correlation), 
  #   error_LI = 0, error_Tensor = 1 - r^2
  cfa_model <- sprintf("OVERALL =~ 1*li_composite + %s", tensor_col)
  fit <- tryCatch(
    cfa(cfa_model, data = df_std, std.lv = TRUE),
    error = function(e) NULL
  )
  
  # --- PCA (always works, exact weights) ---
  pca <- prcomp(df_std %>% select(all_of(ind_cols)), center = FALSE, scale. = FALSE)
  pc1_scores <- pca$x[,1]
  pc_loadings <- pca$rotation[,1]
  if (mean(pc_loadings) < 0) {
    pc1_scores <- -pc1_scores
    pc_loadings <- -pc_loadings
  }
  
  pca_result <- list(
    method = "PCA",
    scores = tibble(train_id = df_std$train_id, joint_composite = as.numeric(pc1_scores)),
    loadings = pc_loadings,
    variance_explained = summary(pca)$importance[2,1]
  )
  
  if (is.null(fit) || lavInspect(fit, "converged") == FALSE) {
    message(sprintf("[Joint-%s] CFA did not converge. Using PCA only.", suffix))
    pca_result
  } else {
    scores <- lavPredict(fit, type = "lv") %>% as.data.frame()
    names(scores) <- "joint_composite"
    scores$train_id <- df_std$train_id
    
    list(
      method = "CFA",
      scores = scores %>% select(train_id, joint_composite),
      loadings = parameterestimates(fit, standardized = TRUE) %>%
        filter(op == "=~") %>%
        select(lhs, rhs, est, std.all, pvalue),
      pca_loadings = pc_loadings,
      fit = fit,
      gof = fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    )
  }
}

# ---------------------------------------------------------------------------
# 3. Run for Tensor Full
# ---------------------------------------------------------------------------
result_full <- run_joint_model(joint_df, "tensor_full_composite", "full")
message(sprintf("\n[Joint-full] Method: %s", result_full$method))
if (result_full$method == "CFA") {
  message("[Joint-full] Standardized loadings (weights):")
  print(result_full$loadings)
  message("[Joint-full] Fit measures:")
  print(result_full$gof)
} else {
  message("[Joint-full] PCA loadings (PC1):")
  print(result_full$loadings)
  message(sprintf("[Joint-full] Variance explained: %.2f%%", result_full$variance_explained * 100))
}

# ---------------------------------------------------------------------------
# 4. Run for Tensor Final20
# ---------------------------------------------------------------------------
result_final <- run_joint_model(joint_df, "tensor_final20_composite", "final20")
message(sprintf("\n[Joint-final20] Method: %s", result_final$method))
if (result_final$method == "CFA") {
  message("[Joint-final20] Standardized loadings (weights):")
  print(result_final$loadings)
  message("[Joint-final20] Fit measures:")
  print(result_final$gof)
} else {
  message("[Joint-final20] PCA loadings (PC1):")
  print(result_final$loadings)
  message(sprintf("[Joint-final20] Variance explained: %.2f%%", result_final$variance_explained * 100))
}

# ---------------------------------------------------------------------------
# 5. Print both CFA and PCA weights
# ---------------------------------------------------------------------------
print_joint_results <- function(res, label) {
  message(sprintf("\n[Joint-%s] CFA standardized loadings:", label))
  print(res$loadings)
  message(sprintf("[Joint-%s] CFA fit measures:", label))
  print(res$gof)
  message(sprintf("[Joint-%s] PCA loadings (PC1):", label))
  print(res$pca_loadings)
  message(sprintf("[Joint-%s] PCA variance explained: %.2f%%", label, res$variance_explained * 100))
}

print_joint_results(result_full, "full")
print_joint_results(result_final, "final20")

# Build comparison table: CFA vs PCA weights
make_weight_df <- function(res, label, method_name) {
  if (method_name == "CFA") {
    res$loadings %>%
      mutate(model = paste(label, "CFA"), weight = std.all) %>%
      select(model, rhs, weight)
  } else {
    tibble(
      model = paste(label, "PCA"),
      rhs = names(res$pca_loadings),
      weight = as.numeric(res$pca_loadings)
    )
  }
}

weight_df <- bind_rows(
  make_weight_df(result_full, "LI + Tensor Full", "CFA"),
  make_weight_df(result_full, "LI + Tensor Full", "PCA"),
  make_weight_df(result_final, "LI + Tensor Final20", "CFA"),
  make_weight_df(result_final, "LI + Tensor Final20", "PCA")
)

message("\n[Joint] Combined weights table (CFA vs PCA):")
print(as.data.frame(weight_df))

p_weights <- ggplot(weight_df, aes(x = rhs, y = weight, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", weight)),
            position = position_dodge(width = 0.8), vjust = -0.5, size = 3) +
  labs(title = "CFA vs PCA Loadings (Weights): LI vs Tensor Composites",
       x = "Composite Indicator", y = "Loading / Weight",
       fill = "Model") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "joint_factor_weights.png"), p_weights, width = 8, height = 5, dpi = 150)
message(sprintf("[Joint] Weight plot saved to: %s", file.path(out_dir, "joint_factor_weights.png")))

# ---------------------------------------------------------------------------
# 6. Scatter: joint composite vs LI / Tensor
# ---------------------------------------------------------------------------
joint_output <- joint_df %>%
  left_join(result_full$scores %>% rename(joint_full = joint_composite), by = "train_id") %>%
  left_join(result_final$scores %>% rename(joint_final20 = joint_composite), by = "train_id")

p_scatter_full <- ggplot(joint_output, aes(x = li_composite, y = tensor_full_composite, color = joint_full)) +
  geom_point(size = 4) +
  scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "LI vs Tensor Full Composite (colored by Joint Factor Score)",
       x = "LI Composite", y = "Tensor Full Composite", color = "Joint Factor") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "joint_scatter_full.png"), p_scatter_full, width = 6, height = 5, dpi = 150)

p_scatter_final <- ggplot(joint_output, aes(x = li_composite, y = tensor_final20_composite, color = joint_final20)) +
  geom_point(size = 4) +
  scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "LI vs Tensor Final20 Composite (colored by Joint Factor Score)",
       x = "LI Composite", y = "Tensor Final20 Composite", color = "Joint Factor") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "joint_scatter_final20.png"), p_scatter_final, width = 6, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# 7. Summary report
# ---------------------------------------------------------------------------
report <- tibble(
  metric = c(
    "N_trains",
    "joint_full_cfa_li_loading", "joint_full_cfa_tensor_loading",
    "joint_full_pca_li_loading", "joint_full_pca_tensor_loading",
    "joint_final20_cfa_li_loading", "joint_final20_cfa_tensor_loading",
    "joint_final20_pca_li_loading", "joint_final20_pca_tensor_loading"
  ),
  value = c(
    nrow(joint_df),
    result_full$loadings$std.all[1], result_full$loadings$std.all[2],
    result_full$pca_loadings[1], result_full$pca_loadings[2],
    result_final$loadings$std.all[1], result_final$loadings$std.all[2],
    result_final$pca_loadings[1], result_final$pca_loadings[2]
  )
)

write_csv(report, file.path(out_dir, "joint_factor_summary.csv"))
message(sprintf("[Joint] Summary report saved to: %s", file.path(out_dir, "joint_factor_summary.csv")))

message("[Joint] Done.")
