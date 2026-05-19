# =============================================================================
# 02_tensor_analysis.R
# Tensor CSV Analysis Pipeline
# =============================================================================
# For each train, reads 6 tensor CSVs, computes two sets of train-level
# aggregates (full-train IQM and final-20% IQM), then runs CFA / PCA
# to obtain Tensor factor / composite scores.
#
# Usage:
#   Rscript 02_tensor_analysis.R --run_dir ../../results/run_01 --out_dir ../output
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(lavaan)

# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

run_dir  <- ifelse("--run_dir" %in% args, args[which(args == "--run_dir") + 1], "C:/Users/infra/OneDrive/Desktop/forCFA/run01")
out_dir  <- ifelse("--out_dir" %in% args, args[which(args == "--out_dir") + 1], file.path("../output", basename(run_dir)))

run_dir  <- normalizePath(run_dir,  mustWork = FALSE)
out_dir  <- normalizePath(out_dir, mustWork = FALSE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

TENSOR_TAGS <- c(
  "Cumulative Reward",
  "Episode Length",
  "Learning Rate",
  "Policy Loss",
  "Value Loss"
)

# ---------------------------------------------------------------------------
# 1. Discover trains
# ---------------------------------------------------------------------------
discover_trains <- function(rdir) {
  dirs <- list.dirs(rdir, recursive = FALSE, full.names = TRUE)
  trains <- dirs[basename(dirs) %>% str_detect("^train\\d+")]
  tibble(train_id = basename(trains), train_path = trains) %>% arrange(train_id)
}

# ---------------------------------------------------------------------------
# 2. IQM (Interquartile Mean)
# ---------------------------------------------------------------------------
compute_iqm <- function(x) {
  x <- na.omit(x)
  if (length(x) == 0) return(NA_real_)
  q1 <- quantile(x, 0.25, names = FALSE)
  q3 <- quantile(x, 0.75, names = FALSE)
  subset <- x[x >= q1 & x <= q3]
  if (length(subset) == 0) return(mean(x))  # fallback
  mean(subset)
}

# ---------------------------------------------------------------------------
# 3. Process one train's tensor CSVs
# ---------------------------------------------------------------------------
process_train_tensors <- function(train_path, train_id) {
  tensor_dir <- file.path(train_path, "tensor")
  if (!dir.exists(tensor_dir)) {
    warning(sprintf("No tensor dir for %s", train_id))
    return(NULL)
  }
  
  results <- map_dfr(TENSOR_TAGS, function(tag) {
    csv_path <- file.path(tensor_dir, sprintf("%s.csv", tag))
    if (!file.exists(csv_path)) {
      warning(sprintf("Missing %s for %s", tag, train_id))
      return(tibble(train_id = train_id, tag = tag, iqm_full = NA, iqm_final20 = NA))
    }
    
    df <- read_csv(csv_path, show_col_types = FALSE, col_types = cols(
      `Wall time` = col_double(),
      Step = col_double(),
      Value = col_double()
    ))
    
    if (nrow(df) == 0) {
      return(tibble(train_id = train_id, tag = tag, iqm_full = NA, iqm_final20 = NA))
    }
    
    df <- df %>% arrange(Step)
    max_step <- max(df$Step, na.rm = TRUE)
    threshold <- max_step * 0.8
    
    iqm_full <- compute_iqm(df$Value)
    df_final <- df %>% filter(Step >= threshold)
    iqm_final20 <- if (nrow(df_final) > 0) compute_iqm(df_final$Value) else iqm_full
    
    tibble(train_id = train_id, tag = tag, iqm_full = iqm_full, iqm_final20 = iqm_final20)
  })
  
  results
}

# ---------------------------------------------------------------------------
# 4. Cronbach's Alpha & CFA / PCA for Tensor indicators
# ---------------------------------------------------------------------------
cronbach_alpha <- function(mat, auto_reverse = TRUE) {
  k <- ncol(mat)
  if (k < 2) return(NA_real_)
  mat <- as.data.frame(mat)
  
  if (auto_reverse) {
    for (i in seq_len(k)) {
      it_cor <- cor(mat[[i]], rowSums(mat[, -i, drop = FALSE], na.rm = TRUE), use = "pairwise.complete.obs")
      if (!is.na(it_cor) && it_cor < 0) {
        mat[[i]] <- -mat[[i]]
      }
    }
  }
  
  item_vars <- apply(mat, 2, var, na.rm = TRUE)
  total_var <- var(rowSums(mat, na.rm = TRUE), na.rm = TRUE)
  if (total_var <= 0) return(NA_real_)
  (k / (k - 1)) * (1 - sum(item_vars) / total_var)
}

run_tensor_cfa <- function(df_wide, suffix) {
  ind_cols <- TENSOR_TAGS %>% str_replace_all(" ", "_") %>% str_to_lower()
  # sanitise column names to match
  names(df_wide) <- names(df_wide) %>% str_replace_all(" ", "_") %>% str_to_lower()
  
  avail <- ind_cols[ind_cols %in% names(df_wide)]
  if (length(avail) < 3) {
    stop(sprintf("[Tensor-%s] Too few valid indicators (%d). Need at least 3.", suffix, length(avail)))
  }
  
  df_std <- df_wide %>%
    select(train_id, all_of(avail)) %>%
    mutate(across(all_of(avail), ~ scale(.)[,1]))
  
  # Build lavaan syntax dynamically
  model_str <- paste("TENSOR_FACTOR =~", paste(avail, collapse = " + "))
  
  fit <- tryCatch(
    cfa(model_str, data = df_std, std.lv = TRUE, missing = "fiml"),
    error = function(e) NULL
  )
  
  if (is.null(fit) || lavInspect(fit, "converged") == FALSE) {
    message(sprintf("[Tensor-%s] CFA did not converge. Falling back to PCA.", suffix))
    pca <- prcomp(df_std %>% select(all_of(avail)), center = FALSE, scale. = FALSE)
    pc1_scores <- pca$x[,1]
    loadings <- pca$rotation[,1]
    if (mean(loadings) < 0) pc1_scores <- -pc1_scores
    
    list(
      method = "PCA_fallback",
      scores = tibble(train_id = df_std$train_id, tensor_composite = as.numeric(pc1_scores)),
      loadings = loadings,
      fit = NULL,
      variance_explained = summary(pca)$importance[2,1]
    )
  } else {
    scores <- lavPredict(fit, type = "lv") %>% as.data.frame()
    names(scores) <- "tensor_composite"
    scores$train_id <- df_std$train_id
    
    list(
      method = "CFA",
      scores = scores %>% select(train_id, tensor_composite),
      loadings = parameterestimates(fit, standardized = TRUE) %>% filter(op == "=~") %>% select(lhs, rhs, est, std.all, pvalue),
      fit = fit,
      gof = fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    )
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
message(sprintf("[Tensor] Scanning run directory: %s", run_dir))

trains <- discover_trains(run_dir)
message(sprintf("[Tensor] Discovered %d train(s)", nrow(trains)))

tensor_long <- trains %>%
  mutate(data = map2(train_path, train_id, process_train_tensors)) %>%
  select(data) %>%
  unnest(data)

if (nrow(tensor_long) == 0) {
  stop("[Tensor] No tensor data found. Exiting.")
}

# Pivot to wide format
make_wide <- function(df, value_col) {
  df %>%
    select(train_id, tag, !!sym(value_col)) %>%
    pivot_wider(names_from = tag, values_from = !!sym(value_col))
}

tensor_full_wide  <- make_wide(tensor_long, "iqm_full")
tensor_final_wide <- make_wide(tensor_long, "iqm_final20")

message("[Tensor] Full-train IQM wide:")
print(tensor_full_wide, n = Inf)
message("[Tensor] Final-20% IQM wide:")
print(tensor_final_wide, n = Inf)

# --- Full-train CFA ---
result_full <- run_tensor_cfa(tensor_full_wide, "full")
message(sprintf("[Tensor-full] Method: %s", result_full$method))
if (result_full$method == "CFA") {
  print(result_full$loadings)
  print(result_full$gof)
} else {
  print(result_full$loadings)
  message(sprintf("[Tensor-full] Variance explained: %.2f%%", result_full$variance_explained * 100))
}

# --- Final-20% CFA ---
result_final <- run_tensor_cfa(tensor_final_wide, "final20")
message(sprintf("[Tensor-final20] Method: %s", result_final$method))
if (result_final$method == "CFA") {
  print(result_final$loadings)
  print(result_final$gof)
} else {
  print(result_final$loadings)
  message(sprintf("[Tensor-final20] Variance explained: %.2f%%", result_final$variance_explained * 100))
}

# Merge outputs
tensor_output <- tensor_full_wide %>%
  left_join(result_full$scores %>% rename(tensor_full_composite = tensor_composite), by = "train_id") %>%
  left_join(result_final$scores %>% rename(tensor_final20_composite = tensor_composite), by = "train_id")

out_csv <- file.path(out_dir, "tensor_scores.csv")
write_csv(tensor_output, out_csv)
message(sprintf("[Tensor] Scores written to: %s", out_csv))

# Save long-form for traceability
out_long <- file.path(out_dir, "tensor_iqm_long.csv")
write_csv(tensor_long, out_long)

if (!is.null(result_full$fit)) {
  sink(file.path(out_dir, "tensor_cfa_full_summary.txt"))
  print(summary(result_full$fit, standardized = TRUE, fit.measures = TRUE))
  sink()
}
if (!is.null(result_final$fit)) {
  sink(file.path(out_dir, "tensor_cfa_final20_summary.txt"))
  print(summary(result_final$fit, standardized = TRUE, fit.measures = TRUE))
  sink()
}

message("[Tensor] Done.")
