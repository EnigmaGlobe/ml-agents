# =============================================================================
# 01_li_analysis.R
# Learning Improvement (LI) Analysis Pipeline
# =============================================================================
# Extracts 4 train-level indicators from each train's learning_metrics summary,
# runs Confirmatory Factor Analysis (CFA) or PCA fallback,
# and outputs LI factor / composite scores per train.
#
# Usage:
#   Rscript 01_li_analysis.R --run_dir ../../results/run_01 --out_dir ../output
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

# ---------------------------------------------------------------------------
# 1. Discover trains
# ---------------------------------------------------------------------------
discover_trains <- function(rdir) {
  dirs <- list.dirs(rdir, recursive = FALSE, full.names = TRUE)
  trains <- dirs[basename(dirs) %>% str_detect("^train\\d+")]
  tibble(
    train_id = basename(trains),
    train_path = trains
  ) %>% arrange(train_id)
}

# ---------------------------------------------------------------------------
# 2. Read LI summary CSV (multi-section text file)
# ---------------------------------------------------------------------------
read_li_summary <- function(filepath) {
  if (!file.exists(filepath)) return(NULL)
  lines <- readLines(filepath, warn = FALSE)
  start <- which(grepl("^=== SUMMARY ===", lines))
  if (length(start) == 0) return(NULL)
  start <- start[1] + 1
  end_candidates <- which(grepl("^===", lines[start:length(lines)]))
  end <- if (length(end_candidates) > 0) start + end_candidates[1] - 2 else length(lines)
  csv_text <- paste(lines[start:end], collapse = "\n")
  read_csv(csv_text, show_col_types = FALSE, na = c("", "NA", "NULL"))
}

# ---------------------------------------------------------------------------
# 3. Extract 4 train-level indicators
# ---------------------------------------------------------------------------
extract_li_indicators <- function(summary_df, train_id) {
  if (is.null(summary_df) || nrow(summary_df) == 0) {
    warning(sprintf("Empty summary for %s", train_id))
    return(NULL)
  }
  summary_df <- summary_df %>% mutate(metric = metric %>% str_squish())
  
  get_val <- function(m) {
    v <- summary_df %>% filter(metric == m) %>% pull(value)
    if (length(v) == 0) NA_real_ else as.numeric(v[1])
  }
  
  tibble(
    train_id         = train_id,
    auc_norm         = get_val("AUC (normalized)"),
    final_iqm        = get_val("Final IQM progress"),
    stability_sd     = get_val("Stability SD"),
    learning_slope   = get_val("Learning slope (per 100k)"),
    episodes         = get_val("Episodes")
  )
}

# ---------------------------------------------------------------------------
# 4. Cronbach's Alpha & CFA model (1-factor, 4 indicators)
# ---------------------------------------------------------------------------
cronbach_alpha <- function(mat, auto_reverse = TRUE) {
  # mat: data frame/matrix of items (columns), observations (rows)
  # auto_reverse: flip items that are negatively correlated with the total score
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

run_li_cfa <- function(df) {
  ind_cols <- c("auc_norm", "final_iqm", "stability_sd", "learning_slope")
  df_std <- df %>% mutate(across(all_of(ind_cols), ~ scale(.)[,1]))
  
  cfa_model <- "
    LI_FACTOR =~ auc_norm + final_iqm + stability_sd + learning_slope
  "
  
  fit <- tryCatch(
    cfa(cfa_model, data = df_std, std.lv = TRUE, missing = "fiml"),
    error = function(e) NULL
  )
  
  if (is.null(fit) || lavInspect(fit, "converged") == FALSE) {
    message("[LI] CFA did not converge (likely N too small). Falling back to PCA-based composite score.")
    
    pca <- prcomp(df_std %>% select(all_of(ind_cols)), center = FALSE, scale. = FALSE)
    pc1_scores <- pca$x[,1]
    loadings <- pca$rotation[,1]
    if (mean(loadings) < 0) pc1_scores <- -pc1_scores
    
    list(
      method = "PCA_fallback",
      scores = tibble(train_id = df$train_id, li_composite = as.numeric(pc1_scores)),
      loadings = loadings,
      fit = NULL,
      variance_explained = summary(pca)$importance[2,1]
    )
  } else {
    scores <- lavPredict(fit, type = "lv") %>% as.data.frame()
    names(scores) <- "li_composite"
    scores$train_id <- df$train_id
    
    list(
      method = "CFA",
      scores = scores %>% select(train_id, li_composite),
      loadings = parameterestimates(fit, standardized = TRUE) %>% filter(op == "=~") %>% select(lhs, rhs, est, std.all, pvalue),
      fit = fit,
      gof = fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    )
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
message(sprintf("[LI] Scanning run directory: %s", run_dir))

trains <- discover_trains(run_dir)
message(sprintf("[LI] Discovered %d train(s)", nrow(trains)))

li_data <- trains %>%
  mutate(
    summary_file = file.path(train_path, "metric", "learning improvement",
                              sprintf("learning_metrics_%s.csv", str_replace(train_id, "^train(\\d+)$", "train_\\1"))),
    summary_df   = map(summary_file, read_li_summary),
    indicators   = map2(summary_df, train_id, extract_li_indicators)
  ) %>%
  select(indicators) %>%
  unnest(indicators)

if (nrow(li_data) == 0) {
  stop("[LI] No valid learning improvement data found. Exiting.")
}

message("[LI] Indicators extracted:")
print(li_data, n = Inf)

# Cronbach's alpha on raw indicators
alpha_li <- cronbach_alpha(li_data %>% select(all_of(c("auc_norm", "final_iqm", "stability_sd", "learning_slope"))))
message(sprintf("[LI] Cronbach's Alpha (4 indicators): %.4f", alpha_li))

li_result <- run_li_cfa(li_data)

message(sprintf("[LI] Method used: %s", li_result$method))
if (li_result$method == "CFA") {
  message("[LI] CFA loadings (standardised):")
  print(li_result$loadings)
  message("[LI] CFA fit measures:")
  print(li_result$gof)
} else {
  message("[LI] PCA loadings (PC1):")
  print(li_result$loadings)
  message(sprintf("[LI] Variance explained by PC1: %.2f%%", li_result$variance_explained * 100))
}

li_output <- li_data %>%
  left_join(li_result$scores, by = "train_id")

out_csv <- file.path(out_dir, "li_scores.csv")
write_csv(li_output, out_csv)
message(sprintf("[LI] Scores written to: %s", out_csv))

sink(file.path(out_dir, "li_cfa_summary.txt"))
cat("=== CRONBACH'S ALPHA ===\n")
cat(sprintf("Cronbach's Alpha (4 indicators, auto-reversed): %.4f\n\n", alpha_li))
if (li_result$method == "CFA") {
  cat("=== CFA SUMMARY ===\n")
  print(summary(li_result$fit, standardized = TRUE, fit.measures = TRUE))
} else {
  cat("=== PCA FALLBACK ===\n")
  cat("Method: PCA\n")
  cat(sprintf("Variance explained by PC1: %.2f%%\n", li_result$variance_explained * 100))
  cat("\nLoadings (PC1):\n")
  print(li_result$loadings)
}
sink()

message("[LI] Done.")
