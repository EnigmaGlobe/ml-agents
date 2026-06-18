# PushBlock Learning Improvement v3 FRD

## 1. Purpose

This document defines the requirements and technical approach for the **third revision** of the PushBlock learning-improvement analysis pipeline.

The v2 pipeline introduced robust episode-level exports and a basic metric quartet (AUC, final success rate, stability SD, learning slope). However, empirical analysis of those metrics revealed severe structural redundancy: **AUC, stability, and learning slope are all driven by the same underlying variable (`normalized_task_progress`)**, producing correlations near 0.9 and Heywood-case pathologies in factor analysis.

The goal of v3 is to:

1. Replace the redundant v2 quartet with a **low-collinearity, process-first metric set**
2. Use **all available episode data families** (`success`, `final_goal_zone_error_xz`, `time_to_goal`), not just `normalized_task_progress`
3. Ensure every headline metric is **progressive** (computed over the full training trajectory), not final-window-only
4. Provide a **confirmable measurement model** (CFA-ready) with at least two latent dimensions

This FRD is code-backed. Every metric is traceable to existing `learning_improvement.csv` columns and to specific functions in `config/learning_metrics.py`.

---

## 2. Background & Problem Statement

### 2.1 v2 Metric Redundancy

The v2 pipeline computes four headline metrics:

| v2 Metric | Primary Input | Window |
|---|---|---|
| Normalized AUC | `training_step` + `normalized_task_progress` | Rolling mean across full run |
| Learning slope | `training_step` + `normalized_task_progress` | Rolling mean across full run |
| Stability SD | `normalized_task_progress` | Final 10% of episodes only |
| Final success rate | `success` | Final 10% of episodes only |

**Problem:** AUC, slope, and stability all depend on `normalized_task_progress`. Empirically, their inter-correlations approach **r ≈ 0.9**, because:

- AUC integrates the rolling progress curve; the late-training plateau dominates the area
- Stability SD uses the same progress values, just in the final window
- Both reward the same latent pattern: "high progress late in training + low variance late in training"

This creates **Heywood-like factor-analysis pathology**: if you treat the four metrics as indicators of one construct, the model is under-identified or produces impossible estimates.

### 2.2 Underutilized Data

`PushAgentBasic.cs` exports 11 episode columns, but the v2 headline metrics consume only **3** of them:

- `training_step`
- `success`
- `normalized_task_progress`

These columns are logged but **not used as primary evidence** in v2:

- `episode_reward`
- `episode_length`
- `time_to_goal`
- `start_goal_zone_error_xz`
- `final_goal_zone_error_xz`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`

The v3 pipeline must exploit these unused signals to build statistically independent indicators.

### 2.3 Final-Window Bias

v2 labels like "final success rate" and "final goal-error stability" bake in an end-state framing. For a metric suite named **learning improvement**, the emphasis should be on **how competence develops over time**, not only where the agent ends up. v3 therefore eliminates "final" semantics and computes all core metrics on rolling trajectories.

---

## 3. Terminology & Definitions

| Term | Definition |
|---|---|
| **Episode** | One complete PushBlock trial, from `OnEpisodeBegin` to success or `MaxStep` timeout |
| **Training step** | The global step count at episode end, drawn from `Academy.Instance.StepCount` |
| **Normalized task progress** | `(start_goal_zone_error_xz - final_goal_zone_error_xz) / max(start_goal_zone_error_xz, 0.0001)` |
| **Rolling window** | A trailing mean (or SD) computed over the last `W` episodes ending at index `i` |
| **Trapezoidal AUC** | Integration of a rolling series against `training_step` using the trapezoid rule |
| **Heywood case** | In factor analysis, a pathological solution where variance estimates are negative or loadings are impossible, often caused by excessive collinearity |
| **CFA** | Confirmatory Factor Analysis; a structural equation model used to test whether observed indicators load on hypothesized latent dimensions |
| **IQM** | Interquartile Mean; the mean of values between the 25th and 75th percentiles |
| **CV** | Coefficient of Variation; `sd / abs(mean)`. In v2 this was used for stability but is **deprecated in v3** because it explodes when the mean is near zero |
| **Progressive metric** | A metric computed over the full training trajectory (rolling windows across all episodes), not just the final window |

---

## 4. Data Sources & Contract

### 4.1 Primary input

`learning_improvement.csv`, exported by `PushAgentBasic.cs` at episode end.

### 4.2 Required columns for v3 analysis

| Column | Type | Source | Used in v3 by |
|---|---|---|---|
| `training_step` | int | `Academy.Instance.StepCount` | All rolling and AUC computations |
| `success` | int (0/1) | `GoalDetect.cs` trigger | Competence arrival, rolling success attainment |
| `normalized_task_progress` | float | Derived in `PushAgentBasic.cs` | Learning exposure, retention |
| `final_goal_zone_error_xz` | float | Euclidean distance in xz plane at episode end | Rolling goal-error consistency |
| `time_to_goal` | int | Steps from episode start to success; `-1` if fail | Rolling time-to-goal efficiency |

### 4.3 Diagnostic columns (optional)

| Column | Type | Used for |
|---|---|---|
| `episode_reward` | float | Reward stability diagnostics (optional) |
| `episode_length` | int | Episode duration diagnostics |
| `start_goal_zone_error_xz` | float | Upstream of `normalized_task_progress` (indirect) |

### 4.4 Deprecated for headline metrics

- `final_goal_zone_error_xz` used **only** in a final window → replaced by rolling trajectory
- `normalized_task_progress` used as the **sole** signal for >1 headline metric → now restricted to Learning Dynamics dimension only

---

## 5. Metric Architecture

### 5.1 Measurement model (CFA target)

The v3 suite is designed to support a **correlated two-factor CFA** with two first-order dimensions. A second-order "Learning Improvement" factor is **not recommended** until at least three first-order dimensions are available.

```
Learning_Dynamics  <----->  Attainment_Consistency
       |                           |
   +---+---+                   +---+---+---+
   |   |   |                   |   |   |
  LES CAS LRS                RSA RGEC RTGE
```

Where:

- **LES** = Learning Exposure Score
- **CAS** = Competence Arrival Step
- **LRS** = Learning Retention Score
- **RSA** = Rolling Success Attainment
- **RGEC** = Rolling Goal-Error Consistency
- **RTGE** = Rolling Time-to-Goal Efficiency

### 5.2 Why two dimensions?

| Dimension | Question it answers | Metric family |
|---|---|---|
| **Learning Dynamics** | "How did learning unfold over time?" | Curve shape, speed, persistence |
| **Attainment Consistency** | "How reliably does the agent accomplish the task?" | Completion rate, spatial precision, operational efficiency |

An agent can learn fast and still be inconsistent (high dynamics, low attainment), or learn slowly and end up reliable (low dynamics, high attainment). Treating all indicators as one factor ignores this distinction and invites Heywood cases.

### 5.3 Progressive design rule

> Every core metric must be computable from a rolling trajectory over `training_step`. No core metric may be defined as "final window only".

The only exception is diagnostic output (e.g., raw final-window IQM for human inspection), which does not enter the CFA model.

---

## 6. Core Metrics (6 Headline + 2 Backup)

### 6.1 Learning Exposure Score (LES)

| Attribute | Value |
|---|---|
| **Dimension** | Learning Dynamics |
| **Purpose** | Whole-training quality: how much sustained good performance the agent accumulates across the run |
| **Inputs** | `training_step`, `normalized_task_progress` |
| **Formula** | 1. Build rolling progress: `rp_i = mean(normalized_task_progress over trailing W episodes ending at i)` <br> 2. Trapezoidal integration: `AUC = Σ(i=2..n) (step_i - step_{i-1}) * (rp_i + rp_{i-1}) / 2` <br> 3. Normalize by span: `LES = AUC / (step_n - step_1)` |
| **Direction** | Higher is better (max 1.0 if progress stays at 1.0 across the entire span) |
| **Output columns** | `learning_exposure_score`, `learning_exposure_raw_auc`, `learning_exposure_span` |
| **Implementation notes** | Reuse existing `compute_auc()` in `learning_metrics.py`. Rename output conceptually; formula is identical to v2 normalized AUC. |
| **Heywood risk** | Low when used alone; **high** if bundled with other progress-derived metrics (which v3 avoids) |

---

### 6.2 Competence Arrival Step (CAS)

| Attribute | Value |
|---|---|
| **Dimension** | Learning Dynamics |
| **Purpose** | Learning speed: the first training step where the policy becomes reliably usable |
| **Inputs** | `training_step`, `success` |
| **Formula** | 1. Build rolling success: `rs_i = mean(success over trailing W episodes)` <br> 2. Choose threshold `T` (default `0.80`) and persistence `K` (default `3` consecutive rolling points) <br> 3. `CAS_start = first(training_step_i)` where the sustained period begins <br> 4. `CAS_confirm = training_step_{i+K-1}` where the sustained period is confirmed <br> 5. **Headline metric:** `CAS_ratio = CAS_confirm / total_training_steps` (0–1, lower is better) <br> 6. If never reached: `CAS_ratio = NaN`, `competence_arrival_reached = false` |
| **Direction** | Lower is better (earlier competence, relative to total run length) |
| **Output columns** | `competence_arrival_start_step`, `competence_arrival_start_episode`, `competence_arrival_confirm_step`, `competence_arrival_confirm_episode`, `competence_arrival_ratio`, `competence_arrival_reached` |
| **Implementation notes** | Add `rolling_success_rate` to `compute_rolling()`. Scan forward for sustained crossing. Using `success` instead of `normalized_task_progress` decouples this metric from LES. |
| **Heywood risk** | Medium. Correlates with final success trajectory but measures a different concept (speed vs. area). |

---

### 6.3 Learning Retention Score (LRS)

| Attribute | Value |
|---|---|
| **Dimension** | Learning Dynamics |
| **Purpose** | Collapse / forgetting detection: whether the agent learned and then regressed |
| **Inputs** | `normalized_task_progress` (rolling series) |
| **Formula** | 1. Use same `rp_i` series as LES <br> 2. `best = max_i(rp_i)` <br> 3. `end = rp_n` <br> 4. `retention_drop = best - end` <br> 5. Optionally convert to higher-is-better score: `LRS = 1 - max(0, retention_drop)` |
| **Direction** | Higher LRS (or lower raw drop) is better |
| **Output columns** | `learning_retention_drop`, `learning_retention_score`, `best_rolling_progress`, `end_rolling_progress` |
| **Implementation notes** | Single pass over existing rolling series. No new data needed. Detects policies that peak early and then degrade. |
| **Heywood risk** | Medium-High with LES (both use `rp_i`), but conceptually distinct (peak-to-end drop vs. accumulated area). Acceptable as the third indicator in a 3-indicator factor because the construct is "Learning Dynamics," not a single scalar. |

---

### 6.4 Rolling Success Attainment (RSA)

| Attribute | Value |
|---|---|
| **Dimension** | Attainment Consistency |
| **Purpose** | Progressive task-completion behavior: not "final success rate" but "how successfully the agent behaves across the whole training period" |
| **Inputs** | `training_step`, `success` |
| **Formula** | 1. Build rolling success: `rs_i = mean(success over trailing W episodes)` <br> 2. Trapezoidal integration over `training_step`: `success_auc = Σ(i=2..n) (step_i - step_{i-1}) * (rs_i + rs_{i-1}) / 2` <br> 3. Normalize: `RSA = success_auc / (step_n - step_1)` |
| **Direction** | Higher is better (max 1.0) |
| **Output columns** | `rolling_success_attainment`, `rolling_success_raw_auc`, `rolling_success_span` |
| **Implementation notes** | Reuse `compute_auc()` structure, but pass `rs_i` as the y-series instead of `rp_i`. This is the "success-family" counterpart to LES. |
| **Heywood risk** | Low with LES (different y variable: binary completion vs. continuous progress). Medium with CAS because both use `success`, but CAS measures a threshold-crossing step while RSA measures accumulated area. |

---

### 6.5 Rolling Goal-Error Consistency (RGEC)

| Attribute | Value |
|---|---|
| **Dimension** | Attainment Consistency |
| **Purpose** | Progressive spatial consistency: how reliably the agent keeps the block close to the target across training, using direct geometry instead of derived progress |
| **Inputs** | `training_step`, `final_goal_zone_error_xz` |
| **Formula** | 1. Rolling error mean: `ge_i = mean(final_goal_zone_error_xz over trailing W episodes)` <br> 2. Rolling error SD: `gs_i = sd(final_goal_zone_error_xz over trailing W episodes)` <br> 3. Summarize consistency: `mean_rolling_goal_error_sd = mean(gs_i)` <br> 4. Soft-normalized consistency score (unit-invariant): `RGEC = 1 / (1 + mean_rolling_goal_error_sd)` |
| **Direction** | Higher RGEC is better (closer to 1.0 = perfectly consistent); lower `mean_rolling_goal_error_sd` is also better |
| **Output columns** | `rolling_goal_error_consistency`, `mean_rolling_goal_error_sd` |
| **Implementation notes** | Extend `compute_rolling()` to emit `mean_final_goal_zone_error_xz` and `sd_final_goal_zone_error_xz`. Summarize the rolling SD series with a simple mean or with its own AUC. |
| **Heywood risk** | Low. Uses a different raw field (`final_goal_zone_error_xz`) and a dispersion summary rather than a mean or area metric. |

---

### 6.6 Rolling Time-to-Goal Efficiency (RTGE)

| Attribute | Value |
|---|---|
| **Dimension** | Attainment Consistency |
| **Purpose** | Progressive operational efficiency: distinguishes "can solve" from "solves quickly" across training |
| **Inputs** | `training_step`, `time_to_goal`, `success`, and training-config `MaxStep` |
| **Formula** | 1. In each rolling window, keep only successful episodes where `time_to_goal >= 0` <br> 2. Compute rolling mean `time_to_goal`: `rtg_i = mean(time_to_goal over successful episodes in trailing W)` <br> 3. Convert to efficiency: `eff_i = 1 - (rtg_i / MaxStep)` <br> 4. Summarize: `RTGE = mean(eff_i)` <br> **MaxStep must come from training config (CLI `--max-step`), not inferred from observed episode lengths.** |
| **Direction** | Higher is better (closer to 1.0 means faster successful episodes) |
| **Output columns** | `rolling_time_to_goal_efficiency_score`, `rolling_mean_time_to_goal`, `rolling_time_to_goal_valid_windows` |
| **Implementation notes** | If a window has zero successes, `rtg_i` is undefined; skip or propagate backward. `MaxStep` is the episode timeout (known from training config). |
| **Heywood risk** | Low. Uses `time_to_goal`, a distinct data family from success, progress, and goal error. |

---

### 6.7 Backup: Learning Velocity (LV)

| Attribute | Value |
|---|---|
| **Dimension** | Learning Dynamics (backup) |
| **Purpose** | Reserve indicator for learning speed if CAS or LRS behaves badly |
| **Inputs** | `training_step`, `normalized_task_progress` (rolling series) |
| **Formula** | OLS linear regression on `training_step` vs `rp_i`; slope, `r_squared`, `p_value` |
| **Output columns** | `learning_velocity_slope`, `learning_velocity_r2`, `learning_velocity_per_100k` |
| **When to use** | Only if one of the three primary Learning Dynamics indicators must be dropped due to poor loading or excessive residual correlation |
| **Heywood risk** | High with LES (same x-y pair). Use strictly as backup. |

---

### 6.8 Backup: Downside Robustness Score (DRS)

| Attribute | Value |
|---|---|
| **Dimension** | Attainment Consistency (backup) |
| **Purpose** | Tail-risk indicator: captures worst-case behavior hidden by mean summaries |
| **Inputs** | `final_goal_zone_error_xz` or `success` (rolling series) |
| **Formula** | Preferred: upper-tail rolling goal error (e.g., 90th percentile of `ge_i` or CVaR-10%) <br> Alternative: lower-tail rolling success (e.g., 10th percentile of `rs_i`) |
| **Output columns** | `downside_robustness_score`, `downside_robustness_metric` |
| **When to use** | Only if one of the three primary Attainment Consistency indicators must be dropped |
| **Heywood risk** | Medium if computed on `normalized_task_progress` (avoid). Low if computed on `final_goal_zone_error_xz` or `success`. |

---

## 7. Data-to-Metric Mapping

### 7.1 Direct input matrix

| Metric | `training_step` | `success` | `normalized_task_progress` | `final_goal_zone_error_xz` | `time_to_goal` |
|---|---|---|---|---|---|
| Learning Exposure Score (LES) | X | | X | | |
| Competence Arrival Step (CAS) | X | X | | | |
| Learning Retention Score (LRS) | | | X | | |
| Rolling Success Attainment (RSA) | X | X | | | |
| Rolling Goal-Error Consistency (RGEC) | X | | | X | |
| Rolling Time-to-Goal Efficiency (RTGE) | X | X | | | X |
| Learning Velocity (backup) | X | | X | | |
| Downside Robustness Score (backup) | X | X | | X | |

### 7.2 Indirect upstream dependencies

`normalized_task_progress` itself is derived from `start_goal_zone_error_xz` and `final_goal_zone_error_xz` in `PushAgentBasic.cs`. These matter upstream but are not direct inputs to the Python metric pipeline.

---

## 8. CFA / Measurement Model Design

### 8.1 Why not one factor?

A single-factor CFA with 6 indicators is technically identified, but in this domain it is not theoretically defensible:

- LES, CAS, and LRS describe **trajectory shape over time**
- RSA, RGEC, and RTGE describe **task execution quality**

An agent can have steep learning dynamics but poor attainment consistency, or vice versa. Forcing them into one dimension produces high residual correlations and unstable loadings.

### 8.2 Recommended CFA structure

**Model:** Correlated two-factor CFA (no higher-order factor yet)

```
Learning_Dynamics  <----->  Attainment_Consistency
       |                           |
   +---+---+                   +---+---+---+
   |   |   |                   |   |   |
  LES CAS LRS                RSA RGEC RTGE
```

**Constraints for identification:**
- Fix one loading per factor to 1.0 (e.g., LES on Learning_Dynamics; RSA on Attainment_Consistency)
- Estimate factor correlation freely
- Allow unique variances to be freely estimated

### 8.3 When to add a higher-order factor

A second-order "Learning Improvement" factor requires **at least 3 first-order dimensions**. With only 2 dimensions, the higher-order construct is empirically underidentified. If a third dimension (e.g., **Operational Efficiency** or **Robustness**) is added later, then a second-order model becomes viable.

### 8.4 CFA input: continuous values, not verdict labels

> **Do not use verdict labels (EXCELLENT / STRONG / PASS / WARN / FAIL) as CFA indicators.**

Verdicts are ordinal simplifications for human inspection. CFA should use the underlying continuous scalars:
- Learning Dynamics: `learning_exposure_score`, `competence_arrival_ratio`, `learning_retention_score`
- Attainment Consistency: `rolling_success_attainment`, `rolling_goal_error_consistency`, `rolling_time_to_goal_efficiency_score`

If a discrete indicator is absolutely required, use the binary `competence_arrival_reached` flag, not the ordinal verdict.

### 8.5 Sample-size rule of thumb

For a 2-factor CFA with 6 indicators, aim for **at least 100–150 independent training runs** (or episodes, depending on the unit of analysis) for stable parameter estimates. If the sample is smaller, treat the CFA as exploratory and report robust standard errors.

---

## 9. Implementation Plan

### 9.1 Target file

`config/learning_metrics.py`

### 9.2 Step-by-step changes

#### Step 1: Extend `compute_rolling()`

Current `compute_rolling()` emits rolling mean of `normalized_task_progress`. Extend it to emit:

- `rolling_success_rate`: trailing mean of `success`
- `rolling_mean_goal_error`: trailing mean of `final_goal_zone_error_xz`
- `rolling_sd_goal_error`: trailing SD of `final_goal_zone_error_xz`
- `rolling_mean_time_to_goal`: trailing mean of `time_to_goal` (successful episodes only; `NaN` if no successes)

#### Step 2: Reuse `compute_auc()` for multiple series

Refactor `compute_auc()` to accept a generic `x` (training_step) and `y` (rolling series) pair. Call it for:

- `y = rolling_progress` → LES
- `y = rolling_success_rate` → RSA

Optionally call it for:
- `y = rolling_efficiency` → RTGE (if using AUC summarization instead of mean)

#### Step 0: Add required-column validation and CLI parameters

- Validate that `learning_improvement.csv` contains required columns: `training_step`, `success`, `normalized_task_progress`, `final_goal_zone_error_xz`, `time_to_goal`. Raise an error if any are missing or empty.
- Sort episodes by `(training_step, source_row)` and validate `training_step` monotonicity.
- Add CLI arguments:
  - `--max-step <int>`: Episode MaxStep from training config. **Required for RTGE.** Do not infer from observed `episode_length`.
  - `--rolling-window <int>`: Override rolling window size. Default is `max(25, ceil(0.01 * n))`.

#### Step 3: Add new computation functions

```python
def compute_competence_arrival(
    rolling_success: list[dict],
    threshold: float = 0.80,
    sustain: int = 3,
    min_window_size: int | None = None,
) -> dict: ...

def compute_retention_drop(rolling_progress: list[dict]) -> dict: ...

def compute_goal_error_consistency(rolling_error_sd: list[dict]) -> dict: ...
```

#### Step 4: Update top-level summary output

Replace the v2 headline quartet with the v6 headline sextet:

```python
summary = {
    "learning_exposure_score": les,
    "competence_arrival_start_step": cas_start,
    "competence_arrival_confirm_step": cas_confirm,
    "competence_arrival_reached": cas_reached,
    "learning_retention_score": lrs,
    "rolling_success_attainment": rsa,
    "mean_rolling_goal_error_sd": rgec_sd,
    "rolling_goal_error_consistency": rgec,
    "rolling_time_to_goal_efficiency_score": rtge,
}
```

Keep v2 metrics as **diagnostic rows** in the output CSV (not headline), so historical comparisons remain possible:

- `final_success_rate` (diagnostic)
- `final_iqm_progress` (diagnostic)
- `stability_sd_progress` (diagnostic)
- `learning_slope` (diagnostic)

#### Step 4.5: Add CFA-ready key-value summary section

In addition to the human-readable segmented CSV, append a machine-readable `[V3 CFA SUMMARY (KEY-VALUE)]` section at the end of the output. Each row is `[column_name, value]`. This allows downstream scripts to parse the six headline metrics and their verdicts without heuristically skipping header rows.

Example rows:
- `learning_exposure_score`
- `competence_arrival_confirm_step`
- `learning_retention_score`
- `rolling_success_attainment`
- `mean_rolling_goal_error_sd`
- `rolling_goal_error_consistency`
- `rolling_time_to_goal_efficiency_score`
- plus corresponding `*_verdict` fields

#### Step 5: Verdict thresholds

Define verdict bands for each new metric. Example defaults:

| Metric | EXCELLENT | STRONG | PASS | WARN | FAIL |
|---|---|---|---|---|---|
| LES | >= 0.60 | >= 0.45 | >= 0.30 | >= 0.15 | < 0.15 |
| CAS (ratio) | <= 0.25 | <= 0.35 | <= 0.50 | <= 0.70 | > 0.70 or never |
| LRS (score) | >= 0.95 | >= 0.85 | >= 0.70 | >= 0.50 | < 0.50 |
| RSA | >= 0.60 | >= 0.45 | >= 0.30 | >= 0.15 | < 0.15 |
| RGEC | >= 0.35 | >= 0.28 | >= 0.20 | >= 0.15 | < 0.15 |
| RTGE | >= 0.80 | >= 0.65 | >= 0.50 | >= 0.30 | < 0.30 |

> **Note:** CAS and RGEC thresholds have been calibrated against an initial batch of 32 training runs (run_02 + run_03). LES / LRS / RSA / RTGE thresholds remain provisional and should also be calibrated once a larger representative sample is available.

---

## 10. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Residual collinearity between LES and LRS** | Medium | High (Heywood) | Keep both in Learning Dynamics but monitor factor correlation. If >0.85, drop LRS and promote Learning Velocity as backup. |
| **CAS ceiling effect** (all runs reach competence early) | Medium | Medium (loss of variance) | If >80% of runs reach competence in the first window, lower threshold or add a stricter threshold (e.g., 0.90). Report `competence_arrival_reached` as a binary auxiliary variable. |
| **RGEC undefined windows** (no valid `time_to_goal` in early training) | High in early runs | Low | Skip undefined windows in rolling mean; do not backfill. This is valid: early training simply has no efficiency data. |
| **CFA misfit with 6 indicators** | Medium | Medium | Use EFA first to verify 2-factor structure. If fit is poor, drop the weakest-loading indicator per factor and use backups. |
| **Threshold mis-calibration** | High initially | Medium | Treat v3 verdicts as **provisional** for the first N runs. Collect run statistics and set thresholds at empirical quartiles or IQM-based cutoffs. |
| **Rolling window width `W` too small / too large** | Medium | Medium | Make `W` a configurable parameter. Default to 25 or 1% of total episodes, whichever is larger. Document sensitivity analysis in the first report. |

---

## 11. Migration from v2

| v2 Artifact | v3 Status | Action |
|---|---|---|
| `normalized_auc` | Renamed → `learning_exposure_score` | Keep formula; rename output column |
| `final_success_rate` | Demoted to diagnostic | Keep computation; move to diagnostic section of output CSV |
| `stability_sd_progress` | Demoted to diagnostic | Keep computation; do not use in headline summary |
| `learning_slope` | Demoted to backup (Learning Velocity) | Keep computation; activate only if a primary dynamics indicator drops |
| `auc_verdict` | Renamed → `learning_exposure_verdict` | Re-label; keep thresholds as initial defaults |
| `overall_verdict` | Deprecated | Replace with dimensional verdicts or a composite index after CFA calibration |

---

## 12. Open Questions

1. ~~Rolling window width `W`~~ **Resolved**: default `max(25, ceil(0.01 * n))`, overridable via `--rolling-window`.
2. **Competence threshold `T`**: Is 0.80 success rate the right bar for "usable competence," or should it be task-dependent?
3. ~~RTGE summarization~~ **Resolved**: use `mean(eff_i)` for simplicity and to avoid over-alignment with LES/RSA.
4. **CFA unit of analysis**: Should the model be fit per **training run** (n = number of runs) or per **episode** (n = number of episodes)? The construct "Learning Improvement" is typically a run-level property.
5. **Downside robustness metric family**: Should DRS use CVaR (conditional value at risk) or a simple percentile? CVaR is more principled but requires a larger sample.

---

## 13. References

- Agarwal, R., et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS 2021. (RLiable robust evaluation framework)
- v2 FRD: `PushBlock_Learning_Improvement_V2_FRD.md`
- v2 Tech Spec: `PushBlock_Learning_Improvement_V2_TechSpec.md`
- AUC calculation proposal: `auc calculation proposal.md` (source conversation for this revision)
