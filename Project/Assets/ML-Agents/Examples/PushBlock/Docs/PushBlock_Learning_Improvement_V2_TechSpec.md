# PushBlock Learning Improvement v2 Technical Specification

This document defines the concrete metric construction for the PushBlock learning-improvement revision.

The scope is intentionally narrow:

1. AUC of performance curve
2. Final performance window
3. Learning stability
4. Learning slope

The first three are primary metrics. Learning slope is secondary and diagnostic.

## 1. Data Inputs

### 1.1 Unity episode source

Primary source:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\learning_improvement_20260508_110321.csv`

Required fields:

- `episode_id`
- `training_step`
- `success`
- `normalized_task_progress`
- `episode_reward`
- `final_goal_zone_error_xz`
- `end_reason`

Recommended supporting fields:

- `episode_length`
- `time_to_goal`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `success_goal_consistency`
- `final_goal_error`

### 1.2 TensorBoard source folder

Primary TensorBoard folder:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\TFModels\learning_improvement02`

TensorBoard scalar files in this folder are treated as reference diagnostics.

They are not the primary source for the four metrics in this specification.

## 2. Canonical Ordering

Before computing any metric:

1. Load all episode rows.
2. Filter out rows with missing or non-finite values in the fields required by the target metric.
3. Sort rows by `training_step` ascending.
4. If two rows share the same `training_step`, keep their original order by `episode_id` as a tie-breaker.

The ordered episode sequence is the base record for all four metrics.

## 3. Shared Curve Construction

Curve-based metrics use a smoothed performance series built from `normalized_task_progress`.

### 3.1 Rolling window rule

Default rolling window:

- `W = 50` episodes

Fallback rules:

- if the run has fewer than `50` episodes, use `W = max(5, floor(N / 10))`
- if that still exceeds the episode count, use all available episodes

### 3.2 Rolling series definition

For each episode index `i`, define:

- `rolling_progress[i] = mean(normalized_task_progress over the trailing W episodes ending at i)`

This is a trailing moving average, not a centered window.

Why trailing:

- it preserves training causality
- it avoids using future episodes to define a point on the curve
- it is stable enough for noisy single-run data

### 3.3 X-axis definition

The x-axis for curve metrics is `training_step`.

If duplicate `training_step` values exist in a future multi-agent or merged run, aggregate duplicates before curve calculation:

- `y(step) = mean(rolling_progress values at that step)`

For the current single-run file, unique steps are expected and no special aggregation should usually be needed.

## 4. Metric 1: AUC of Performance Curve

### 4.1 Question answered

- How much did the agent learn overall?

### 4.2 Metric definition

Use the trapezoidal area under the rolling progress curve:

```text
raw_auc = integral of rolling_progress over training_step
normalized_auc = raw_auc / (last_step - first_step)
```

### 4.3 Implementation details

Given ordered points `(x_i, y_i)` where:

- `x_i = training_step`
- `y_i = rolling_progress[i]`

Compute:

```text
raw_auc = sum over i=1..n-1 of ((x_i - x_{i-1}) * (y_i + y_{i-1}) / 2)
span = x_last - x_first
normalized_auc = raw_auc / span, if span > 0, otherwise 0
```

### 4.4 Reporting rules

Report both:

- `raw_auc`
- `normalized_auc`

If only one value is shown in UI, show `normalized_auc`.

### 4.5 Interpretation

- higher AUC means the agent spent more of training at better performance
- low AUC means slow learning, weak performance, or late collapse
- high AUC with a weak final window suggests the agent improved earlier but did not hold the result

## 5. Metric 2: Final Performance Window

### 5.1 Question answered

- How good is the agent now?

### 5.2 Window definition

Use the last segment of the ordered episode list.

Default final window size:

- last `10%` of episodes

Minimum final window size:

- at least `25` episodes when available

Final window rule:

- `final_window = last max(25, ceil(0.10 * N)) episodes`

If the run is shorter than the minimum, use all available episodes.

### 5.3 Point estimates

The primary final-window point estimate is:

```text
final_window_iqm = IQM(normalized_task_progress in final_window)
```

Supporting estimates:

- `final_window_mean_progress`
- `final_window_mean_reward`
- `final_window_mean_goal_error`
- `final_window_success_rate`

### 5.4 Bootstrap confidence interval

Compute a bootstrap confidence interval for the same robust estimator used for the point estimate.

Recommended estimator:

- IQM of `normalized_task_progress`

Bootstrap settings:

- resamples: `10,000`
- confidence level: `95%`
- random seed: `42` for reproducibility

Bootstrap procedure:

1. Sample `final_window` rows with replacement.
2. Compute IQM on each resample.
3. Sort bootstrap IQM values.
4. Take the `2.5%` and `97.5%` percentiles as the interval.

### 5.5 Reporting rules

Report:

- `final_window_iqm_progress`
- `final_window_ci_lower`
- `final_window_ci_upper`
- `final_window_mean_progress`
- `final_window_mean_goal_error`

### 5.6 Interpretation

- higher final-window IQM means the agent is currently performing better
- a narrow confidence interval indicates a more stable estimate
- a single high episode is not sufficient if the rest of the window is weak

## 6. Metric 3: Learning Stability

### 6.1 Question answered

- Can we trust this result?

### 6.2 Window definition

Use the same final window as the final performance metric.

### 6.3 Stability statistics

Compute these on the raw final-window values of `normalized_task_progress`:

- standard deviation
- interquartile range
- coefficient of variation
- minimum
- maximum
- median
- Q1
- Q3

### 6.4 Formula

```text
sd = standard deviation of final_window values
iqr = Q3 - Q1
cv = sd / abs(mean), if mean != 0
```

### 6.5 Reporting rules

Report at minimum:

- `final_window_sd_progress`
- `final_window_iqr_progress`
- `final_window_cv_progress`

Optional companion stability values:

- reward stability
- success stability
- final goal error stability

### 6.6 Interpretation

- low SD / low IQR means stable final behavior
- high SD / high IQR means the result is noisy or unstable
- CV helps compare stability across runs with different absolute scale

## 7. Metric 4: Learning Slope

### 7.1 Question answered

- How fast did it learn?

### 7.2 Metric definition

Learning slope is the OLS slope of rolling progress versus training step.

```text
learning_slope = slope(rolling_progress ~ training_step)
```

### 7.3 Implementation details

Use the same `training_step` axis as the AUC calculation.

Fit a simple linear regression:

- x = `training_step`
- y = `rolling_progress`

Report:

- slope
- intercept
- `R^2`
- standard error
- p-value if available

### 7.4 Diagnostic status

This metric is secondary.

It should never override the three primary metrics.

### 7.5 Interpretation

- positive slope means the agent improved over time
- near-zero slope means learning stalled
- negative slope means performance degraded

## 8. Metric Priority Rules

The report must always follow this order:

1. AUC of performance curve
2. Final performance window
3. Learning stability
4. Learning slope

The first three are the actual decision metrics.

Learning slope is only supplementary.

## 9. Output Fields

The implementation should expose the following fields in the report object:

- `AUC_Raw`
- `AUC_Normalized`
- `FinalWindow_IQMProgress`
- `FinalWindow_MeanProgress`
- `FinalWindow_CI_Lower`
- `FinalWindow_CI_Upper`
- `Stability_SD_Progress`
- `Stability_IQR_Progress`
- `Stability_CV_Progress`
- `LearningSlope`
- `LearningSlopeIntercept`
- `LearningSlopeR2`
- `LearningSlopePValue`

Recommended supporting fields:

- `FinalWindow_MeanReward`
- `FinalWindow_MeanGoalError`
- `FinalWindow_SuccessRate`
- `FinalWindow_MinProgress`
- `FinalWindow_MaxProgress`
- `FinalWindow_MedianProgress`

## 10. Output Artifacts

All derived outputs for this revision should be written under:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\TFModels\trianercheckfix`

Recommended file types:

- one summary CSV
- one plain-text snapshot report
- optional plot only if it is genuinely needed for inspection

Keep the output set minimal. Do not generate extra derived files unless they are required by the implementation or needed for validation.

The deprecated aligned file is not the canonical source of truth for this revision.

## 11. Implementation Mapping

The current `TrainingCheckerWindow.cs` already contains:

- CSV parsing
- rolling alignment logic
- slope calculation
- final-window reporting

The revision should refactor this into:

- a Unity episode loader
- a metric engine for the four chosen metrics
- a report builder
- a UI presentation layer

The metric engine should not depend on TensorBoard alignment for the primary metrics unless a later version explicitly adds a comparison view.

## 12. Validation Rules

Before accepting the report:

1. Verify that the episode ordering is correct.
2. Verify that the rolling window size matches the spec.
3. Verify that the final window uses the last episodes, not the last TensorBoard points.
4. Verify that the bootstrap CI is reproducible with the fixed seed.
5. Verify that slope and AUC use the same curve basis.
6. Verify that the report clearly labels slope as secondary.

## 13. Acceptance Criteria

This specification is satisfied when:

- the code computes the four metrics consistently
- the report and UI present them in the correct priority order
- the final-window metric is robust, not a single point estimate
- stability is computed from the same final window
- slope remains available as a supplementary diagnostic
- outputs are written into the current run folder
