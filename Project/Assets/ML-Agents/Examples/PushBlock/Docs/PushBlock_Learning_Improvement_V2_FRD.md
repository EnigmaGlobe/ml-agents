# PushBlock Learning Improvement v2 FRD

## 1. Purpose

This document defines the requirements and technical approach for the next revision of the PushBlock learning-improvement analysis pipeline.

The goal of this revision is to replace the older, loosely aligned analysis flow with a clearer and more defensible evaluation process built from:

- episode-level Unity export data
- TensorBoard scalar exports from the current training run
- robust learning-curve statistics
- reproducible, code-backed report generation

This revision is intended to support a single completed training run first. The design should still be extensible to future multi-seed and multi-run analysis.

## 2. Problem Statement

The current analysis flow mixes legacy outputs, partial alignment tables, and training diagnostics that are not clearly tied to one data contract.

That creates three problems:

1. The primary data source is not explicit enough.
2. The relationship between Unity-side episode steps and TensorBoard trainer steps is easy to misinterpret.
3. The current learning-improvement checks rely too heavily on point estimates and do not clearly separate primary learning evidence from secondary diagnostics.

The new pipeline must fix these issues by defining one stable source of truth for the Unity episode export and one stable source of truth for TensorBoard scalar data.

## 3. Data Sources

### 3.1 Unity episode export

Primary episode-level source:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\learning_improvement_20260508_110321.csv`

This file contains the complete episode-level record for one full training run.

### 3.2 TensorBoard scalar exports

Primary TensorBoard source folder:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\TFModels\learning_improvement02`

This folder contains the exported scalar series for the same training run, including:

- `Cumulative Reward.csv`
- `Episode Length.csv`
- `Extrinsic Reward.csv`
- `Learning Rate.csv`
- `Policy Loss.csv`
- `Value Loss.csv`

### 3.3 Deprecated alignment artifact

The older aligned file is not the source of truth for this revision.

- `learning_improvement_aligned.csv` must not be treated as the canonical input

If any derived alignment table is needed again, it must be regenerated from the two primary sources above.

## 4. Data Contract

### 4.1 Unity episode CSV contract

The Unity export is episode-oriented.

Required columns for the new analysis:

- `episode_id`
- `training_step`
- `success`
- `normalized_task_progress`
- `episode_reward`
- `final_goal_zone_error_xz`
- `end_reason`

Useful diagnostic columns:

- `episode_length`
- `time_to_goal`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `success_goal_consistency`
- `final_goal_error`
- `final_center_distance_xz`
- `final_center_distance_xyz`

### 4.2 TensorBoard scalar contract

TensorBoard scalar files are trainer-oriented.

Each scalar series is expected to provide:

- `Wall time`
- `Step`
- `Value`

The `Step` field is the trainer x-axis for those exported summaries.
It is not identical to the Unity episode-level `training_step` field and must not be assumed to match one-to-one without explicit alignment logic.

## 5. Technical Goals

This revision must achieve the following goals:

1. Define the learning-improvement pipeline around the current full training run.
2. Separate primary learning evidence from secondary diagnostics.
3. Make all step alignment logic explicit and reproducible.
4. Support robust statistical summaries that are less sensitive to single-episode noise.
5. Store new derived outputs under the current TensorBoard run folder.
6. Keep the implementation traceable to existing code and exported files.

## 6. Non-Goals

This revision does not aim to:

- change reward shaping
- change the agent architecture
- change the Unity scene layout
- tune trainer hyperparameters
- claim multi-seed statistical significance from a single run

The current run is valuable, but it is still a single-seed observation.
Any paper-style claims must therefore include that caveat.

## 7. Output Location

All new derived artifacts for this revision must be stored under:

- `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\TFModels\trianercheckfix`

This folder should become the canonical home for the minimum necessary derived outputs:

- one summary CSV
- one human-readable report snapshot, if needed
- only the files required to reproduce or inspect the result

Avoid generating a large set of intermediate or duplicate files unless they are strictly necessary for the implementation.

The older `learning_improvement_aligned.csv` file should be treated as legacy and not reused as the main output for this revision.

## 8. Metric Strategy

This revision focuses on four metrics only:

- AUC of performance curve
- Final performance window
- Learning stability
- Learning slope

The first three are the primary metrics. Learning slope is secondary and diagnostic.

### 8.1 Primary metric 1: AUC of performance curve

Question answered:

- How much did the agent learn overall?

Recommended meaning:

- integrate normalized task progress over training step

Construction rules:

- use the aligned training-step axis
- compute trapezoidal area under the curve
- normalize by step span so runs of different duration can still be compared
- report the raw area and the normalized area

Interpretation:

- higher AUC means the agent performed better across more of training
- low AUC with a strong final window may indicate late learning
- high AUC with a weak final window may indicate collapse near the end

### 8.2 Primary metric 2: Final performance window

Question answered:

- How good is the agent now?

Recommended meaning:

- summarize the last segment of the run, typically the final 10 percent of episodes or the final aligned window

Construction rules:

- compute the mean of the primary performance signal
- compute IQM for robustness against outliers
- compute a bootstrap confidence interval for the final-window estimate
- include final goal error as a supporting quality signal

Interpretation:

- this is the best indicator of current competency
- a single good final episode is not enough
- the final window must be stable, not just high on one point

### 8.3 Primary metric 3: Learning stability

Question answered:

- Can we trust this result?

Recommended meaning:

- measure how noisy or consistent the final-window performance is

Construction rules:

- compute standard deviation
- compute interquartile range
- compute coefficient of variation when the mean is non-zero
- apply the same stability logic to progress, reward, and success if those series are available

Interpretation:

- lower variability means a more trustworthy result
- a high mean with high variance is weaker evidence than a slightly lower but stable result

### 8.4 Secondary metric: Learning slope

Question answered:

- How fast did it learn?

Recommended meaning:

- estimate the linear trend of performance versus training step

Construction rules:

- treat this as supplementary, not primary
- compute it on the same aligned step axis used by the primary metrics
- report slope, intercept, and fit quality if available

Interpretation:

- positive slope suggests improvement over time
- a strong slope is useful, but it does not replace AUC, final-window quality, or stability

### 8.5 Explicit metric hierarchy

The report and UI should always present the metrics in this order:

1. AUC of performance curve
2. Final performance window
3. Learning stability
4. Learning slope

This ordering matters because the first three are the real decision metrics.
Slope is a supporting diagnostic only.

## 9. Benchmark Strategy

The metric values alone are not sufficient for decision-making.
Each metric must be compared against an explicit benchmark before the report can be interpreted as good, marginal, or poor.

The benchmark design follows these principles:

- papers define the evaluation philosophy, not universal numeric cutoffs
- thresholds are task-specific operational benchmarks for PushBlock
- each core metric gets its own benchmark
- the overall verdict is derived from metric-level verdicts, not from a single total score
- stability affects confidence, not just success/failure

### 9.1 Benchmark sources

The benchmark design should be informed by the following ideas:

- Agarwal et al.: use robust aggregate metrics, confidence intervals, and distribution-aware summaries
- Henderson et al.: report variance, reproducibility, and instability explicitly
- QT-Opt / robotic manipulation benchmarks: use final success rate as a task-completion signal
- PPO / Schulman-style optimization: inspect learning trend and training stability, but do not use slope alone as the main verdict

These papers support the benchmark structure, but they do not directly prescribe the PushBlock numeric thresholds.

### 9.2 Metric-level benchmark matrix

The Learning Improvement topic should use the following benchmark logic:

#### AUC of performance curve

Question:

- How much did the agent learn overall?

Thresholds:

- `Fail`: `auc_normalized < 0.15`
- `Warn`: `0.15 <= auc_normalized < 0.30`
- `Pass`: `0.30 <= auc_normalized < 0.45`
- `Strong`: `0.45 <= auc_normalized < 0.60`
- `Excellent`: `auc_normalized >= 0.60`

Evaluation order:

- these ranges are mutually exclusive
- evaluate from `Excellent` down to `Fail`
- assign exactly one AUC verdict

Interpretation:

- this is a task-specific operational scale
- higher is better
- the benchmark should be read as cumulative learning quality, not final-only quality

#### Final performance window

Question:

- How good is the agent now?

Thresholds:

- `Fail`: `final_success_rate < 0.60` and `final_IQM_progress < 0.50`
- `Warn`: `final_success_rate >= 0.60` or `final_IQM_progress >= 0.50`
- `Pass`: `final_success_rate >= 0.80` and `final_IQM_progress >= 0.70`
- `Strong`: `final_success_rate >= 0.90` and `final_IQM_progress >= 0.85`
- `Excellent`: `final_success_rate >= 0.95` and `final_IQM_progress >= 0.90`

Evaluation order:

- evaluate from `Excellent` down to `Fail`
- assign the highest matching tier only
- `Excellent` implies `Strong`, `Pass`, and `Warn`, but it must still be reported as `Excellent`
- `Strong` implies `Pass` and `Warn`, but it must still be reported as `Strong`
- `Pass` implies `Warn`, but it must still be reported as `Pass`

Interpretation:

- success rate is the main task-completion signal
- IQM is the robust quality signal
- mean progress and mean goal error are supporting diagnostics

#### Learning stability

Question:

- Can we trust this result?

Thresholds:

- `Outlier warning`: `IQR <= 0.10` but `SD > 0.50` or `CV > 1.00`
- `High confidence`: `IQR <= 0.10` and `SD <= 0.25`
- `Medium confidence`: `IQR <= 0.25`
- `Low confidence`: `IQR > 0.50` or `CV > 1.00`
- `Unstable / Fail`: none of the above, or the distribution is too noisy to trust

Interpretation:

- stability should not be reduced to a single number
- robust central stability and tail risk should both be visible
- the report should be able to say `PASS_WITH_WARNING` when the center is stable but outliers remain

Evaluation order:

- evaluate `Outlier warning` first
- then evaluate `High confidence`
- then `Medium confidence`
- then `Low confidence`
- finally fall back to `Unstable / Fail`
- `Outlier warning` is intentionally separate from `Low confidence`
- if `IQR <= 0.10` but `SD > 0.50` or `CV > 1.00`, classify as `OUTLIER_WARNING`, not `LOW_CONFIDENCE`

#### Learning slope

Question:

- How fast did it learn?

Thresholds:

- `Positive`: `slope_per_100k_steps > 0`
- `Weak`: `0 < slope_per_100k_steps < 0.05`
- `Strong`: `slope_per_100k_steps >= 0.05`
- `Flat`: approximately zero
- `Negative`: below zero

Interpretation:

- slope is diagnostic only
- slope should be displayed in scientific notation or in scaled form
- slope must not override the primary benchmark verdict

### 9.3 Overall verdict rules

The topic-level verdict should combine metric verdicts as follows:

- `PASS`: all primary metrics pass, and no strong stability warning exists
- `PASS_WITH_WARNING`: primary metrics pass, but stability has outliers or confidence is not high
- `WARN`: some primary metrics pass, but at least one primary metric is weak
- `FAIL`: one or more primary metrics fail

The overall verdict must be evaluated in priority order:

1. Check for any primary metric `FAIL`
2. Check for `Excellent` / `Strong` / `Pass` primary outcomes
3. Apply stability confidence
4. Derive `PASS_WITH_WARNING` when primary metrics pass but stability is not high
5. Use `WARN` when primary metrics are mixed
6. Use `FAIL` when any primary metric is below the fail threshold

Suggested final status wording:

- `Training Quality`
- `Final Performance`
- `Confidence`
- `Warning`

This keeps the report readable and avoids collapsing all meaning into a single binary label.

## 10. Benchmark Popup UI

The benchmark should be exposed through a popup window rather than embedded permanently in the main results area.

### 10.1 Button placement

For each topic tab, add a `Benchmark` button in the `Results` area.

For the `Learning Improvement` tab, the button should be placed on the right side of the status strip, next to the current status summary.

This keeps the main panel compact while still making benchmarks one click away.

### 10.2 Popup behavior

Clicking the button opens a shared popup template.

The popup should:

- reuse the same window structure for all topics
- swap only the benchmark content based on topic
- remain read-only
- avoid creating extra files or duplicate UI flows

### 10.3 Popup content structure

The popup should present benchmark information in a compact, tabular style:

- metric name
- threshold tiers
- verdict meaning
- short note

Recommended order for `Learning Improvement`:

1. AUC of performance curve
2. Final performance window
3. Learning stability
4. Learning slope

### 10.4 Popup visual rules

The popup should:

- fit within the editor without dominating the screen
- use a scroll view if content exceeds the visible area
- separate primary metrics from secondary diagnostics
- highlight the currently selected topic

The popup must be simple enough to be read quickly, but structured enough to support future topic-specific benchmarks.

### 10.5 Shared template requirement

All topics should use the same popup template.

That means:

- same window class or shared UI renderer
- same layout sections
- same button behavior
- topic-specific benchmark data only

This prevents divergence between topics and keeps the checker maintainable.

## 10. Alignment Strategy

The Unity episode export and TensorBoard scalar exports will not be compared by simple row equality.

The alignment strategy must be explicit and documented.

Recommended approach:

1. Treat Unity episodes as the episode-level base record.
2. Treat TensorBoard scalars as trainer-side reference curves.
3. Build a shared analysis axis from training progress or aligned training-step buckets.
4. Aggregate episodes into buckets before computing curve metrics when a direct step match is not available.
5. Preserve the original episode step and TensorBoard step for traceability.

The implementation must record which alignment method was used so the result can be audited later.

## 11. Technical Approach

### 11.1 Data ingestion

The pipeline should load:

- the Unity episode CSV
- the TensorBoard scalar CSV files required for the current training run

### 11.2 Derived table construction

The pipeline should then build:

- an episode-level working table
- an aligned analysis table for the selected metric axis
- a TensorBoard scalar working table for reference and comparison

### 11.3 Statistical computation

The report layer should compute exactly the following metric families:

- AUC of performance curve
- final performance window
- learning stability
- learning slope

Within those families, the implementation should support:

- trapezoidal AUC
- final-window mean
- final-window IQM
- final-window bootstrap confidence interval
- standard deviation
- interquartile range
- coefficient of variation
- linear regression slope

### 11.4 Report generation

The UI and report output should clearly separate:

- primary metrics
- secondary diagnostics
- raw data references
- caveats and interpretation notes

## 12. Implementation Phases

### Phase 1: Requirement lock

Confirm the data source paths, metric definitions, output location, axis-alignment rule, benchmark tiers, and popup UI behavior.

### Phase 2: Metric engine design

Define the exact formulas for:

- AUC
- final performance window
- learning stability
- learning slope

### Phase 3: Parser and alignment design

Define how the Unity CSV and TensorBoard CSV files will be loaded, bucketed, and matched.

### Phase 4: UI shell

Add the `Benchmark` button in the `Results` area and create the shared popup scaffold.

### Phase 5: Benchmark content model

Define a topic-to-benchmark mapping that supplies the popup with threshold rows.

### Phase 6: Statistical engine

Implement the metric calculations and confidence intervals in code.

### Phase 7: Learning Improvement wiring

Populate the `Learning Improvement` topic with the four metric benchmarks and connect metric values to verdicts.

### Phase 8: Report integration

Update the Unity editor window, summary export, and diagnostic output.

### Phase 9: Validation

Compare the generated report against the raw data and verify that the results are internally consistent.

## 13. Acceptance Criteria

This revision is complete when all of the following are true:

- the new FRD is approved
- the implementation uses the current training-run folder as the canonical output location
- the analysis pipeline no longer depends on the deprecated aligned file as its source of truth
- the report presents the four chosen metrics in the correct priority order
- the metric definitions are implemented consistently in code and documentation
- the output can be traced back to the Unity export and TensorBoard scalar files
- the final report is reproducible from the same input files

## 14. Risks and Caveats

Important caveats for this run:

- this is a single run, so variance across seeds is not measured
- TensorBoard trainer steps and Unity episode steps are related but not identical
- any bootstrap interval or significance-style output is only as strong as the underlying sample structure
- if the alignment method changes later, the report must say so explicitly

## 15. Next Step

The next implementation step is to translate this FRD into the updated `TrainingCheckerWindow.cs` workflow and define the exact data-loading and alignment functions before any metric computation is changed.
