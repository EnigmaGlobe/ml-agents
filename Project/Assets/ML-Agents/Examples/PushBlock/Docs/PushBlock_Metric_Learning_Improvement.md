# Learning Improvement Metrics

This document defines the metrics used to measure whether a PushBlock agent is improving over training.

The goal is to evaluate learning as a trajectory, not as a single final reward number. For this environment, a useful learning metric must answer four questions:

- is the agent improving?
- how fast is it improving?
- is the improvement stable?
- does the policy collapse after reaching a peak?

## 1. Scope

This metric group covers episode-level learning behavior for a single agent, with later reuse for multi-agent comparison.

In scope:

- training slope
- learning-curve area
- time-to-threshold
- final-window performance
- final-window stability
- performance drop after peak
- metric-validity checks for distance-based task progress

Out of scope:

- detailed movement efficiency
- collision quality outside the success condition
- control smoothness
- multi-agent ranking logic

Those items belong to the other metric docs.

## 2. Code Basis

The current `PushAgentBasic.cs` script already defines the episode lifecycle needed for learning analysis.

Relevant runtime signals:

- `m_episodeSteps`
- `m_episodeCumulativeReward`
- `MaxStep`
- `ScoredAGoal()`
- `OnActionReceived()`
- `OnEpisodeBegin()`

Current recorded stats:

- `PushBlock/episode_reward`
- `PushBlock/success`
- `PushBlock/episode_length`
- `PushBlock/time_to_goal`

These values are enough for a basic training curve, but not enough for a research-grade learning analysis.

To support the revised learning metric, the episode export should include:

- `agent_id`
- `episode_id`
- `training_step`
- `start_goal_zone_error_xz`
- `final_goal_zone_error_xz`
- `normalized_task_progress`
- `success_goal_consistency`

Recommended debug-only fields:

- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `final_goal_error`
- `final_center_distance_xz`
- `final_center_distance_xyz`
- `goal_instance_id`
- `block_instance_id`
- `scored_goal_name`

## 3. Why The Old Distance Metrics Are No Longer Primary

The previous learning-improvement draft treated these fields as primary:

- `normalized_block_progress`
- `final_goal_error`

Those metrics used center-to-center distance:

```text
final_goal_error =
distance(block.transform.position, goal.transform.position)
```

That definition does not match the actual success condition in PushBlock.

Current success semantics:

- `success = 1` when the block collider touches the goal collider

Current legacy distance semantics:

- `final_goal_error = 3D distance from the block transform center to the goal transform center`

These are not the same thing. Because of that mismatch, successful episodes can still show:

- large `final_goal_error`
- negative `normalized_block_progress`
- misleading reward-progress correlation

This is why old center-distance metrics must be downgraded to debug-only fields.

## 4. Primary Learning Signal

The primary learning metric should now be `normalized_task_progress`.

Definition:

```text
normalized_task_progress =
(start_goal_zone_error_xz - final_goal_zone_error_xz)
/ max(start_goal_zone_error_xz, epsilon)
```

Where:

- `start_goal_zone_error_xz` = the XZ-plane distance from the block collider boundary to the goal collider boundary at episode start
- `final_goal_zone_error_xz` = the same XZ-plane goal-zone error at episode end
- `epsilon` = a very small positive value used only to avoid division by zero

Why it is primary:

- it is continuous
- it works even when the agent fails
- it is consistent with the success condition
- it measures progress toward the success zone, not toward an arbitrary transform center
- it is less ambiguous than reward

Interpretation:

- `1.00` means the block reached the success zone
- `0.50` means the remaining goal-zone error was reduced by half
- `0.00` means no useful progress
- `< 0` means the block ended farther from the success zone

## 5. Goal-Zone Error Definition

Distance for learning-improvement should be defined on the XZ plane, not by 3D center-to-center distance.

Recommended definition:

```text
goal_zone_error_xz =
distance between block collider boundary and goal collider boundary on the XZ plane
```

Interpretation:

- `0` means the block is touching or overlapping the success zone
- `> 0` means the block has not yet reached the success zone

Recommended implementation concept:

- use collider bounds or another collider-aware geometric check
- measure boundary-to-boundary gap on the XZ plane
- do not use `Vector3.Distance(block.position, goal.position)` as the primary task error

Recommended success-aligned rule:

```text
if success == 1,
final_goal_zone_error_xz should be 0
```

## 6. Why Reward Is Not Enough

Reward in PushBlock can improve for several different reasons:

- the agent reaches the goal more often
- the agent reaches it faster
- the agent receives fewer step penalties
- the spawn conditions are easier
- the block moves closer by chance

Because of that, reward should be treated as a secondary signal, not the main proof of learning.

TensorBoard metrics are still useful, but they answer a different question. They show whether the training process is changing in a healthy way. Unity metrics show whether the agent is actually learning the block-pushing task.

Recommended secondary signals:

- `rolling_success_rate`
- `episode_reward`
- `episode_length`
- `time_to_goal`

## 7. Metrics

### 7.1 Learning Slope

Definition:

```text
learning_slope = slope(metric ~ training_step)
```

Recommended input metric:

- rolling `normalized_task_progress`

Secondary inputs:

- rolling success rate
- episode reward

What it measures:

- how fast the agent improves over training

Interpretation:

- positive slope: learning is improving
- flat slope: learning is stalled
- negative slope: performance is degrading

Implementation note:

- compute slope on a rolling-window metric, not on raw noisy episodes
- keep the rolling window fixed across agents

### 7.2 AUC of Performance Curve

Definition:

```text
AUC(metric) = integral of rolling_metric over training_step
normalized_AUC = AUC / (last_step - first_step)
```

Recommended input metric:

- rolling `normalized_task_progress`

What it measures:

- the total amount of useful learning across the whole run

Interpretation:

- higher AUC means better learning over time
- low AUC means the agent improved slowly, stayed weak, or collapsed

Implementation note:

- use rolling mean values
- compute AUC per agent
- normalize by training length so runs of different duration can be compared

### 7.3 Time-to-Threshold

Definition:

```text
time_to_threshold =
first training_step where rolling_metric >= threshold
```

Recommended thresholds:

- `rolling_success_rate >= 0.80`
- `rolling_normalized_task_progress >= 0.80`

What it measures:

- how quickly the agent becomes competent

Interpretation:

- lower time-to-threshold is better
- no threshold reached means the agent did not become competent within the budget

Implementation note:

- use sustained threshold detection if possible
- require several consecutive windows above threshold to avoid a one-window spike

### 7.4 Final Performance Window

Definition:

```text
final_performance = mean(metric in final 10% of episodes)
```

Recommended metrics:

- `normalized_task_progress`
- `success`
- `episode_reward`
- `final_goal_zone_error_xz`

What it measures:

- how well the final policy is performing near the end of training

Why it matters:

- one final episode can be noisy
- a final window is more reliable than a single point

Implementation note:

- also consider a final fixed-step window if two runs have different episode counts

### 7.5 Learning Stability

Definition:

```text
learning_stability_sd = sd(metric in final 10% episodes)
learning_stability_iqr = IQR(metric in final 10% episodes)
```

Recommended metrics:

- `normalized_task_progress`
- `episode_reward`
- `final_goal_zone_error_xz`
- `success`

What it measures:

- whether the policy converged or is still unstable

Interpretation:

- low SD or IQR means stable final behavior
- high SD or IQR means inconsistent behavior

Implementation note:

- report a mean together with SD or IQR
- do not report stability alone

### 7.6 Performance Drop Index

Definition:

```text
performance_drop_index = max(rolling_metric) - final(rolling_metric)
relative_performance_drop = (max_rolling_metric - final_rolling_metric) / max_rolling_metric
```

Recommended metrics:

- rolling `normalized_task_progress`
- rolling `success_rate`
- rolling `episode_reward`

What it measures:

- whether the agent improved and then got worse

Interpretation:

- small drop: stable training
- large drop: collapse or forgetting

Implementation note:

- report both absolute and relative drop when possible

## 8. Metric Validity Check

Before assigning PASS, WARN, or FAIL, the learning-improvement analysis should run a metric-validity check.

Purpose:

- confirm that success and distance-based metrics use the same task semantics
- detect cases where the success trigger fires but the distance metric still claims the block is far from the goal
- prevent false failure conclusions caused by invalid behavior metrics

Required validity checks:

- `mean(final_goal_zone_error_xz where success == 1)`
- `proportion(success == 1 and final_goal_zone_error_xz > tolerance)`
- `proportion(success == 1 and normalized_task_progress < 0)`

Recommended debug checks:

- `mean(final_center_distance_xz where success == 1)`
- `mean(final_center_distance_xyz where success == 1)`
- `proportion(success == 1 and normalized_block_progress < 0)`

Recommended warning rule:

```text
If successful episodes often have large goal-zone error
or successful episodes often have negative normalized task progress,
flag: FAIL with metric-validity warning
```

Recommended interpretation text:

```text
Current status: FAIL with metric-validity warning.
Success was recorded, but the distance-based task-progress metrics remain semantically inconsistent with the success condition.
```

## 9. TensorBoard Comparison

Learning improvement should be compared against trainer-side diagnostics, but not treated as identical to them.

Useful comparisons:

- `normalized_task_progress` vs cumulative reward
- `rolling_success_rate` vs cumulative reward
- `learning_stability` vs value loss
- `final_performance_window` vs value estimate
- `performance_drop_index` vs policy loss or entropy

These comparisons help answer whether the training signal matches actual behavior.

Important rule:

- correlation does not prove causation

## 10. Logging Requirements

Each episode row should contain:

- `agent_id`
- `episode_id`
- `training_step`
- `episode_reward`
- `episode_length`
- `success`
- `start_goal_zone_error_xz`
- `final_goal_zone_error_xz`
- `normalized_task_progress`

Recommended additional fields:

- `time_to_goal`
- `run_id`
- `seed`
- `environment_id`
- `spawn_condition_id`

Recommended debug-only fields:

- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `final_goal_error`
- `final_center_distance_xz`
- `final_center_distance_xyz`
- `goal_instance_id`
- `block_instance_id`
- `scored_goal_name`

Derived later in analysis:

- `rolling_success_rate`
- `rolling_normalized_task_progress`
- `rolling_reward`
- `learning_slope`
- `AUC`
- `time_to_threshold`
- `final_performance`
- `learning_stability`
- `performance_drop_index`

## 11. Code Path

The learning-improvement pipeline should follow this flow:

1. `OnEpisodeBegin()`
   - reset counters
   - capture `start_goal_zone_error_xz`
   - capture debug center-distance fields if needed
   - assign episode identity

2. `OnActionReceived()`
   - increment `m_episodeSteps`
   - accumulate `m_episodeCumulativeReward`
   - continue until success or timeout

3. `ScoredAGoal()`
   - mark success
   - capture final block state before `EndEpisode()`
   - capture `final_goal_zone_error_xz`
   - compute `normalized_task_progress`
   - export the episode row
   - end the episode

4. Timeout branch in `OnActionReceived()`
   - mark failure
   - capture final block state
   - capture `final_goal_zone_error_xz`
   - compute `normalized_task_progress`
   - export the episode row
   - end the episode

Critical rules:

- final metrics must be captured before reset occurs
- success logic and distance logic must use the same goal definition
- center-distance values are debug fields only if success is defined by collider contact

## 12. Safety Rule

If `start_goal_zone_error_xz` is zero or very close to zero, exclude the episode or mark it invalid.

That prevents division errors and avoids inflating progress on degenerate spawn cases.

## 13. Multi-Agent Extension

For later multi-agent comparison, the same metric definitions must be reused for every agent.

Use:

- `agent_id = Agent_01`
- `agent_id = Agent_02`
- `agent_id = Agent_03`

Then compare:

- learning slope by agent
- AUC by agent
- time-to-threshold by agent
- final performance by agent
- stability by agent
- performance drop by agent

## 14. Recommended Reporting

For each agent, report:

- mean learning slope
- normalized AUC
- time-to-threshold
- final-window mean
- final-window SD or IQR
- performance drop index
- metric-validity warning status

This gives a complete picture of learning speed, stability, robustness, and metric health.

## 15. Milestone Plan

This section defines the implementation roadmap for this metric group.

### Milestone 1: MVP

Goal:

- get the minimum useful learning-analysis pipeline working for a single PushBlock agent

Scope:

- keep the current episode reward and success logging
- add episode export for `agent_id`, `episode_id`, and `training_step`
- log `start_goal_zone_error_xz` and `final_goal_zone_error_xz`
- compute `normalized_task_progress`
- compute `rolling_success_rate`
- compute `learning_slope`

Success criteria:

- we can generate a basic learning curve
- we can compare reward with actual task progress
- the spec is backed by real episode data

### Milestone 2: Full Single-Agent Metrics

Goal:

- expand the single-agent analysis into the full learning-improvement spec

Scope:

- compute AUC
- compute time-to-threshold
- compute final performance window
- compute learning stability
- compute performance drop index
- add metric-validity checks
- report normalized task progress alongside reward, success, and episode length

Success criteria:

- the learning curve is no longer just one number
- we can judge learning speed, stability, and collapse
- the outputs are suitable for research-style comparison

### Milestone 3: Multi-Agent Comparison

Goal:

- compare multiple agents one by one using the same definitions

Scope:

- preserve `agent_id` for every episode row
- compute all learning-improvement metrics per agent
- compare slopes, AUC, thresholds, stability, and performance drop across agents
- keep analysis consistent across seeds and training runs

Success criteria:

- agents can be ranked fairly
- individual learning curves are not hidden by averages
- the same metric definitions work across all agents

## 16. Current Interpretation Rule

Until the Unity-side metric definitions are updated in code, any learning-improvement result that depends heavily on:

- `normalized_block_progress`
- `final_goal_error`
- `goal_error_reduction_raw`
- `reward_progress_correlation`

should be labeled:

```text
FAIL with metric-validity warning
```

This is the correct interim interpretation when success is defined by collider contact but the distance-based metrics still use transform-center distance.
