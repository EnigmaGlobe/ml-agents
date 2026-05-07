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

Out of scope:

- detailed movement efficiency
- collision quality
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

To support the primary learning metric, the episode export must also include:

- `agent_id`
- `episode_id`
- `training_step`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `final_goal_error`

## 3. Primary Learning Signal

The primary learning metric should be `normalized_block_progress`.

Definition:

```text
normalized_block_progress =
(start_block_goal_distance - final_block_goal_distance)
/ start_block_goal_distance
```

Why it is primary:

- it is continuous
- it works even when the agent fails
- it measures task progress directly
- it is less ambiguous than reward

Interpretation:

- `1.00` means the block reached or nearly reached the goal
- `0.50` means the block covered about half the original distance
- `0.00` means no useful progress
- `< 0` means the block ended farther from the goal

## 4. Why Reward Is Not Enough

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

## 5. Metrics

### 5.1 Learning Slope

Definition:

```text
learning_slope = slope(metric ~ training_step)
```

Recommended input metric:

- rolling `normalized_block_progress`

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

### 5.2 AUC of Performance Curve

Definition:

```text
AUC(metric) = integral of rolling_metric over training_step
normalized_AUC = AUC / (last_step - first_step)
```

Recommended input metric:

- rolling `normalized_block_progress`

What it measures:

- the total amount of useful learning across the whole run

Interpretation:

- higher AUC means better learning over time
- low AUC means the agent improved slowly, stayed weak, or collapsed

Implementation note:

- use rolling mean values
- compute AUC per agent
- normalize by training length so runs of different duration can be compared

### 5.3 Time-to-Threshold

Definition:

```text
time_to_threshold =
first training_step where rolling_metric >= threshold
```

Recommended thresholds:

- `rolling_success_rate >= 0.80`
- `rolling_progress >= 0.80`

What it measures:

- how quickly the agent becomes competent

Interpretation:

- lower time-to-threshold is better
- no threshold reached means the agent did not become competent within the budget

Implementation note:

- use sustained threshold detection if possible
- require several consecutive windows above threshold to avoid a one-window spike

### 5.4 Final Performance Window

Definition:

```text
final_performance = mean(metric in final 10% of episodes)
```

Recommended metrics:

- `normalized_block_progress`
- `success`
- `episode_reward`
- `final_goal_error`

What it measures:

- how well the final policy is performing near the end of training

Why it matters:

- one final episode can be noisy
- a final window is more reliable than a single point

Implementation note:

- also consider a final fixed-step window if two runs have different episode counts

### 5.5 Learning Stability

Definition:

```text
learning_stability_sd = sd(metric in final 10% episodes)
learning_stability_iqr = IQR(metric in final 10% episodes)
```

Recommended metrics:

- `normalized_block_progress`
- `episode_reward`
- `final_goal_error`
- `success`

What it measures:

- whether the policy converged or is still unstable

Interpretation:

- low SD or IQR means stable final behavior
- high SD or IQR means inconsistent behavior

Implementation note:

- report a mean together with SD or IQR
- do not report stability alone

### 5.6 Performance Drop Index

Definition:

```text
performance_drop_index = max(rolling_metric) - final(rolling_metric)
relative_performance_drop = (max_rolling_metric - final_rolling_metric) / max_rolling_metric
```

Recommended metrics:

- rolling `normalized_block_progress`
- rolling `success_rate`
- rolling `episode_reward`

What it measures:

- whether the agent improved and then got worse

Interpretation:

- small drop: stable training
- large drop: collapse or forgetting

Implementation note:

- report both absolute and relative drop when possible

## 6. TensorBoard Comparison

Learning improvement should be compared against trainer-side diagnostics, but not treated as identical to them.

Useful comparisons:

- `normalized_block_progress` vs cumulative reward
- `rolling_success_rate` vs cumulative reward
- `learning_stability` vs value loss
- `final_performance_window` vs value estimate
- `performance_drop_index` vs policy loss or entropy

These comparisons help answer whether the training signal matches actual behavior.

Important rule:

- correlation does not prove causation

## 7. Logging Requirements

Each episode row should contain:

- `agent_id`
- `episode_id`
- `training_step`
- `episode_reward`
- `episode_length`
- `success`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `final_goal_error`

Recommended additional fields:

- `run_id`
- `seed`
- `environment_id`
- `spawn_condition_id`

Derived later in analysis:

- `rolling_success_rate`
- `rolling_progress`
- `rolling_reward`
- `learning_slope`
- `AUC`
- `time_to_threshold`
- `final_performance`
- `learning_stability`
- `performance_drop_index`

## 8. Code Path

The learning-improvement pipeline should follow this flow:

1. `OnEpisodeBegin()`
   - reset counters
   - capture the start block-goal distance
   - assign episode identity

2. `OnActionReceived()`
   - increment `m_episodeSteps`
   - accumulate `m_episodeCumulativeReward`
   - continue until success or timeout

3. `ScoredAGoal()`
   - mark success
   - capture the final block-goal distance
   - compute normalized progress
   - export the episode row
   - end the episode

4. Timeout branch in `OnActionReceived()`
   - mark failure
   - capture the final block-goal distance
   - compute normalized progress
   - export the episode row
   - end the episode

## 9. Safety Rule

If `start_block_goal_distance` is zero or very close to zero, exclude the episode or mark it invalid.

That prevents division errors and avoids inflating progress on degenerate spawn cases.

## 10. Multi-Agent Extension

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

## 11. Recommended Reporting

For each agent, report:

- mean learning slope
- normalized AUC
- time-to-threshold
- final-window mean
- final-window SD or IQR
- performance drop index

This gives a complete picture of learning speed, stability, and robustness.

## 12. Milestone Plan

This section defines the implementation roadmap for this metric group.

### Milestone 1: MVP

Goal:

- get the minimum useful learning-analysis pipeline working for a single PushBlock agent

Scope:

- keep the current episode reward and success logging
- add episode export for `agent_id`, `episode_id`, and `training_step`
- log `start_block_goal_distance` and `final_block_goal_distance`
- compute `normalized_block_progress`
- compute `rolling_success_rate`
- compute `learning_slope`

Success criteria:

- we can generate a basic learning curve
- we can compare reward with actual block progress
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
- report normalized progress alongside reward, success, and episode length

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
