# PushBlock Metrics FRD

## 1. Purpose

This document defines the metrics used to evaluate the PushBlock environment in Unity ML-Agents.

The goal is to support:

- single-agent training analysis
- comparison against TensorBoard training curves
- later comparison across multiple concurrently trained agents

The emphasis is on measurable episode-level statistics, not just success/fail reporting.

This FRD is intentionally code-backed. Every metric below should be traceable to an existing variable, a small amount of new logging, or a clearly defined derived calculation.

## 2. Scope

In scope:

- per-episode metrics
- block, agent, and goal motion metrics
- learning-curve statistics
- comparisons between Unity-side behavior and TensorBoard training signals
- per-agent comparison in multi-agent training

Out of scope:

- model architecture changes
- reward shaping design
- scene layout changes
- trainer hyperparameter tuning

## 3. Metric Naming Rules

Use stable, explicit names.

- Unity stats should use the `PushBlock/` prefix.
- Episode-level exports should use snake_case column names.
- Per-agent records should include `agent_id`.
- Training-step alignment should include `training_step` or `global_step`.

Example:

- Unity stat key: `PushBlock/normalized_block_progress`
- CSV column: `normalized_block_progress`

## 4. Current Implemented Metrics

The current `PushAgentBasic` script already records these values through `StatsRecorder`.

| Variable | Key | Type | Definition | When recorded |
|---|---|---:|---|---|
| Episode reward | `PushBlock/episode_reward` | float | Total reward accumulated during an episode | On success or max-step failure |
| Success | `PushBlock/success` | int | `1` if the block reaches the goal, otherwise `0` | On success or max-step failure |
| Episode length | `PushBlock/episode_length` | int | Number of agent steps in the episode | On success or max-step failure |
| Time to goal | `PushBlock/time_to_goal` | int | Number of steps taken to reach the goal; `-1` if the goal was not reached | On success or max-step failure |

These are useful baseline metrics, but they are not enough for a research-grade comparison between agents. The rest of this document defines how to extend them into a real evaluation framework.

## 5. Code-to-Metric Mapping

This section ties the FRD directly to the current scripts in `Scripts/`.

### 5.1 Existing code signals

The current `PushAgentBasic` implementation already gives us the following raw signals:

- `m_episodeSteps`
- `m_episodeCumulativeReward`
- `MaxStep`
- success event from `GoalDetect.OnCollisionEnter`
- block start and reset positions through `OnEpisodeBegin()` and `ResetBlock()`
- agent position and rigidbody state through `transform.position`, `m_AgentRb.linearVelocity`, and `m_AgentRb.angularVelocity`
- block position and rigidbody state through `block.transform.position`, `m_BlockRb.linearVelocity`, and `m_BlockRb.angularVelocity`
- environment randomization through `PushBlockSettings` and `EnvironmentParameters`

The multi-agent controller in `PushBlockEnvController` adds:

- `SimpleMultiAgentGroup`
- group reward
- group episode interruption on max environment steps
- per-agent reset positions and rotations
- per-block reset positions and rotations

### 5.2 What each script contributes

| Script | Useful fields / methods | Metric support |
|---|---|---|
| `PushAgentBasic.cs` | `m_episodeSteps`, `m_episodeCumulativeReward`, `MaxStep`, `ScoredAGoal()`, `OnEpisodeBegin()` | Episode reward, episode length, time-to-goal, success, reset-time baselines |
| `GoalDetect.cs` | `OnCollisionEnter(Collision col)` | Success event, goal contact timestamp |
| `PushBlockEnvController.cs` | `SimpleMultiAgentGroup`, `MaxEnvironmentSteps`, `ScoredAGoal(Collider col, float score)` | Multi-agent reward, group timeout, per-agent comparison structure |
| `PushBlockSettings.cs` | `agentRunSpeed`, `agentRotationSpeed`, `spawnAreaMarginMultiplier`, block mass/size/friction levels | Experimental condition metadata and environment parameter logs |
| `PushAgentCollab.cs` | `OnActionReceived()`, `MoveAgent()` | Per-agent action traces for multi-agent experiments |

### 5.3 Metrics directly supported by existing code

These metrics can be computed now or with minimal additional logging:

- `episode_reward`
- `success`
- `episode_length`
- `time_to_goal`
- `max_step_timeout` from `m_episodeSteps >= MaxStep`
- `spawn_configuration` from `PushBlockSettings`
- `block_mass`, `block_size`, `static_friction`, `dynamic_friction`, `block_drag`

### 5.4 Metrics that require added instrumentation

These metrics are not yet fully available in the current scripts and would require step-level history or extra collision tracking:

- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `agent_path_length`
- `block_path_length`
- `contact_latency`
- `push_ratio`
- `idle_ratio`
- `action_switch_rate`
- `oscillation_index`
- `velocity_variance`
- `acceleration_variance`
- `jerk`
- `failure_mode_distribution`

The FRD should treat these as derived metrics, computed from logged trajectories rather than from one scalar in the current script.

## 6. Proposed Metrics

Supporting documents:

- [Learning Improvement Metrics](PushBlock_Metric_Learning_Improvement.md)
- [Block Progress Metrics](PushBlock_Metric_Block_Progress.md)
- [Agent Efficiency Metrics](PushBlock_Metric_Agent_Efficiency.md)
- [Control Quality Metrics](PushBlock_Metric_Control_Quality.md)
- [Reliability Metrics](PushBlock_Metric_Reliability.md)
- [TensorBoard Comparison Metrics](PushBlock_Metric_TensorBoard_Comparison.md)

### 6.1 Learning Improvement Metrics

These answer: is the agent improving, how fast, and how stable is that improvement?

| Metric | Definition | Implementation concept | Why useful |
|---|---|---|---|
| Learning slope | Slope of `performance_metric ~ training_step` | Run linear regression on a rolling-window metric such as normalized progress or success probability | Measures improvement speed, not only final result |
| AUC of performance curve | Area under the learning curve | Compute the area under normalized progress, success probability, or reward across training steps | Captures total learning quality across training, not only the end point |
| Time-to-threshold | First episode/window where metric reaches target | Use rolling success probability or rolling normalized progress | Measures how quickly the agent becomes competent |
| Final performance window | Mean performance in the last 10% of episodes | Calculate mean normalized progress, success probability, and final goal error in the final training segment | More reliable than a single final episode |
| Learning stability | SD, IQR, or coefficient of variation in the final window | Compute variability of reward, progress, and goal error in the final 10% of episodes | Shows whether learning converges or remains unstable |
| Performance drop index | `max_rolling_score - final_rolling_score` | Compare best rolling performance against final rolling performance | Detects collapse, instability, or forgetting |

Recommended formulas:

```text
learning_slope = slope(metric ~ training_step)
AUC = area_under_curve(metric over training_step)
time_to_threshold = first training_step where rolling_success >= 0.80
final_performance = mean(metric in final 10% episodes)
learning_stability = sd(metric in final 10% episodes)
performance_drop_index = max(rolling_metric) - final(rolling_metric)
```

#### AUC specification

When this document says "AUC of performance curve," it means the following:

- choose one performance variable, such as normalized block progress, rolling success rate, or episode reward
- choose the x-axis as episode index or training step
- compute the numerical integral of the curve across the chosen interval
- optionally normalize by the interval width so that different training runs can be compared on the same scale

In plain language:

- if performance rises quickly and stays high, AUC is larger
- if performance stays low or improves only late, AUC is smaller

The spec should always state:

- what metric is integrated
- over which x-axis
- whether the curve is raw, rolling-window smoothed, or normalized
- whether the result is divided by total training span

Recommended AUC spec for this project:

```text
AUC(metric) = integral of rolling_metric over training_step
normalized_AUC = AUC / (last_step - first_step)
```

Suggested default:

- use a rolling window of `50` episodes for noisy metrics
- use normalized block progress or rolling success probability for learning-curve AUC
- use reward AUC only as a secondary signal, because reward can improve without actual task competence

### 6.2 Block Progress Metrics

These answer: is the block actually moving toward the goal, even when the agent does not fully succeed?

| Metric | Definition | Implementation concept | Why useful |
|---|---|---|---|
| Normalized block progress | `(start_distance - final_distance) / start_distance` | Log block-goal distance at episode start and end | Measures partial task progress, even in failed episodes |
| Progress rate | `normalized_progress / episode_steps` | Divide task progress by episode length | Measures efficiency of progress |
| Final goal error | `final_block_goal_distance` | Log distance between block and goal at episode end | Gives a continuous outcome instead of binary success/failure |
| Backtracking distance | Sum of block movement away from goal | At each step, compare current distance to previous distance | Captures bad pushing or pushing in the wrong direction |
| Monotonic progress ratio | Percentage of steps where block gets closer to goal | Count steps where `distance_t < distance_t-1` | Measures consistency of block progress |
| Useful displacement ratio | Goal-directed block displacement / total block displacement | Project block movement vector onto goal direction | Detects whether movement is useful or random |

Recommended formulas:

```text
normalized_block_progress =
  (start_block_goal_distance - final_block_goal_distance)
  / start_block_goal_distance

monotonic_progress_ratio =
  number_of_steps_block_gets_closer_to_goal
  / total_episode_steps

useful_displacement_ratio =
  sum(goal_directed_block_displacement)
  / sum(total_block_displacement)
```

Interpretation:

- `1.00` = block reached goal or nearly perfect progress
- `0.50` = block moved halfway toward goal
- `0.00` = no useful progress
- `< 0` = block ended farther from goal

#### How code maps to block-progress metrics

The current code already gives us the starting point for these calculations:

- `OnEpisodeBegin()` is where a new episode starts, so that is the correct place to capture the starting block-goal distance
- `ResetBlock()` and `GetRandomSpawnPos()` define the initial spatial state
- `GoalDetect.OnCollisionEnter()` defines the success event
- `ScoredAGoal()` is the place where the episode ends successfully and where we already know the final step count and reward

To compute block progress metrics, we need to log or reconstruct:

- `block.transform.position` at episode start
- `goal.transform.position` at episode start and end
- `block.transform.position` at episode end

Then:

- start distance is `distance(block_start, goal)`
- final distance is `distance(block_end, goal)`
- normalized progress is the distance reduction ratio

That means the code already contains the episode boundaries. We only need to store the spatial values across the episode.

### 6.3 Agent Efficiency Metrics

These answer: is the agent learning an efficient strategy, or is it just getting lucky after many steps?

| Metric | Definition | Implementation concept | Why useful |
|---|---|---|---|
| Path efficiency | `shortest_useful_path / actual_agent_path_length` | Track agent position each step and compare to an ideal route to the block/goal | Measures movement efficiency |
| Push efficiency | `useful_block_displacement / agent_path_length` | Compare how much the block moves toward the goal per unit of agent movement | Measures how much agent movement produces useful pushing |
| Contact latency | Steps before first block contact | Detect first collision/contact between agent and block | Measures how quickly the agent identifies the task object |
| Push ratio | `contact_steps / total_episode_steps` | Count steps where the agent is touching/pushing the block | Measures productive task engagement |
| Idle ratio | `no_progress_steps / total_episode_steps` | Count steps with no agent movement, no block movement, or no distance improvement | Detects wasted behavior |
| Action switch rate | `number_of_action_changes / total_steps` | Compare action at step `t` with action at step `t-1` | Measures unstable or jittery decision-making |
| Oscillation index | Repeated opposite-direction actions per step | Detect left-right-left, forward-back-forward, or rotate-left/rotate-right cycles | Detects confused movement or unstable control |

Recommended formulas:

```text
path_efficiency = shortest_relevant_distance / actual_agent_path_length
push_efficiency = useful_block_displacement / agent_path_length
contact_latency = first_contact_step - episode_start_step
idle_ratio = no_progress_steps / total_episode_steps
```

#### How code maps to agent-efficiency metrics

The current script already logs the episode step counter via `m_episodeSteps`.

To support the efficiency metrics, we need additional per-step logging for:

- agent position
- block position
- first contact with the block
- action selected each step

`MoveAgent(ActionSegment<int> act)` is the right place to capture the chosen action each step, because that is where the discrete action is interpreted.

`OnActionReceived()` is the right place to increment per-step counters such as:

- action changes
- idle steps
- no-progress steps

Because the current code moves the agent with rigidbody force and rotation, path efficiency can be measured from the cumulative distance traveled by the agent rigidbody.

### 6.4 Control Quality Metrics

These answer: did the agent learn smooth, stable pushing behavior, or is it bumping around randomly?

| Metric | Definition | Implementation concept | Why useful |
|---|---|---|---|
| Velocity toward goal | Mean block velocity projected onto goal direction | Use block position change per step and project onto the goal vector | Captures goal-directed pushing speed |
| Acceleration variance | Variance of agent or block acceleration | Use velocity changes across steps | Measures unstable movement |
| Jerk | Change in acceleration over time | Compute third-order movement change from position logs | Measures sudden movement changes |
| Rotation variance | Variance in agent angular velocity or rotation direction | Track rotation delta per step | Captures unstable turning behavior |
| Action entropy | Entropy of action distribution within an episode/window | Compute from selected actions, not only TensorBoard policy entropy | Shows whether actions are diverse, random, or overly repetitive |
| Repeated-action ratio | Long repeated action sequences / total actions | Count repeated action runs | Detects freezing or over-committed behavior |
| Policy smoothness | Mean action difference between steps | For discrete actions: switch rate. For continuous actions: mean absolute action change | Measures how sharply behavior changes |

Recommended formulas:

```text
goal_velocity = mean(dot(block_velocity_t, unit_vector_block_to_goal_t))
action_switch_rate = count(action_t != action_t-1) / total_steps
oscillation_index = count(opposite_action_pairs) / total_steps
policy_smoothness = mean(abs(action_t - action_t-1))
```

#### How code maps to control-quality metrics

The current code already supports motion logging through:

- `transform.Rotate(...)`
- `m_AgentRb.AddForce(...)`
- `m_AgentRb.linearVelocity`
- `m_AgentRb.angularVelocity`
- `m_BlockRb.linearVelocity`
- `m_BlockRb.angularVelocity`

That means:

- velocity toward goal can be derived from the block rigidbody
- acceleration variance can be derived from changes in velocity over time
- jerk can be derived from changes in acceleration over time
- rotation variance can be derived from angular velocity
- action entropy can be derived from the action sequence recorded per episode

In other words, the current scripts already contain the physics sources. The missing part is step-history capture.

For a block-pushing task, the strongest control-quality metrics are:

- `mean_goal_velocity`
- `action_switch_rate`
- `oscillation_index`

### 6.5 Reliability Metrics

These answer: can the agent perform consistently, or does it only sometimes succeed?

| Metric | Definition | Implementation concept | Why useful |
|---|---|---|---|
| Success probability | Rolling success rate | Use a rolling window, for example last 50 or 100 episodes | Interpretable task-completion measure |
| Reward consistency | SD/IQR of reward over rolling window | Calculate reward variability per agent | Shows whether reward is stable |
| Worst-case performance | Bottom 10% of performance, e.g. CVaR-10 | Sort episode scores and average the worst 10% | Measures robustness, not only average performance |
| Failure severity | Mean final goal error for failed episodes | Filter failed episodes and compute final distance | Separates small failures from severe failures |
| Failure mode distribution | Timeout / wrong direction / stuck / unstable | Classify failure using rule-based labels | Explains why failures happen |

Recommended formulas:

```text
rolling_success_probability = sum(success in rolling_window) / window_size
reward_consistency = sd(episode_reward in rolling_window)
CVaR_10 = mean(worst 10% of episode_performance_scores)
failure_severity = mean(final_goal_error where success == 0)
```

#### How code maps to reliability metrics

`PushAgentBasic` already distinguishes success and timeout:

- success is recorded in `ScoredAGoal()`
- timeout is recorded in `OnActionReceived()` when `m_episodeSteps >= MaxStep`

That means reliability metrics can be computed directly from the episode log:

- rolling success probability uses the `success` column
- reward consistency uses `episode_reward`
- worst-case performance uses the distribution of per-episode scores
- failure severity uses `final_block_goal_distance`
- failure mode distribution needs extra labels, but the success/timeout split is already present

This is important because the code already provides a clean binary completion signal, which is the foundation for more advanced robustness statistics.

### 6.6 TensorBoard Comparison Metrics

This is the bridge between Unity behavior and training diagnostics.

| Unity behavior metric | TensorBoard metric | Statistical test | Why this pairing is useful |
|---|---|---|---|
| Normalized block progress | Cumulative reward | Correlation, regression, lagged correlation | Tests whether reward increase reflects actual task progress |
| Progress rate | Episode length | Correlation, slope comparison | Tests whether shorter episodes reflect efficient progress or early termination |
| Push efficiency | Policy loss / entropy | Correlation, lagged correlation | Tests whether policy updates and exploration become useful pushing behavior |
| Learning stability | Value loss | Rolling correlation, final-window comparison | Tests whether stable value prediction aligns with stable behavior |
| Success probability | Cumulative reward | Rolling correlation, threshold analysis | Tests whether reward maps to task completion |
| Oscillation index | Entropy / action distribution | Correlation, lagged correlation | Tests whether randomness or unstable policy outputs produce jittery behavior |
| Final goal error | Reward / value estimate | Regression, correlation | Tests whether training signals reflect actual final accuracy |

Recommended statistics:

```text
correlation(Unity_metric, TensorBoard_metric)
cross_correlation(Unity_metric_t, TensorBoard_metric_t-lag)
Unity_metric ~ cumulative_reward + policy_loss + value_loss + entropy
Unity_metric ~ TensorBoard_metric + training_step
```

Example lag question:

- Does policy loss decrease first, then push efficiency improve 5,000 training steps later?

#### What the AUC spec means in practice

You asked what the spec means. In this FRD, a metric spec means the exact rule for how the metric is built.

For AUC, the spec should answer:

1. Which variable is being integrated?
2. Is the variable raw or rolling-window smoothed?
3. Is the x-axis episode number or training step?
4. Is the result normalized by training length?
5. Is the metric computed per agent or across all agents?

Example spec:

- `AUC(normalized_block_progress)` per agent
- use rolling mean over 50 episodes
- integrate against `training_step`
- normalize by total training-step span

That produces a number that rewards both speed and consistency of improvement.

If we omit the spec, "AUC" is ambiguous and two people can compute different values from the same logs.

## 7. Environment Parameters to Log

The PushBlock scripts also expose configuration values that should be logged alongside the episode metrics because they change task difficulty.

### 7.1 From `PushBlockSettings.cs`

| Parameter | Meaning | Why it should be logged |
|---|---|---|
| `agentRunSpeed` | Walking/movement speed of the agent | Strongly affects episode length, path efficiency, and push efficiency |
| `agentRotationSpeed` | Agent turning speed | Affects control smoothness and oscillation |
| `spawnAreaMarginMultiplier` | How far from the edge objects can spawn | Changes initial difficulty and start distance statistics |
| `blockMassLevel` / selected block mass | Mass of the block | Changes push difficulty and reward progression |
| `blockSizeLevel` / selected block size | Size of the block | Changes contact surface and pushing stability |
| `surfaceFrictionLevel` / selected friction values | Static and dynamic friction | Affects slip, backtracking, and goal reachability |
| `defaultBlockDrag` | Block damping / drag | Affects how long the block continues moving after contact |

### 7.2 Why these matter

These are not just environment settings. They are explanatory variables.

If two agents produce different metrics, we need to know whether the difference came from the policy or from a harder environment configuration.

This is especially important for:

- learning slope
- time-to-threshold
- push efficiency
- oscillation index
- success probability

## 8. Episode Data Schema

Each episode should be logged as one row in a structured dataset.

Suggested columns:

- `agent_id`
- `episode_id`
- `training_step`
- `episode_reward`
- `success`
- `episode_length`
- `time_to_goal`
- `start_block_goal_distance`
- `final_block_goal_distance`
- `normalized_block_progress`
- `progress_rate`
- `block_path_length`
- `agent_path_length`
- `useful_block_displacement`
- `push_efficiency`
- `contact_latency`
- `push_ratio`
- `idle_ratio`
- `action_switch_rate`
- `oscillation_index`
- `mean_goal_directed_velocity`
- `velocity_variance`
- `rolling_success_rate`
- `final_window_mean`
- `final_window_std_dev`
- `policy_loss`
- `value_loss`
- `entropy`

If we want the FRD to be usable for analysis later, each episode row should also carry the environment condition values:

- `agent_run_speed`
- `agent_rotation_speed`
- `spawn_area_margin_multiplier`
- `block_mass`
- `block_size`
- `static_friction`
- `dynamic_friction`
- `block_drag`

That lets us compare metrics fairly across runs.

## 9. Recording Rules

### 9.1 Episode boundary

Finalized metrics should be recorded once per episode:

- on success
- on timeout or max-step failure

### 9.2 Single-agent logging

For single-agent experiments, metrics should be recorded without aggregation across agents.

### 9.3 Multi-agent logging

For multi-agent experiments, each agent must keep its own `agent_id` and per-episode metric row.

This prevents TensorBoard averaging from hiding the behavior of individual agents.

## 10. Analysis Plan

Recommended comparisons for single-agent analysis:

- trend over training steps
- final-window mean and variance
- rolling success probability
- correlation with TensorBoard reward, loss, and entropy
- AUC of normalized progress and reward
- time-to-threshold

Recommended comparisons for multi-agent analysis:

- per-agent learning slope
- per-agent final-window mean
- per-agent stability
- per-agent robustness metrics such as CVaR-10
- statistical comparison across agents using confidence intervals and effect sizes

The important point is that we should not compare agents only by final reward. Two agents can end with the same reward but have very different stability, efficiency, and failure behavior.

## 11. Acceptance Criteria

This documentation is complete when:

- every metric used in the experiment has a name, definition, and formula
- implemented stats are separated from proposed stats
- each variable can be traced to a logging source or data field
- the schema supports later comparison across agents
- AUC and related curve metrics have an explicit spec
- the metrics can be derived from the current code with documented additions

## 12. Milestones

This project should be implemented in three stages so the metrics stay grounded in the code and we can validate each layer before moving on.

### Milestone 1: MVP

Goal:

- get the minimum useful evaluation pipeline working for a single PushBlock agent

Scope:

- keep the current reward and success logging
- add episode export for `agent_id`, `episode_id`, and `training_step`
- compute `normalized_block_progress`
- compute `rolling_success_rate`
- compute `learning_slope` and `final_performance_window`

Why this milestone matters:

- it proves that the learning-curve analysis works end to end
- it gives us a baseline before adding more complex trajectory metrics

### Milestone 2: Full Single-Agent Metrics

Goal:

- expand the single-agent analysis into a full research-grade metric set

Scope:

- add block-progress metrics
- add agent-efficiency metrics
- add control-quality metrics
- add reliability metrics
- compute AUC, time-to-threshold, stability, and performance-drop metrics

Why this milestone matters:

- it gives us a full behavioral description of how the bot learns to push blocks
- it separates task progress from motion quality and reward shape

### Milestone 3: Multi-Agent Comparison

Goal:

- compare agents one by one without losing individual behavior in averaged TensorBoard curves

Scope:

- log each agent separately
- preserve `agent_id` in all exports
- compute the same metric set per agent
- compare agents using slopes, AUC, thresholds, stability, and robustness

Why this milestone matters:

- it supports fair comparison across multiple training agents
- it makes the analysis usable for ranking and ablation studies

## 13. References

The following references were used as the basis for the metric design and should be verified against the final bibliography format before publication.

- Agarwal et al. (2021)
- Ahmed et al. (2019)
- Andrychowicz et al. (2017)
- Anderson et al. (2018)
- Christmann et al. (2024)
- Henderson et al. (2018)
- Kalashnikov et al. (2018)
- Lei et al. (2025)
- Mysore et al. (2021)
- Ng et al. (1999)
- Schulman et al. (2015)
- Schulman et al. (2017)
- Shen et al. (2020)
- Sutton (1988)
- Unity ML-Agents docs
- Wang et al. (2023)
