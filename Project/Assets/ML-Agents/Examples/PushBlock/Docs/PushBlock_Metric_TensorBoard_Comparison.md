# TensorBoard Comparison Metrics

This document defines the bridge between the Unity-side behavior metrics and the TensorBoard training diagnostics.

The goal is to check whether the training curves shown by ML-Agents actually match what the agent is doing in the environment.

## Code Basis

The current code already emits the Unity-side episode summaries:

- `PushBlock/episode_reward`
- `PushBlock/success`
- `PushBlock/episode_length`
- `PushBlock/time_to_goal`

From the trainer side, TensorBoard typically gives us:

- cumulative reward
- policy loss
- value loss
- entropy

If multiple agents are training at the same time, we need per-agent rows on the Unity side so the TensorBoard plots do not hide individual behavior.

## Why This File Exists

TensorBoard curves are useful, but they are not the same thing as actual gameplay behavior.

For example:

- reward may increase because episodes end faster
- entropy may decrease while the agent is still physically inefficient
- value loss may stabilize before the task behavior becomes better

This file defines which TensorBoard signals should be compared against which Unity metrics, and why.

## Comparison Pairs

### Normalized Block Progress vs Cumulative Reward

Use this pairing to check whether reward tracks real task improvement.

If reward rises but block progress does not, then the reward signal is not reflecting actual task competence very well.

### Progress Rate vs Episode Length

Use this pairing to check whether shorter episodes mean better efficiency.

If episodes are shorter but progress is not better, the policy may simply be terminating more quickly without being more effective.

### Push Efficiency vs Policy Loss / Entropy

Use this pairing to check whether policy updates produce more useful movement.

This is especially helpful when the policy seems to train well numerically but still moves awkwardly in the environment.

### Learning Stability vs Value Loss

Use this pairing to check whether better value prediction aligns with more stable behavior.

If value loss stabilizes and behavior also becomes less noisy, that is a good sign.

### Success Probability vs Cumulative Reward

Use this pairing to check whether reward matches actual completion.

This is one of the simplest and most useful sanity checks.

### Oscillation Index vs Entropy / Action Distribution

Use this pairing to check whether action randomness matches physical jitter.

If entropy drops but oscillation stays high, the policy may still be physically unstable.

### Final Goal Error vs Reward / Value Estimate

Use this pairing to check whether reward reflects actual task accuracy.

The final distance to the goal is often a better behavioral outcome than reward alone.

## Recommended Statistics

Recommended tests:

- correlation
- lagged correlation
- regression
- rolling-window comparison

Examples:

- Does reward improve before behavior improves, or after?
- Does entropy fall before action switching becomes smoother?
- Does value loss stabilization line up with lower final goal error?

## Recommended Formulas

```text
correlation(Unity_metric, TensorBoard_metric)
cross_correlation(Unity_metric_t, TensorBoard_metric_t-lag)
Unity_metric ~ cumulative_reward + policy_loss + value_loss + entropy
Unity_metric ~ TensorBoard_metric + training_step
```

## Code Spec

The comparison needs a stable join key.

At minimum, the dataset should store:

- `agent_id`
- `episode_id`
- `training_step`
- Unity-side episode metrics
- TensorBoard export values

The most important part is alignment. The metrics should be compared at the same episode boundary or the same training step. If the x-axis is inconsistent, the comparison is not valid.

In multi-agent training, this becomes even more important because TensorBoard may average statistics across agents. If we want to compare individual agents later, we must preserve each agent's rows separately.

## Interpretation Rules

Good behavior alignment usually looks like this:

- cumulative reward rises
- normalized progress rises
- success probability rises
- final goal error falls
- action oscillation falls

If those signals disagree, the doc should treat that as a warning, not as success.

## Logging Requirements

To make this comparison possible, store:

- per-episode Unity metrics
- TensorBoard export data
- `training_step` or `global_step`
- `agent_id` for multi-agent runs
- environment configuration values if the task difficulty changes

## Implementation Notes

The best comparison strategy is usually a combination of:

- same-step correlation
- rolling-window trend comparison
- lagged correlation

That lets us answer not just whether two metrics are related, but whether one tends to lead the other.

This is valuable because policy updates do not always show up in behavior immediately. Sometimes a loss curve changes first, and the gameplay curve follows later.
