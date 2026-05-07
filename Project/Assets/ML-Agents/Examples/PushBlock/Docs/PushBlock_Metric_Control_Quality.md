# Control Quality Metrics

This document defines the metrics used to measure whether the agent learned smooth, stable control instead of noisy or jittery movement.

These metrics focus on motion quality. A policy can still finish the task while being unstable, spinning, or oscillating. Control quality metrics make that visible.

## Code Basis

The current code exposes the physics state we need:

- `m_AgentRb.linearVelocity`
- `m_AgentRb.angularVelocity`
- `m_BlockRb.linearVelocity`
- `m_BlockRb.angularVelocity`
- `transform.Rotate(...)`
- `m_AgentRb.AddForce(...)`
- discrete action selection in `MoveAgent()`

Because the agent is controlled through rigidbody motion, we can compute higher-order motion statistics from the step history.

## Metrics

### Velocity Toward Goal

Definition:

- mean block velocity projected onto the goal direction

Meaning:

- how strongly the block motion aligns with the task
- higher values usually mean more useful pushing

### Acceleration Variance

Definition:

- variance of velocity changes over time

Meaning:

- how stable the movement is
- high variance indicates jerky control

### Jerk

Definition:

- change in acceleration over time

Meaning:

- how sudden motion changes are
- high jerk usually indicates abrupt and less controlled behavior

### Rotation Variance

Definition:

- variance in angular velocity or turning direction

Meaning:

- how stable the agent's heading is
- helps detect spinning or unnecessary turning

### Action Entropy

Definition:

- entropy of the action distribution over an episode or rolling window

Meaning:

- how diverse or random the action choice is
- can reveal whether the policy is focused or scattered

### Repeated-Action Ratio

Definition:

- long repeated action sequences divided by total actions

Meaning:

- how much the agent gets stuck in repetitive behavior
- useful for spotting freeze-like or over-committed policies

### Policy Smoothness

Definition:

- mean difference between consecutive actions

Meaning:

- how smoothly the policy changes over time
- lower abruptness usually means cleaner control

## Recommended Formulas

```text
goal_velocity = mean(dot(block_velocity_t, unit_vector_block_to_goal_t))
action_switch_rate = count(action_t != action_t-1) / total_steps
oscillation_index = count(opposite_action_pairs) / total_steps
policy_smoothness = mean(abs(action_t - action_t-1))
```

## Code Spec

The motion source is split into two parts:

1. `MoveAgent()` interprets the action and applies force or rotation.
2. The rigidbody state records the resulting physical response.

That means the doc should treat the control signal and the physical outcome separately.

For this metric family, the analysis layer should log:

- action index
- agent linear velocity
- agent angular velocity
- block linear velocity
- block angular velocity
- step index

From these logs, compute:

- velocity toward goal
- acceleration variance
- jerk
- rotation variance
- action entropy
- repeated-action ratio
- policy smoothness

## Why These Metrics Matter

Reward is not enough to tell us if the policy is smooth.

For example, two agents may both succeed:

- one agent pushes with controlled, steady motion
- another agent spins around and eventually hits the block into the goal

Both can earn reward, but only one is really learning a clean control policy.

These metrics let us compare:

- stability
- smoothness
- randomness
- action repetition

That makes them useful both for training diagnostics and for later agent-by-agent comparison.

## Logging Requirements

To compute these metrics reliably, store:

- action each step
- linear velocity each step
- angular velocity each step
- step index

If the analysis uses a rolling window, store the window size and keep it consistent across agents.

## Implementation Notes

For this task, the strongest summary metrics are usually:

- `mean_goal_velocity`
- `action_switch_rate`
- `oscillation_index`

Those three together tell us whether the agent is pushing effectively, changing actions too often, or bouncing between conflicting commands.
