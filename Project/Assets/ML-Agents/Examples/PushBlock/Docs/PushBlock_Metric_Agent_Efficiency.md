# Agent Efficiency Metrics

This document defines the metrics used to measure how efficiently the agent moves while solving PushBlock.

The purpose of this metric group is to answer a different question from learning improvement. Learning improvement tells us whether the agent is getting better over training. Agent efficiency tells us whether the learned behavior is physically clean, direct, and productive.

An agent can succeed and still be inefficient. It may wander, spin, stall, or push in the wrong direction before eventually reaching the goal. Efficiency metrics are meant to expose that behavior.

## 1. Scope

This metric group covers step-level behavior within an episode.

In scope:

- movement distance
- block-contact timing
- action switching
- idle behavior
- oscillation
- task-relevant motion efficiency

Out of scope:

- learning slope
- AUC
- final-window stability
- reward trajectory analysis

Those belong to the learning-improvement spec.

## 2. Code Basis

The current `PushAgentBasic.cs` script already provides the movement and episode-loop machinery needed for this metric group.

Relevant runtime signals:

- `MoveAgent(ActionSegment<int> act)`
- `OnActionReceived()`
- `m_episodeSteps`
- `m_AgentRb.linearVelocity`
- `m_AgentRb.angularVelocity`
- `m_BlockRb.linearVelocity`
- `m_BlockRb.angularVelocity`
- `transform.Rotate(...)`
- `m_AgentRb.AddForce(...)`

The discrete action space is already interpretable:

- forward
- backward
- rotate left
- rotate right
- strafe left
- strafe right

That makes it possible to count action changes and identify repeated back-and-forth control patterns.

## 3. Why This Metric Group Exists

Reward does not tell us whether movement was efficient.

For example, two agents can both reach the goal:

- one may move directly, touch the block quickly, and push with minimal waste
- the other may wander for a long time and only succeed after many useless actions

Both can earn reward, but only one is clearly efficient.

This metric group exists to measure the quality of the agent's motion, not just the outcome.

## 4. Metrics

### 4.1 Path Efficiency

Definition:

```text
path_efficiency = shortest_relevant_distance / actual_agent_path_length
```

Meaning:

- how direct the agent's movement was
- values closer to 1 mean less wasted travel

Interpretation:

- high path efficiency means the agent moved with purpose
- low path efficiency means the agent wasted time moving around

Implementation note:

- `shortest_relevant_distance` is the ideal distance from agent start to the useful interaction point
- `actual_agent_path_length` is the sum of step-to-step agent displacement

### 4.2 Push Efficiency

Definition:

```text
push_efficiency = useful_block_displacement / agent_path_length
```

Meaning:

- how much useful block movement was produced per unit of agent movement

Interpretation:

- high push efficiency means the agent's movement is being converted into useful block progress
- low push efficiency means the agent is moving a lot without helping the task much

### 4.3 Contact Latency

Definition:

- the number of steps before the agent first touches the block

Meaning:

- how quickly the agent begins interacting with the task object

Interpretation:

- low contact latency is better
- high contact latency suggests the agent is slow to locate or approach the block

### 4.4 Push Ratio

Definition:

```text
push_ratio = contact_steps / total_episode_steps
```

Meaning:

- how much of the episode was spent in productive contact

Interpretation:

- high push ratio usually means the agent spends more time doing useful task work
- low push ratio means the agent spends too much time not engaged with the block

### 4.5 Idle Ratio

Definition:

```text
idle_ratio = no_progress_steps / total_episode_steps
```

Meaning:

- how much of the episode was wasted

Interpretation:

- high idle ratio means the agent is standing still, circling, or making no useful progress
- low idle ratio means the agent is staying active in a task-relevant way

### 4.6 Action Switch Rate

Definition:

```text
action_switch_rate = number_of_action_changes / total_steps
```

Meaning:

- how often the agent changes behavior

Interpretation:

- a very high value can indicate indecision or unstable control
- a moderate value can be normal if the task requires frequent correction

### 4.7 Oscillation Index

Definition:

- repeated opposite-direction action sequences per step

Meaning:

- whether the agent is bouncing back and forth
- whether the policy is jittering instead of acting smoothly

Interpretation:

- low oscillation is usually better
- high oscillation often means poor control or unstable action selection

## 5. Recommended Formulas

```text
path_efficiency = shortest_relevant_distance / actual_agent_path_length
push_efficiency = useful_block_displacement / agent_path_length
contact_latency = first_contact_step - episode_start_step
idle_ratio = no_progress_steps / total_episode_steps
action_switch_rate = action_changes / total_steps
oscillation_index = opposite_direction_reversals / total_steps
```

## 6. Code Spec

The movement logic is defined in `MoveAgent()`.

That is the correct place to interpret the selected action and map it to motion.

The per-step bookkeeping is defined in `OnActionReceived()`.

That is the correct place to:

- increment the episode step counter
- record the current action
- compare the current action with the previous action
- detect no-progress steps
- record step-level positions

To compute this metric group properly, the analysis layer should also store:

- agent position at each step
- block position at each step
- selected action at each step
- first block contact step
- whether the step produced useful progress

From those values, we can derive:

- agent path length
- block path length
- useful block displacement
- action change count
- repeated action runs
- contact latency
- push ratio
- idle ratio

## 7. Logging Requirements

To support this metric group, the Unity output should log one row per episode, and preferably a step-level trace as well.

Episode-level fields:

- `agent_id`
- `episode_id`
- `training_step`
- `episode_length`
- `success`
- `path_efficiency`
- `push_efficiency`
- `contact_latency`
- `push_ratio`
- `idle_ratio`
- `action_switch_rate`
- `oscillation_index`

Step-level fields:

- `step_index`
- `agent_position`
- `block_position`
- `action`
- `block_goal_distance`
- `agent_block_distance`
- `contact_flag`
- `progress_flag`

Without step history, several of these metrics cannot be computed accurately.

## 8. Debug Logging

If we test this in Unity, the console logs should use a separate prefix so they are easy to filter.

Recommended prefix:

- `[agent_efficiency]`

Recommended log stages:

- `start`
- `step`
- `end`
- `export`

Example end log:

```text
[agent_efficiency] end agent=Agent episode=12 reason=goal steps=87 path_efficiency=0.63 push_efficiency=0.41 contact_latency=14 push_ratio=0.52 idle_ratio=0.21 action_switch_rate=0.18 oscillation_index=0.07
```

## 9. Implementation Notes

The most important design choice is how to define "useful" movement.

For this task, useful movement should mean movement that reduces block-to-goal distance.

That makes the metric task-specific and prevents random motion from counting as productive behavior.

For later comparisons between agents, the same definitions must be reused across all runs. Otherwise the values will not be comparable.

## 10. Relationship to TensorBoard

Agent efficiency is not a TensorBoard metric by itself.

It is a Unity-side behavioral metric that can be compared later against trainer-side diagnostics such as:

- policy loss
- entropy
- cumulative reward

Those comparisons belong in a later spec section, after the step-level data exists.

