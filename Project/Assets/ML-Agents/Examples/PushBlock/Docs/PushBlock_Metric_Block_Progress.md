# Block Progress Metrics

This document defines the metrics used to measure whether the orange block is actually moving toward the goal.

This is one of the most important spec groups for a pushing task because it gives us a continuous measure of task progress. A success flag tells us only whether the goal was reached. Block-progress metrics tell us how close the agent got, how efficiently it moved, and whether it moved in the right direction.

## Code Basis

The current scripts already expose the spatial state needed to compute these metrics.

Useful code hooks:

- `OnEpisodeBegin()` for reset and start-state capture
- `ResetBlock()` for the block spawn position
- `GoalDetect.OnCollisionEnter()` for success detection
- `ScoredAGoal()` for terminal success handling
- `block.transform.position` for block location
- `goal.transform.position` for goal location

The code already resets the block and the agent to randomized positions, which means every episode starts from a different spatial configuration. That makes normalized distance-based metrics especially useful because they can be compared across episodes with different spawn positions.

## Metrics

### Normalized Block Progress

Formula:

```text
(start_distance - final_distance) / start_distance
```

Meaning:

- how much of the original block-to-goal distance was removed
- positive values mean the agent moved the block in the right direction
- zero means no useful progress
- negative values mean the block ended farther away

Why it matters:

- works even for failed episodes
- gives a continuous measure instead of a binary one

### Progress Rate

Formula:

```text
normalized_progress / episode_steps
```

Meaning:

- progress per step
- measures how efficiently the agent converts time into task progress

Why it matters:

- two agents may have the same final progress, but one gets there much faster

### Final Goal Error

Definition:

- final distance between the block and the goal

Meaning:

- how close the agent ended up
- useful when the agent does not fully succeed

Why it matters:

- gives a continuous completion measure
- can be used in failed episodes, not just successful ones

### Backtracking Distance

Definition:

- the sum of all increases in distance from the goal over the episode

Meaning:

- how much the block moved the wrong way
- a large value suggests wasted pushing or bad correction

### Monotonic Progress Ratio

Definition:

- the fraction of steps where the distance to the goal decreased

Meaning:

- whether the block generally moved forward instead of oscillating around the same area

### Useful Displacement Ratio

Definition:

- goal-directed displacement divided by total displacement

Meaning:

- how much of the block motion was actually helpful

This is one of the strongest measures in a pushing task because it distinguishes random motion from productive motion.

## Recommended Formulas

```text
normalized_block_progress =
  (start_block_goal_distance - final_block_goal_distance)
  / start_block_goal_distance

progress_rate =
  normalized_block_progress / episode_steps

monotonic_progress_ratio =
  number_of_steps_where_distance_decreased / total_episode_steps

useful_displacement_ratio =
  sum(goal_directed_block_displacement) / sum(total_block_displacement)
```

## Code Spec

This spec is built around episode start, movement, and episode end.

At the start of an episode:

- record the block position
- record the goal position
- compute `start_block_goal_distance`

During the episode:

- optionally store block position at every step
- compute distance to goal at each step
- track whether the distance is decreasing

At the end of the episode:

- record the final block position
- compute `final_block_goal_distance`
- derive normalized progress

The current code already gives the boundaries:

- `OnEpisodeBegin()` marks the start
- `GoalDetect.OnCollisionEnter()` marks success
- `OnActionReceived()` marks the step loop and timeout path

That means the only missing part is trajectory logging.

## Practical Interpretation

- `1.00` means the block reached the goal or nearly did
- `0.50` means the block covered about half the original distance
- `0.00` means no useful progress
- negative values mean the agent pushed the block the wrong way

This matters because early in training many episodes will fail, but some failed episodes still show genuine improvement. These metrics capture that improvement instead of throwing it away.

## Logging Requirements

To compute these metrics, store:

- block position at episode start
- block position at episode end
- goal position
- optionally, block position at every step
- episode length
- success flag

If step-level logging is available, the analysis can also compute:

- backtracking distance
- monotonic progress ratio
- useful displacement ratio

## Implementation Notes

If you only log success/fail, you cannot see whether the agent is learning to push better over time.

If you log only reward, you can miss cases where the agent becomes more efficient but the reward stays noisy.

This is why block-progress metrics are important. They give a direct behavioral view of the task.

For comparison across agents, the same progress formula should be used for every run so that the scores are truly comparable.
