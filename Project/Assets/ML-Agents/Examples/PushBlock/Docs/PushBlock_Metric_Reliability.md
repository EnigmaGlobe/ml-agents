# Reliability Metrics

This document defines the metrics used to measure whether an agent performs consistently across many episodes.

Average performance is not enough. A policy that succeeds sometimes and fails badly at other times is less reliable than a policy that performs slightly less well on average but is stable.

## Code Basis

The current code already gives us the reliable episode outcome signals:

- `ScoredAGoal()` for success
- `OnActionReceived()` with `m_episodeSteps >= MaxStep` for timeout failure
- `m_episodeCumulativeReward` for reward history
- `m_episodeMetricsRecorded` to avoid duplicate logging

The environment also randomizes spawn positions and can vary physics parameters through `PushBlockSettings` and `EnvironmentParameters`. That means reliability needs to be measured across many episodes, not from one run.

## Metrics

### Success Probability

Definition:

- rolling success rate over a fixed window

Meaning:

- how often the agent completes the task
- the clearest reliability measure for this environment

### Reward Consistency

Definition:

- standard deviation or interquartile range of reward over a rolling window

Meaning:

- how stable the reward is across episodes
- low variability is usually a good sign

### Worst-Case Performance

Definition:

- mean of the bottom 10 percent of episode outcomes

Meaning:

- how bad the weakest episodes are
- useful for measuring robustness

### Failure Severity

Definition:

- mean final goal error among failed episodes

Meaning:

- how severe failures are
- separates near misses from major misses

### Failure Mode Distribution

Definition:

- percent of failures that are timeout, stuck, wrong direction, unstable, or unknown

Meaning:

- why failures happen
- useful for debugging behavior and training setup

## Recommended Formulas

```text
rolling_success_probability = sum(success in rolling_window) / window_size
reward_consistency = sd(episode_reward in rolling_window)
CVaR_10 = mean(worst 10% of episode_scores)
failure_severity = mean(final_goal_error where success == 0)
```

## Code Spec

The code already produces the clean labels needed for this file:

- success episodes are logged in `ScoredAGoal()`
- failure episodes are logged in the max-step branch

That makes reliability metrics naturally episode-level and easy to aggregate.

However, the environment randomization means the analysis should also store:

- block mass
- block size
- surface friction
- agent speed
- spawn area margin

Otherwise, changes in reliability might come from environment difficulty rather than policy quality.

## Why These Metrics Matter

Two agents can have the same mean reward and still behave very differently:

- one may succeed steadily and rarely fail
- the other may be unstable, with a few very good episodes and several very bad ones

Reliability metrics expose that difference.

This is especially important if we later compare multiple agents one by one. A good agent is not only one that scores high. It is one that performs predictably.

## Logging Requirements

To compute these metrics, store:

- `success`
- `episode_reward`
- `final_block_goal_distance`
- `failure_mode` if available
- environment configuration values

If failure mode tagging is not available in code, it can be added later with a simple rule-based classifier.

## Implementation Notes

Recommended analysis windows:

- short rolling window for quick monitoring
- final window for reporting stable results

Recommended summary outputs:

- mean success probability
- reward standard deviation
- CVaR-10
- failure severity
- failure mode percentages

These are the numbers that tell us whether the policy is dependable, not just whether it can occasionally do the task.
