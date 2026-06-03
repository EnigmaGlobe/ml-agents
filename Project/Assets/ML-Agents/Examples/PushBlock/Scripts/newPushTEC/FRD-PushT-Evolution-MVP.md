# FRD: PushT Evolution Integration Plan

## Why This Exists

You already have a PushT-style ML-Agents scene with a trained or trainable
robot agent. You want to reuse the smallest useful part of ABL's experiment
thinking to evolve environment and block parameters, without importing the
planner, GP tree system, or UI framework.

This folder is the starting point for that.

This FRD is written from your standpoint:

- what you are trying to build
- what you need to wire up
- what you should ship first
- what comes in later phases

## Your Goal

You want a repeatable workflow where you can:

1. choose a set of PushT block/environment parameters to optimize
2. run multiple episodes for one parameter set
3. score that parameter set using your PushT agent's outcomes
4. repeat this across a population
5. keep the best candidates and mutate them
6. inspect the best settings at the end

## Scope Clarification: Environment Parameter EC Only

This FRD and the code in this folder are scoped to **Environment Parameter
Evolutionary Computation (EC)**. The agent's neural-network policy is treated
as **frozen** during the entire EC run.

The EC searches for environment configurations (block mass, size, friction,
obstacle placement, goal position, etc.) that produce the best, most stable,
and most useful agent performance.

The agent does **not** learn during EC. It is a fixed test subject.

## What You Are Building

You are building a small evolution module for PushT with these responsibilities:

- store one candidate parameter set (genome)
- apply that candidate to the scene before an episode starts
- run a PushT episode with a **fixed, pre-trained agent**
- read success, progress, stability, and error signals from the agent/environment
- compute a fitness score using a multi-objective formula
- run multiple episodes per genome to reduce noise
- evolve candidates over generations using elite selection + mutation

## What You Are Not Building

For all phases in this FRD, you are **not** building:

- ABL's HTN planner
- strongly-typed GP trees
- graph visualization
- custom inspector UI
- distributed evaluation
- complex crossover logic
- **agent training or policy updates**

## Critical Distinction: EC Does Not Train The Agent

The most common misunderstanding of this system is that it somehow makes the
agent "smarter" or "learn" over generations. It does not.

| Aspect | What This EC Does | What ML-Agents Training Does |
|---|---|---|
| What changes | Environment parameters (block, ground, layout) | Agent neural-network weights |
| Agent state | Frozen `.onnx` / `.nn` policy, read-only | Active learning, weights updated via PPO/SAC |
| Who learns | **Nothing learns.** The environment parameters are searched. | The agent learns from reward feedback |
| Python trainer | Not required | Required |
| Episode purpose | Evaluate a static environment config | Generate training experience |

**Verification:** If you binary-diff the agent's model file before and after a
full EC run, it will be **identical byte-for-byte**. The agent's "brain" is
never touched.

## Files In This Folder

- `PushTBlockGenome.cs`
  This is your candidate parameter container. Extend this when you want to
evolve more than basic block properties.
- `PushTEvolutionConfig.cs`
  This stores your search settings and allowed parameter ranges.
- `PushTEvolutionResult.cs`
  This stores episode-level, genome-level, and run-level outputs.
  
  **Three result layers:**
  
  1. `PushTEpisodeResult` — one episode's raw outcome.
  2. `PushTGenomeEvaluation` — aggregated stats for one genome across
     `episodesPerGenome` runs (includes `SuccessRate`, `IQMProgress`,
     `SDProgress`, `IQRProgress`, `MeanGoalError`, and the final `fitness`).
  3. `PushTEvolutionRunResult` — the full run summary (`bestGenome`,
     `bestFitness`, per-generation history, and the complete population
     evaluation list).
- `PushTBlockEvaluator.cs`
  This is the base class for evaluating one genome over multiple episodes.
- `PushTEvolutionRunner.cs`
  This is the search loop.
- `PushTSceneEvaluatorTemplate.cs`
  This is the scene-side evaluator template you adapt to your PushT setup.
- `PushTApplyTestGenome.cs`
  This is the Phase 0 helper for proving one hand-written parameter set can
change the block.
- `PushTOneScriptEc.cs`
  This is the simplest first integration: add one script to the existing
PushT agent GameObject and press Play.
- `samplepushagent.md`
  This is your sample PushT agent reference.

## Fitness Function Specification

Fitness is the **only signal** the EC runner uses to rank candidates. The
formula must translate your environment-quality goals into a single scalar.

### 3-Tier Fitness Formula

The formula is intentionally designed as **three tiers** to prevent gaming:

| Tier | Condition | Formula | Purpose |
|---|---|---|---|
| **Tier 1: Gatekeeper** | `SuccessRate < successRateThreshold` (default `0.9`) | `fitness = 80 × SuccessRate` | Hard gate. If the agent cannot reliably succeed, nothing else matters. Maximum possible fitness in this tier is `72` (well below Tier 2 minimum). |
| **Tier 2: Quality** | `SuccessRate ≥ successRateThreshold` | `fitness = 100 + 40×IQMProgress + 25×Difficulty + 30×TimeScore − StabilityPenalty − 20×GoalErrorScore − 50×InvalidPenalty` | Unlocks only when the environment is reliably solvable. Base `100` ensures any Tier 2 genome always outranks any Tier 1 genome. |
| **Tier 3: Time Target Zone** | Embedded inside Tier 2 via `TimeScore` | See **Time Score** below | Graduated time scoring replaces the old hard "too-easy" threshold, preventing EC from gaming the boundary. |

**Why this structure?**

- **Old approach** (hard threshold at 5s with `TooEasyPenalty`) caused the EC to
  converge just above the threshold — e.g. blocks that took 5.1s instead of
  genuinely interesting 8–12s configurations.
- **New approach** uses a **target zone** (`idealTimeMin = 5s` to
  `idealTimeMax = 12s`) with maximum reward inside the zone, partial reward on
  the edges, and negative reward outside. This creates a smooth, ungameable
  incentive.

### Term Definitions

| Term | Range | How To Compute | Why It Matters |
|---|---|---|---|
| `SuccessRate` | 0..1 | `successfulEpisodes / totalEpisodes` | **Tier 1 gatekeeper.** Must be ≥ `successRateThreshold` (default `0.9`) to unlock Tier 2. |
| `IQMProgress` | 0..1 | Interquartile mean of `normalized_task_progress` across episodes | Robust central tendency. Discards outlier episodes (lucky flukes and unlucky bugs). |
| `Difficulty` | 0..1 | See **Difficulty Score** below | Weak incentive to favor intrinsically harder parameter combinations. |
| `TimeScore` | −2.0..+1.0 | See **Time Score** below | **Replaces the old `TooEasyPenalty`.** Graduated scoring based on `meanTimeToGoal`. |
| `IQRProgress` | 0..1 | `Q3(progress) − Q1(progress)` | Stability penalty. High IQR means the environment produces inconsistent results. |
| `SDProgress` | 0..1 | Standard deviation of `normalized_task_progress` | Stability penalty. Captures overall spread, including tail risk that IQR ignores. |
| `GoalErrorScore` | 0..1 | `Clamp01(MeanGoalError / MaxAcceptableGoalError)` | Penalty for ending far from the goal, even on partial successes. |
| `InvalidPenalty` | 0..1 | `1` if physics explode or constraints violated, else `0` | Hard penalty for physically impossible configurations (negative mass, intersecting colliders, etc.). |

> **Note on `MeanProgress`:** The old formula included `MeanProgress` as a
> secondary signal. In practice it correlates (`r > 0.98`) with `IQMProgress`,
> so it has been removed to simplify the formula without losing signal.

#### Genome Validation Rules (`IsGenomeValid`)

Implement an explicit validator so the evaluator can reject genomes before
running expensive episodes:

```csharp
public static bool IsGenomeValid(PushTBlockGenome genome)
{
    if (genome == null) return false;

    // Physical positivity
    if (genome.width      <= 0f) return false;
    if (genome.height     <= 0f) return false;
    if (genome.depth      <= 0f) return false;
    if (genome.mass       <= 0f) return false;
    if (genome.blockDrag  <  0f) return false;
    if (genome.friction   <  0f) return false;
    if (genome.bounciness <  0f) return false;

    // Stability constraint: extremely tall, thin blocks can tip over
    // before the agent even acts, producing invalid physics noise.
    if (genome.height / Mathf.Max(genome.depth, 0.0001f) > 12f)
        return false;

    return true;
}
```

**Where to call it:**

1. **Before `CreateRandom`** — if random sampling produces an invalid genome,
   reject and resample.
2. **After `Mutate`** — if mutation pushes a value out of bounds, clamp or
   reject. Prefer clamping to avoid discarding elite genomes.
3. **Inside `ApplyGenome`** — if physics state becomes invalid after applying
   (e.g., inertia tensor NaN), set `InvalidPenalty = 1` and abort the episode
   bundle early.

### Difficulty Score

Because the search space has been simplified to block-only evolution (Phase 5
simplified), the difficulty score uses only the block physics parameters:

```
Difficulty =
    0.30 * MassDifficulty
  + 0.20 * WidthDifficulty
  + 0.20 * FrictionDifficulty
  + 0.15 * HeightDifficulty
  + 0.15 * DepthDifficulty
```

Each sub-difficulty is `Clamp01((value - min) / (max - min))`. For friction,
invert the scale because lower friction (slippery) is harder:

```
FrictionDifficulty = 1 - Clamp01((friction - minFriction) / (maxFriction - minFriction))
```

> **Note:** `DistanceDifficulty` and `ObstacleDifficulty` were removed because
> goal position and obstacle evolution were removed from Phase 5 (see
> **Phase 5: Expand The Search Space** below). If you later re-add layout
> evolution, restore those terms.

### Time Score

`TimeScore` replaces the old `TooEasyPenalty`. Instead of a hard binary
threshold, it uses a **graduated target zone**:

```csharp
float ComputeTimeScore(float meanTimeToGoal, PushTEvolutionConfig config)
{
    if (meanTimeToGoal <= 0f)
        return -2.0f;  // never succeeded

    if (meanTimeToGoal < config.tooEasyTimeMax)   // < 3s
        return -1.0f;  // way too easy

    if (meanTimeToGoal < config.easyTimeThreshold) // 3s .. 5s
        return -0.3f;  // slightly too easy

    if (meanTimeToGoal <= config.idealTimeMax)    // 5s .. 12s
        return +1.0f;  // ideal target zone

    if (meanTimeToGoal <= config.hardTimeMax)     // 12s .. 20s
        return +0.3f;  // still acceptable, but getting hard

    return -0.5f;  // > 20s or timeout — too hard
}
```

**Why this mapping works:**

| Zone | Time | Score | Rationale |
|---|---|---|---|
| Never succeeded | — | `−2.0` | Should not happen in Tier 2 (SR ≥ 0.9), but guards against edge cases. |
| Way too easy | `< 3s` | `−1.0` | Agent wins instantly; no learning signal. |
| Slightly too easy | `3s – 5s` | `−0.3` | Mild discouragement below ideal zone. |
| **Ideal zone** | `5s – 12s` | `+1.0` | Maximum reward. Agent must work but succeeds reliably. |
| Getting hard | `12s – 20s` | `+0.3` | Partial reward. Configurations at the edge of agent capability. |
| Too hard / timeout | `> 20s` | `−0.5` | Agent barely makes it or times out; not useful for training. |

Suggested thresholds:

| Parameter | Value | Note |
|---|---|---|
| `tooEasyTimeMax` | `3.0f` | Below this, environment is trivial regardless of other factors. |
| `easyTimeThreshold` | `5.0f` | Lower bound of the ideal zone. |
| `idealTimeMax` | `12.0f` | Upper bound of the ideal zone. |
| `hardTimeMax` | `20.0f` | Above this, time pressure dominates and episodes become unstable. |

### GoalErrorScore

```csharp
float GoalErrorScore = Mathf.Clamp01(meanGoalError / maxAcceptableGoalError);
```

Suggested `maxAcceptableGoalError`: `2.0f`.

### C# Pseudocode

```csharp
float ComputeEnvironmentFitness(PushTGenomeEvaluation eval, PushTEvolutionConfig config)
{
    float sr = eval.successRate;

    // Tier 1: Gatekeeper
    if (sr < config.successRateThreshold)
    {
        eval.fitness = 80f * sr;
        return eval.fitness;
    }

    // Tier 2: Quality (unlocked only when SR >= threshold)
    float iqm          = eval.iqmProgress;
    float sd           = eval.sdProgress;
    float iqr          = eval.iqrProgress;
    float difficulty   = ComputeDifficulty(eval.genome, config);
    float timeScore    = ComputeTimeScore(eval.meanTimeToGoal, config);
    float invalid      = eval.invalidPenalty;
    float goalError    = Mathf.Clamp01(eval.meanGoalError / config.maxAcceptableGoalError);

    float stabilityPenalty = 15f * sd + 10f * iqr;

    eval.fitness =
        100f
      + 40f * iqm
      + 25f * difficulty
      + 30f * timeScore
      - stabilityPenalty
      - 20f * goalError
      - 50f * invalid;

    return eval.fitness;
}
```

### RewardScore Is Optional

`episode_reward` is **not included** in the recommended formulas above. For a
fixed agent, reward is typically highly correlated (`r > 0.9`) with
`SuccessRate` and `Progress`. Adding it rarely improves the search signal and
may introduce noise from reward-shaping artifacts.

If you later verify that `correlation(Reward, Progress) < 0.7`, add:

```
+ 10..20 * RewardScore
```

where `RewardScore = Clamp01((MeanReward - RewardMin) / (RewardMax - RewardMin))`.

## Episode Completion Protocol

The evaluator must reliably detect when an episode has finished before reading
results. The recommended implementation is a **Coroutine-based polling loop**.

### Recommended Pattern (Coroutine)

```csharp
private IEnumerator EvaluateEpisodeCoroutine(PushTBlockGenome genome, int seed)
{
    // 1. Trigger reset + apply genome
    agent.ResetEpisodeForEvolution(genome, seed);

    float elapsed = 0f;
    float timeout = episodeTimeoutSeconds;  // e.g., 15f

    // 2. Poll every frame until episode completes or times out
    while (!agent.IsEpisodeComplete && elapsed < timeout)
    {
        elapsed += Time.deltaTime;
        yield return null;
    }

    // 3. Read results immediately
    var result = new PushTEpisodeResult
    {
        success        = agent.WasLastEpisodeSuccessful,
        reward         = agent.LastEpisodeReward,
        progress       = agent.LastEpisodeNormalizedTaskProgress,
        finalGoalError = agent.LastEpisodeFinalGoalError,
        timeToGoal     = agent.LastEpisodeTimeToGoal,
        timedOut       = (elapsed >= timeout && !agent.IsEpisodeComplete),
        invalidPhysics = false  // set by evaluator if physics checks fail
    };

    // 4. Return result to genome aggregator
    evaluation.episodes.Add(result);
}
```

### Why Coroutine Instead Of async/await?

- Unity's frame lifecycle maps naturally to `yield return null`.
- No threading complexity.
- Easy to debug with `Debug.Log` per frame if needed.
- Works identically in Phase 0 through Phase 3 without architecture changes.

### Timeout Handling

If `elapsed >= timeout` and `IsEpisodeComplete` is still `false`:

- Mark the episode as `timedOut = true`.
- Treat `success = false`.
- Use the last known progress value (do not discard partial data).
- Log a warning: `[PushT EC] Episode timed out after {timeout}s`.

This prevents the evaluator from hanging forever if the agent gets stuck or the
episode-end signal is lost.

## Data Contract And Result Classes

The pipeline must store more than a single fitness float. It needs **traceable
process data** so you can audit whether a high-fitness genome is genuinely good
or just lucky.

### 1. Episode Result (`PushTEpisodeResult`)

One row per episode:

```csharp
public struct PushTEpisodeResult
{
    public int    episodeIndex;        // 0..episodesPerGenome-1
    public int    episodeSeed;         // random seed used for this reset
    public bool   success;             // did the agent reach the goal?
    public float  reward;              // cumulative episode reward
    public float  progress;            // normalized task progress [0..1]
    public float  finalGoalError;      // final distance to goal
    public float  timeToGoal;          // seconds to success, or -1 if failed
    public bool   timedOut;            // did we hit episodeTimeoutSeconds?
    public bool   invalidPhysics;      // did colliders overlap / mass <= 0?
    public string notes;               // free-form diagnostic string
}
```

### 2. Genome Evaluation (`PushTGenomeEvaluation`)

One object per genome, aggregating all its episodes:

```csharp
public class PushTGenomeEvaluation
{
    public PushTBlockGenome           genome;           // the evaluated params
    public List<PushTEpisodeResult>   episodes;         // raw episode list

    // Aggregates (computed after all episodes finish)
    public float  successRate;       // successes / total
    public float  meanProgress;      // arithmetic mean of progress
    public float  iqmProgress;       // interquartile mean of progress
    public float  sdProgress;        // standard deviation of progress
    public float  iqrProgress;       // interquartile range of progress
    public float  meanGoalError;     // arithmetic mean of finalGoalError
    public float  meanTimeToGoal;    // mean of timeToGoal (failed = -1)
    public float  fitness;           // final scalar from the fitness formula

    // Diagnostics
    public int    generationIndex;
    public int    genomeIndex;
}
```

### 3. Evolution Run Result (`PushTEvolutionRunResult`)

One object for the entire EC run:

```csharp
public class PushTEvolutionRunResult
{
    public PushTBlockGenome              bestGenome;
    public float                         bestFitness;
    public List<PushTGenomeEvaluation>   allResults;       // every genome tested
    public List<PushTGenomeEvaluation>   bestPerGeneration; // top genome each gen
    public int                           totalEpisodesRun;  // for sanity checks
}
```

### Why Keep Raw Episode Data?

Because after the run you will ask:

- "Is this genome actually stable, or did it get lucky on 2 out of 10 runs?"
- "Why did fitness drop in generation 12?"
- "Can I plot the progress distribution for the winner?"

If you only keep `fitness`, these questions are unanswerable.

## Metric Strategy For Environment EC

### Metrics That Belong In Environment EC Fitness

These metrics require **no temporal learning axis**. They describe the
**distribution of outcomes** when a fixed agent is evaluated multiple times in
one environment configuration:

- **Success Rate** — fraction of episodes that end in goal completion.
- **Final Performance Window (IQM, Mean)** — central tendency of progress across
  the evaluation bundle.
- **Learning Stability (SD, IQR, CV)** — variability of outcomes across the
  evaluation bundle. In environment EC, this is **episode stability**, not
  training stability.
- **Goal Error** — final distance to target.
- **Difficulty Score** — intrinsic hardness of the environment parameters.

### Metrics That Do NOT Belong In Environment EC Fitness

These metrics describe **how an agent learns over time**. They require a
training curve (performance vs. training step). A fixed agent evaluated in a
static environment has no such curve.

| Metric | Why It Is Excluded |
|---|---|
| **AUC of performance curve** | AUC integrates performance over training steps. EC has no training steps; it has independent episode replications. |
| **Learning Slope** | Slope estimates improvement over time. A fixed agent does not improve during an EC evaluation bundle. |
| **Policy Loss / Value Loss** | These are trainer-internal diagnostics. EC has no access to them and they do not describe environment quality. |
| **Learning Rate** | This is a training hyperparameter, not an environment parameter. |

### Mapping FRD Metrics To EC Fitness

If you are cross-referencing the `PushBlock_Learning_Improvement_V2_FRD`, use
this mapping:

| Learning Improvement FRD Metric | Environment EC Equivalent |
|---|---|
| AUC of performance curve | **Not applicable.** Replace with `SuccessRate` and `IQMProgress`. |
| Final performance window | `IQMProgress` computed over the `episodesPerGenome` bundle. (`MeanProgress` was removed from the fitness formula because it correlates `r > 0.98` with IQM.) |
| Learning stability (SD, IQR, CV) | `SDProgress`, `IQRProgress`, and `CVProgress` computed over the episode bundle. |
| Learning slope | **Not applicable.** |

## Episodes Per Genome

Environment parameters must be evaluated over **multiple independent episodes**
because PushBlock randomizes spawn positions every `OnEpisodeBegin`. A single
episode is a noisy point estimate.

| Scenario | `episodesPerGenome` | Rationale |
|---|---|---|
| Smoke test / sanity check | 5 | Minimum viable. Accept high variance. Use the **minimal fitness formula** (no IQR). |
| Standard environment EC | **10** | Recommended default. IQM and IQR are reliable. Good balance of accuracy and speed. |
| High-variance environments | 20 | If you evolve obstacle placement, goal position, or other high-impact layout changes. |
| Publication / benchmark | 20+ | When you need defensible, reproducible results with tight confidence intervals. |

**Recommendation:** Set the default to `10`. PushBlock's spawn randomization
alone justifies this over `5`.

## Simplest User Flow

The first usable workflow should be:

1. clone the existing PushT scene
2. add `PushTOneScriptEc` to the same GameObject as `PushAgentBasic`
3. press Play
4. watch the `PushT EC Validator` overlay
5. inspect `bestCandidate` on the component

This is the preferred first path because it avoids creating a separate runner,
separate evaluator, or custom public API before you know the idea works.

## How You Know It Is Working

The first version needs visible validation, not just console logs.

When you press Play, `PushTOneScriptEc` should show a small on-screen validator
panel with:

- current status
- current validation check
- current generation
- current candidate
- last fitness
- last success
- best candidate so far

The validator is working if you can see it move through these states:

1. resolving scene references
2. resetting episode
3. candidate applied
4. candidate scored
5. finished with best fitness

If it gets stuck, the panel should make the stuck point obvious, such as:

- missing agent
- missing block
- missing block Rigidbody
- waiting for episode result

## How This Maps To Your Sample Push Agent

Your sample file shows a `PushAgentBasic : Agent` that already gives you most
of the signals needed for evolution.

Useful signals already present:

- success path:
  `ScoredAGoal()`
- reward:
  `m_episodeCumulativeReward`
- end reason:
  `m_episodeEndReason`
- final error:
  `m_finalBlockGoalDistance`
- progress:
  `m_normalizedTaskProgress`
- parameter application:
  `SetBlockProperties()`
- friction application:
  `SetGroundMaterialFriction()`
- grouped reset-time parameter application:
  `SetResetParameters()`

This is important because it means your first integration should fit around the
existing agent flow rather than replacing it.

## Delivery Plan

### Phase 0: One Manual Block Change

This is the real first step.

In Phase 0, you are only proving the one-script flow:

`Can I clone the scene, add one script to the agent, press Play, and see EC logs?`

By the end of Phase 0, you should be able to run the tiny smoke test without
changing `PushAgentBasic`.

#### Phase 0 Scope

Use `PushTOneScriptEc` with tiny defaults.

Only search obvious block/environment parameters:

- block scale
- block mass
- block drag
- ground dynamic friction
- ground static friction

Do not include:

- multiple seeds
- shape switching
- result logging

#### Phase 0 User Requirements

As the implementer, you want the first action to feel almost trivial:

1. open the PushT scene
2. duplicate or clone it
3. add `PushTOneScriptEc` to the agent GameObject
4. press Play
5. see the `PushT EC Validator` overlay update

#### Phase 0 Functional Requirements

##### P0-FR-1 One Script On The Agent

You need the first integration to require only one new component:

- `PushTOneScriptEc`

It should be attached to the same GameObject as `PushAgentBasic`.

##### P0-FR-2 Automatic Reference Discovery

You need the script to find the existing `block` and `ground` references from
the agent.

If automatic discovery fails, you should be able to assign:

- `blockOverride`
- `groundOverride`

##### P0-FR-3 Tiny EC Defaults

You need defaults small enough that pressing Play is not scary:

- population size: `4`
- elites: `1`
- generations: `2`
- episode timeout: `15` seconds

##### P0-FR-4 Visible Confirmation

You need immediate confirmation that the first step worked.

Good enough confirmations:

- the console prints `[PushT EC]`
- the validator panel appears
- the validator panel advances through candidates
- the block parameters change during Play mode
- `bestCandidate` is filled in after the smoke test

### Phase 1: One Genome, One Episode

In this phase, your goal is to run exactly one candidate through exactly one
PushT episode and get a score back.

By the end of Phase 1, you should be able to:

1. define one candidate genome
2. apply that genome to the PushT scene
3. run one episode
4. read fitness signals from the finished episode
5. print or inspect one score

#### Phase 1 Scope

Still do not evolve a population.

Only connect:

- one candidate
- one reset
- one episode
- one score

#### Phase 1 User Requirements

As the implementer, you want to answer:

`Can one parameter set run through my PushT agent and produce a score?`

#### Phase 1 Functional Requirements

##### P1-FR-1 Candidate Definition

You need one candidate to mean one concrete PushT setup.

In Phase 1, one candidate should contain:

- block scale
- block mass
- block drag
- ground dynamic friction
- ground static friction

`PushTBlockGenome` is the place to keep this data, even if you trim out the
extra unused fields after copying it into your PushT project.

##### P1-FR-2 Scene Application

You need to take one candidate and apply it into the scene before the episode
starts.

For your sample agent, the cleanest version is to route these values through the
same logic used by:

- `SetBlockProperties()`
- `SetGroundMaterialFriction()`
- `SetResetParameters()`

If that is awkward through `EnvironmentParameters`, add a small adapter in your
PushT project that applies the genome directly to:

- the block rigidbody
- the block transform
- the ground physic material

##### P1-FR-3 Episode Execution

You need to run one candidate through one episode.

For the candidate:

1. apply the parameters
2. reset the environment
3. let the PushT episode run
4. wait until the episode finishes
5. read the result

##### P1-FR-4 Fitness Readout

You need to score the candidate using values already produced by your sample
agent.

For Phase 1, your evaluator should read:

- success
- episode reward
- normalized task progress
- final goal error

Recommended first formula (single-episode prototype):

`fitness = 100 * success + episode_reward + 25 * normalized_task_progress - final_goal_error`

Mapping to your sample agent:

- `success` comes from the goal-completion path
- `episode_reward` maps to `m_episodeCumulativeReward`
- `normalized_task_progress` maps to `m_normalizedTaskProgress`
- `final_goal_error` maps to `m_finalBlockGoalDistance`

##### P1-FR-5 One Result

At the end of Phase 1, you only need one printed or inspectable result:

- genome values
- success
- reward
- progress
- final error
- fitness

### Phase 2: Tiny Evolution Smoke Test

In this phase, your goal is to run the smallest useful evolution loop.

By the end of Phase 2, you should be able to:

1. create a tiny population
2. evaluate each candidate once
3. sort by score
4. mutate the best candidates
5. run a few generations

#### Phase 2 Scope

Use tiny numbers first:

- population size: `4`
- elites: `1`
- generations: `2`
- episodes per genome: `1`

This phase is for proving the machinery works, not finding a great solution.

#### Phase 2 Functional Requirements

##### P2-FR-1 Candidate Sampling

You need to generate an initial population of candidate parameter sets from
bounded ranges you control.

You will use `PushTEvolutionConfig` to define:

- min/max scale
- min/max mass
- min/max drag
- min/max dynamic friction
- min/max static friction
- population size
- generation count
- mutation rate
- mutation strength

##### P2-FR-2 Evolution Loop

You need the runner to:

1. create an initial population
2. evaluate all candidates
3. sort by fitness
4. keep the elites
5. refill the population by mutation
6. repeat for multiple generations

##### P2-FR-3 Result Review

At the end of a run, you want to inspect:

- best candidate overall
- best candidate per generation
- average fitness
- success rate

### Phase 3: Smallest Useful EC Loop

In this phase, your goal is not to evolve everything. Your goal is to prove the
loop is useful enough to run for real.

By the end of Phase 3, you should be able to:

1. evaluate each genome over multiple episodes
2. reduce noise by using multiple spawn seeds
3. track variance and stability
4. compute the full multi-objective fitness formula
5. inspect the best candidate overall

#### Phase 3 Scope

Only evolve parameters your sample agent already supports cleanly.

**First-version genome (7 fields, block-only):**

The genome controls block physics directly. Width, height, and depth are
independent (non-uniform scale) so the EC can explore tall narrow blocks,
wide flat blocks, and everything in between:

```csharp
public class PushTBlockGenome
{
    public float width;        // X scale
    public float height;       // Y scale
    public float depth;        // Z scale
    public float mass;         // Rigidbody.mass
    public float friction;     // controls BOTH static and dynamic friction
    public float blockDrag;    // maps to Rigidbody.linearDamping
    public float bounciness;   // PhysicMaterial.bounciness
}
```

Applying the genome to the scene:

```csharp
block.transform.localScale = new Vector3(
    genome.width, genome.height, genome.depth);

blockRb.mass = genome.mass;
blockRb.linearDamping = genome.blockDrag;

runtimeGroundMaterial.staticFriction = genome.friction;
runtimeGroundMaterial.dynamicFriction = genome.friction;
runtimeGroundMaterial.bounciness = genome.bounciness;
```

**Why split scale into width/height/depth?**

- Uniform `blockScale` cannot represent tall thin blocks or wide flat slabs.
- Non-uniform scale adds only 2 dimensions (height, depth) but dramatically
  expands the viable search space.
- The agent's RayPerceptionSensor observes the block from multiple angles, so
  shape variation is a legitimate difficulty axis.

**Why keep friction merged?**

- It avoids the physical constraint `staticFriction >= dynamicFriction`.
- Your scene's Low/Medium/High presets already use identical values for both
  (`0.2`, `0.6`, `1.0`), so a single friction field is faithful to the
  existing design.

**Phase 3 formal search ranges:**

These ranges were calibrated during implementation to cover useful variation
without producing physically broken configurations:

| Parameter | Min | Max | Default | Rationale |
|---|---|---|---|---|
| `width` | `0.5f` | `3.0f` | `1.0f` | Wide blocks are harder to maneuver around; very narrow blocks may be invisible to sensors. |
| `height` | `0.5f` | `5.0f` | `1.0f` | Tall blocks raise center of mass and may tip; very short blocks reduce contact area. |
| `depth` | `0.25f` | `2.0f` | `1.0f` | Thin depth makes the block unstable; thick depth increases inertia. |
| `mass` | `0.5f` | `6.0f` | `1.0f` | Light blocks are easy to push; heavy blocks test agent persistence. |
| `blockDrag` | `0.05f` | `1.5f` | `0.5f` | Low drag makes the block slide far; high drag stops it quickly. |
| `friction` | `0.1f` | `1.0f` | `0.5f` | Low friction is slippery and harder; high friction gives agent more purchase. |
| `bounciness` | `0.0f` | `0.3f` | `0.0f` | Zero is predictable; moderate bounce adds chaos without making the task impossible. |

**Phase-specific ranges:**

Use narrower windows for early phases to avoid wasting evaluations on extreme
configurations:

| Phase | width | height | depth | mass | drag | friction | bounciness |
|---|---|---|---|---|---|---|---|
| Phase 0–1 (manual / one-shot) | `0.8` – `1.5` | `0.8` – `1.5` | `0.5` – `1.2` | `1.0` – `2.5` | `0.2` – `0.8` | `0.3` – `0.8` | `0.0` – `0.1` |
| Phase 2 (smoke test) | `0.6` – `2.0` | `0.6` – `2.5` | `0.3` – `1.5` | `0.8` – `4.0` | `0.1` – `1.0` | `0.2` – `0.9` | `0.0` – `0.2` |
| Phase 3 (formal EC) | `0.5` – `3.0` | `0.5` – `5.0` | `0.25` – `2.0` | `0.5` – `6.0` | `0.05` – `1.5` | `0.1` – `1.0` | `0.0` – `0.3` |

Do not evolve these (removed from scope):

- ~~shape switching~~ — removed; block is always a BoxCollider
- ~~block color~~ — removed; does not affect learning signal
- ~~collider replacement~~ — removed; fixed BoxCollider
- ~~mesh replacement~~ — removed; fixed cube mesh
- ~~obstacle geometry~~ — removed from Phase 5 simplified
- ~~goal position~~ — removed from Phase 5 simplified
- separate static vs dynamic friction — kept merged for simplicity

#### Phase 3 Configuration

Use these defaults for the first real experiment:

**Search hyperparameters:**

- population size: `24`
- elites: `6`
- generations: `20`
- episodes per genome: **`10`** (default; use `5` only for fast smoke tests)
- mutation rate: `0.35`
- mutation strength: `0.15`

**Fitness thresholds (calibrated during implementation):**

| Parameter | Value | Meaning |
|---|---|---|
| `successRateThreshold` | `0.9` | Minimum success rate to unlock Tier 2 fitness. |
| `easyTimeThreshold` | `5.0f` | Lower bound of ideal time zone. |
| `tooEasyTimeMax` | `3.0f` | Below this, environment is trivially easy. |
| `idealTimeMin` | `5.0f` | Lower bound of ideal time zone (same as `easyTimeThreshold`). |
| `idealTimeMax` | `12.0f` | Upper bound of ideal time zone. |
| `hardTimeMax` | `20.0f` | Above this, episodes are too hard / unstable. |
| `minUsefulDifficulty` | `0.25f` | Minimum difficulty score considered useful. |
| `maxAcceptableGoalError` | `2.0f` | Largest acceptable final distance to goal. |

#### Phase 3 User Requirements

As the implementer, you want to:

- copy this folder into your PushT project
- hook one evaluator into your PushT scene
- point that evaluator at your agent and block
- run evolution from one GameObject
- inspect the best parameter set after the run

#### Phase 3 Functional Requirements

##### P3-FR-1 Repeated Episode Evaluation

You need to run one candidate across multiple episodes with independent spawn
seeds.

For each candidate:

1. apply the parameters
2. reset the environment (new random seed)
3. let the PushT episode run
4. wait until the episode finishes
5. read the results
6. repeat for `episodesPerGenome`
7. aggregate statistics (SuccessRate, IQM, Mean, SD, IQR)
8. compute the full fitness score

##### P3-FR-2 Statistics Aggregation

The evaluator must compute per-genome:

- `SuccessRate`
- `IQMProgress` (interquartile mean of normalized task progress)
- `SDProgress` (standard deviation)
- `IQRProgress` (interquartile range)
- `MeanGoalError`
- `MeanTimeToGoal`

`MeanProgress` is no longer used in the fitness formula (it correlates `r > 0.98`
with `IQMProgress`). If `episodesPerGenome < 10`, skip `IQRProgress` because the
sample size is too small for reliable quartile estimation.

##### P3-FR-3 Result Review

At the end of a run, you want to inspect:

- best candidate overall
- best candidate per generation
- average fitness
- success rate
- variance across episodes
- stability metrics (SD, IQR)

### Phase 4: Better Scene Integration

In this phase, you make the evaluator feel native to your PushT project instead
of just "good enough."

By the end of Phase 4, you should have:

- a concrete PushT-specific evaluator class
- cleaner reset hooks
- explicit episode-complete flags
- public accessors for reward/success/progress/final error

#### Phase 4 User Requirements

As the implementer, you want to stop relying on vague template hooks and use
explicit APIs from your PushT scene.

#### Phase 4 Functional Requirements

##### P4-FR-1 Public Episode State

You must expose a small, read-only public API on `PushAgentBasic`. The
evaluator should **never** read private fields (e.g., `m_episodeCumulativeReward`,
`m_finalBlockGoalDistance`) via reflection.

**Required properties:**

```csharp
public bool  IsEpisodeComplete               { get; private set; }
public bool  WasLastEpisodeSuccessful        { get; private set; }
public float LastEpisodeReward               { get; private set; }
public float LastEpisodeNormalizedTaskProgress { get; private set; }
public float LastEpisodeFinalGoalError       { get; private set; }
public float LastEpisodeTimeToGoal           { get; private set; }
```

**Where to set them:**

In `CaptureEpisodeEndMetrics(bool success, string endReason)`, at the point
where the episode is definitively ending, set:

```csharp
IsEpisodeComplete        = true;
WasLastEpisodeSuccessful = success;
LastEpisodeReward        = m_episodeCumulativeReward;
LastEpisodeNormalizedTaskProgress = m_normalizedTaskProgress;
LastEpisodeFinalGoalError = m_finalBlockGoalDistance;
LastEpisodeTimeToGoal    = success ? m_episodeSteps * Time.fixedDeltaTime : -1f;
```

In `OnEpisodeBegin()`, clear them:

```csharp
IsEpisodeComplete        = false;
WasLastEpisodeSuccessful = false;
LastEpisodeReward        = 0f;
LastEpisodeNormalizedTaskProgress = 0f;
LastEpisodeFinalGoalError = -1f;
LastEpisodeTimeToGoal    = -1f;
```

**Rationale:**

- Reflection is fragile. If you rename `m_episodeCumulativeReward` later, the
evaluator breaks silently.
- Public properties are discoverable in IntelliSense and documented in code.
- They make unit testing the evaluator possible without a running agent scene.

##### P4-FR-2 Public Reset API

You must expose one clear reset entry point on `PushAgentBasic`:

```csharp
public void ResetEpisodeForEvolution(PushTBlockGenome genome, int seed)
{
    // 1. Apply genome parameters to the scene
    ApplyGenome(genome);

    // 2. Set random seed for reproducible spawn
    Random.InitState(seed);

    // 3. Reset positions (uses the seeded Random)
    ResetBlock();
    ResetAgent();

    // 4. Clear physics state
    m_BlockRb.linearVelocity    = Vector3.zero;
    m_BlockRb.angularVelocity   = Vector3.zero;
    m_AgentRb.linearVelocity    = Vector3.zero;
    m_AgentRb.angularVelocity   = Vector3.zero;

    // 5. Reset episode tracking
    m_episodeSteps             = 0;
    m_episodeCumulativeReward  = 0f;
    m_episodeMetricsRecorded   = false;
    m_episodeEndReason         = "unknown";

    // 6. Clear public result surface (see P4-FR-1)
    IsEpisodeComplete               = false;
    WasLastEpisodeSuccessful        = false;
    LastEpisodeReward               = 0f;
    LastEpisodeNormalizedTaskProgress = 0f;
    LastEpisodeFinalGoalError       = -1f;
    LastEpisodeTimeToGoal           = -1f;

    // 7. Start the episode
    OnEpisodeBegin();
}
```

**Why this exact sequence matters:**

| Step | Failure mode if skipped |
|---|---|
| Apply genome | Parameters never change. EC evaluates the same environment every time. |
| Set seed | Results are non-reproducible. You cannot debug why genome X failed. |
| Reset positions | Block/agent start in stale locations from previous episode. |
| Clear velocities | Block carries momentum from previous episode into the new one. |
| Clear tracking | `m_episodeSteps` counts across episode boundaries. Reward leaks. |
| Clear public API | Evaluator reads stale `success = true` from previous episode. |
| Start episode | `OnEpisodeBegin()` must fire so `m_episodeId` increments and spawn logic runs. |

**Note:** Do **not** call `EndEpisode()` inside `ResetEpisodeForEvolution`. That
marks the *previous* episode done, which is fine if you call it *before* this
method, but calling it inside creates a race condition where the evaluator might
read completion flags prematurely.

##### P4-FR-3 Stable Repeated Evaluation

You should be able to evaluate the same candidate over multiple seeds and get a
stable aggregate score.

### Phase 5: Expand The Search Space (Simplified)

> **Note:** In the current implementation, Phase 5 was **simplified** to block-only
> evolution. Color, shape, goal position, and obstacle evolution were explicitly
> removed because they did not improve the learning signal for the fixed agent.
> The genome expanded from 4 fields (`blockScale`, `mass`, `drag`, `friction`) to
> 7 fields (`width`, `height`, `depth`, `mass`, `friction`, `drag`, `bounciness`).

What Phase 5 **actually** added:

- **Independent width/height/depth** — replaced uniform `blockScale`
- **Bounciness** — new physics axis via `PhysicMaterial.bounciness`

What was **removed from scope**:

- ~~block color~~ — no impact on RayPerceptionSensor observations
- ~~shape type~~ — fixed BoxCollider; mesh/collider replacement adds complexity
  without clear benefit for this agent
- ~~obstacle placement / size~~ — removed to keep search focused
- ~~goal position~~ — removed to keep search focused
- ~~separate static vs dynamic friction~~ — kept merged; single field is sufficient

#### Phase 5 User Requirements

As the implementer, you want the EC to explore block shape and material bounce
once the basic loop (Phase 3) is already working.

#### Phase 5 Functional Requirements

##### P5-FR-1 Non-Uniform Scale

Replace `blockScale` with independent `width`, `height`, and `depth`:

```csharp
block.transform.localScale = new Vector3(
    genome.width, genome.height, genome.depth);
```

Update `IsGenomeValid` to enforce a maximum `height / depth` ratio (e.g., `12`)
to prevent extremely tall, thin blocks that tip over before the agent acts.

##### P5-FR-2 Bounciness

Add `bounciness` to the genome:

```csharp
runtimeGroundMaterial.bounciness = genome.bounciness;
```

Keep the range conservative (`0.0f` – `0.3f`). High bounciness can make the
block unpredictable or cause it to bounce out of the goal zone.

##### P5-FR-3 Difficulty Score Update

Update the difficulty formula to include `WidthDifficulty`, `HeightDifficulty`,
and `DepthDifficulty` instead of a single `SizeDifficulty`:

```
Difficulty =
    0.30 * MassDifficulty
  + 0.20 * WidthDifficulty
  + 0.20 * FrictionDifficulty
  + 0.15 * HeightDifficulty
  + 0.15 * DepthDifficulty
```

`DistanceDifficulty` and `ObstacleDifficulty` remain removed unless you later
re-add layout evolution.

### Phase 6: Better Search And Analysis

Once the workflow is useful, you can improve quality rather than just
functionality.

Potential additions:

- crossover
- tournament selection
- novelty/diversity pressure
- JSON/CSV export of per-generation statistics
- batch experiment configs
- stop/resume support
- post-run correlation analysis (e.g., validate that `RewardScore` is not
  redundant with `SuccessRate` + `Progress`)

## Integration Steps In Your PushT Project

### Step 0: Do The Simplest Possible Proof

Start with the one-script path.

In the PushT scene:

1. clone the scene
2. add `PushTOneScriptEc` to the same GameObject as `PushAgentBasic`
3. press Play
4. watch the `PushT EC Validator` overlay
5. inspect `bestCandidate`

You are done with this step when the component completes a tiny run.

### Step 1: Copy The Folder

Copy `PushTEvolutionMvp/` into your PushT project after Phase 0 feels clear.

Suggested location:

`Assets/Scripts/Evolution/`

### Step 2: Create A Concrete Evaluator

Start from `PushTSceneEvaluatorTemplate.cs`.

In your PushT project, rename it to something explicit, for example:

- `PushTBasicEvolutionEvaluator`

Implement the full fitness formula in `EvaluateEpisodeAsync` or in an
aggregation step after all episodes for one genome are complete.

### Step 3: Connect It To PushAgentBasic

Your evaluator should reference:

- the block transform
- the block rigidbody
- the block renderer if needed
- the ground physic material if needed
- the GameObject or component containing `PushAgentBasic`

### Step 4: Add Or Expose The Missing Hooks

Your current sample agent is rich, but the evaluator still needs a cleaner way
to know:

- when an episode ended
- whether it succeeded
- what reward it got
- what progress it made
- what final error it ended with

The best fix is to expose these as public properties in your PushT project.

### Step 5: Add One Evolution Runner GameObject

Create a GameObject like `EvolutionRunner`.

Attach:

- `PushTEvolutionRunner`
- your concrete evaluator

Assign the evaluator into the runner.

Set `episodesPerGenome` to `10` for the first real run.

### Step 6: Trim The Genome For The First Real Loop

Do not carry around unnecessary fields just because the template has them.

For the first real loop, keep only what you are truly evolving.

Recommended initial genome (Phase 3, 7 fields):

- `width` — X scale
- `height` — Y scale
- `depth` — Z scale
- `mass` — Rigidbody mass
- `friction` — merged static + dynamic friction
- `blockDrag` — Rigidbody linear damping
- `bounciness` — PhysicMaterial bounciness

After Phase 3 is stable, Phase 5 simplified adds no new fields — it only
replaces uniform `blockScale` with independent width/height/depth and adds
`bounciness`.

If you later split friction into static/dynamic, enforce the physical
constraint in `Mutate` or `ApplyGenome`:

```csharp
if (genome.staticFriction < genome.dynamicFriction)
    genome.staticFriction = genome.dynamicFriction;
```

### Step 7: Run One Genome, One Episode

Before a population exists, run one candidate through one episode.

You are done with this step when you can inspect or print:

- success
- reward
- progress
- final error
- fitness

### Step 8: Run A Tiny Evolution Smoke Test

Use the tiniest useful evolution configuration:

- population size: `4`
- elites: `1`
- generations: `2`
- episodes per genome: `1`

This is just to verify the loop.

### Step 9: Run A Real First Experiment

After the smoke test works, try:

- population size: `24`
- elites: `6`
- generations: `20`
- episodes per genome: **`10`**
- mutation rate: `0.35`
- mutation strength: `0.15`

Use the **standard fitness formula** with IQM, SD, and IQR enabled.

## Concrete Guidance For Your Sample Agent

### Best First Parameters To Evolve

Start with:

- `width` / `height` / `depth` — independent scale axes (replaces uniform `blockScale`)
- `mass` — block mass
- `friction` — merged static + dynamic friction
- `blockDrag` — linear damping
- `bounciness` — material bounciness

Why these first:

- they all map directly to existing Unity physics properties
- they do not require scene restructuring
- the merged friction field matches your scene's Low/Medium/High preset design
- non-uniform scale dramatically expands the viable search space with only 2
  extra dimensions

Why **not** separate static/dynamic friction yet:

- adds a hidden constraint (`static >= dynamic`) that EC must respect
- your scene presets already keep them equal
- one less dimension makes Phase 0–3 debugging faster

### Best First Fitness Inputs

Read and combine:

- success
- normalized task progress
- final goal zone error
- episode reward (optional, see **RewardScore Is Optional**)

### Best First Code Change In Your PushT App

The most useful thing you can add in your PushT project is a tiny public result
surface on top of `PushAgentBasic`, for example:

- `public bool IsEpisodeComplete`
- `public bool WasLastEpisodeSuccessful`
- `public float LastEpisodeReward`
- `public float LastEpisodeNormalizedTaskProgress`
- `public float LastEpisodeFinalGoalError`

That one change will make the evaluator much cleaner.

## Post-Run Analysis Checklist

After your first real EC run completes, validate the fitness function before
trusting its results:

1. **Correlation check**
   - Compute `correlation(SuccessRate, IQMProgress)`.
   - If it is below `0.5`, your environment produces "high success but low
     progress" or vice versa. This is unusual; inspect the raw episodes.

2. **Redundancy check** (historical — `MeanProgress` was already removed)
   - Compute `correlation(IQMProgress, MeanProgress)`.
   - If it is above `0.98`, `MeanProgress` is redundant with IQM. This is why the
     current fitness formula no longer includes `MeanProgress`.

3. **Reward redundancy check**
   - Compute `correlation(SuccessRate + IQMProgress, MeanReward)`.
   - If above `0.90`, `RewardScore` is redundant. Do not add it to the formula.

4. **Stability check**
   - Plot `SDProgress` vs `SuccessRate` for the final generation.
   - High-success genomes should cluster in the low-SD region. If they do not,
     your `episodesPerGenome` may be too low or the environment is inherently
     unstable.

5. **Time-zone audit**
   - List the top-5 genomes by fitness.
   - Check their `meanTimeToGoal`. Ideally they should fall in the **ideal zone**
     (`5s` – `12s`).
   - If top genomes cluster at `< 3s` (way too easy) or `> 20s` (too hard),
     adjust `TimeScore` weights or tighten the zone boundaries.
   - If any have `difficulty < 0.2` and `successRate > 0.95`, the difficulty
     score may need recalibration for the current search ranges.

## Acceptance Criteria By Phase

### Phase 0 Done Means

You can:

1. press or call one method
2. apply one hand-written parameter set
3. see the block or inspector values change
4. do all of this without running a full episode

### Phase 1 Done Means

You can:

1. apply one candidate to the PushT environment
2. run one episode
3. read success, reward, progress, and final error
4. compute one fitness score

### Phase 2 Done Means

You can:

1. create a tiny population
2. evaluate each candidate once
3. sort candidates by fitness
4. mutate the best candidate
5. run two generations without manual intervention

### Phase 3 Done Means

You can:

1. evaluate one candidate over multiple episodes (default: 10)
2. compute the full multi-objective fitness (3-tier formula with SuccessRate
   gatekeeper, IQM, SD, IQR, Difficulty, TimeScore, GoalError, InvalidPenalty)
3. track average fitness, success rate, and stability
4. identify the best candidate overall with defensible statistics
5. auto-export results to CSV (`summary`, `all_evaluations`, `episodes`)

### Phase 4 Done Means

You can:

1. read episode status through explicit public APIs
2. reset the episode through one clean entry point
3. run repeated evaluations reliably without manual scene intervention

### Phase 5 Done Means

You can:

1. evolve independent width/height/depth and bounciness
2. compute a meaningful `Difficulty` score for the expanded block-only genome
3. validate genomes with the height/depth ratio constraint

### Phase 6 Done Means

You can:

1. export experiment data cleanly (per-generation CSV/JSON)
2. run more serious sweeps
3. compare multiple evolution settings or fitness definitions
4. run the **post-run analysis checklist** and act on its findings
