# FRD: PushT Curriculum EC — MVP

## 1. Overview & Goal

**The problem:**

Training an agent on a fixed default PushBlock environment causes it to overfit to a single block configuration. The agent learns to push *one specific block* well, but fails to generalize to different sizes, masses, or friction levels.

**The solution (Curriculum Learning + Evolutionary Computation):**

1. Train the agent on the default environment for a short warm-up period.
2. **Freeze** the current agent into an evaluator (inference-only, no learning).
3. Use EC to evolve many block genomes, testing each with the frozen evaluator.
4. Keep only **"Goldilocks" genomes** — environments that are neither too easy nor too hard for the current agent.
5. Place these genomes into a **Genome Pool**.
6. The agent resumes training, sampling a random genome from the pool at the start of each episode.
7. Repeat steps 2–6 at fixed intervals, generating new environment pools matched to the agent's improving skill level.

The agent progressively encounters configurations that are **slightly harder than its current capability**, creating a natural curriculum.

---

## 2. Core Concepts

| Term | Definition |
|---|---|
| **Genome** | A parameter vector describing a block configuration (width, height, depth, mass, friction, bounciness, drag). |
| **Frozen Evaluator** | An inference-only agent loaded with a fixed `.onnx` checkpoint. It evaluates genomes without learning. |
| **Genome Pool** | The set of genomes currently available for training. Sampled randomly each episode. |
| **Goldilocks Zone** | The difficulty sweet spot: the agent's success rate on a genome is neither too high nor too low. |
| **EC Run** | One evolutionary computation cycle: generate candidates → evaluate with frozen agent → filter → output pool. |
| **Curriculum Stage** | One training stage = one EC run followed by N training steps on the resulting pool. |
| **Training Arena** | The environment where the training agent learns. |
| **Evaluation Arena** | A separate, isolated environment where the frozen evaluator runs EC. |

---

## 3. MVP Architecture

The MVP is **semi-automatic**: you manually export/load the `.onnx` model and trigger EC via a context menu. No Python-side modifications are required.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Unity Scene                               │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   TrainingArena     │    │   EvaluationArena   │            │
│  │                     │    │                     │            │
│  │  TrainingAgent      │    │  FrozenEvaluator    │            │
│  │  (Learning)         │    │  (Inference Only)   │            │
│  │      │              │    │      │              │            │
│  │      ▼              │    │      ▼              │            │
│  │  TrainingBlock      │    │  EvalBlock          │            │
│  │  TrainingGround     │    │  EvalGround         │            │
│  │      ▲              │    │      ▲              │            │
│  │      │              │    │      │              │            │
│  │  PushTCurriculum    │    │  PushTBasicEvolution│            │
│  │  Environment        │    │  Evaluator          │            │
│  │  (reads Pool)       │    │  (drives Frozen)    │            │
│  └──────────┬──────────┘    └──────────┬──────────┘            │
│             │                          │                       │
│             │        ┌─────────────────┘                       │
│             │        │                                         │
│             │        ▼                                         │
│             │   PushTCurriculumEC                               │
│             │   (evolution loop + Goldilocks filter)            │
│             │        │                                         │
│             │   writes selected                                │
│             │        ▼                                         │
│             │   GenomePool (ScriptableObject + JSON)            │
│             │        │                                         │
│             └────────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Why Semi-Automatic?

Fully automatic curriculum generation would require modifying the ML-Agents Python trainer or building complex cross-process communication. The MVP validates the core hypothesis first:

> *"Do Goldilocks genomes selected by a frozen evaluator actually help the agent generalize?"*

Once validated, automation can be layered on top.

---

## 4. Component Design

### 4.1 Arena Separation (Critical)

**Training and evaluation must not share the same block, ground, or agent.**

If the frozen evaluator runs EC while the training agent is active in the same arena, they will fight over the same environment objects:

- EC repeatedly resets block parameters and episode states.
- The training agent sees inconsistent physics mid-episode.
- Episode statistics become meaningless.

**Required structure:**

```
Unity Scene
│
├── TrainingArena
│   ├── TrainingAgent
│   ├── TrainingBlock
│   ├── TrainingGround
│   └── PushTCurriculumEnvironment
│
├── EvaluationArena
│   ├── FrozenEvaluatorAgent
│   ├── EvalBlock
│   ├── EvalGround
│   └── PushTBasicEvolutionEvaluator
│
└── PushTCurriculumEC
    └── writes to GenomePool
```

Each arena has its own block, ground, and goal. The two agents never interact.

### 4.2 GenomePool — Training Pool

**Dual storage:**

- **ScriptableObject** for runtime reference and Inspector visibility.
- **JSON file** for persistent, version-controlled, cross-session storage.

```csharp
[CreateAssetMenu(fileName = "GenomePool", menuName = "PushT/GenomePool")]
public class GenomePool : ScriptableObject
{
    public List<PushTBlockGenome> genomes = new List<PushTBlockGenome>();
    public int generationId = 0;
    public float targetSuccessRateMin = 0.30f;
    public float targetSuccessRateMax = 0.75f;
    public int evalEpisodesPerGenome = 10;

    public PushTBlockGenome SampleRandom(System.Random random)
    {
        if (genomes.Count == 0) return null;
        return genomes[random.Next(genomes.Count)].Clone();
    }

    public bool IsEmpty => genomes.Count == 0;

    // Persist to JSON
    public void SaveToJson(string path);
    public void LoadFromJson(string path);
}
```

**Why both?**

- ScriptableObject allows drag-and-drop assignment, runtime inspection, and easy swapping in the Editor.
- JSON survives play-mode exits, scene changes, and version control.
- Training should read from JSON (or a copy synced from JSON) to avoid play-mode asset mutation issues.

**Output files per EC run:**

```
genome_pool_stage_01.json          — pool for training
genome_pool_stage_01_summary.csv   — one row per selected genome
genome_pool_stage_01_all_evals.csv — one row per evaluated genome
genome_pool_stage_01_episodes.csv  — one row per episode
```

### 4.3 PushTCurriculumEnvironment — Training Environment Sampler

A standalone component in the **TrainingArena** that applies a random genome from the pool before each episode begins.

**Do not hook into `Academy.Instance.OnEnvironmentReset` as the primary mechanism.** That event is global and fires for all arenas simultaneously. Instead, inject the genome application into the per-agent episode lifecycle.

**Recommended approach:**

Option A (preferred): Add a lightweight event to `PushAgentBasic`:

```csharp
public event Action OnBeforeEpisodeReset;

public override void OnEpisodeBegin()
{
    OnBeforeEpisodeReset?.Invoke();  // apply genome first
    // ... existing reset logic ...
}
```

Option B (no code change to agent): Use `Script Execution Order` to ensure `PushTCurriculumEnvironment` runs before `PushAgentBasic`, and call apply from a component that knows when the agent's episode begins.

```csharp
public class PushTCurriculumEnvironment : MonoBehaviour
{
    public GenomePool genomePool;
    public PushTBlockGenome fallbackGenome;

    [Header("Scene References (Training Arena Only)")]
    public Transform blockTransform;
    public Rigidbody blockRigidbody;
    public Collider groundCollider;      // reference to apply runtime material

    private PhysicMaterial m_runtimeGroundMaterial;
    private System.Random m_random;

    void Awake()
    {
        m_random = new System.Random();

        // Clone ground material so we don't mutate the shared project asset
        if (groundCollider != null && groundCollider.material != null)
        {
            m_runtimeGroundMaterial = new PhysicMaterial(groundCollider.material.name + "_Runtime");
            m_runtimeGroundMaterial.dynamicFriction = groundCollider.material.dynamicFriction;
            m_runtimeGroundMaterial.staticFriction = groundCollider.material.staticFriction;
            m_runtimeGroundMaterial.bounciness = groundCollider.material.bounciness;
            groundCollider.material = m_runtimeGroundMaterial;
        }
    }

    /// <summary>
    /// Called before PushAgentBasic.OnEpisodeBegin() resets positions.
    /// </summary>
    public void ApplyRandomGenomeFromPool()
    {
        var genome = (genomePool != null && !genomePool.IsEmpty)
            ? genomePool.SampleRandom(m_random)
            : fallbackGenome;

        ApplyGenome(genome);
    }

    void ApplyGenome(PushTBlockGenome genome)
    {
        if (blockTransform != null)
            blockTransform.localScale = new Vector3(genome.width, genome.height, genome.depth);

        if (blockRigidbody != null)
        {
            blockRigidbody.mass = genome.mass;
            blockRigidbody.linearDamping = genome.blockDrag;
        }

        if (m_runtimeGroundMaterial != null)
        {
            m_runtimeGroundMaterial.dynamicFriction = genome.friction;
            m_runtimeGroundMaterial.staticFriction = genome.friction;
            m_runtimeGroundMaterial.bounciness = genome.bounciness;
        }
    }
}
```

**Critical verification:**

- `ApplyGenome()` must run **before** `ResetBlock()` in `PushAgentBasic.OnEpisodeBegin()`.
- Each arena must apply its genome independently.
- The runtime material must be a **clone**, not the shared project asset.

### 4.4 PushTFrozenEvaluatorAgent — Frozen Evaluator

A duplicate of the training agent GameObject in the **EvaluationArena** with these differences:

- `BehaviorParameters.BehaviorType = InferenceOnly`
- `BehaviorParameters.Model` assigned to a `.onnx` checkpoint
- All logging and screenshot capture optionally disabled to speed up EC
- Does **not** communicate with the Python trainer

This agent is driven exclusively by `PushTBasicEvolutionEvaluator` during EC runs.

### 4.5 PushTCurriculumEC — Curriculum EC Runner

Extends `PushTEvolutionRunner` with two additions:

1. Uses the **EvaluationArena** evaluator (not the training agent).
2. Filters evaluated genomes through the **Goldilocks + Diversity** criteria before writing to the pool.

```csharp
public class PushTCurriculumEC : PushTEvolutionRunner
{
    [Header("Curriculum")]
    public GenomePool outputPool;
    public float goldilocksMinSuccessRate = 0.30f;
    public float goldilocksMaxSuccessRate = 0.75f;
    public float minMeanProgress = 0.40f;
    public float maxSdProgress = 0.35f;
    public float maxInvalidPenalty = 0.10f;
    public int maxPoolSize = 50;
    public float minGenomeDistance = 0.15f;  // diversity threshold

    [Header("Evaluator (EvaluationArena)")]
    public PushTBlockEvaluator frozenEvaluator;

    [Header("Export")]
    public bool exportJson = true;
    public bool exportCsv = true;

    [ContextMenu("Run Curriculum EC")]
    public async void RunCurriculumEC()
    {
        if (frozenEvaluator == null)
        {
            Debug.LogError("[Curriculum EC] No frozen evaluator assigned.");
            return;
        }

        // 1. Run full EC
        var runResult = await RunEvolutionAsync(frozenEvaluator);

        // 2. Select Goldilocks + Diverse genomes
        var selected = SelectGoldilocksGenomes(runResult.allEvaluations);

        // 3. Write to pool (never empty the pool if selection fails)
        WriteToPool(selected, runResult);

        // 4. Export
        if (exportJson || exportCsv)
            ExportResults(runResult, selected);

        Debug.Log($"[Curriculum EC] Selected {selected.Count} genomes for training pool.");
    }

    List<PushTGenomeEvaluation> SelectGoldilocksGenomes(List<PushTGenomeEvaluation> allEvals)
    {
        // Step 1: Hard filters
        var candidates = allEvals
            .Where(e => e.episodes.Count >= 5)
            .Where(e => e.invalidPenalty <= maxInvalidPenalty)
            .Where(e => e.successRate >= goldilocksMinSuccessRate)
            .Where(e => e.successRate <= goldilocksMaxSuccessRate)
            .Where(e => e.meanProgress >= minMeanProgress)
            .Where(e => e.sdProgress <= maxSdProgress)
            .OrderByDescending(e => e.fitness)
            .ToList();

        // Step 2: Diversity filter
        var selected = new List<PushTGenomeEvaluation>();
        foreach (var candidate in candidates)
        {
            if (selected.Count >= maxPoolSize) break;

            float minDist = float.MaxValue;
            foreach (var existing in selected)
            {
                float d = NormalizedGenomeDistance(candidate.genome, existing.genome);
                if (d < minDist) minDist = d;
            }

            if (selected.Count == 0 || minDist >= minGenomeDistance)
            {
                selected.Add(candidate);
            }
        }

        return selected;
    }

    float NormalizedGenomeDistance(PushTBlockGenome a, PushTBlockGenome b)
    {
        // Normalize each dimension to [0,1] using config ranges, then L2
        var c = config;
        float dw = Normalize(a.width,  c.widthRange.min,  c.widthRange.max)  - Normalize(b.width,  c.widthRange.min,  c.widthRange.max);
        float dh = Normalize(a.height, c.heightRange.min, c.heightRange.max) - Normalize(b.height, c.heightRange.min, c.heightRange.max);
        float dd = Normalize(a.depth,  c.depthRange.min,  c.depthRange.max)  - Normalize(b.depth,  c.depthRange.min,  c.depthRange.max);
        float dm = Normalize(a.mass,   c.massRange.min,   c.massRange.max)   - Normalize(b.mass,   c.massRange.min,   c.massRange.max);
        float df = Normalize(a.friction,c.frictionRange.min,c.frictionRange.max) - Normalize(b.friction,c.frictionRange.min,c.frictionRange.max);
        float ddg= Normalize(a.blockDrag,c.dragRange.min, c.dragRange.max)   - Normalize(b.blockDrag,c.dragRange.min, c.dragRange.max);
        float db = Normalize(a.bounciness,c.bouncinessRange.min,c.bouncinessRange.max) - Normalize(b.bounciness,c.bouncinessRange.min,c.bouncinessRange.max);

        return Mathf.Sqrt(dw*dw + dh*dh + dd*dd + dm*dm + df*df + ddg*ddg + db*db) / Mathf.Sqrt(7f);
    }

    float Normalize(float value, float min, float max)
    {
        return (max <= min) ? 0.5f : Mathf.Clamp01((value - min) / (max - min));
    }

    void WriteToPool(List<PushTGenomeEvaluation> selected, PushTEvolutionRunResult run)
    {
        if (outputPool == null) return;

        // NEVER empty the pool if EC produced no valid genomes
        if (selected.Count == 0)
        {
            Debug.LogWarning("[Curriculum EC] No Goldilocks genomes selected. Keeping previous pool.");
            return;
        }

        outputPool.genomes.Clear();
        foreach (var eval in selected)
        {
            outputPool.genomes.Add(eval.genome.Clone());
        }

        outputPool.generationId++;
        outputPool.targetSuccessRateMin = goldilocksMinSuccessRate;
        outputPool.targetSuccessRateMax = goldilocksMaxSuccessRate;
        outputPool.evalEpisodesPerGenome = config.episodesPerGenome;
    }

    void ExportResults(PushTEvolutionRunResult run, List<PushTGenomeEvaluation> selected)
    {
        var outputDir = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "Scripts", "newPushTEC", "curriculum_outputs");
        Directory.CreateDirectory(outputDir);
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var stage = $"stage_{outputPool.generationId:D3}";

        if (exportJson && outputPool != null)
            outputPool.SaveToJson(Path.Combine(outputDir, $"genome_pool_{stage}_{timestamp}.json"));

        if (exportCsv)
        {
            // summary, all_evaluations, episodes — same format as PushTEvolutionRunner
            // plus a "selected" marker column
        }
    }
}
```

---

## 5. Goldilocks Zone Design

### 5.1 Why Success Rate Alone Is Insufficient

Two genomes can have identical success rates but wildly different training utility:

| Genome | Success Rate | Mean Progress | SD Progress | Verdict |
|---|---|---|---|---|
| A | 50% | 0.85 | 0.10 | **Good.** Consistent near-success. |
| B | 50% | 0.30 | 0.45 | **Bad.** Unstable physics or erratic behavior. |

Genome B would confuse the agent because the outcome is not deterministic.

### 5.2 MVP Goldilocks Filter

```csharp
bool IsGoldilocks(PushTGenomeEvaluation eval)
{
    // Data quality
    if (eval.episodes.Count < 5) return false;

    // Physics stability
    if (eval.invalidPenalty > 0.10f) return false;

    // Difficulty sweet spot
    if (eval.successRate < 0.30f) return false;
    if (eval.successRate > 0.75f) return false;

    // Agent must make meaningful progress on failures
    if (eval.meanProgress < 0.40f) return false;

    // Environment must be consistent across episodes
    if (eval.sdProgress > 0.35f) return false;

    return true;
}
```

### 5.3 Diversity Filter

EC populations often converge to a narrow region of the search space. A pool of 50 nearly identical genomes provides less training diversity than 15 distinct ones.

**Greedy diversity selection:**

1. Sort candidates by fitness (descending).
2. Add the best candidate to the pool.
3. For each subsequent candidate, compute normalized distance to all already-selected genomes.
4. If minimum distance ≥ `minGenomeDistance` (default 0.15), add it.
5. Stop when `maxPoolSize` is reached.

Distance is computed over all 7 normalized genome dimensions.

### 5.4 Why Use a Frozen Evaluator?

A training agent's policy changes every step. If you evaluate a genome with a learning agent, your difficulty measurement is a moving target. A frozen evaluator provides a **stable, reproducible difficulty baseline** against which genomes can be reliably compared.

---

## 6. Step-by-Step MVP Workflow

### Stage 0: Scene Setup

1. Open your PushBlock scene.
2. **Duplicate** the agent GameObject. Rename the copy to `FrozenEvaluator` and place it in a new parent `EvaluationArena`.
3. On `FrozenEvaluator`:
   - Set `BehaviorParameters.BehaviorType = InferenceOnly`
   - Leave `BehaviorParameters.Model` empty for now
   - Optionally disable screenshot capture and verbose logging
4. Create a second copy of block + ground under `EvaluationArena`. Ensure the frozen evaluator references these copies, not the training arena objects.
5. Create `GenomePool` Asset: `Assets → Create → PushT → GenomePool`.
6. Create `PushTCurriculumEnvironment` in `TrainingArena`:
   - Assign the `GenomePool` asset
   - Assign TrainingArena's block, rigidbody, and ground collider
   - Set `fallbackGenome` to default parameters
7. Create `PushTCurriculumEC` in the scene root:
   - Assign `frozenEvaluator` → `PushTBasicEvolutionEvaluator` on `EvaluationArena`
   - Assign `outputPool` → the `GenomePool` asset
   - Tune EC down for fast MVP iteration: `populationSize=12`, `eliteCount=3`, `generationCount=5`, `episodesPerGenome=5`

### Stage 1: Warm-Up Training

```bash
mlagents-learn config/ppo/PushBlock.yaml --run-id=pushblock_curriculum_base
```

Train until the agent reliably solves the default environment (e.g., 500k–1M steps).

### Stage 2: Freeze & Evaluate (First EC)

1. Export or locate the trained `.onnx` model in `results/pushblock_curriculum_base/`.
2. Drag the `.onnx` into `FrozenEvaluator.BehaviorParameters.Model`.
3. Right-click `PushTCurriculumEC` → **Run Curriculum EC**.
4. Wait for EC to finish (minutes, depending on parameters).
5. Inspect the `GenomePool` asset. It should now contain 5–20 genomes.
6. Check the output folder for JSON and CSV exports.

### Stage 3: Curriculum Training

```bash
mlagents-learn config/ppo/PushBlock.yaml --run-id=pushblock_curriculum_s1
```

At the start of each episode, `PushTCurriculumEnvironment` samples a genome from the pool and applies it to the training block. The agent now trains on a distribution of configurations rather than a single default.

### Stage 4: Iterate

1. After N training steps (e.g., 1M), pause training.
2. Update `FrozenEvaluator` with the latest checkpoint `.onnx`.
3. Run **Run Curriculum EC** again.
4. The new pool overwrites the old (or merges, depending on your strategy).
5. Resume training.

---

## 7. Data Flow

```
Python Trainer
   │
   │ trains
   ▼
TrainingAgent (TrainingArena)
   │
   │ exports .onnx checkpoint
   ▼
FrozenEvaluator (EvaluationArena)
   │
   │ PushTBasicEvolutionEvaluator evaluates genomes
   ▼
PushTCurriculumEC
   │
   │ filters Goldilocks + Diversity
   ▼
GenomePool (ScriptableObject + JSON)
   │
   │ sampled each episode
   ▼
PushTCurriculumEnvironment (TrainingArena)
   │
   │ applies genome to block/ground
   ▼
TrainingAgent learns from varied environments
   │
   └──────────────────────────────────────► (loop)
```

---

## 8. Integration with Existing Code

| Existing Code | Role in New System | Modified? |
|---|---|---|
| `PushTBlockGenome` | Genotype definition. Reused as-is. | ❌ No |
| `PushTEvolutionConfig` | Search hyperparameters. Reused as-is. | ❌ No |
| `PushTEvolutionResult` | Evaluation data structures. Reused as-is. | ❌ No |
| `PushTBlockEvaluator` | Abstract evaluator base. Reused as-is. | ❌ No |
| `PushTEvolutionRunner` | Evolution loop base. Extended by `PushTCurriculumEC`. | 🟡 Inherited |
| `PushTBasicEvolutionEvaluator` | Concrete evaluator. Reused, but pointed at frozen agent in EvaluationArena. | 🟡 Reference change |
| `PushAgentBasic` | Training agent + frozen evaluator (dual role, separate instances). | ❌ No |
| `PushTOneScriptEc` | Not used in curriculum mode. | ❌ No |

**New files required (~3 files, < 400 lines):**

1. `GenomePool.cs` — ScriptableObject with JSON serialization
2. `PushTCurriculumEnvironment.cs` — Per-episode genome sampler for TrainingArena
3. `PushTCurriculumEC.cs` — Curriculum runner extending `PushTEvolutionRunner`

---

## 9. Fallback & Edge Case Handling

| Case | Behavior |
|---|---|
| **EC produces zero Goldilocks genomes** | Pool is **not cleared**. Previous pool is retained. Log warning. |
| **Pool has fewer than 5 genomes** | Mix pool genomes with `fallbackGenome` at a configurable ratio (e.g., 50% pool, 50% fallback). |
| **Pool is empty on first run** | Use `fallbackGenome` exclusively. Log info. |
| **JSON load fails** | Fall back to ScriptableObject runtime data. If both empty, use fallback genome. |
| **PhysicsMaterial is a shared asset** | Always clone at runtime. Never mutate the project asset. |
| **Frozen evaluator `.onnx` mismatches** | Log error, abort EC. Require behavior name match between model and agent. |

---

## 10. Experimental Validation Design

To verify that EC curriculum is actually beneficial (not just "random environment variation"), run three control groups:

| Group | Condition | Purpose |
|---|---|---|
| **A — Fixed Baseline** | Train only on default PushBlock parameters. | Baseline: what happens without any variation. |
| **B — Random DR** | Each episode samples block parameters uniformly from the full search range. | Control: is *any* randomization enough? |
| **C — EC Curriculum** | Use the full pipeline: frozen evaluator → Goldilocks filter → diversity pool → training. | Test: is *selected* difficulty progression better than random? |

**Key metrics to compare:**

- Learning speed (steps to threshold success rate)
- Final success rate on default environment
- Generalization test: success rate on a held-out set of 20 extreme genomes
- Reward stability (std dev of episode reward over last 100 episodes)
- Time-to-goal distribution

The critical claim being tested:

> *"It is not environment variation itself that helps, but the systematic selection of genomes at the edge of the agent's current capability."*

---

## 11. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Frozen evaluator model mismatch | Medium | EC evaluation invalid | Verify behavior name before EC. Log clear error if mismatch detected. |
| Goldilocks filter too strict, empty pool | Medium | Training reverts to default | Widen thresholds dynamically if selection count < 5. Never clear old pool. |
| Pool genomes have high variance, training destabilizes | Low | Agent performance drops | Add `sdProgress` and `invalidPenalty` filters. Monitor reward variance. |
| EC evaluation too slow for iteration | High | Developer friction | MVP uses small EC params (pop=12, gen=5, eps=5). Parallel evaluation can be added later. |
| Frozen evaluator too strong, all genomes >75% SR | Medium | No curriculum generated | Expand search ranges (mass, width). Lower SR ceiling to 0.60. Increase episode timeout. |
| Frozen evaluator too weak, all genomes <30% SR | Medium | No initial curriculum | Train longer before first EC. Widen SR floor to 0.10. Seed pool with hand-written easy genomes. |
| Shared PhysicsMaterial mutated | Low | All scenes affected | Always clone material at runtime. Assert original asset is unchanged. |

---

## 12. Verification Checklist (Post-MVP)

After completing your first curriculum loop, verify:

- [ ] `GenomePool` contains 5–20 genomes and is not empty.
- [ ] Every pooled genome has `successRate` between 30% and 75%.
- [ ] `sdProgress` for each pooled genome is ≤ 0.35.
- [ ] Training episodes show visibly different block sizes/masses across resets.
- [ ] Training reward does not crash after switching to pool sampling.
- [ ] Second EC run produces a pool with different parameter distribution than the first (evidence of curriculum progression).
- [ ] Frozen evaluator console shows no `Behavior Name mismatch` warnings.
- [ ] EvaluationArena and TrainingArena block/ground objects are independent (modifying one does not affect the other).
- [ ] Output JSON and CSV files are created and non-empty.

---

## 13. Future Extensions

### Phase 2: Automated Checkpoint Swap

- Python script or Unity Editor script auto-copies latest `.onnx` to a fixed path.
- Unity `FileSystemWatcher` detects new model and reloads it into `FrozenEvaluator.BehaviorParameters.Model`.

### Phase 3: Weighted Sampling

Instead of uniform random sampling from the pool, weight genomes by their distance from the agent's current estimated skill boundary:

```
weight ∝ exp(-|successRate - 0.50| / temperature)
```

Genomes near 50% success rate are sampled more often.

### Phase 4: Dynamic Goldilocks Bounds

Adjust SR bounds based on the training agent's recent performance:

- If agent's 100-episode average SR > 70% → raise floor to 0.40.
- If agent's 100-episode average SR < 40% → lower floor to 0.15.

### Phase 5: Expanded Genome

Evolve beyond block physics:

- Initial agent-to-block distance
- Goal position
- Obstacle presence/layout
- Arena size

This requires extending `PushTBlockGenome` and updating the scene application logic.

---

**Document Version:** MVP v1.0  
**Based On:** `newPushTEC` EC framework + `PushAgentBasic` existing interfaces  
**Next Step:** Implement `GenomePool.cs`, `PushTCurriculumEnvironment.cs`, `PushTCurriculumEC.cs`.
