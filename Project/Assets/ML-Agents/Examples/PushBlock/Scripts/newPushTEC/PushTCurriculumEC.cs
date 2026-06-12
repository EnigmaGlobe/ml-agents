using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Curriculum EC runner.
    /// Extends the base evolution loop with Goldilocks filtering and diversity selection,
    /// writing the resulting genomes into a GenomePool for training.
    /// </summary>
    public class PushTCurriculumEC : PushTEvolutionRunner
    {
        [Header("Curriculum")]
        public GenomePool outputPool;

        [Range(0f, 1f)] public float goldilocksMinSuccessRate = 0.30f;
        [Range(0f, 1f)] public float goldilocksMaxSuccessRate = 0.75f;
        [Range(0f, 1f)] public float minMeanProgress = 0.40f;
        [Range(0f, 1f)] public float maxSdProgress = 0.35f;
        [Range(0f, 1f)] public float maxInvalidPenalty = 0.10f;
        public int maxPoolSize = 50;
        [Range(0f, 1f)] public float minGenomeDistance = 0.15f;

        [Header("Evaluator (EvaluationArena)")]
        public PushTBasicEvolutionEvaluator frozenEvaluator;

        [Header("Export")]
        public bool exportJson = true;
        public bool exportCsv = true;

        /// <summary>
        /// True while Curriculum EC is actively running.
        /// </summary>
        public bool IsRunning { get; private set; }

        /// <summary>
        /// Result of the most recent Curriculum EC run.
        /// </summary>
        public PushTCurriculumECResult LastResult { get; private set; }

        private string CurriculumOutputDir
        {
            get
            {
                return Path.Combine(
                    Application.dataPath,
                    "ML-Agents",
                    "Examples",
                    "PushBlock",
                    "Scripts",
                    "newPushTEC",
                    "curriculum_outputs");
            }
        }

        /// <summary>
        /// Context-menu entry point for manual runs. Internally awaits RunCurriculumECAsync().
        /// </summary>
        [ContextMenu("Run Curriculum EC")]
        public async void RunCurriculumEC()
        {
            await RunCurriculumECAsync();
        }

        /// <summary>
        /// Runs the full curriculum EC loop asynchronously and returns a result object.
        /// Sets IsRunning while active and populates LastResult on completion.
        /// </summary>
        public async Task<PushTCurriculumECResult> RunCurriculumECAsync()
        {
            var result = new PushTCurriculumECResult();

            if (IsRunning)
            {
                result.success = false;
                result.errorMessage = "Curriculum EC is already running.";
                Debug.LogError("[Curriculum EC] " + result.errorMessage);
                LastResult = result;
                return result;
            }

            IsRunning = true;

            try
            {
                if (frozenEvaluator == null)
                {
                    result.success = false;
                    result.errorMessage = "No frozen evaluator assigned. Assign the evaluator from the EvaluationArena.";
                    Debug.LogError("[Curriculum EC] " + result.errorMessage);
                    return result;
                }

                if (outputPool == null)
                {
                    result.success = false;
                    result.errorMessage = "No output GenomePool assigned.";
                    Debug.LogError("[Curriculum EC] " + result.errorMessage);
                    return result;
                }

                // Validate frozen evaluator configuration BEFORE running EC
                if (!ValidateFrozenEvaluator())
                {
                    result.success = false;
                    result.errorMessage = "ABORTED: Frozen evaluator is not properly configured. Fix the errors above before running EC.";
                    Debug.LogError("[Curriculum EC] " + result.errorMessage);
                    return result;
                }

                // 1. Run full EC using the frozen evaluator
                this.status = "Running EC with frozen evaluator...";
                var runResult = await RunEvolutionAsync(frozenEvaluator);
                this.lastRun = runResult;

                Debug.Log(
                    $"[Curriculum EC] Run complete. Total evaluations={runResult.allEvaluations?.Count ?? 0}, " +
                    $"Generations={runResult.bestPerGeneration?.Count ?? 0}, " +
                    $"TotalEpisodes={runResult.totalEpisodesRun}");

                // 2. Select Goldilocks + diverse genomes
                var selected = SelectGoldilocksGenomes(runResult.allEvaluations);

                // 3. Write to pool (never empty the pool if selection fails)
                WriteToPool(selected, runResult);

                result.selectedCount = selected.Count;
                result.generationId = outputPool.generationId;

                // 4. Export JSON + CSV
                if (exportJson || exportCsv)
                {
                    ExportResults(runResult, selected, result);
                }

                if (selected.Count == 0)
                {
                    result.success = true;
                    result.errorMessage = "No genomes passed Goldilocks filter; previous pool retained.";
                    Debug.LogWarning("[Curriculum EC] " + result.errorMessage);
                }
                else
                {
                    result.success = true;
                }

                this.status = $"Finished. Selected {selected.Count} genomes for pool (gen {outputPool.generationId}).";
                Debug.Log($"[Curriculum EC] {this.status}");
            }
            catch (Exception ex)
            {
                result.success = false;
                result.errorMessage = $"Exception during Curriculum EC: {ex.Message}";
                Debug.LogError("[Curriculum EC] " + result.errorMessage);
            }
            finally
            {
                IsRunning = false;
                LastResult = result;
            }

            return result;
        }

        /// <summary>
        /// Validates that the frozen evaluator is properly configured for EC.
        /// Returns true if everything is OK, false if there are errors.
        /// </summary>
        bool ValidateFrozenEvaluator()
        {
            bool allGood = true;

            // Check the agent reference
            if (frozenEvaluator.agent == null)
            {
                Debug.LogError("[EC-Validate] frozenEvaluator.agent is null! Assign the FrozenEvaluator agent.");
                allGood = false;
            }
            else
            {
                var agent = frozenEvaluator.agent;
                Debug.Log($"[EC-Validate] Checking agent: '{agent.name}'");

                // Check BehaviorParameters
                var bp = agent.GetComponent<BehaviorParameters>();
                if (bp == null)
                {
                    Debug.LogError("[EC-Validate] No BehaviorParameters on frozen agent!");
                    allGood = false;
                }
                else
                {
                    // Check BehaviorType
                    if (bp.BehaviorType != BehaviorType.InferenceOnly)
                    {
                        Debug.LogError(
                            $"[EC-Validate] FAIL: BehaviorType = '{bp.BehaviorType}' (must be 'InferenceOnly'). " +
                            $"EC will hang waiting for Python trainer decisions!");
                        Debug.Log("[EC-Validate] FIX: Select FrozenEvaluator agent → BehaviorParameters → Behavior Type = Inference Only");
                        allGood = false;
                    }
                    else
                    {
                        Debug.Log("[EC-Validate] PASS: BehaviorType = InferenceOnly ✓");
                    }

                    // Check Model
                    if (bp.Model == null)
                    {
                        Debug.LogError("[EC-Validate] FAIL: No .onnx model assigned to BehaviorParameters.Model!");
                        Debug.Log("[EC-Validate] FIX: Train a model first, then drag the .onnx file to Model field.");
                        allGood = false;
                    }
                    else
                    {
                        Debug.Log($"[EC-Validate] PASS: Model = '{bp.Model.name}' ✓");
                    }
                }

                // Check DecisionRequester — CRITICAL: without this, the agent never makes decisions
                var dr = agent.GetComponent<Unity.MLAgents.DecisionRequester>();
                if (dr == null)
                {
                    Debug.LogError("[EC-Validate] FAIL: No DecisionRequester on frozen agent!");
                    Debug.Log("[EC-Validate] FIX: Select FrozenEvaluator → Add Component → Decision Requester");
                    allGood = false;
                }
                else
                {
                    Debug.Log($"[EC-Validate] PASS: DecisionRequester (period={dr.DecisionPeriod}) ✓");
                }

                // Check agent has block reference
                if (agent.block == null)
                {
                    Debug.LogError("[EC-Validate] FAIL: frozenAgent.block is null!");
                    allGood = false;
                }
                else
                {
                    Debug.Log($"[EC-Validate] PASS: block = '{agent.block.name}' ✓");
                }
            }

            // Check timeout setting
            if (frozenEvaluator.episodeTimeoutSeconds < 5f)
            {
                Debug.LogWarning($"[EC-Validate] WARN: episodeTimeoutSeconds = {frozenEvaluator.episodeTimeoutSeconds}s may be too short.");
            }
            else
            {
                Debug.Log($"[EC-Validate] PASS: episodeTimeoutSeconds = {frozenEvaluator.episodeTimeoutSeconds}s ✓");
            }

            return allGood;
        }

        /// <summary>
        /// Two-step selection: hard Goldilocks filters, then greedy diversity filtering.
        /// </summary>
        List<PushTGenomeEvaluation> SelectGoldilocksGenomes(List<PushTGenomeEvaluation> allEvals)
        {
            if (allEvals == null || allEvals.Count == 0)
            {
                Debug.LogWarning("[Curriculum EC] allEvaluations is null or empty. Nothing to select.");
                return new List<PushTGenomeEvaluation>();
            }

            int minEpisodeCount = Mathf.Min(5, config?.episodesPerGenome ?? 5);

            int stage0 = allEvals.Count;
            int stage1 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount);
            int stage2 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount && e.invalidPenalty <= maxInvalidPenalty);
            int stage3 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount && e.invalidPenalty <= maxInvalidPenalty && e.successRate >= goldilocksMinSuccessRate);
            int stage4 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount && e.invalidPenalty <= maxInvalidPenalty && e.successRate >= goldilocksMinSuccessRate && e.successRate <= goldilocksMaxSuccessRate);
            int stage5 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount && e.invalidPenalty <= maxInvalidPenalty && e.successRate >= goldilocksMinSuccessRate && e.successRate <= goldilocksMaxSuccessRate && e.meanProgress >= minMeanProgress);
            int stage6 = allEvals.Count(e => e.episodes != null && e.episodes.Count >= minEpisodeCount && e.invalidPenalty <= maxInvalidPenalty && e.successRate >= goldilocksMinSuccessRate && e.successRate <= goldilocksMaxSuccessRate && e.meanProgress >= minMeanProgress && e.sdProgress <= maxSdProgress);

            Debug.Log(
                $"[Curriculum EC] Goldilocks filter stages: total={stage0}, hasEpisodes>={minEpisodeCount}={stage1}, " +
                $"invalid<={maxInvalidPenalty}={stage2}, SR>={goldilocksMinSuccessRate}={stage3}, " +
                $"SR<={goldilocksMaxSuccessRate}={stage4}, progress>={minMeanProgress}={stage5}, sd<={maxSdProgress}={stage6}");

            // Print sample of failed genomes for diagnosis
            if (stage1 == 0)
            {
                var sample = allEvals.FirstOrDefault();
                if (sample != null)
                {
                    Debug.LogWarning(
                        $"[Curriculum EC] All genomes have insufficient episodes. Sample: episodes={sample.episodes?.Count ?? 0}, " +
                        $"fitness={sample.fitness:F2}, successRate={sample.successRate:F2}, " +
                        $"genome={(sample.genome != null ? $"w={sample.genome.width}" : "null")}");
                }
            }
            else if (stage6 == 0 && stage1 > 0)
            {
                var bestSurvivor = allEvals
                    .Where(e => e.episodes != null && e.episodes.Count >= minEpisodeCount)
                    .OrderByDescending(e => e.fitness)
                    .FirstOrDefault();
                if (bestSurvivor != null)
                {
                    Debug.LogWarning(
                        $"[Curriculum EC] Best genome after episode filter but before full Goldilocks: " +
                        $"fitness={bestSurvivor.fitness:F2}, SR={bestSurvivor.successRate:F2}, " +
                        $"progress={bestSurvivor.meanProgress:F2}, sd={bestSurvivor.sdProgress:F2}, " +
                        $"invalid={bestSurvivor.invalidPenalty:F2}");
                }
            }

            // Step 1: Hard Goldilocks filters
            var candidates = allEvals
                .Where(e => e.episodes != null && e.episodes.Count >= minEpisodeCount)
                .Where(e => e.invalidPenalty <= maxInvalidPenalty)
                .Where(e => e.successRate >= goldilocksMinSuccessRate)
                .Where(e => e.successRate <= goldilocksMaxSuccessRate)
                .Where(e => e.meanProgress >= minMeanProgress)
                .Where(e => e.sdProgress <= maxSdProgress)
                .OrderByDescending(e => e.fitness)
                .ToList();

            if (candidates.Count == 0)
            {
                Debug.LogWarning(
                    $"[Curriculum EC] No genomes passed strict Goldilocks filter. " +
                    $"SR range [{goldilocksMinSuccessRate:P0},{goldilocksMaxSuccessRate:P0}], " +
                    $"minProgress={minMeanProgress}, maxSD={maxSdProgress}, minEpisodes={minEpisodeCount}. " +
                    "Falling back to hardest genomes with SR >= min threshold.");

                // Fallback 1: relax max success rate cap (agent may be too skilled)
                candidates = allEvals
                    .Where(e => e.episodes != null && e.episodes.Count >= minEpisodeCount)
                    .Where(e => e.invalidPenalty <= maxInvalidPenalty)
                    .Where(e => e.successRate >= goldilocksMinSuccessRate)
                    .Where(e => e.meanProgress >= minMeanProgress)
                    .OrderByDescending(e => e.difficulty)
                    .ThenByDescending(e => e.fitness)
                    .ToList();
            }

            if (candidates.Count == 0)
            {
                Debug.LogWarning(
                    "[Curriculum EC] No genomes passed relaxed filter either. " +
                    "Falling back to top genomes by fitness from all valid evaluations.");

                // Fallback 2: any genome that has valid episodes and isn't penalized
                candidates = allEvals
                    .Where(e => e.episodes != null && e.episodes.Count >= minEpisodeCount)
                    .Where(e => e.invalidPenalty <= maxInvalidPenalty)
                    .OrderByDescending(e => e.fitness)
                    .ToList();
            }

            if (candidates.Count == 0)
            {
                Debug.LogWarning(
                    "[Curriculum EC] No valid genomes found at all. " +
                    "All evaluations may have timed out or had invalid physics.");
                return new List<PushTGenomeEvaluation>();
            }

            // Step 2: Greedy diversity filter
            var selected = new List<PushTGenomeEvaluation>();
            foreach (var candidate in candidates)
            {
                if (selected.Count >= maxPoolSize) break;

                if (selected.Count == 0)
                {
                    selected.Add(candidate);
                    continue;
                }

                float minDist = float.MaxValue;
                foreach (var existing in selected)
                {
                    float d = NormalizedGenomeDistance(candidate.genome, existing.genome);
                    if (d < minDist) minDist = d;
                }

                if (minDist >= minGenomeDistance)
                {
                    selected.Add(candidate);
                }
            }

            Debug.Log(
                $"[Curriculum EC] Goldilocks: {candidates.Count} passed filters, " +
                $"{selected.Count} selected after diversity filter (target max={maxPoolSize}).");

            return selected;
        }

        /// <summary>
        /// Normalized Euclidean distance between two genomes in [0,1] feature space.
        /// </summary>
        float NormalizedGenomeDistance(PushTBlockGenome a, PushTBlockGenome b)
        {
            if (a == null || b == null) return 1f;
            if (config == null) return 0f;

            float dw = Normalize(a.width,       config.widthRange.min,       config.widthRange.max)
                     - Normalize(b.width,       config.widthRange.min,       config.widthRange.max);
            float dh = Normalize(a.height,      config.heightRange.min,      config.heightRange.max)
                     - Normalize(b.height,      config.heightRange.min,      config.heightRange.max);
            float dd = Normalize(a.depth,       config.depthRange.min,       config.depthRange.max)
                     - Normalize(b.depth,       config.depthRange.min,       config.depthRange.max);
            float dm = Normalize(a.mass,        config.massRange.min,        config.massRange.max)
                     - Normalize(b.mass,        config.massRange.min,        config.massRange.max);
            float df = Normalize(a.friction,    config.frictionRange.min,    config.frictionRange.max)
                     - Normalize(b.friction,    config.frictionRange.min,    config.frictionRange.max);
            float ddg = Normalize(a.blockDrag,  config.dragRange.min,        config.dragRange.max)
                      - Normalize(b.blockDrag,  config.dragRange.min,        config.dragRange.max);
            float db = Normalize(a.bounciness,  config.bouncinessRange.min,  config.bouncinessRange.max)
                     - Normalize(b.bounciness,  config.bouncinessRange.min,  config.bouncinessRange.max);

            float sq = dw * dw + dh * dh + dd * dd + dm * dm + df * df + ddg * ddg + db * db;
            return Mathf.Sqrt(sq / 7f);
        }

        float Normalize(float value, float min, float max)
        {
            return (max <= min) ? 0.5f : Mathf.Clamp01((value - min) / (max - min));
        }

        /// <summary>
        /// Writes selected genomes to the pool.
        /// CRITICAL: if selection is empty, the previous pool is retained.
        /// </summary>
        void WriteToPool(List<PushTGenomeEvaluation> selected, PushTEvolutionRunResult run)
        {
            if (outputPool == null) return;

            if (selected == null || selected.Count == 0)
            {
                Debug.LogWarning(
                    "[Curriculum EC] No Goldilocks genomes selected. " +
                    $"Keeping previous pool with {outputPool.Count} genomes.");
                return;
            }

            outputPool.Clear();
            foreach (var eval in selected)
            {
                if (eval?.genome != null)
                {
                    outputPool.Add(eval.genome);
                }
            }

            outputPool.generationId++;
            outputPool.targetSuccessRateMin = goldilocksMinSuccessRate;
            outputPool.targetSuccessRateMax = goldilocksMaxSuccessRate;
            outputPool.evalEpisodesPerGenome = config?.episodesPerGenome ?? 10;

            Debug.Log(
                $"[Curriculum EC] Pool updated: generation={outputPool.generationId}, " +
                $"count={outputPool.Count}, bestFitness={selected[0].fitness:F2}, " +
                $"bestSR={selected[0].successRate:P0}.");
        }

        /// <summary>
        /// Exports JSON pool + CSV diagnostics and populates the result object with export paths.
        /// </summary>
        void ExportResults(PushTEvolutionRunResult run, List<PushTGenomeEvaluation> selected, PushTCurriculumECResult result)
        {
            try
            {
                Directory.CreateDirectory(CurriculumOutputDir);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Curriculum EC] Failed to create output dir: {ex.Message}");
                return;
            }

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var stage = $"stage_{outputPool.generationId:D3}";

            if (exportJson && outputPool != null)
            {
                try
                {
                    var jsonPath = Path.Combine(CurriculumOutputDir, $"genome_pool_{stage}_{timestamp}.json");
                    outputPool.SaveToJson(jsonPath);
                    result.poolPath = jsonPath;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Curriculum EC] JSON export failed: {ex.Message}");
                }
            }

            if (exportCsv && run != null)
            {
                try
                {
                    var summaryPath = Path.Combine(CurriculumOutputDir, $"summary_{stage}_{timestamp}.csv");
                    var allEvalPath = Path.Combine(CurriculumOutputDir, $"all_evaluations_{stage}_{timestamp}.csv");
                    var episodesPath = Path.Combine(CurriculumOutputDir, $"episodes_{stage}_{timestamp}.csv");

                    ExportSummaryCsv(summaryPath, selected);
                    ExportAllEvaluationsCsv(allEvalPath, run.allEvaluations);
                    ExportEpisodesCsv(episodesPath, run.allEvaluations);

                    result.summaryCsvPath = summaryPath;
                    result.allEvaluationsCsvPath = allEvalPath;
                    result.episodesCsvPath = episodesPath;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Curriculum EC] CSV export failed: {ex.Message}");
                }
            }
        }

        void ExportSummaryCsv(string path, List<PushTGenomeEvaluation> selected)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,fitness,success_rate,mean_progress,iqm_progress,sd_progress,iqr_progress,difficulty,time_score,invalid_penalty,goal_error_score,sd_goal_error,width,height,depth,mass,block_drag,friction,bounciness,selected");

            foreach (var eval in selected)
            {
                var g = eval.genome;
                lines.Add(
                    $"{eval.generationIndex},{eval.genomeIndex},{eval.fitness:F6},{eval.successRate:F6},{eval.meanProgress:F6},{eval.iqmProgress:F6},{eval.sdProgress:F6},{eval.iqrProgress:F6},{eval.difficulty:F6},{eval.timeScore:F6},{eval.invalidPenalty:F6},{eval.goalErrorScore:F6},{eval.sdGoalError:F6},{g.width:F4},{g.height:F4},{g.depth:F4},{g.mass:F4},{g.blockDrag:F4},{g.friction:F4},{g.bounciness:F4},1");
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[Curriculum EC] Summary CSV: {path}");
        }

        void ExportAllEvaluationsCsv(string path, List<PushTGenomeEvaluation> allEvals)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,fitness,success_rate,mean_progress,iqm_progress,sd_progress,iqr_progress,cv_progress,mean_goal_error,sd_goal_error,mean_time_to_goal,mean_reward,difficulty,time_score,goal_error_score,invalid_penalty,block_scale,block_mass,block_drag,friction,width,height,depth,mass,bounciness");

            foreach (var eval in allEvals)
            {
                var g = eval.genome;
                lines.Add(
                    $"{eval.generationIndex},{eval.genomeIndex},{eval.fitness:F6},{eval.successRate:F6},{eval.meanProgress:F6},{eval.iqmProgress:F6},{eval.sdProgress:F6},{eval.iqrProgress:F6},{eval.cvProgress:F6},{eval.meanGoalError:F6},{eval.sdGoalError:F6},{eval.meanTimeToGoal:F6},{eval.meanReward:F6},{eval.difficulty:F6},{eval.timeScore:F6},{eval.goalErrorScore:F6},{eval.invalidPenalty:F6},{g.blockScale:F4},{g.blockMass:F4},{g.blockDrag:F4},{g.friction:F4},{g.width:F4},{g.height:F4},{g.depth:F4},{g.mass:F4},{g.bounciness:F4}");
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[Curriculum EC] All evaluations CSV: {path}");
        }

        void ExportEpisodesCsv(string path, List<PushTGenomeEvaluation> allEvals)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,episode_index,episode_seed,success,reward,progress,final_goal_error,time_to_goal,timed_out,invalid_physics");

            foreach (var eval in allEvals)
            {
                foreach (var ep in eval.episodes)
                {
                    lines.Add(
                        $"{eval.generationIndex},{eval.genomeIndex},{ep.episodeIndex},{ep.episodeSeed},{(ep.success ? 1 : 0)},{ep.reward:F6},{ep.progress:F6},{ep.finalGoalError:F6},{ep.timeToGoal:F6},{(ep.timedOut ? 1 : 0)},{(ep.invalidPhysics ? 1 : 0)}");
                }
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[Curriculum EC] Episodes CSV: {path}");
        }
    }
}
