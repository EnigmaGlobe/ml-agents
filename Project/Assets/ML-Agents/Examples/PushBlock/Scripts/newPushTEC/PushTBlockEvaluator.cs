using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace PushTEvolutionMvp
{
    public abstract class PushTBlockEvaluator : MonoBehaviour
    {
        public virtual async Task<PushTGenomeEvaluation> EvaluateGenomeAsync(
            PushTBlockGenome genome,
            PushTEvolutionConfig config,
            int generationIndex,
            int genomeIndex,
            System.Random random)
        {
            var evaluation = new PushTGenomeEvaluation
            {
                generationIndex = generationIndex,
                genomeIndex = genomeIndex,
                genome = genome.Clone()
            };

            for (int episodeIndex = 0; episodeIndex < config.episodesPerGenome; episodeIndex++)
            {
                var episodeSeed = random.Next();
                var episode = await this.EvaluateEpisodeAsync(genome, episodeIndex, episodeSeed);
                evaluation.episodes.Add(episode);
            }

            ComputeAggregates(evaluation, config);
            return evaluation;
        }

        protected abstract Task<PushTEpisodeResult> EvaluateEpisodeAsync(
            PushTBlockGenome genome,
            int episodeIndex,
            int episodeSeed);

        private static void ComputeAggregates(PushTGenomeEvaluation eval, PushTEvolutionConfig config)
        {
            var episodes = eval.episodes;
            int n = episodes.Count;
            if (n == 0)
            {
                eval.fitness = -999f;
                return;
            }

            // Basic counts
            int successes = episodes.Count(e => e.success);
            eval.successRate = successes / (float)n;

            // Progress distribution
            var progressList = episodes.Select(e => e.progress).ToList();
            progressList.Sort();
            eval.meanProgress = progressList.Average();
            eval.sdProgress = CalculateStdDev(progressList);
            eval.iqmProgress = CalculateIQM(progressList);
            eval.iqrProgress = CalculateIQR(progressList);
            eval.cvProgress = eval.meanProgress > 0.0001f ? eval.sdProgress / eval.meanProgress : 0f;

            // Goal error distribution
            var goalErrorList = episodes.Select(e => e.finalGoalError).ToList();
            eval.meanGoalError = goalErrorList.Average();
            eval.sdGoalError = CalculateStdDev(goalErrorList);

            // Time to goal (diagnostic only, not used in fitness)
            eval.meanTimeToGoal = episodes
                .Where(e => e.timeToGoal > 0f)
                .Select(e => e.timeToGoal)
                .DefaultIfEmpty(-1f)
                .Average();
            eval.meanReward = episodes.Average(e => e.reward);

            // Invalid penalty from episode physics
            eval.invalidPenalty = episodes.Any(e => e.invalidPhysics) ? 1f : 0f;

            // M4 spatial behavior profile and sub-scores.
            // Computed for all valid evaluations so that CSV diagnostics are complete
            // and M4 can influence selection across success-rate tiers.
            eval.spatialProfile = ComputeSpatialProfile(episodes);

            eval.learnabilityScore = eval.iqmProgress;
            eval.challengeScore = config.useGenerationDependentChallenge
                ? ComputeChallengeScore(eval.successRate, eval.generationIndex, config)
                : 1f - Mathf.Abs(eval.successRate - 0.5f);
            eval.spatialBehaviorScore = config.useM4Fitness
                ? ComputeSpatialBehaviorScore(eval.spatialProfile, config)
                : 0f;
            eval.noveltyScore = 0f; // reserved for Phase 3

            float m4WeightedScore = 0f;
            if (config.useM4Fitness)
            {
                // No novelty in Phase 2; renormalize across the three available components.
                m4WeightedScore =
                    0.40f * eval.learnabilityScore +
                    0.35f * eval.challengeScore +
                    0.25f * eval.spatialBehaviorScore;
            }

            if (config.useM4Fitness && eval.genomeIndex == 0)
            {
                Debug.Log(
                    $"[PushTBlockEvaluator] Generation {eval.generationIndex}: M4 fitness ENABLED " +
                    $"(bonusWeight={config.m4FitnessBonusWeight}, targetSpread={config.targetBlockRadialSpread}, " +
                    $"challenge={(config.useGenerationDependentChallenge ? "generation-dependent" : "fixed 0.5")}).");
            }

            // Tier 1: Invalid physics or invalid genome → fitness = 0, hard return
            if (eval.invalidPenalty > 0.5f || !PushTBlockGenome.IsGenomeValid(eval.genome))
            {
                eval.fitness = 0f;
                return;
            }

            // Tier 2: Success rate below threshold but otherwise valid.
            // Add M4 bonus so spatial behavior can still influence low-success genomes.
            if (eval.successRate < config.successRateThreshold)
            {
                eval.fitness = 80f * eval.successRate + 15f * eval.iqmProgress
                    + (config.useM4Fitness ? config.m4FitnessBonusWeight * m4WeightedScore : 0f);
                return;
            }

            // Tier 3: Success rate >= threshold
            float goalAccuracyScore = 1f / (1f + eval.meanGoalError);
            float goalConsistencyScore = 1f / (1f + eval.sdGoalError);

            eval.fitness = 100f
                + 30f * eval.iqmProgress
                + 25f * eval.successRate
                + 25f * goalAccuracyScore
                + 20f * goalConsistencyScore
                - 15f * eval.sdProgress
                - 10f * eval.iqrProgress
                + (config.useM4Fitness ? config.m4FitnessBonusWeight * m4WeightedScore : 0f);

            // Diagnostic metrics (not used in fitness, kept for CSV export)
            eval.difficulty = ComputeDifficulty(eval.genome, config);
            eval.goalErrorScore = goalAccuracyScore;
            eval.timeScore = 0f;
        }

        /// <summary>
        /// Aggregates per-episode M4 spatial metrics into a genome-level profile.
        /// </summary>
        private static PushTSpatialBehaviorProfile ComputeSpatialProfile(List<PushTEpisodeResult> episodes)
        {
            var profile = new PushTSpatialBehaviorProfile();
            int n = episodes.Count;
            if (n == 0) return profile;

            profile.meanBlockNetDisplacement    = episodes.Average(e => e.blockNetDisplacement);
            profile.meanAgentNetDisplacement    = episodes.Average(e => e.agentNetDisplacement);
            profile.meanBlockRadialSpread       = episodes.Average(e => e.blockRadialSpread);
            profile.meanAgentRadialSpread       = episodes.Average(e => e.agentRadialSpread);
            profile.meanTaskCentroidCentrality  = episodes.Average(e => e.taskCentroidCentrality);
            profile.meanSceneCentroidCentrality = episodes.Average(e => e.sceneCentroidCentrality);

            profile.sdBlockNetDisplacement      = PushTSpatialMetrics.CalculateStdDev(episodes.Select(e => e.blockNetDisplacement).ToList());
            profile.sdBlockRadialSpread         = PushTSpatialMetrics.CalculateStdDev(episodes.Select(e => e.blockRadialSpread).ToList());
            profile.sdTaskCentroidCentrality    = PushTSpatialMetrics.CalculateStdDev(episodes.Select(e => e.taskCentroidCentrality).ToList());

            return profile;
        }

        /// <summary>
        /// Linear normalization of a value to [0, 1] given a min/max range.
        /// </summary>
        private static float Normalize01(float value, float min, float max)
        {
            if (max <= min) return 0.5f;
            return Mathf.Clamp01((value - min) / (max - min));
        }

        /// <summary>
        /// Reward radial spread that falls inside a useful [min, max] band around a target.
        /// Too low = no exploration; too high = random wandering.
        /// </summary>
        private static float OptimalRadialSpreadScore(float spread, float target, float min, float max)
        {
            if (spread < min)
            {
                return spread / Mathf.Max(min, 0.0001f);
            }
            if (spread > max)
            {
                float overshoot = spread - max;
                return Mathf.Max(0f, 1f - overshoot / Mathf.Max(max, 0.0001f));
            }
            float range = Mathf.Max(target - min, max - target);
            return range > 0.0001f ? 1f - Mathf.Abs(spread - target) / range : 1f;
        }

        /// <summary>
        /// Computes the M4 spatial-behavior score for a genome profile.
        /// </summary>
        private static float ComputeSpatialBehaviorScore(PushTSpatialBehaviorProfile profile, PushTEvolutionConfig config)
        {
            if (profile == null) return 0f;

            float netDisplacementScore = Normalize01(
                profile.meanBlockNetDisplacement,
                0f,
                config.maxExpectedBlockDisplacement);

            float taskCentralityScore = Normalize01(
                profile.meanTaskCentroidCentrality,
                -config.maxExpectedTaskCentrality,
                0f);

            float spreadScore = OptimalRadialSpreadScore(
                profile.meanBlockRadialSpread,
                config.targetBlockRadialSpread,
                config.minUsefulBlockRadialSpread,
                config.maxUsefulBlockRadialSpread);

            return 0.40f * netDisplacementScore
                 + 0.30f * taskCentralityScore
                 + 0.30f * spreadScore;
        }

        /// <summary>
        /// Returns the target success rate for the current generation.
        /// Early generations target easier tasks; later generations target harder tasks.
        /// </summary>
        private static float GetGenerationTargetSuccessRate(int generationIndex, PushTEvolutionConfig config)
        {
            if (config.generationCount <= 1)
            {
                return (config.earlyTargetSuccessRate + config.lateTargetSuccessRate) * 0.5f;
            }
            float t = generationIndex / (float)(config.generationCount - 1);
            return Mathf.Lerp(config.earlyTargetSuccessRate, config.lateTargetSuccessRate, t);
        }

        /// <summary>
        /// Challenge score: peaks when success rate matches the generation-dependent target.
        /// </summary>
        private static float ComputeChallengeScore(float successRate, int generationIndex, PushTEvolutionConfig config)
        {
            float target = GetGenerationTargetSuccessRate(generationIndex, config);
            return 1f - Mathf.Abs(successRate - target);
        }

        /// <summary>
        /// Time target scoring based on reviewer recommendation.
        /// Ideal completion time is 5-12 seconds.
        /// </summary>
        private static float ComputeTimeScore(float meanTimeToGoal, PushTEvolutionConfig config)
        {
            if (meanTimeToGoal <= 0f)
            {
                // No successful episodes
                return -2.0f;
            }
            else if (meanTimeToGoal <= config.timeTooFast)
            {
                // Too fast (0-3s): too easy
                return -1.0f;
            }
            else if (meanTimeToGoal <= config.timeEasy)
            {
                // Easy (3-5s): slightly too easy
                return -0.3f;
            }
            else if (meanTimeToGoal <= config.timeIdeal)
            {
                // Ideal (5-12s): best range
                return 1.0f;
            }
            else if (meanTimeToGoal <= config.timeHard)
            {
                // Hard (12-20s): still acceptable but getting tough
                return 0.3f;
            }
            else
            {
                // Too hard (>20s): likely unstable or frustrating
                return -0.5f;
            }
        }

        private static float ComputeDifficulty(PushTBlockGenome genome, PushTEvolutionConfig config)
        {
            float massDiff     = NormalizeDifficulty(genome.mass,     config.massRange.min,     config.massRange.max);
            float widthDiff    = NormalizeDifficulty(genome.width,    config.widthRange.min,    config.widthRange.max);
            float frictionDiff = 1f - NormalizeDifficulty(genome.friction, config.frictionRange.min, config.frictionRange.max);
            float heightDiff   = NormalizeDifficulty(genome.height,   config.heightRange.min,   config.heightRange.max);
            float depthDiff    = NormalizeDifficulty(genome.depth,    config.depthRange.min,    config.depthRange.max);

            return
                0.30f * massDiff +
                0.20f * widthDiff +
                0.20f * frictionDiff +
                0.15f * heightDiff +
                0.15f * depthDiff;
        }

        private static float NormalizeDifficulty(float value, float min, float max)
        {
            if (max <= min) return 0.5f;
            return Mathf.Clamp01((value - min) / (max - min));
        }

        private static float CalculateStdDev(List<float> values)
        {
            if (values.Count == 0) return 0f;
            float mean = values.Average();
            float sumSq = values.Sum(v =>
            {
                float d = v - mean;
                return d * d;
            });
            return Mathf.Sqrt(sumSq / values.Count);
        }

        private static float CalculateIQM(List<float> sorted)
        {
            int n = sorted.Count;
            if (n == 0) return 0f;

            float q1Index = (n - 1) * 0.25f;
            float q3Index = (n - 1) * 0.75f;
            int startIdx = Mathf.CeilToInt(q1Index);
            int endIdx = Mathf.FloorToInt(q3Index);

            if (startIdx > endIdx)
            {
                return sorted.Average();
            }

            float sum = 0f;
            int count = 0;
            for (int i = startIdx; i <= endIdx && i < n; i++)
            {
                sum += sorted[i];
                count++;
            }

            return count > 0 ? sum / count : sorted.Average();
        }

        private static float CalculateIQR(List<float> sorted)
        {
            int n = sorted.Count;
            if (n == 0) return 0f;

            float q1 = Percentile(sorted, 0.25f);
            float q3 = Percentile(sorted, 0.75f);
            return q3 - q1;
        }

        private static float Percentile(List<float> sorted, float p)
        {
            int n = sorted.Count;
            if (n == 1) return sorted[0];

            float index = (n - 1) * p;
            int lower = Mathf.FloorToInt(index);
            int upper = Mathf.CeilToInt(index);

            if (lower == upper) return sorted[lower];

            float weight = index - lower;
            return sorted[lower] * (1f - weight) + sorted[upper] * weight;
        }
    }
}
