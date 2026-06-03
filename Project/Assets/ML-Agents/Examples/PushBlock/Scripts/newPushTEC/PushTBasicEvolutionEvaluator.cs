using System.Threading.Tasks;
using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Concrete evaluator for the PushBlock scene.
    /// Hooks into PushAgentBasic's public EC API and ResetEpisodeForEvolution.
    /// </summary>
    public class PushTBasicEvolutionEvaluator : PushTBlockEvaluator
    {
        [Header("Scene References")]
        public PushAgentBasic agent;

        [Header("Episode Settings")]
        public float episodeTimeoutSeconds = 15f;

        public override async Task<PushTGenomeEvaluation> EvaluateGenomeAsync(
            PushTBlockGenome genome,
            PushTEvolutionConfig config,
            int generationIndex,
            int genomeIndex,
            System.Random random)
        {
            if (this.agent == null)
            {
                Debug.LogError("[PushT EC] PushTBasicEvolutionEvaluator has no agent assigned.");
                return await base.EvaluateGenomeAsync(genome, config, generationIndex, genomeIndex, random);
            }

            bool originalScreenshot = this.agent.enableScreenshotCapture;
            bool originalGlobalDisable = PushAgentBasic.disableAllScreenshotCapture;
            this.agent.enableScreenshotCapture = false;
            PushAgentBasic.disableAllScreenshotCapture = true;

            try
            {
                return await base.EvaluateGenomeAsync(genome, config, generationIndex, genomeIndex, random);
            }
            finally
            {
                this.agent.enableScreenshotCapture = originalScreenshot;
                PushAgentBasic.disableAllScreenshotCapture = originalGlobalDisable;
            }
        }

        protected override async Task<PushTEpisodeResult> EvaluateEpisodeAsync(
            PushTBlockGenome genome,
            int episodeIndex,
            int episodeSeed)
        {
            if (this.agent == null)
            {
                Debug.LogError("[PushT EC] PushTBasicEvolutionEvaluator has no agent assigned.");
                return new PushTEpisodeResult
                {
                    episodeIndex = episodeIndex,
                    episodeSeed = episodeSeed,
                    notes = "Missing agent reference"
                };
            }

            // 1. Reset episode with genome and seed
            this.agent.ResetEpisodeForEvolution(genome, episodeSeed);

            // 2. Poll until episode completes or times out
            float elapsed = 0f;
            while (!this.agent.IsEpisodeComplete && elapsed < this.episodeTimeoutSeconds)
            {
                elapsed += Time.deltaTime;
                await Task.Yield();
            }

            bool timedOut = elapsed >= this.episodeTimeoutSeconds && !this.agent.IsEpisodeComplete;

            if (timedOut)
            {
                Debug.LogWarning($"[PushT EC] Episode timed out after {this.episodeTimeoutSeconds}s");
            }

            // 3. Read results from public API
            var result = new PushTEpisodeResult
            {
                episodeIndex = episodeIndex,
                episodeSeed = episodeSeed,
                success = this.agent.WasLastEpisodeSuccessful,
                reward = this.agent.LastEpisodeReward,
                progress = this.agent.LastEpisodeNormalizedTaskProgress,
                finalGoalError = this.agent.LastEpisodeFinalGoalError,
                timeToGoal = this.agent.LastEpisodeTimeToGoal,
                timedOut = timedOut,
                invalidPhysics = false // set by caller if physics checks fail
            };

            return result;
        }
    }
}
