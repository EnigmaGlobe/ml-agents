using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

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

        [Header("Diagnostics")]
        public bool enableVerboseLogging = true;

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

            // Validate frozen agent configuration
            if (enableVerboseLogging)
            {
                ValidateFrozenAgentConfig();
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

        /// <summary>
        /// Validates that the agent is properly configured for inference-only evaluation.
        /// </summary>
        void ValidateFrozenAgentConfig()
        {
            var bp = agent.GetComponent<BehaviorParameters>();
            if (bp == null)
            {
                Debug.LogError("[PushT EC-Frozen] Agent has no BehaviorParameters component!");
                return;
            }

            if (bp.BehaviorType != BehaviorType.InferenceOnly)
            {
                Debug.LogWarning($"[PushT EC-Frozen] Agent BehaviorType is {bp.BehaviorType}, should be InferenceOnly. " +
                    $"EC will hang waiting for Python trainer decisions.");
            }

            if (bp.Model == null)
            {
                Debug.LogWarning("[PushT EC-Frozen] Agent has no .onnx model assigned. " +
                    "Assign a trained model to BehaviorParameters.Model for inference.");
            }
            else
            {
                if (enableVerboseLogging)
                {
                    Debug.Log($"[PushT EC-Frozen] Model loaded: {bp.Model.name}");
                }
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

            if (enableVerboseLogging)
            {
                Debug.Log($"[EC-Ep{episodeIndex}] === START on agent='{agent.name}' seed={episodeSeed}");
            }

            // 1. Reset episode with genome and seed
            this.agent.ResetEpisodeForEvolution(genome, episodeSeed);

            if (enableVerboseLogging)
            {
                Debug.Log(
                    $"[EC-Ep{episodeIndex}] POST-RESET: " +
                    $"EcCaptured={agent.EcEpisodeCaptured} " +
                    $"EcReward={agent.EcReward:F2} " +
                    $"EcSuccess={agent.EcSuccess}");
            }

            // 2. Poll until EC episode capture completes or times out.
            // We use agent.EcEpisodeCaptured instead of agent.IsEpisodeComplete
            // because EndEpisode() → OnEpisodeBegin() resets the normal flags
            // before our async loop can observe them.
            //
            // IMPORTANT: use real-time (DateTime.UtcNow) for the timeout. In Edit Mode
            // Time.deltaTime is not reliably updated across async/await continuations,
            // so episodes that never complete would hang forever.
            var episodeStartUtc = DateTime.UtcNow;
            int pollCount = 0;
            int prevStepCount = -1;

            // Unity 6+ requires SimulationMode.Script before calling Physics.Simulate().
            // Save the original mode so we can restore it after the evaluation.
            var originalSimulationMode = Physics.simulationMode;
            if (!Application.isPlaying)
            {
                Physics.simulationMode = SimulationMode.Script;
            }

            try
            {
                while (!this.agent.EcEpisodeCaptured && (DateTime.UtcNow - episodeStartUtc).TotalSeconds < this.episodeTimeoutSeconds)
                {
                    pollCount++;

                    // In Edit Mode the Academy/Physics do not step automatically, so drive
                    // the evaluation frame manually. In Play Mode this is handled by Unity.
                    if (!Application.isPlaying)
                    {
                        StepEvaluationFrame();
                    }

                    // Use WaitForEndOfFrame to properly wait for Unity frame processing
                    await WaitForNextFrame();

                    // Periodic log every ~2s
                    if (enableVerboseLogging && pollCount % 120 == 0)
                    {
                        int curStep = GetAcademyStepCount(agent);
                        float elapsed = (float)(DateTime.UtcNow - episodeStartUtc).TotalSeconds;
                        Debug.Log(
                            $"[EC-Ep{episodeIndex}] POLL#{pollCount} t={elapsed:F1}s " +
                            $"step={curStep} stepChanged={curStep != prevStepCount} " +
                            $"EcCaptured={agent.EcEpisodeCaptured}");

                        prevStepCount = curStep;
                    }
                }
            }
            finally
            {
                if (!Application.isPlaying)
                {
                    Physics.simulationMode = originalSimulationMode;
                }
            }

            float elapsedFinal = (float)(DateTime.UtcNow - episodeStartUtc).TotalSeconds;
            bool timedOut = elapsedFinal >= this.episodeTimeoutSeconds && !this.agent.EcEpisodeCaptured;

            // Read from EC cache (written by PushAgentBasic before EndEpisode())
            float finalReward = this.agent.EcReward;
            bool finalSuccess = this.agent.EcSuccess;
            float finalProgress = this.agent.EcProgress;
            float finalGoalError = this.agent.EcGoalError;
            float finalTimeToGoal = this.agent.EcTimeToGoal;

            if (enableVerboseLogging)
            {
                Debug.Log(
                    $"[EC-COLLECT] captured={agent.EcEpisodeCaptured} " +
                    $"success={agent.EcSuccess} " +
                    $"reward={agent.EcReward:F3} " +
                    $"progress={agent.EcProgress:F3} " +
                    $"goalError={agent.EcGoalError:F3} " +
                    $"timeToGoal={agent.EcTimeToGoal:F2}");

                Debug.Log(
                    $"[EC-Ep{episodeIndex}] === DONE === " +
                    $"elapsed={elapsedFinal:F2}s timedOut={timedOut} " +
                    $"success={finalSuccess} reward={finalReward:F2} " +
                    $"progress={finalProgress:F2} goalError={finalGoalError:F2} " +
                    $"timeToGoal={finalTimeToGoal:F2}");
            }

            if (timedOut)
            {
                Debug.LogWarning(
                    $"[EC-Ep{episodeIndex}] TIMEOUT after {this.episodeTimeoutSeconds}s! " +
                    $"EcCaptured={agent.EcEpisodeCaptured}. " +
                    "The agent did not write EC cache before timeout.");
            }

            // 4. Capture spatial trajectories and compute per-episode M4 metrics
            var result = new PushTEpisodeResult
            {
                episodeIndex = episodeIndex,
                episodeSeed = episodeSeed,
                success = finalSuccess,
                reward = finalReward,
                progress = finalProgress,
                finalGoalError = finalGoalError,
                timeToGoal = finalTimeToGoal,
                timedOut = timedOut,
                invalidPhysics = false
            };

            // Fall back to live buffers if EC cache is empty (defensive).
            var agentPositions = (this.agent.EcAgentPositions != null && this.agent.EcAgentPositions.Count > 0)
                ? this.agent.EcAgentPositions
                : this.agent.EpisodeAgentPositions;
            var blockPositions = (this.agent.EcBlockPositions != null && this.agent.EcBlockPositions.Count > 0)
                ? this.agent.EcBlockPositions
                : this.agent.EpisodeBlockPositions;

            result.agentPositions = new List<Vector3>(agentPositions ?? new List<Vector3>());
            result.blockPositions = new List<Vector3>(blockPositions ?? new List<Vector3>());
            result.goalPosition = this.agent.goal != null ? this.agent.goal.transform.position : Vector3.zero;

            Vector3 sceneCenter = Vector3.zero;
            if (this.agent.ground != null)
            {
                var groundCollider = this.agent.ground.GetComponent<Collider>();
                sceneCenter = groundCollider != null ? groundCollider.bounds.center : this.agent.ground.transform.position;
            }
            result.sceneCenter = sceneCenter;

            // Compute M4 metrics for every episode, regardless of success/failure.
            // Centrality can be computed with a single sample; displacement/spread need >= 2.
            if (result.blockPositions.Count >= 1)
            {
                result.taskCentroidCentrality = PushTSpatialMetrics.TaskCentroidCentrality(result.blockPositions, result.goalPosition);
                result.sceneCentroidCentrality = PushTSpatialMetrics.SceneCentroidCentrality(result.blockPositions, result.sceneCenter);
            }
            if (result.blockPositions.Count >= 2)
            {
                result.blockNetDisplacement = PushTSpatialMetrics.NetDisplacementMagnitude(result.blockPositions);
                result.blockRadialSpread = PushTSpatialMetrics.RadialSpread(result.blockPositions, result.sceneCenter);
            }

            if (result.agentPositions.Count >= 1)
            {
                // Agent centrality is diagnostic only; no field for it currently.
            }
            if (result.agentPositions.Count >= 2)
            {
                result.agentNetDisplacement = PushTSpatialMetrics.NetDisplacementMagnitude(result.agentPositions);
                result.agentRadialSpread = PushTSpatialMetrics.RadialSpread(result.agentPositions, result.sceneCenter);
            }

            return result;
        }

        /// <summary>
        /// Advances the Academy and physics by one evaluation frame.
        /// Required for running EC in Edit Mode where Unity does not tick automatically.
        /// </summary>
        static void StepEvaluationFrame()
        {
            try
            {
                if (!Academy.IsInitialized)
                {
                    _ = Academy.Instance; // lazily create the Academy
                }
                Academy.Instance.EnvironmentStep();
                Physics.Simulate(Time.fixedDeltaTime);
            }
            catch (System.Exception ex)
            {
                UnityEngine.Debug.LogError($"[PushT EC] Manual evaluation step failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Wait for the next Unity frame (FixedUpdate + rendering) to complete.
        /// More reliable than Task.Yield() for detecting agent state changes.
        /// </summary>
        static async Task WaitForNextFrame()
        {
            await Task.Delay(16); // ~60fps frame duration as fallback
        }

        /// <summary>
        /// Read the agent's academy step count via reflection.
        /// </summary>
        static int GetAcademyStepCount(Agent a)
        {
            try
            {
                var field = typeof(Agent).GetField(
                    "m_AcademyStepCount",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                if (field != null) return (int)field.GetValue(a);
            }
            catch { }
            return -1;
        }
    }
}
