using System;
using UnityEngine;
using Unity.MLAgents.Policies;

namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// Optional component that auto-triggers PushTCurriculumEC after a successful
    /// FrozenEvaluator model load. Default is disabled. Never triggers if model loading
    /// failed or if the EC is already believed to be running.
    /// </summary>
    public class CurriculumECAutoTrigger : MonoBehaviour
    {
        [Header("Curriculum EC")]
        public PushTCurriculumEC curriculumEC;

        [Header("Model Loader")]
        public FrozenEvaluatorModelLoader modelLoader;

        [Header("Auto Run")]
        [Tooltip("If true, automatically run Curriculum EC after a successful model load.")]
        public bool autoRunEcAfterModelLoad = false;

        [Header("Status (read-only)")]
        public bool isEcRunning = false;
        public string lastRunStatus = "";

        bool m_Subscribed = false;

        void OnEnable()
        {
            if (!m_Subscribed)
            {
                FrozenEvaluatorModelLoader.OnModelLoadSuccess += OnModelLoaded;
                FrozenEvaluatorModelLoader.OnModelLoadFailed += OnModelLoadFailed;
                m_Subscribed = true;
            }
        }

        void OnDisable()
        {
            if (m_Subscribed)
            {
                FrozenEvaluatorModelLoader.OnModelLoadSuccess -= OnModelLoaded;
                FrozenEvaluatorModelLoader.OnModelLoadFailed -= OnModelLoadFailed;
                m_Subscribed = false;
            }
        }

        void OnModelLoaded()
        {
            if (!autoRunEcAfterModelLoad)
            {
                return;
            }

            RunEC();
        }

        void OnModelLoadFailed(string message)
        {
            Debug.LogError(
                $"[Checkpoint Swap] EC auto-trigger skipped because model load failed: {message}");
        }

        void RunEC()
        {
            if (curriculumEC == null)
            {
                Debug.LogError("[Checkpoint Swap] CurriculumECAutoTrigger has no PushTCurriculumEC assigned.");
                return;
            }

            if (isEcRunning)
            {
                Debug.LogWarning("[Checkpoint Swap] EC is already running. Skipping auto-trigger.");
                return;
            }

            if (modelLoader != null && modelLoader.status != CheckpointSwapStatus.ModelReady)
            {
                Debug.LogWarning("[Checkpoint Swap] Model is not ready. Skipping auto-trigger.");
                return;
            }

            var evaluator = curriculumEC.frozenEvaluator;
            if (evaluator == null || evaluator.agent == null)
            {
                Debug.LogError("[Checkpoint Swap] Frozen evaluator not configured. Skipping auto-trigger.");
                return;
            }

            var bp = evaluator.agent.GetComponent<BehaviorParameters>();
            if (bp == null || bp.Model == null)
            {
                Debug.LogError("[Checkpoint Swap] Frozen evaluator has no model. Skipping auto-trigger.");
                return;
            }

            if (bp.BehaviorType != BehaviorType.InferenceOnly)
            {
                Debug.LogError("[Checkpoint Swap] Frozen evaluator is not InferenceOnly. Skipping auto-trigger.");
                return;
            }

            isEcRunning = true;
            lastRunStatus = "Started";
            Debug.Log("[Checkpoint Swap] Auto-triggering Curriculum EC...");

            try
            {
                curriculumEC.RunCurriculumEC();
                lastRunStatus = "Triggered (async)";
                Debug.Log("[Checkpoint Swap] Curriculum EC triggered.");
            }
            catch (Exception ex)
            {
                lastRunStatus = $"Failed: {ex.Message}";
                Debug.LogError($"[Checkpoint Swap] Failed to trigger Curriculum EC: {ex.Message}");
            }
            finally
            {
                // RunCurriculumEC is async void, so it returns immediately.
                // A true IsRunning flag would require modifying PushTCurriculumEC.
                // For V2 we reset to allow the next auto-trigger cycle.
                isEcRunning = false;
            }
        }
    }
}
