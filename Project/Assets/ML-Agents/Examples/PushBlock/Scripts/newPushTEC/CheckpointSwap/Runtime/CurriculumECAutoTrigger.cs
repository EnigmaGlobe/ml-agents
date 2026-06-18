using System;
using UnityEngine;
using Unity.MLAgents.Policies;
using PushTEvolutionMvp.CurriculumOrchestration;

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

        [Header("Orchestration (optional)")]
        [Tooltip("If assigned, auto-trigger will process a pending ec_request.json through the orchestration pipeline instead of running EC directly. This writes ec_done.json for the Python orchestrator.")]
        public CurriculumECRequestProcessor processor;

        [Header("Model Loader")]
        public FrozenEvaluatorModelLoader modelLoader;

        [Header("Auto Run")]
        [Tooltip("If true, automatically run Curriculum EC after a successful model load.")]
        public bool autoRunEcAfterModelLoad = false;

        [Header("Status (read-only)")]
        public bool isEcRunning = false;
        public string lastRunStatus = "";

        bool m_Subscribed = false;

        void OnValidate()
        {
            if (curriculumEC == null)
            {
#if UNITY_2021_2_OR_NEWER
                curriculumEC = FindFirstObjectByType<PushTCurriculumEC>(FindObjectsInactive.Include);
#else
                curriculumEC = FindObjectOfType<PushTCurriculumEC>();
#endif
            }

            if (processor == null)
            {
#if UNITY_2021_2_OR_NEWER
                processor = FindFirstObjectByType<CurriculumECRequestProcessor>(FindObjectsInactive.Include);
#else
                processor = FindObjectOfType<CurriculumECRequestProcessor>();
#endif
            }

            if (modelLoader == null)
            {
                modelLoader = GetComponent<FrozenEvaluatorModelLoader>();
#if UNITY_2021_2_OR_NEWER
                if (modelLoader == null)
                    modelLoader = FindFirstObjectByType<FrozenEvaluatorModelLoader>(FindObjectsInactive.Include);
#else
                if (modelLoader == null)
                    modelLoader = FindObjectOfType<FrozenEvaluatorModelLoader>();
#endif
            }
        }

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

            if (Application.isPlaying)
            {
                Debug.Log("[Checkpoint Swap] Model loaded while in Play Mode; skipping auto-trigger until Edit Mode.");
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

            // If we are being driven by the file-based orchestrator, process the pending
            // request through the processor so ec_done.json / ec_error.json are written.
            if (processor != null)
            {
                Debug.Log("[Checkpoint Swap] Auto-triggering via CurriculumECRequestProcessor...");
                var request = CurriculumECRequestManifest.Load(processor.RequestManifestFullPath);
                if (request != null && request.IsRunCurriculumRequest)
                {
                    processor.ProcessRequest(request);
                }
                else
                {
                    Debug.LogWarning("[Checkpoint Swap] No pending orchestrator request found; falling back to direct EC.");
                    curriculumEC.RunCurriculumEC();
                }
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
