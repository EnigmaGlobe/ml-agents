using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using Unity.MLAgents.Policies;
using PushTEvolutionMvp.CheckpointSwap;

namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// Reads ec_request.json, validates the scene, runs PushTCurriculumEC, and writes
    /// ec_done.json or ec_error.json. This component should live on the scene root or
    /// on the same GameObject as the watcher.
    /// </summary>
    public class CurriculumECRequestProcessor : MonoBehaviour
    {
        [Header("Settings")]
        public CurriculumECOrchestrationSettings settings;

        [Header("Scene References")]
        public PushTCurriculumEC curriculumEC;
        public FrozenEvaluatorModelLoader modelLoader;

        [Header("Status (read-only)")]
        public CurriculumECStatus status = CurriculumECStatus.Idle;
        public string lastRequestId = "";
        public string lastMessage = "";

        /// <summary>
        /// Full path to the request manifest on disk.
        /// </summary>
        public string RequestManifestFullPath => ResolveFullPath(settings != null ? settings.requestManifestPath : "Assets/Models/FrozenEvaluator/ec_request.json");

        /// <summary>
        /// Full path to the done manifest on disk.
        /// </summary>
        public string DoneManifestFullPath => ResolveFullPath(settings != null ? settings.doneManifestPath : "Assets/Models/FrozenEvaluator/ec_done.json");

        /// <summary>
        /// Full path to the error manifest on disk.
        /// </summary>
        public string ErrorManifestFullPath => ResolveFullPath(settings != null ? settings.errorManifestPath : "Assets/Models/FrozenEvaluator/ec_error.json");

        void OnValidate()
        {
            if (curriculumEC == null)
            {
                curriculumEC = FindFirstObjectByType<PushTCurriculumEC>();
            }
        }

        /// <summary>
        /// Public entry point used by the watcher. Kicks off async processing.
        /// </summary>
        public void ProcessRequest(CurriculumECRequestManifest request)
        {
            _ = ProcessRequestAsync(request);
        }

        async Task ProcessRequestAsync(CurriculumECRequestManifest request)
        {
            if (request == null)
            {
                Debug.LogError("[Curriculum Orchestration] Received null request.");
                return;
            }

            if (string.Equals(request.request_id, lastRequestId, StringComparison.Ordinal))
            {
                Debug.Log($"[Curriculum Orchestration] Ignoring duplicate request_id '{request.request_id}'.");
                return;
            }

            status = CurriculumECStatus.Validating;
            lastRequestId = request.request_id;
            string startedAt = DateTime.UtcNow.ToString("O");

            if (!ValidateRequest(request, out string errorCode, out string errorMessage))
            {
                status = MapErrorStatus(errorCode);
                lastMessage = errorMessage;
                Debug.LogError($"[Curriculum Orchestration] Validation failed: {errorCode} - {errorMessage}");
                WriteErrorManifest(request, errorCode, errorMessage);
                return;
            }

            Debug.Log($"[Curriculum Orchestration] Request '{request.request_id}' validated. Starting Curriculum EC...");
            status = CurriculumECStatus.RunningEC;

            try
            {
                var result = await curriculumEC.RunCurriculumECAsync();

                if (result.success)
                {
                    status = CurriculumECStatus.WritingDone;
                    WriteDoneManifest(request, result, startedAt);
                    status = CurriculumECStatus.Completed;
                    lastMessage = $"Request {request.request_id} completed. Selected {result.selectedCount} genomes.";
                    Debug.Log($"[Curriculum Orchestration] {lastMessage}");
                }
                else
                {
                    status = CurriculumECStatus.ErrorExecutionFailed;
                    lastMessage = result.errorMessage;
                    Debug.LogError($"[Curriculum Orchestration] EC failed: {lastMessage}");
                    WriteErrorManifest(request, "ExecutionFailed", lastMessage);
                }
            }
            catch (Exception ex)
            {
                status = CurriculumECStatus.ErrorExecutionFailed;
                lastMessage = $"Exception during EC: {ex.Message}";
                Debug.LogError($"[Curriculum Orchestration] {lastMessage}");
                WriteErrorManifest(request, "ExecutionFailed", lastMessage);
            }
        }

        bool ValidateRequest(CurriculumECRequestManifest request, out string errorCode, out string errorMessage)
        {
            if (string.IsNullOrWhiteSpace(request.request_id))
            {
                errorCode = "MissingRequestId";
                errorMessage = "request_id is empty.";
                return false;
            }

            if (!request.IsRunCurriculumRequest)
            {
                errorCode = "InvalidActionOrStatus";
                errorMessage = $"Invalid action='{request.action}' or status='{request.status}'. Expected action='run_curriculum_ec' and status='pending'.";
                return false;
            }

            if (curriculumEC == null)
            {
                errorCode = "MissingCurriculumEC";
                errorMessage = "PushTCurriculumEC reference is missing.";
                return false;
            }

            if (curriculumEC.outputPool == null)
            {
                errorCode = "MissingOutputPool";
                errorMessage = "PushTCurriculumEC.outputPool is not assigned.";
                return false;
            }

            if (curriculumEC.frozenEvaluator == null || curriculumEC.frozenEvaluator.agent == null)
            {
                errorCode = "MissingFrozenEvaluator";
                errorMessage = "PushTCurriculumEC.frozenEvaluator or its agent is not assigned.";
                return false;
            }

            var bp = curriculumEC.frozenEvaluator.agent.GetComponent<BehaviorParameters>();
            if (bp == null)
            {
                errorCode = "MissingBehaviorParameters";
                errorMessage = "FrozenEvaluator agent has no BehaviorParameters.";
                return false;
            }

            if (bp.Model == null)
            {
                errorCode = "MissingFrozenEvaluatorModel";
                errorMessage = "FrozenEvaluator BehaviorParameters.Model is null.";
                return false;
            }

            if (bp.BehaviorType != BehaviorType.InferenceOnly)
            {
                Debug.LogWarning($"[Curriculum Orchestration] FrozenEvaluator BehaviorType was {bp.BehaviorType}. Forcing to InferenceOnly.");
                bp.BehaviorType = BehaviorType.InferenceOnly;
            }

            bool strictNameCheck = settings != null && settings.strictBehaviorNameCheck;
            if (strictNameCheck && !string.IsNullOrEmpty(request.expected_behavior_name))
            {
                if (!string.Equals(bp.BehaviorName, request.expected_behavior_name, StringComparison.OrdinalIgnoreCase))
                {
                    errorCode = "BehaviorNameMismatch";
                    errorMessage = $"Expected behavior name '{request.expected_behavior_name}' but found '{bp.BehaviorName}'.";
                    return false;
                }
            }

            if (curriculumEC.IsRunning)
            {
                errorCode = "EcAlreadyRunning";
                errorMessage = "Curriculum EC is already running.";
                return false;
            }

            errorCode = "";
            errorMessage = "";
            return true;
        }

        void WriteDoneManifest(CurriculumECRequestManifest request, PushTCurriculumECResult result, string startedAt)
        {
            try
            {
                var done = new CurriculumECDoneManifest
                {
                    request_id = request.request_id,
                    stage = request.stage,
                    status = "done",
                    started_at = startedAt,
                    completed_at = DateTime.UtcNow.ToString("O"),
                    selected_count = result.selectedCount,
                    generation_id = result.generationId,
                    pool_path = ToProjectRelativePath(result.poolPath),
                    summary_csv_path = ToProjectRelativePath(result.summaryCsvPath),
                    all_evaluations_csv_path = ToProjectRelativePath(result.allEvaluationsCsvPath),
                    episodes_csv_path = ToProjectRelativePath(result.episodesCsvPath),
                    message = result.selectedCount == 0
                        ? "Curriculum EC completed but no genomes passed Goldilocks filter; previous pool retained."
                        : "Curriculum EC completed successfully."
                };

                CurriculumECDoneManifest.Save(DoneManifestFullPath, done);
                Debug.Log($"[Curriculum Orchestration] Wrote ec_done.json: {DoneManifestFullPath}");
            }
            catch (Exception ex)
            {
                lastMessage = $"Failed to write ec_done.json: {ex.Message}";
                Debug.LogError($"[Curriculum Orchestration] {lastMessage}");
            }
        }

        void WriteErrorManifest(CurriculumECRequestManifest request, string errorCode, string message)
        {
            try
            {
                status = CurriculumECStatus.WritingError;
                var error = new CurriculumECErrorManifest
                {
                    request_id = request != null ? request.request_id : "",
                    stage = request != null ? request.stage : -1,
                    status = "error",
                    error_code = errorCode,
                    message = message,
                    created_at = DateTime.UtcNow.ToString("O")
                };

                CurriculumECErrorManifest.Save(ErrorManifestFullPath, error);
                Debug.Log($"[Curriculum Orchestration] Wrote ec_error.json: {ErrorManifestFullPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Curriculum Orchestration] Failed to write ec_error.json: {ex.Message}");
            }
        }

        CurriculumECStatus MapErrorStatus(string errorCode)
        {
            switch (errorCode)
            {
                case "MissingRequestId":
                case "InvalidActionOrStatus":
                    return CurriculumECStatus.ErrorInvalidRequest;
                case "MissingCurriculumEC":
                    return CurriculumECStatus.ErrorMissingCurriculumEC;
                case "MissingOutputPool":
                    return CurriculumECStatus.ErrorMissingOutputPool;
                case "MissingFrozenEvaluator":
                case "MissingBehaviorParameters":
                    return CurriculumECStatus.ErrorMissingFrozenEvaluator;
                case "MissingFrozenEvaluatorModel":
                    return CurriculumECStatus.ErrorMissingModel;
                case "BehaviorNameMismatch":
                    return CurriculumECStatus.ErrorBehaviorNameMismatch;
                case "EcAlreadyRunning":
                    return CurriculumECStatus.ErrorEcAlreadyRunning;
                default:
                    return CurriculumECStatus.ErrorExecutionFailed;
            }
        }

        string ResolveFullPath(string assetRelativePath)
        {
            return Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), assetRelativePath));
        }

        string ToProjectRelativePath(string fullPath)
        {
            if (string.IsNullOrEmpty(fullPath))
            {
                return "";
            }

            bool useRelative = settings == null || settings.writeProjectRelativePaths;
            if (!useRelative)
            {
                return fullPath;
            }

            string projectPath = Path.GetDirectoryName(Path.GetFullPath(Application.dataPath));
            string normalizedProject = projectPath.Replace('/', '\\');
            string normalizedPath = Path.GetFullPath(fullPath).Replace('/', '\\');

            if (normalizedPath.StartsWith(normalizedProject + "\\", StringComparison.OrdinalIgnoreCase))
            {
                return normalizedPath.Substring(normalizedProject.Length + 1).Replace('\\', '/');
            }

            return fullPath.Replace('\\', '/');
        }
    }
}
