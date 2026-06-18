using System;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.InferenceEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// Loads the latest ONNX checkpoint from a fixed Unity asset path and assigns it to
    /// the FrozenEvaluator's BehaviorParameters, forcing InferenceOnly. Keeps the old
    /// model if loading fails. Works only inside the Unity Editor.
    /// </summary>
    public class FrozenEvaluatorModelLoader : MonoBehaviour
    {
        [Header("Target")]
        [Tooltip("Direct reference to the FrozenEvaluator agent (PushAgentBasic).")]
        public PushAgentBasic frozenAgent;

        [Header("Paths")]
        [Tooltip("Project-relative path to the manifest file. Kept outside Assets to avoid .meta files.")]
        public string manifestPath = "Temp/FrozenEvaluator/checkpoint_ready.json";

        [Tooltip("Fallback asset-relative ONNX path if manifest cannot be read.")]
        public string fallbackOnnxAssetPath = "Assets/Models/FrozenEvaluator/FrozenEvaluator_latest.onnx";

        [Header("Validation")]
        [Tooltip("Expected BehaviorParameters.BehaviorName. Leave empty to skip name validation.")]
        public string expectedBehaviorName = "";

        [Tooltip("If true, abort loading when behavior names do not match.")]
        public bool strictCheckBehaviorName = false;

        [Header("Status (read-only)")]
        public CheckpointSwapStatus status = CheckpointSwapStatus.Idle;
        public string lastStatusMessage = "";
        public string lastLoadedModelName = "";

        /// <summary>
        /// Fired after a successful model load.
        /// </summary>
        public static event Action OnModelLoadSuccess;

        /// <summary>
        /// Fired when model loading fails. Argument is the failure message.
        /// </summary>
        public static event Action<string> OnModelLoadFailed;

        void OnValidate()
        {
            // Ensure a direct agent reference is available.
            if (frozenAgent == null)
            {
                // Note: PushTBasicEvolutionEvaluator is in the PushTEvolutionMvp namespace.
                var evaluator = GetComponent<PushTBasicEvolutionEvaluator>();
                if (evaluator != null)
                {
                    frozenAgent = evaluator.agent;
                }
            }
        }

        /// <summary>
        /// Reloads the model from the manifest/fallback path.
        /// </summary>
        public void LoadLatestModel()
        {
#if UNITY_EDITOR
            LoadLatestModelEditor();
#else
            status = CheckpointSwapStatus.ErrorImportFailed;
            lastStatusMessage = "FrozenEvaluatorModelLoader is Editor-only in Version 2.";
            Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
            OnModelLoadFailed?.Invoke(lastStatusMessage);
#endif
        }

#if UNITY_EDITOR
        void LoadLatestModelEditor()
        {
            status = CheckpointSwapStatus.LoadingModel;
            lastStatusMessage = "";

            try
            {
                // Resolve agent reference.
                if (frozenAgent == null)
                {
                    var evaluator = GetComponent<PushTBasicEvolutionEvaluator>();
                    if (evaluator != null)
                    {
                        frozenAgent = evaluator.agent;
                    }
                }

                if (frozenAgent == null)
                {
                    status = CheckpointSwapStatus.ErrorMissingFrozenEvaluator;
                    lastStatusMessage = "Frozen agent reference is missing. Assign PushAgentBasic 'FrozenEvaluator'.";
                    Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                    OnModelLoadFailed?.Invoke(lastStatusMessage);
                    return;
                }

                var bp = frozenAgent.GetComponent<BehaviorParameters>();
                if (bp == null)
                {
                    status = CheckpointSwapStatus.ErrorMissingFrozenEvaluator;
                    lastStatusMessage = $"No BehaviorParameters component on '{frozenAgent.name}'.";
                    Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                    OnModelLoadFailed?.Invoke(lastStatusMessage);
                    return;
                }

                // Safety: force InferenceOnly so EC never accidentally waits for Python.
                if (bp.BehaviorType != BehaviorType.InferenceOnly)
                {
                    Debug.LogWarning(
                        $"[Checkpoint Swap] Forcing BehaviorType from {bp.BehaviorType} to InferenceOnly on '{frozenAgent.name}'.");
                    bp.BehaviorType = BehaviorType.InferenceOnly;
                }

                // Resolve the ONNX asset path.
                string onnxAssetPath = ResolveOnnxAssetPath();
                if (string.IsNullOrEmpty(onnxAssetPath))
                {
                    status = CheckpointSwapStatus.ErrorMissingOnnx;
                    lastStatusMessage = "Could not resolve ONNX asset path from manifest or fallback.";
                    Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                    OnModelLoadFailed?.Invoke(lastStatusMessage);
                    return;
                }

                // Refresh so Unity imports the new/changed .onnx as a ModelAsset.
                status = CheckpointSwapStatus.RefreshingAssetDatabase;
                Debug.Log($"[Checkpoint Swap] Refreshing AssetDatabase for {onnxAssetPath}...");
                AssetDatabase.Refresh();

                var newModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnxAssetPath);
                if (newModel == null)
                {
                    status = CheckpointSwapStatus.ErrorImportFailed;
                    lastStatusMessage = $"Failed to import ONNX as ModelAsset: {onnxAssetPath}";
                    Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                    OnModelLoadFailed?.Invoke(lastStatusMessage);
                    return;
                }

                // Optional behavior-name validation.
                if (!string.IsNullOrEmpty(expectedBehaviorName))
                {
                    bool namesMatch = string.Equals(
                        bp.BehaviorName,
                        expectedBehaviorName,
                        StringComparison.OrdinalIgnoreCase);

                    if (!namesMatch)
                    {
                        string msg =
                            $"Behavior name mismatch: expected '{expectedBehaviorName}', found '{bp.BehaviorName}'.";
                        if (strictCheckBehaviorName)
                        {
                            status = CheckpointSwapStatus.ErrorBehaviorMismatch;
                            lastStatusMessage = msg;
                            Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                            OnModelLoadFailed?.Invoke(lastStatusMessage);
                            return;
                        }
                        else
                        {
                            Debug.LogWarning($"[Checkpoint Swap] {msg} (strictCheckBehaviorName is off, continuing)");
                        }
                    }
                }

                // Save previous model and assign new. If anything below throws, old model stays.
                var previousModel = bp.Model;
                bp.Model = newModel;
                lastLoadedModelName = newModel.name;

                status = CheckpointSwapStatus.ModelReady;
                lastStatusMessage =
                    $"Successfully loaded model '{newModel.name}' into '{frozenAgent.name}'. " +
                    $"Previous model was '{(previousModel != null ? previousModel.name : "null")}'.";
                Debug.Log($"[Checkpoint Swap] {lastStatusMessage}");
                OnModelLoadSuccess?.Invoke();
            }
            catch (Exception ex)
            {
                status = CheckpointSwapStatus.ErrorImportFailed;
                lastStatusMessage = $"Exception during model load: {ex.Message}";
                Debug.LogError($"[Checkpoint Swap] {lastStatusMessage}");
                OnModelLoadFailed?.Invoke(lastStatusMessage);
            }
        }

        string ResolveOnnxAssetPath()
        {
            string manifestFullPath = Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), manifestPath));

            var manifest = CheckpointReadyManifest.Load(manifestFullPath);
            if (manifest != null && !string.IsNullOrEmpty(manifest.unity_onnx_path))
            {
                string relative = MakeRelativeAssetPath(manifest.unity_onnx_path);
                if (!string.IsNullOrEmpty(relative))
                {
                    return relative;
                }
            }

            return fallbackOnnxAssetPath;
        }

        static string MakeRelativeAssetPath(string fullPath)
        {
            string assetsPath = Path.GetFullPath(Application.dataPath);
            string projectPath = Path.GetDirectoryName(assetsPath);
            string normalized = Path.GetFullPath(fullPath).Replace('/', '\\');
            string normalizedProject = projectPath.Replace('/', '\\');

            if (normalized.StartsWith(normalizedProject + "\\", StringComparison.OrdinalIgnoreCase))
            {
                return normalized.Substring(normalizedProject.Length + 1).Replace('\\', '/');
            }

            return fullPath.Replace('\\', '/');
        }
#endif
    }
}
