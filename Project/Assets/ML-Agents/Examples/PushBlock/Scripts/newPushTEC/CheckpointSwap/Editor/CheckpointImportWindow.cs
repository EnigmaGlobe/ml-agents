using System.IO;
using UnityEngine;
using UnityEditor;

namespace PushTEvolutionMvp.CheckpointSwap.Editor
{
    /// <summary>
    /// Editor window for manually importing the latest ONNX checkpoint into Unity.
    /// Menu path: Tools &gt; PushT &gt; Import Latest ONNX.
    /// </summary>
    public class CheckpointImportWindow : EditorWindow
    {
        string m_SourceDirectory = "";
        string m_SearchPattern = "*.onnx";
        string m_DestinationOnnxPath = "Assets/Models/FrozenEvaluator/FrozenEvaluator_latest.onnx";
        string m_ManifestPath = "Assets/Models/FrozenEvaluator/checkpoint_ready.json";
        string m_HistoryDirectory = "Assets/Models/FrozenEvaluator/checkpoint_history";
        bool m_KeepHistory = true;
        bool m_AutoLoadModel = true;

        CheckpointSwapStatus m_Status = CheckpointSwapStatus.Idle;
        string m_StatusMessage = "Idle";

        [MenuItem("Tools/PushT/Import Latest ONNX")]
        static void OpenWindow()
        {
            var window = GetWindow<CheckpointImportWindow>("PushT Checkpoint Import");
            window.minSize = new Vector2(480, 360);
            window.Show();
        }

        void OnEnable()
        {
            TryAutoDetectSourceDirectory();
        }

        void TryAutoDetectSourceDirectory()
        {
            if (!string.IsNullOrEmpty(m_SourceDirectory))
            {
                return;
            }

            string projectRoot = Path.GetDirectoryName(Application.dataPath);
            string[] candidates = new[]
            {
                Path.Combine(projectRoot, "results"),
                Path.Combine(projectRoot, "config", "results"),
                Path.Combine(projectRoot, "Project", "results"),
                Path.Combine(projectRoot, "ml-agents", "results"),
            };

            foreach (var candidate in candidates)
            {
                if (Directory.Exists(candidate))
                {
                    m_SourceDirectory = candidate;
                    break;
                }
            }
        }

        void OnGUI()
        {
            GUILayout.Label("PushT Checkpoint Import (Version 2)", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            EditorGUILayout.BeginHorizontal();
            m_SourceDirectory = EditorGUILayout.TextField("Source Results Folder", m_SourceDirectory);
            if (GUILayout.Button("Browse", GUILayout.Width(80)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select ML-Agents results folder",
                    Directory.Exists(m_SourceDirectory) ? m_SourceDirectory : Application.dataPath,
                    "");
                if (!string.IsNullOrEmpty(selected))
                {
                    m_SourceDirectory = selected;
                }
            }
            EditorGUILayout.EndHorizontal();

            m_SearchPattern = EditorGUILayout.TextField("Search Pattern", m_SearchPattern);
            m_DestinationOnnxPath = EditorGUILayout.TextField("Destination ONNX Path", m_DestinationOnnxPath);
            m_ManifestPath = EditorGUILayout.TextField("Manifest Path", m_ManifestPath);
            m_HistoryDirectory = EditorGUILayout.TextField("History Directory", m_HistoryDirectory);
            m_KeepHistory = EditorGUILayout.Toggle("Keep History", m_KeepHistory);
            m_AutoLoadModel = EditorGUILayout.Toggle("Auto Load Into FrozenEvaluator", m_AutoLoadModel);

            EditorGUILayout.Space();

            GUI.enabled = !string.IsNullOrEmpty(m_SourceDirectory);
            if (GUILayout.Button("Import Latest ONNX", GUILayout.Height(32)))
            {
                ImportLatest();
            }
            GUI.enabled = true;

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("Status", m_Status.ToString(), EditorStyles.boldLabel);
            MessageType mt = m_Status.ToString().StartsWith("Error")
                ? MessageType.Error
                : (m_Status == CheckpointSwapStatus.ModelReady ? MessageType.Info : MessageType.Warning);
            EditorGUILayout.HelpBox(m_StatusMessage, mt);
        }

        void ImportLatest()
        {
            m_Status = CheckpointSwapStatus.SearchingForCheckpoint;
            m_StatusMessage = $"Searching {m_SourceDirectory} for '{m_SearchPattern}'...";
            Repaint();

            var copyResult = CheckpointSafeCopier.CopyLatestCheckpoint(
                m_SourceDirectory,
                m_SearchPattern,
                m_DestinationOnnxPath,
                GetFullManifestPath(),
                m_KeepHistory,
                GetFullHistoryPath()
            );

            if (!copyResult.success)
            {
                m_Status = CheckpointSwapStatus.ErrorMissingOnnx;
                m_StatusMessage = copyResult.errorMessage;
                Debug.LogError($"[Checkpoint Swap] {m_StatusMessage}");
                Repaint();
                return;
            }

            m_Status = CheckpointSwapStatus.CopyingCheckpoint;
            m_StatusMessage = $"Copied checkpoint from {copyResult.sourcePath} to {copyResult.destinationPath}";
            Debug.Log($"[Checkpoint Swap] {m_StatusMessage}");
            Repaint();

            m_Status = CheckpointSwapStatus.WritingManifest;
            m_StatusMessage =
                $"Manifest written: {copyResult.manifest.copied_at}, step={copyResult.manifest.checkpoint_step}, run_id={copyResult.manifest.run_id}";
            Debug.Log($"[Checkpoint Swap] {m_StatusMessage}");
            Repaint();

            if (m_AutoLoadModel)
            {
                var loader = FindLoader();
                if (loader == null)
                {
                    m_Status = CheckpointSwapStatus.ErrorMissingFrozenEvaluator;
                    m_StatusMessage = "No FrozenEvaluatorModelLoader found in scene. ONNX copied but not loaded.";
                    Debug.LogError($"[Checkpoint Swap] {m_StatusMessage}");
                    Repaint();
                    return;
                }

                Debug.Log($"[Checkpoint Swap] Auto-loading via {loader.name}");
                loader.LoadLatestModel();
                m_Status = loader.status;
                m_StatusMessage = loader.lastStatusMessage;
            }
            else
            {
                m_Status = CheckpointSwapStatus.CheckpointDetected;
                m_StatusMessage = "ONNX copied. Enable Auto Load or use a CheckpointWatcher to load it.";
            }

            Repaint();
        }

        string GetFullManifestPath()
        {
            return Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), m_ManifestPath));
        }

        string GetFullHistoryPath()
        {
            return Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), m_HistoryDirectory));
        }

        /// <summary>
        /// Robustly finds the first FrozenEvaluatorModelLoader in loaded scenes,
        /// including inactive GameObjects. Falls back to FindFirstObjectByType.
        /// </summary>
        static FrozenEvaluatorModelLoader FindLoader()
        {
            var all = Resources.FindObjectsOfTypeAll(typeof(FrozenEvaluatorModelLoader));
            if (all != null && all.Length > 0)
            {
                foreach (var obj in all)
                {
                    var loader = obj as FrozenEvaluatorModelLoader;
                    if (loader != null && !string.IsNullOrEmpty(loader.name))
                    {
                        return loader;
                    }
                }
            }

#if UNITY_2021_2_OR_NEWER
            return FindFirstObjectByType<FrozenEvaluatorModelLoader>(FindObjectsInactive.Include);
#else
            return FindObjectOfType<FrozenEvaluatorModelLoader>();
#endif
        }
    }
}
