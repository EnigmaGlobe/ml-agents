using System;
using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// Watches checkpoint_ready.json (not the ONNX file) for changes and triggers
    /// FrozenEvaluatorModelLoader on the main Unity thread. Uses debounce to avoid
    /// duplicate reloads. Editor-only.
    /// </summary>
    public class CheckpointWatcher : MonoBehaviour
    {
        [Header("Manifest to watch")]
        [Tooltip("Asset-relative path to checkpoint_ready.json.")]
        public string manifestPath = "Assets/Models/FrozenEvaluator/checkpoint_ready.json";

        [Header("Loader")]
        public FrozenEvaluatorModelLoader modelLoader;

        [Header("Debounce")]
        public float debounceSeconds = 1.0f;

        [Header("Status")]
        public bool isWatching = false;

        FileSystemWatcher m_Watcher;
        DateTime m_LastEventUtc = DateTime.MinValue;
        bool m_ReloadRequested = false;
        bool m_UpdateRegistered = false;

        void OnEnable()
        {
#if UNITY_EDITOR
            StartWatching();
#endif
        }

        void OnDisable()
        {
            StopWatching();
        }

        void OnDestroy()
        {
            StopWatching();
        }

        public void StartWatching()
        {
#if UNITY_EDITOR
            StopWatching();

            string fullManifestPath = GetFullManifestPath();
            string directory = Path.GetDirectoryName(fullManifestPath);
            string fileName = Path.GetFileName(fullManifestPath);

            if (!Directory.Exists(directory))
            {
                Debug.LogWarning($"[Checkpoint Swap] Cannot watch missing directory: {directory}");
                isWatching = false;
                return;
            }

            m_Watcher = new FileSystemWatcher(directory, fileName)
            {
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.CreationTime | NotifyFilters.FileName,
                EnableRaisingEvents = true,
                IncludeSubdirectories = false
            };

            m_Watcher.Changed += OnManifestEvent;
            m_Watcher.Created += OnManifestEvent;
            m_Watcher.Renamed += OnManifestEvent;
            m_Watcher.Error += OnWatcherError;

            if (!m_UpdateRegistered)
            {
                EditorApplication.update += OnEditorUpdate;
                m_UpdateRegistered = true;
            }

            isWatching = true;
            Debug.Log($"[Checkpoint Swap] Watching manifest: {fullManifestPath}");
#endif
        }

        public void StopWatching()
        {
            if (m_Watcher != null)
            {
                m_Watcher.Changed -= OnManifestEvent;
                m_Watcher.Created -= OnManifestEvent;
                m_Watcher.Renamed -= OnManifestEvent;
                m_Watcher.Error -= OnWatcherError;
                m_Watcher.EnableRaisingEvents = false;
                m_Watcher.Dispose();
                m_Watcher = null;
            }

#if UNITY_EDITOR
            if (m_UpdateRegistered)
            {
                EditorApplication.update -= OnEditorUpdate;
                m_UpdateRegistered = false;
            }
#endif

            isWatching = false;
        }

        void OnManifestEvent(object sender, FileSystemEventArgs e)
        {
            // NEVER touch Unity objects here: this callback runs on a FileSystemWatcher thread.
            m_LastEventUtc = DateTime.UtcNow;
            m_ReloadRequested = true;
        }

        void OnWatcherError(object sender, ErrorEventArgs e)
        {
            Debug.LogWarning($"[Checkpoint Swap] FileSystemWatcher error: {e.GetException()?.Message}");
        }

        void OnEditorUpdate()
        {
            if (!m_ReloadRequested)
            {
                return;
            }

            double elapsed = (DateTime.UtcNow - m_LastEventUtc).TotalSeconds;
            if (elapsed < debounceSeconds)
            {
                return;
            }

            m_ReloadRequested = false;

            string fullManifestPath = GetFullManifestPath();
            var manifest = CheckpointReadyManifest.Load(fullManifestPath);
            if (manifest == null || !string.Equals(manifest.status, "ready", StringComparison.OrdinalIgnoreCase))
            {
                Debug.LogWarning("[Checkpoint Swap] Manifest event received but manifest is not ready. Ignoring.");
                return;
            }

            if (modelLoader == null)
            {
                modelLoader = FindLoader();
            }

            if (modelLoader == null)
            {
                Debug.LogError("[Checkpoint Swap] No FrozenEvaluatorModelLoader found in scene. Cannot reload model.");
                return;
            }

            Debug.Log($"[Checkpoint Swap] Manifest change detected; reloading via {modelLoader.name}...");
            modelLoader.LoadLatestModel();
        }

        string GetFullManifestPath()
        {
            return Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), manifestPath));
        }

        /// <summary>
        /// Robustly finds the first FrozenEvaluatorModelLoader in loaded scenes,
        /// including inactive GameObjects.
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
