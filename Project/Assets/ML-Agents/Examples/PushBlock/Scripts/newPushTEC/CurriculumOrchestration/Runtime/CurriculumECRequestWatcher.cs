using System;
using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// Watches ec_request.json and dispatches processing to CurriculumECRequestProcessor
    /// on the Unity main thread. Uses debounce to avoid duplicate triggers.
    /// </summary>
    public class CurriculumECRequestWatcher : MonoBehaviour
    {
        [Header("Settings")]
        public CurriculumECOrchestrationSettings settings;

        [Header("Processor")]
        public CurriculumECRequestProcessor processor;

        [Header("Status (read-only)")]
        public bool isWatching = false;
        public string lastDetectedRequestId = "";

        FileSystemWatcher m_Watcher;
        DateTime m_LastEventUtc = DateTime.MinValue;
        bool m_ProcessRequested = false;
        bool m_UpdateRegistered = false;
        bool m_StartupCheckPending = false;

        string RequestManifestFullPath => ResolveFullPath(settings != null ? settings.requestManifestPath : "Assets/Models/FrozenEvaluator/ec_request.json");
        float DebounceSeconds => settings != null ? settings.debounceSeconds : 1.0f;

        void OnEnable()
        {
#if UNITY_EDITOR
            StartWatching();
#endif
        }

#if UNITY_EDITOR
        [MenuItem("Tools/PushT/Start EC Request Watcher")]
        static void MenuStartWatcher()
        {
            var watcher = FindFirstObjectByType<CurriculumECRequestWatcher>(FindObjectsInactive.Include);
            if (watcher == null)
            {
                Debug.LogError("[Curriculum Orchestration] No CurriculumECRequestWatcher found in scene.");
                return;
            }
            watcher.StartWatching();
            Debug.Log("[Curriculum Orchestration] EC Request Watcher started from menu.");
        }

        [MenuItem("Tools/PushT/Stop EC Request Watcher")]
        static void MenuStopWatcher()
        {
            var watcher = FindFirstObjectByType<CurriculumECRequestWatcher>(FindObjectsInactive.Include);
            if (watcher == null)
            {
                Debug.LogError("[Curriculum Orchestration] No CurriculumECRequestWatcher found in scene.");
                return;
            }
            watcher.StopWatching();
            Debug.Log("[Curriculum Orchestration] EC Request Watcher stopped from menu.");
        }
#endif

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

            string fullPath = RequestManifestFullPath;
            string directory = Path.GetDirectoryName(fullPath);
            string fileName = Path.GetFileName(fullPath);

            if (!Directory.Exists(directory))
            {
                Debug.LogWarning($"[Curriculum Orchestration] Cannot watch missing directory: {directory}");
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
            m_StartupCheckPending = true;
            Debug.Log($"[Curriculum Orchestration] Watching request manifest: {fullPath}");
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
            // Never touch Unity objects here; this runs on a FileSystemWatcher thread.
            m_LastEventUtc = DateTime.UtcNow;
            m_ProcessRequested = true;
        }

        void OnWatcherError(object sender, ErrorEventArgs e)
        {
            Debug.LogWarning($"[Curriculum Orchestration] FileSystemWatcher error: {e.GetException()?.Message}");
        }

        void OnEditorUpdate()
        {
            if (m_StartupCheckPending)
            {
                m_StartupCheckPending = false;
                CheckPendingRequestOnStartup();
            }

            if (!m_ProcessRequested)
            {
                return;
            }

            double elapsed = (DateTime.UtcNow - m_LastEventUtc).TotalSeconds;
            if (elapsed < DebounceSeconds)
            {
                return;
            }

            m_ProcessRequested = false;

            var request = CurriculumECRequestManifest.Load(RequestManifestFullPath);
            if (request == null)
            {
                Debug.LogWarning("[Curriculum Orchestration] Request manifest event received but file could not be read.");
                return;
            }

            if (!request.IsRunCurriculumRequest)
            {
                Debug.Log($"[Curriculum Orchestration] Ignoring request '{request.request_id}' (action='{request.action}', status='{request.status}').");
                return;
            }

            if (string.Equals(request.request_id, lastDetectedRequestId, StringComparison.Ordinal))
            {
                Debug.Log($"[Curriculum Orchestration] Already detected request '{request.request_id}'. Ignoring.");
                return;
            }

            lastDetectedRequestId = request.request_id;
            Debug.Log($"[Curriculum Orchestration] Detected request '{request.request_id}' (stage {request.stage}).");

            if (processor == null)
            {
                processor = FindProcessor();
            }

            if (processor == null)
            {
                Debug.LogError("[Curriculum Orchestration] No CurriculumECRequestProcessor found in scene.");
                return;
            }

            processor.ProcessRequest(request);
        }

        void CheckPendingRequestOnStartup()
        {
            var request = CurriculumECRequestManifest.Load(RequestManifestFullPath);
            if (request == null)
            {
                return;
            }

            if (!request.IsRunCurriculumRequest)
            {
                Debug.Log($"[Curriculum Orchestration] Startup check: existing request '{request.request_id}' is not pending (action='{request.action}', status='{request.status}').");
                return;
            }

            if (string.Equals(request.request_id, lastDetectedRequestId, StringComparison.Ordinal))
            {
                return;
            }

            Debug.Log($"[Curriculum Orchestration] Startup check: pending request '{request.request_id}' found. Will process.");
            m_ProcessRequested = true;
            m_LastEventUtc = DateTime.UtcNow - TimeSpan.FromSeconds(DebounceSeconds + 1f);
        }

        static CurriculumECRequestProcessor FindProcessor()
        {
            var all = Resources.FindObjectsOfTypeAll(typeof(CurriculumECRequestProcessor));
            if (all != null && all.Length > 0)
            {
                foreach (var obj in all)
                {
                    var processor = obj as CurriculumECRequestProcessor;
                    if (processor != null && !string.IsNullOrEmpty(processor.name))
                    {
                        return processor;
                    }
                }
            }

#if UNITY_2021_2_OR_NEWER
            return FindFirstObjectByType<CurriculumECRequestProcessor>(FindObjectsInactive.Include);
#else
            return FindObjectOfType<CurriculumECRequestProcessor>();
#endif
        }

        string ResolveFullPath(string assetRelativePath)
        {
            return Path.GetFullPath(
                Path.Combine(Path.GetDirectoryName(Application.dataPath), assetRelativePath));
        }
    }
}
