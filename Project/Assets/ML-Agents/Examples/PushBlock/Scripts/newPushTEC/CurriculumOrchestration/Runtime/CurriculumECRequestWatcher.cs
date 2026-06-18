using System;
using System.Collections.Generic;
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
    ///
    /// Uses a static EditorApplication.update callback so watching survives Play Mode
    /// transitions. This is important for the V3.2 training loop: mlagents-learn puts
    /// Unity into Play Mode; after training ends the watcher must still process the
    /// ec_request.json written from Python.
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

        static readonly List<CurriculumECRequestWatcher> s_RegisteredWatchers = new List<CurriculumECRequestWatcher>();
        static bool s_StaticUpdateRegistered = false;

        FileSystemWatcher m_Watcher;
        DateTime m_LastEventUtc = DateTime.MinValue;
        bool m_ProcessRequested = false;
        bool m_StartupCheckPending = false;

        // Fallback poll: FileSystemWatcher can miss events across Play Mode transitions
        // or under heavy load, so we also check the manifest file periodically.
        DateTime m_LastPollUtc = DateTime.MinValue;
        const float POLL_INTERVAL_SECONDS = 2.0f;

        string RequestManifestFullPath => ResolveFullPath(settings != null ? settings.requestManifestPath : "Temp/FrozenEvaluator/ec_request.json");
        float DebounceSeconds => settings != null ? settings.debounceSeconds : 1.0f;

#if UNITY_EDITOR
        [InitializeOnLoadMethod]
        static void InitializeOnLoad()
        {
            // Register one static callback that updates all active watchers.
            // This survives Play Mode enter/exit because it is not tied to a
            // MonoBehaviour instance lifetime.
            if (!s_StaticUpdateRegistered)
            {
                EditorApplication.update += StaticUpdate;
                EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
                s_StaticUpdateRegistered = true;
                Debug.Log("[Curriculum Orchestration] Static watcher update registered.");
            }

            // After a domain reload (script recompilation), OnEnable is NOT called again
            // for objects that were already enabled, so our static watcher list is empty.
            // Re-register them on the next editor update.
            EditorApplication.delayCall += ReRegisterWatchers;
        }

        static void ReRegisterWatchers()
        {
#if UNITY_2021_2_OR_NEWER
            var watchers = FindObjectsByType<CurriculumECRequestWatcher>(FindObjectsInactive.Include, FindObjectsSortMode.None);
#else
            var watchers = FindObjectsOfType<CurriculumECRequestWatcher>(true);
#endif

            lock (s_RegisteredWatchers)
            {
                foreach (var watcher in watchers)
                {
                    if (watcher == null) continue;
                    if (!s_RegisteredWatchers.Contains(watcher))
                    {
                        s_RegisteredWatchers.Add(watcher);
                    }
                    // Always restart watching after a domain reload to make sure the
                    // FileSystemWatcher and EditorApplication.update are wired up.
                    watcher.StartWatching();
                }
            }

            Debug.Log($"[Curriculum Orchestration] Re-registered {watchers.Length} watcher(s) after domain reload.");
        }

        static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.EnteredEditMode)
            {
                // After returning from Play Mode, re-check pending requests on all watchers.
                lock (s_RegisteredWatchers)
                {
                    int count = 0;
                    foreach (var watcher in s_RegisteredWatchers)
                    {
                        if (watcher == null) continue;
                        watcher.m_StartupCheckPending = true;
                        count++;
                    }
                    Debug.Log($"[Curriculum Orchestration] Returned to Edit Mode; {count} watcher(s) will re-check pending requests.");
                }
            }
            else if (state == PlayModeStateChange.ExitingPlayMode)
            {
                Debug.Log("[Curriculum Orchestration] Exiting Play Mode; watcher will resume polling in Edit Mode.");
            }
        }

        static void StaticUpdate()
        {
            lock (s_RegisteredWatchers)
            {
                for (int i = s_RegisteredWatchers.Count - 1; i >= 0; i--)
                {
                    var watcher = s_RegisteredWatchers[i];
                    if (watcher == null)
                    {
                        s_RegisteredWatchers.RemoveAt(i);
                        continue;
                    }

                    try
                    {
                        watcher.OnEditorUpdate();
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[Curriculum Orchestration] Watcher update error: {ex.Message}");
                    }
                }
            }
        }
#endif

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

            lock (s_RegisteredWatchers)
            {
                s_RegisteredWatchers.Remove(this);
            }
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

        [MenuItem("Tools/PushT/Force Poll EC Request")]
        static void MenuForcePollRequest()
        {
            var watcher = FindFirstObjectByType<CurriculumECRequestWatcher>(FindObjectsInactive.Include);
            if (watcher == null)
            {
                Debug.LogError("[Curriculum Orchestration] No CurriculumECRequestWatcher found in scene.");
                return;
            }
            watcher.PollRequestManifest();
        }

        [MenuItem("Tools/PushT/Force Process EC Request Now")]
        static void MenuForceProcessRequest()
        {
            var watcher = FindFirstObjectByType<CurriculumECRequestWatcher>(FindObjectsInactive.Include);
            if (watcher == null)
            {
                Debug.LogError("[Curriculum Orchestration] No CurriculumECRequestWatcher found in scene.");
                return;
            }
            watcher.ForceProcessRequestNow();
        }
#endif

        public void StartWatching()
        {
#if UNITY_EDITOR
            StopWatching();

            string fullPath = RequestManifestFullPath;
            string directory = Path.GetDirectoryName(fullPath);
            string fileName = Path.GetFileName(fullPath);

            if (!Directory.Exists(directory))
            {
                try
                {
                    Directory.CreateDirectory(directory);
                    Debug.Log($"[Curriculum Orchestration] Created manifest directory: {directory}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[Curriculum Orchestration] Cannot watch missing directory: {directory} ({ex.Message})");
                    isWatching = false;
                    return;
                }
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

            lock (s_RegisteredWatchers)
            {
                if (!s_RegisteredWatchers.Contains(this))
                {
                    s_RegisteredWatchers.Add(this);
                }
            }

            m_StartupCheckPending = true;
            isWatching = true;
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

            // NOTE: We intentionally do NOT remove this watcher from s_RegisteredWatchers here.
            // When exiting Play Mode, OnDisable is called; we want the watcher to remain
            // registered so the PlayModeStateChanged callback can set m_StartupCheckPending.
            // Removal happens in OnDestroy.
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

        void Update()
        {
            // Also poll during Play Mode; EditorApplication.update does not run while
            // Unity is playing, so requests written before Edit Mode is fully entered
            // would otherwise be missed.
            if (Application.isPlaying && (DateTime.UtcNow - m_LastPollUtc).TotalSeconds >= POLL_INTERVAL_SECONDS)
            {
                m_LastPollUtc = DateTime.UtcNow;
                PollRequestManifest();
            }
        }

        void OnEditorUpdate()
        {
            if (m_StartupCheckPending)
            {
                m_StartupCheckPending = false;
                CheckPendingRequestOnStartup();
            }

            // Periodic poll fallback in case FileSystemWatcher missed the write.
            if ((DateTime.UtcNow - m_LastPollUtc).TotalSeconds >= POLL_INTERVAL_SECONDS)
            {
                m_LastPollUtc = DateTime.UtcNow;
                PollRequestManifest();
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
            Debug.Log("[Curriculum Orchestration] Startup check: looking for pending request...");
            var request = CurriculumECRequestManifest.Load(RequestManifestFullPath);
            if (request == null)
            {
                Debug.Log("[Curriculum Orchestration] Startup check: no request manifest found.");
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

        /// <summary>
        /// Polls the request manifest file directly. Used as a fallback when the
        /// FileSystemWatcher does not raise an event (common across Play Mode transitions).
        /// </summary>
        public void PollRequestManifest()
        {
            string fullPath = RequestManifestFullPath;
            var request = CurriculumECRequestManifest.Load(fullPath);
            if (request == null)
            {
                return;
            }

            if (!request.IsRunCurriculumRequest)
            {
                return;
            }

            if (string.Equals(request.request_id, lastDetectedRequestId, StringComparison.Ordinal))
            {
                // Already detected; don't spam logs every poll interval.
                return;
            }

            if (processor == null)
            {
                processor = FindProcessor();
            }

            // Only skip if this exact request is the one already running. If a previous
            // EC run left IsRunning stuck, we still want to process a new request.
            if (processor != null && processor.curriculumEC != null && processor.curriculumEC.IsRunning)
            {
                if (string.Equals(request.request_id, processor.lastRequestId, StringComparison.Ordinal))
                {
                    Debug.Log($"[Curriculum Orchestration] Poll found request '{request.request_id}' but it is already running. Skipping.");
                    return;
                }
                Debug.LogWarning($"[Curriculum Orchestration] Poll found new request '{request.request_id}' while EC IsRunning is still true. Will process anyway; processor will reject if truly busy.");
            }

            Debug.Log($"[Curriculum Orchestration] Poll detected pending request '{request.request_id}'. Will process.");
            m_ProcessRequested = true;
            m_LastEventUtc = DateTime.UtcNow - TimeSpan.FromSeconds(DebounceSeconds + 1f);
        }

        /// <summary>
        /// Loads and processes the pending request immediately, bypassing the event queue.
        /// Useful for debugging when the watcher event pipeline is not responding.
        /// </summary>
        public void ForceProcessRequestNow()
        {
            string fullPath = RequestManifestFullPath;
            Debug.Log($"[Curriculum Orchestration] Force processing manifest: {fullPath} (exists={File.Exists(fullPath)})");

            var request = CurriculumECRequestManifest.Load(fullPath);
            if (request == null)
            {
                Debug.LogError("[Curriculum Orchestration] Force process: no manifest found or failed to load.");
                return;
            }

            Debug.Log($"[Curriculum Orchestration] Force process loaded: id='{request.request_id}', action='{request.action}', status='{request.status}'");

            if (!request.IsRunCurriculumRequest)
            {
                Debug.LogError($"[Curriculum Orchestration] Force process: invalid request (action='{request.action}', status='{request.status}').");
                return;
            }

            if (processor == null)
            {
                processor = FindProcessor();
            }

            if (processor == null)
            {
                Debug.LogError("[Curriculum Orchestration] Force process: no processor found in scene.");
                return;
            }

            lastDetectedRequestId = request.request_id;
            Debug.Log($"[Curriculum Orchestration] Force process: dispatching request '{request.request_id}' to processor.");
            processor.ProcessRequest(request);
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
