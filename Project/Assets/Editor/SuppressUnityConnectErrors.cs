#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using System.Reflection;
using System;

[InitializeOnLoad]
public static class SuppressUnityConnectErrors
{
    private static float m_nextCheckTime = 5f;
    private static bool m_autoClean = false; // disabled — was clearing debug logs every 5s

    static SuppressUnityConnectErrors()
    {
        EditorApplication.update += OnEditorUpdate;
        Application.logMessageReceived += OnLogMessageReceived;
    }

    private static void OnLogMessageReceived(string condition, string stacktrace, LogType type)
    {
        if (condition != null && condition.Contains("UnityConnectWebRequestException"))
        {
            // Mark for cleanup on next frame
            m_nextCheckTime = Time.realtimeSinceStartup + 0.1f;
        }
    }

    private static void OnEditorUpdate()
    {
        if (!m_autoClean) return;
        if (Time.realtimeSinceStartup < m_nextCheckTime) return;
        m_nextCheckTime = Time.realtimeSinceStartup + 5f;

        CleanUnityConnectErrors();
    }

    [MenuItem("Tools/Clean UnityConnect Errors")]
    public static void CleanUnityConnectErrors()
    {
        try
        {
            // Use reflection to access UnityEditor.LogEntries
            var logEntriesType = typeof(EditorWindow).Assembly.GetType("UnityEditor.LogEntries");
            if (logEntriesType == null)
            {
                // Fallback: clear entire console
                ClearConsole();
                return;
            }

            var getCountMethod = logEntriesType.GetMethod("GetEntryCount", BindingFlags.Static | BindingFlags.Public);
            if (getCountMethod == null)
            {
                ClearConsole();
                return;
            }

            int count = (int)getCountMethod.Invoke(null, null);
            if (count == 0) return;

            // Check if any entry contains UnityConnect error
            bool hasConnectError = false;
            var getEntryMethod = logEntriesType.GetMethod("GetEntryInternal", BindingFlags.Static | BindingFlags.Public);
            if (getEntryMethod != null)
            {
                var entryType = typeof(EditorWindow).Assembly.GetType("UnityEditor.LogEntry");
                if (entryType != null)
                {
                    var entry = Activator.CreateInstance(entryType);
                    var conditionField = entryType.GetField("condition", BindingFlags.Public | BindingFlags.Instance);
                    for (int i = 0; i < count; i++)
                    {
                        getEntryMethod.Invoke(null, new object[] { i, entry });
                        if (conditionField != null)
                        {
                            string condition = (string)conditionField.GetValue(entry);
                            if (condition != null && condition.Contains("UnityConnectWebRequestException"))
                            {
                                hasConnectError = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (hasConnectError)
            {
                ClearConsole();
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[SuppressUnityConnect] Could not clean specific entries, will clear console: " + e.Message);
            ClearConsole();
        }
    }

    private static void ClearConsole()
    {
        try
        {
            var logEntriesType = typeof(EditorWindow).Assembly.GetType("UnityEditor.LogEntries");
            if (logEntriesType != null)
            {
                var clearMethod = logEntriesType.GetMethod("Clear", BindingFlags.Static | BindingFlags.Public);
                clearMethod?.Invoke(null, null);
            }
        }
        catch { }
    }

    [MenuItem("Tools/Toggle Auto-Clean UnityConnect Errors")]
    public static void ToggleAutoClean()
    {
        m_autoClean = !m_autoClean;
        EditorUtility.DisplayDialog("Auto-Clean", 
            m_autoClean ? "Auto-clean ENABLED" : "Auto-clean DISABLED", 
            "OK");
    }
}
#endif
