using System;
using System.IO;
using UnityEngine;

namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// Incoming request manifest written by an external orchestrator.
    /// Unity watches for this file and triggers Curriculum EC when status is "pending"
    /// and action is "run_curriculum_ec".
    /// </summary>
    [Serializable]
    public class CurriculumECRequestManifest
    {
        public string request_id = "";
        public int stage = -1;
        public string run_id = "";
        public long checkpoint_step = -1;
        public string onnx_path = "";
        public string requested_at = "";
        public string action = "";
        public string status = "pending";
        public string expected_behavior_name = "";

        public bool IsRunCurriculumRequest =>
            string.Equals(action, "run_curriculum_ec", StringComparison.OrdinalIgnoreCase) &&
            string.Equals(status, "pending", StringComparison.OrdinalIgnoreCase);

        public static CurriculumECRequestManifest Load(string path)
        {
            if (!File.Exists(path))
            {
                return null;
            }

            try
            {
                string json = File.ReadAllText(path);
                return JsonUtility.FromJson<CurriculumECRequestManifest>(json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Curriculum Orchestration] Failed to load request manifest from {path}: {ex.Message}");
                return null;
            }
        }

        public static void Save(string path, CurriculumECRequestManifest manifest)
        {
            if (manifest == null)
            {
                throw new ArgumentNullException(nameof(manifest));
            }

            string directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            string json = JsonUtility.ToJson(manifest, prettyPrint: true);
            File.WriteAllText(path, json);
        }
    }
}
