using System;
using System.IO;
using UnityEngine;

namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// Error manifest written when Curriculum EC validation or execution fails.
    /// </summary>
    [Serializable]
    public class CurriculumECErrorManifest
    {
        public string request_id = "";
        public int stage = -1;
        public string status = "error";
        public string error_code = "";
        public string message = "";
        public string created_at = "";

        public static CurriculumECErrorManifest Load(string path)
        {
            if (!File.Exists(path))
            {
                return null;
            }

            try
            {
                string json = File.ReadAllText(path);
                return JsonUtility.FromJson<CurriculumECErrorManifest>(json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Curriculum Orchestration] Failed to load error manifest from {path}: {ex.Message}");
                return null;
            }
        }

        public static void Save(string path, CurriculumECErrorManifest manifest)
        {
            if (manifest == null)
            {
                throw new ArgumentNullException(nameof(manifest));
            }

            WriteSafe(path, JsonUtility.ToJson(manifest, prettyPrint: true));
        }

        static void WriteSafe(string path, string json)
        {
            string directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            string tempPath = path + ".tmp";
            File.WriteAllText(tempPath, json);

            var tempInfo = new FileInfo(tempPath);
            if (!tempInfo.Exists || tempInfo.Length == 0)
            {
                SafeDelete(tempPath);
                throw new IOException("Failed to write error manifest temp file.");
            }

            if (File.Exists(path))
            {
                File.Replace(tempPath, path, path + ".prev");
            }
            else
            {
                File.Move(tempPath, path);
            }
        }

        static void SafeDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch { }
        }
    }
}
