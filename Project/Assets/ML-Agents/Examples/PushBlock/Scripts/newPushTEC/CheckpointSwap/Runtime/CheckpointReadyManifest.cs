using System;
using System.IO;
using UnityEngine;

namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// Marker file written only after the ONNX file has been fully copied into Unity.
    /// Unity components watch/load this manifest instead of polling the raw .onnx file.
    /// </summary>
    [Serializable]
    public class CheckpointReadyManifest
    {
        public string run_id = "";
        public long checkpoint_step = -1;
        public string source_onnx_path = "";
        public string unity_onnx_path = "";
        public string copied_at = "";
        public string status = "ready";

        public static CheckpointReadyManifest Load(string path)
        {
            if (!File.Exists(path))
            {
                return null;
            }

            try
            {
                string json = File.ReadAllText(path);
                var manifest = JsonUtility.FromJson<CheckpointReadyManifest>(json);
                return manifest;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Checkpoint Swap] Failed to load manifest from {path}: {ex.Message}");
                return null;
            }
        }

        public static void Save(string path, CheckpointReadyManifest manifest)
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
