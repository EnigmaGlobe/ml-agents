#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Editor utility to save GenomePool to JSON and verify asset serialization.
    /// </summary>
    public static class GenomePoolEditor
    {
        [MenuItem("Assets/PushT/Save GenomePool to JSON", true)]
        static bool ValidateSavePool()
        {
            return Selection.activeObject is GenomePool;
        }

        [MenuItem("Assets/PushT/Save GenomePool to JSON")]
        static void SavePoolToJson()
        {
            var pool = Selection.activeObject as GenomePool;
            if (pool == null) return;

            var outputDir = System.IO.Path.Combine(
                Application.dataPath,
                "ML-Agents",
                "Examples",
                "PushBlock",
                "Scripts",
                "newPushTEC",
                "curriculum_outputs");

            System.IO.Directory.CreateDirectory(outputDir);

            var timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var jsonPath = System.IO.Path.Combine(outputDir, $"genome_pool_manual_{timestamp}.json");

            pool.SaveToJson(jsonPath);

            // Also save to a known location that training can load
            var defaultPath = System.IO.Path.Combine(outputDir, "genome_pool_latest.json");
            pool.SaveToJson(defaultPath);

            Debug.Log($"[GenomePool Editor] Saved {pool.Count} genomes to:");
            Debug.Log($"  - {jsonPath}");
            Debug.Log($"  - {defaultPath}");

            EditorUtility.SetDirty(pool);
            AssetDatabase.SaveAssets();
        }

        [MenuItem("Assets/PushT/Log GenomePool Info")]
        static void LogPoolInfo()
        {
            var pool = Selection.activeObject as GenomePool;
            if (pool == null) return;

            Debug.Log($"[GenomePool Info] Generation: {pool.generationId}");
            Debug.Log($"[GenomePool Info] Genome Count: {pool.Count}");
            Debug.Log($"[GenomePool Info] Target SR Range: {pool.targetSuccessRateMin:P0} - {pool.targetSuccessRateMax:P0}");
            Debug.Log($"[GenomePool Info] Eval Episodes Per Genome: {pool.evalEpisodesPerGenome}");

            if (pool.genomes != null && pool.genomes.Count > 0)
            {
                Debug.Log($"[GenomePool Info] First genome sample:");
                var g = pool.genomes[0];
                Debug.Log($"  width={g.width:F2}, height={g.height:F2}, depth={g.depth:F2}");
                Debug.Log($"  mass={g.mass:F2}, friction={g.friction:F2}, drag={g.blockDrag:F2}, bounce={g.bounciness:F2}");
            }
        }
    }
}
#endif
