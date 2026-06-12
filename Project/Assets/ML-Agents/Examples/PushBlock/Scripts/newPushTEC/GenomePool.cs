using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Runtime + persistent storage for curriculum training genomes.
    /// Used as a ScriptableObject for Inspector visibility and drag-and-drop,
    /// and serialized to JSON for cross-session persistence and version control.
    /// </summary>
    [CreateAssetMenu(fileName = "GenomePool", menuName = "PushT/GenomePool")]
    public class GenomePool : ScriptableObject
    {
        [Header("Pool Contents")]
        public List<PushTBlockGenome> genomes = new List<PushTBlockGenome>();

        [Header("Metadata")]
        public int generationId = 0;
        public float targetSuccessRateMin = 0.30f;
        public float targetSuccessRateMax = 0.75f;
        public int evalEpisodesPerGenome = 10;

        public bool IsEmpty => genomes == null || genomes.Count == 0;
        public int Count => genomes?.Count ?? 0;

        public PushTBlockGenome SampleRandom(System.Random random)
        {
            if (IsEmpty) return null;
            int idx = random.Next(genomes.Count);
            var src = genomes[idx];
            Debug.Log($"[GenomePool-DIAG] Sampled index {idx}/{genomes.Count}, src is {(src == null ? "NULL" : $"w={src.width:F2}")}");
            return src?.Clone() ?? new PushTBlockGenome();
        }

        /// <summary>
        /// Sample a genome with optional fallback mixing.
        /// fallbackRatio is 0 by default so the pool is used exclusively unless the caller
        /// explicitly requests fallback blending (e.g., when the pool has fewer than 5 genomes).
        /// </summary>
        public PushTBlockGenome SampleWithFallback(System.Random random, PushTBlockGenome fallback, float fallbackRatio = 0f)
        {
            if (IsEmpty) return fallback?.Clone();
            if (fallback == null) return SampleRandom(random);

            bool useFallback = random.NextDouble() < Mathf.Clamp01(fallbackRatio);
            return useFallback ? fallback.Clone() : SampleRandom(random);
        }

        public void Clear()
        {
            genomes?.Clear();
        }

        public void Add(PushTBlockGenome genome)
        {
            if (genomes == null) genomes = new List<PushTBlockGenome>();
            genomes.Add(genome.Clone());
        }

        #region JSON Serialization

        [Serializable]
        private class GenomePoolJsonData
        {
            public List<PushTBlockGenome> genomes;
            public int generationId;
            public float targetSuccessRateMin;
            public float targetSuccessRateMax;
            public int evalEpisodesPerGenome;
        }

        public void SaveToJson(string path)
        {
            var data = new GenomePoolJsonData
            {
                genomes = new List<PushTBlockGenome>(),
                generationId = this.generationId,
                targetSuccessRateMin = this.targetSuccessRateMin,
                targetSuccessRateMax = this.targetSuccessRateMax,
                evalEpisodesPerGenome = this.evalEpisodesPerGenome
            };

            if (this.genomes != null)
            {
                foreach (var g in this.genomes)
                {
                    data.genomes.Add(g.Clone());
                }
            }

            var json = JsonUtility.ToJson(data, prettyPrint: true);
            File.WriteAllText(path, json);
            Debug.Log($"[GenomePool] Saved {data.genomes.Count} genomes to {path}");
        }

        public void LoadFromJson(string path)
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[GenomePool] JSON not found at {path}. Pool unchanged.");
                return;
            }

            var json = File.ReadAllText(path);
            var data = JsonUtility.FromJson<GenomePoolJsonData>(json);

            if (data == null)
            {
                Debug.LogError($"[GenomePool] Failed to parse JSON at {path}.");
                return;
            }

            this.genomes = new List<PushTBlockGenome>();
            if (data.genomes != null)
            {
                foreach (var g in data.genomes)
                {
                    this.genomes.Add(g?.Clone() ?? new PushTBlockGenome());
                }
            }

            this.generationId = data.generationId;
            this.targetSuccessRateMin = data.targetSuccessRateMin;
            this.targetSuccessRateMax = data.targetSuccessRateMax;
            this.evalEpisodesPerGenome = data.evalEpisodesPerGenome;

            Debug.Log($"[GenomePool] Loaded {this.genomes.Count} genomes from {path} (generation {this.generationId}).");
        }

        #endregion

#if UNITY_EDITOR
        [ContextMenu("Save Pool to JSON")]
        public void Editor_SaveToJson()
        {
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
            
            SaveToJson(jsonPath);
            
            // Also save to a standard location
            var defaultPath = System.IO.Path.Combine(outputDir, "genome_pool_latest.json");
            SaveToJson(defaultPath);
            
            UnityEditor.EditorUtility.SetDirty(this);
            UnityEditor.AssetDatabase.SaveAssets();
        }

        [ContextMenu("Load Pool from JSON")]
        public void Editor_LoadFromJson()
        {
            var defaultPath = System.IO.Path.Combine(
                Application.dataPath,
                "ML-Agents",
                "Examples",
                "PushBlock",
                "Scripts",
                "newPushTEC",
                "curriculum_outputs",
                "genome_pool_latest.json");
            
            LoadFromJson(defaultPath);
            
            UnityEditor.EditorUtility.SetDirty(this);
            UnityEditor.AssetDatabase.SaveAssets();
        }
#endif
    }
}
