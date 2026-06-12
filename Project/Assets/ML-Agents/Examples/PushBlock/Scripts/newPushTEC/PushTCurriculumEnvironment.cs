using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// TrainingArena environment sampler.
    /// Applies a random genome from the GenomePool to the block and ground
    /// at the start of each training episode.
    /// </summary>
    public class PushTCurriculumEnvironment : MonoBehaviour
    {
        [Header("Genome Pool")]
        public GenomePool genomePool;
        public PushTBlockGenome fallbackGenome = new PushTBlockGenome
        {
            width = 0.8f,
            height = 0.8f,
            depth = 0.8f,
            mass = 0.6f,
            friction = 0.8f,
            bounciness = 0f,
            blockDrag = 0.2f
        };
        [Tooltip("Probability of using the fallback genome instead of the evolved pool. " +
                  "Keep this at 0 for normal curriculum training. Only raise it temporarily " +
                  "if the pool has fewer than 5 genomes.")]
        [Range(0f, 1f)] public float fallbackRatio = 0f;

        [Header("Scene References (Training Arena Only)")]
        public Transform blockTransform;
        public Rigidbody blockRigidbody;
        public Collider groundCollider;

        [Header("Optional: Training Agent")]
        public PushAgentBasic trainingAgent;

        private PhysicsMaterial m_runtimeGroundMaterial;
        private System.Random m_random;
        private int m_lastCompletedEpisodes = -1;
        private bool m_eventSubscribed = false;

        /// <summary>
        /// The last genome that was applied to the scene.
        /// Useful for training diagnostics to know which curriculum variant is active.
        /// </summary>
        public PushTBlockGenome LastAppliedGenome { get; private set; }

#if UNITY_EDITOR
        void OnValidate()
        {
            if (genomePool == null)
            {
                var guids = UnityEditor.AssetDatabase.FindAssets("t:GenomePool");
                if (guids.Length > 0)
                {
                    var path = UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]);
                    genomePool = UnityEditor.AssetDatabase.LoadAssetAtPath<GenomePool>(path);
                    UnityEditor.EditorUtility.SetDirty(this);
                }
            }
        }
#endif

        void Awake()
        {
            m_random = new System.Random();

            // Determine JSON path (check multiple possible locations)
            string jsonPath = null;
            var candidatePaths = new string[]
            {
                System.IO.Path.Combine(Application.persistentDataPath, "curriculum_outputs", "genome_pool_latest.json"),
                System.IO.Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "Scripts", "newPushTEC", "curriculum_outputs", "genome_pool_latest.json"),
                System.IO.Path.Combine(Application.dataPath, "..", "curriculum_outputs", "genome_pool_latest.json"),
            };
            
            foreach (var candidate in candidatePaths)
            {
                if (System.IO.File.Exists(candidate))
                {
                    jsonPath = candidate;
                    break;
                }
            }

            // Ensure we have a genomePool instance to load into
            if (genomePool == null)
            {
                genomePool = ScriptableObject.CreateInstance<GenomePool>();
                Debug.Log("[CurriculumEnv] genomePool reference was null, created runtime instance.");
            }

            // Try to load from JSON if pool is empty (or was just created)
            if (genomePool.IsEmpty && !string.IsNullOrEmpty(jsonPath))
            {
                Debug.Log($"[CurriculumEnv] GenomePool empty, loading from JSON: {jsonPath}");
                genomePool.LoadFromJson(jsonPath);
                
                if (!genomePool.IsEmpty)
                {
                    Debug.Log($"[CurriculumEnv] Successfully loaded {genomePool.Count} genomes from JSON.");
                }
                else
                {
                    Debug.LogWarning("[CurriculumEnv] JSON loaded but pool is still empty. Will use fallback genome.");
                }
            }
            else if (genomePool.IsEmpty)
            {
                Debug.LogWarning("[CurriculumEnv] GenomePool empty and no JSON file found. Will use fallback genome.");
            }
            else
            {
                Debug.Log($"[CurriculumEnv] GenomePool ready with {genomePool.Count} genomes.");
            }

            // Clone ground material at runtime so we never mutate the shared project asset.
            if (groundCollider != null && groundCollider.material != null)
            {
                m_runtimeGroundMaterial = new PhysicsMaterial(groundCollider.material.name + "_Runtime");
                m_runtimeGroundMaterial.dynamicFriction = groundCollider.material.dynamicFriction;
                m_runtimeGroundMaterial.staticFriction = groundCollider.material.staticFriction;
                m_runtimeGroundMaterial.bounciness = groundCollider.material.bounciness;
                m_runtimeGroundMaterial.frictionCombine = groundCollider.material.frictionCombine;
                m_runtimeGroundMaterial.bounceCombine = groundCollider.material.bounceCombine;
                groundCollider.material = m_runtimeGroundMaterial;
            }
            else if (groundCollider != null)
            {
                m_runtimeGroundMaterial = new PhysicsMaterial("PushT_Curriculum_Ground_Runtime");
                groundCollider.material = m_runtimeGroundMaterial;
            }

            // Subscribe to agent reset event for precise timing.
            // This requires PushAgentBasic.OnEpisodeBegin() to call OnAfterResetParameters.
            if (trainingAgent != null)
            {
                trainingAgent.OnAfterResetParameters += ApplyRandomGenomeFromPool;
                m_eventSubscribed = true;
            }
        }

        void OnDestroy()
        {
            if (trainingAgent != null)
            {
                trainingAgent.OnAfterResetParameters -= ApplyRandomGenomeFromPool;
            }
        }

        /// <summary>
        /// Fallback detection: only runs if the agent event is not subscribed.
        /// LateUpdate detects episode completion and applies genome afterwards.
        /// Note: this may result in one frame of default parameters before the genome is applied.
        /// </summary>
        void LateUpdate()
        {
            if (trainingAgent == null || m_eventSubscribed) return;

            int current = trainingAgent.CompletedEpisodes;
            if (m_lastCompletedEpisodes < 0)
            {
                m_lastCompletedEpisodes = current;
                return;
            }

            if (current != m_lastCompletedEpisodes)
            {
                m_lastCompletedEpisodes = current;
                ApplyRandomGenomeFromPool();
            }
        }

        /// <summary>
        /// Samples a genome from the pool (or fallback) and applies it to the scene.
        /// Safe to call multiple times per episode; idempotent for the same genome.
        /// </summary>
        public void ApplyRandomGenomeFromPool()
        {
            PushTBlockGenome genome = null;

            Debug.Log($"[CurriculumEnv-DIAG] genomePool is {(genomePool == null ? "NULL" : $"NOT NULL (count={genomePool.Count})")}, IsEmpty={genomePool?.IsEmpty}");

            if (genomePool != null && !genomePool.IsEmpty)
            {
                // Safeguard: with an adequate pool (>= 5 genomes) we should not dilute
                // the curriculum with fallback episodes. Force fallbackRatio to 0 and warn
                // if the Inspector value is still non-zero.
                float effectiveFallbackRatio = fallbackRatio;
                if (genomePool.Count >= 5 && fallbackRatio > 0f)
                {
                    Debug.LogWarning(
                        $"[CurriculumEnv] Pool has {genomePool.Count} genomes (>= 5), but fallbackRatio is {fallbackRatio:F2}. " +
                        "Ignoring fallbackRatio to prevent curriculum dilution. Set it to 0 in the Inspector to silence this warning.");
                    effectiveFallbackRatio = 0f;
                }

                genome = genomePool.SampleWithFallback(m_random, fallbackGenome, effectiveFallbackRatio);
                Debug.Log($"[CurriculumEnv-DIAG] Sampled from pool. Genome w={genome?.width:F2}");
            }
            else
            {
                genome = fallbackGenome?.Clone() ?? new PushTBlockGenome();
                Debug.LogWarning($"[CurriculumEnv-DIAG] Using fallback/default! Pool null={genomePool == null}, empty={genomePool?.IsEmpty}");
            }

            ApplyGenome(genome);
        }

        public void ApplyGenome(PushTBlockGenome genome)
        {
            if (genome == null) return;

            LastAppliedGenome = genome.Clone();

            if (blockTransform != null)
            {
                blockTransform.localScale = new Vector3(genome.width, genome.height, genome.depth);
            }

            if (blockRigidbody != null)
            {
                blockRigidbody.mass = genome.mass;
                blockRigidbody.linearDamping = genome.blockDrag;
            }

            if (m_runtimeGroundMaterial != null)
            {
                m_runtimeGroundMaterial.dynamicFriction = genome.friction;
                m_runtimeGroundMaterial.staticFriction = genome.friction;
                m_runtimeGroundMaterial.bounciness = genome.bounciness;
            }

            Debug.Log(
                $"[CurriculumEnv] Applied genome: w={genome.width:F2} h={genome.height:F2} d={genome.depth:F2} " +
                $"mass={genome.mass:F2} drag={genome.blockDrag:F2} friction={genome.friction:F2} bounce={genome.bounciness:F2}");
        }
    }
}
