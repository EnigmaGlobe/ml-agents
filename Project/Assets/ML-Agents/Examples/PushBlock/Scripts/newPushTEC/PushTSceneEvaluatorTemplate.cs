using System.Threading.Tasks;
using UnityEngine;

namespace PushTEvolutionMvp
{
    public class PushTSceneEvaluatorTemplate : PushTBlockEvaluator
    {
        [Header("Scene References")]
        public Transform blockTransform;
        public Renderer blockRenderer;
        public Rigidbody blockRigidbody;
        public PhysicsMaterial targetPhysicMaterial;
        public MonoBehaviour pushTEnvironment;
        public MonoBehaviour pushTAgent;

        [Header("Episode Settings")]
        public float episodeTimeoutSeconds = 15f;

        protected override async Task<PushTEpisodeResult> EvaluateEpisodeAsync(
            PushTBlockGenome genome,
            int episodeIndex,
            int episodeSeed)
        {
            this.ApplyGenome(genome);
            this.ResetEnvironment(episodeSeed);

            var startTime = Time.time;

            // Replace this polling loop with your own end-of-episode signal.
            while (!this.HasEpisodeFinished())
            {
                if (Time.time - startTime >= this.episodeTimeoutSeconds)
                {
                    break;
                }

                await Task.Yield();
            }

            var success = this.ReadSuccess();
            var completionTime = Mathf.Min(Time.time - startTime, this.episodeTimeoutSeconds);
            var reward = this.ReadCumulativeReward();
            var stabilityPenalty = this.ReadStabilityPenalty();
            var progress = success ? 1f : 0f;

            return new PushTEpisodeResult
            {
                episodeIndex = episodeIndex,
                episodeSeed = episodeSeed,
                success = success,
                reward = reward,
                progress = progress,
                finalGoalError = 0f,
                timeToGoal = success ? completionTime : -1f,
                timedOut = false,
                invalidPhysics = false,
                notes = "Replace template hooks in PushTSceneEvaluatorTemplate with your scene-specific logic."
            };
        }

        private void ApplyGenome(PushTBlockGenome genome)
        {
            if (this.blockTransform != null)
            {
                this.blockTransform.localScale = new Vector3(genome.width, genome.height, genome.depth);
            }

            if (this.blockRigidbody != null)
            {
                this.blockRigidbody.mass = genome.mass;
            }

            if (this.targetPhysicMaterial != null)
            {
                this.targetPhysicMaterial.dynamicFriction = genome.friction;
                this.targetPhysicMaterial.staticFriction = genome.friction;
                this.targetPhysicMaterial.bounciness = genome.bounciness;
            }

            // If shape changes require swapping colliders/meshes, add that logic here.
        }

        private void ResetEnvironment(int episodeSeed)
        {
            Random.InitState(episodeSeed);

            // Replace with your actual reset flow. Typical options:
            // - call Academy.Instance.EnvironmentReset()
            // - call your environment's ResetEpisode method
            // - reposition the block/robot/goal manually
            // - clear velocity and internal counters

            if (this.blockRigidbody != null)
            {
                this.blockRigidbody.linearVelocity = Vector3.zero;
                this.blockRigidbody.angularVelocity = Vector3.zero;
            }
        }

        private bool HasEpisodeFinished()
        {
            // Replace this with a real signal from your environment or agent.
            // For example:
            // - an `EpisodeFinished` bool
            // - `StepCount >= MaxStep`
            // - block reached target area
            return false;
        }

        private bool ReadSuccess()
        {
            // Replace with a real success condition, for example:
            // return environment.BlockReachedGoal;
            return false;
        }

        private float ReadCumulativeReward()
        {
            // Replace with your actual reward source.
            // If your agent inherits from Agent, expose a property or method for this.
            return 0f;
        }

        private float ReadStabilityPenalty()
        {
            // Optional: penalize tipping, jitter, collisions, or rollout variance.
            return 0f;
        }

        private float CalculateFitness(
            bool success,
            float completionTimeSeconds,
            float cumulativeReward,
            float stabilityPenalty)
        {
            var successBonus = success ? 100f : 0f;
            var timePenalty = completionTimeSeconds;
            return successBonus + cumulativeReward - timePenalty - stabilityPenalty;
        }
    }
}
