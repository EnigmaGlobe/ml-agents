using System;
using UnityEngine;

namespace PushTEvolutionMvp
{
    [Serializable]
    public class PushTBlockGenome
    {
        public float width = 1f;
        public float height = 1f;
        public float depth = 1f;
        public float mass = 1f;
        public float friction = 0.5f;
        public float bounciness = 0f;
        public float blockDrag = 0.5f;

        // Compatibility properties for legacy EC API in PushAgentBasic
        public float blockScale => width;
        public float blockMass => mass;
        public float dynamicFriction => friction;
        public float staticFriction => friction;

        public static bool IsValid(PushTBlockGenome genome)
        {
            return genome != null;
        }

        public static bool IsGenomeValid(PushTBlockGenome genome)
        {
            if (genome == null) return false;
            if (genome.width <= 0f) return false;
            if (genome.height <= 0f) return false;
            if (genome.depth <= 0f) return false;
            if (genome.mass <= 0f) return false;
            if (genome.blockDrag < 0f) return false;
            if (genome.friction < 0f) return false;
            if (genome.bounciness < 0f) return false;
            // Reviewer suggestion: avoid extreme aspect ratios
            if (genome.depth > 0.0001f && genome.height / genome.depth > 12f) return false;
            return true;
        }

        public PushTBlockGenome Clone()
        {
            return new PushTBlockGenome
            {
                width = this.width,
                height = this.height,
                depth = this.depth,
                mass = this.mass,
                friction = this.friction,
                bounciness = this.bounciness,
                blockDrag = this.blockDrag
            };
        }

        public static PushTBlockGenome CreateRandom(PushTEvolutionConfig config, System.Random random)
        {
            for (int attempt = 0; attempt < 1000; attempt++)
            {
                var genome = new PushTBlockGenome
                {
                    width = config.widthRange.RandomValue(random),
                    height = config.heightRange.RandomValue(random),
                    depth = config.depthRange.RandomValue(random),
                    mass = config.massRange.RandomValue(random),
                    friction = config.frictionRange.RandomValue(random),
                    blockDrag = config.dragRange.RandomValue(random),
                    bounciness = config.bouncinessRange.RandomValue(random)
                };
                if (IsGenomeValid(genome))
                {
                    return genome;
                }
            }
            Debug.LogWarning("[PushT EC] Failed to generate a valid random genome after 1000 attempts. Returning last attempt.");
            return new PushTBlockGenome
            {
                width = config.widthRange.RandomValue(random),
                height = config.heightRange.RandomValue(random),
                depth = config.depthRange.RandomValue(random),
                mass = config.massRange.RandomValue(random),
                friction = config.frictionRange.RandomValue(random),
                blockDrag = config.dragRange.RandomValue(random),
                bounciness = config.bouncinessRange.RandomValue(random)
            };
        }

        public void Mutate(PushTEvolutionConfig config, System.Random random)
        {
            this.width = MutateFloat(this.width, config.widthRange, config, random);
            this.height = MutateFloat(this.height, config.heightRange, config, random);
            this.depth = MutateFloat(this.depth, config.depthRange, config, random);
            this.mass = MutateFloat(this.mass, config.massRange, config, random);
            this.friction = MutateFloat(this.friction, config.frictionRange, config, random);
            this.blockDrag = MutateFloat(this.blockDrag, config.dragRange, config, random);
            this.bounciness = MutateFloat(this.bounciness, config.bouncinessRange, config, random);
            FixAspectRatio(config);
        }

        private void FixAspectRatio(PushTEvolutionConfig config)
        {
            if (this.depth <= 0.0001f || this.height / this.depth <= 12f)
            {
                return;
            }

            // Constraint violated: height / depth > 12
            // Try reducing height first (least invasive)
            float maxHeight = this.depth * 12f;
            if (maxHeight >= config.heightRange.min)
            {
                this.height = Mathf.Min(this.height, maxHeight);
                return;
            }

            // If height is already at minimum, try increasing depth
            float minDepth = this.height / 12f;
            if (minDepth <= config.depthRange.max)
            {
                this.depth = Mathf.Max(this.depth, minDepth);
                return;
            }

            // Fallback: clamp both to the boundary of feasibility
            this.height = config.heightRange.min;
            this.depth = Mathf.Max(this.depth, config.heightRange.min / 12f);
        }

        private static float MutateFloat(
            float value,
            FloatRange range,
            PushTEvolutionConfig config,
            System.Random random)
        {
            if (!ShouldMutate(config, random))
            {
                return value;
            }

            return range.Clamp(value + RandomDelta(range, config, random));
        }

        private static float RandomDelta(
            FloatRange range,
            PushTEvolutionConfig config,
            System.Random random)
        {
            var maxDelta = (range.max - range.min) * config.mutationStrength;
            var sample = (float)random.NextDouble() * 2f - 1f;
            return sample * maxDelta;
        }

        private static bool ShouldMutate(PushTEvolutionConfig config, System.Random random)
        {
            return random.NextDouble() < config.mutationRate;
        }
    }
}
