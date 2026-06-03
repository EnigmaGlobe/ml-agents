using System;
using UnityEngine;

namespace PushTEvolutionMvp
{
    [Serializable]
    public struct FloatRange
    {
        public float min;
        public float max;

        public float Clamp(float value)
        {
            return Mathf.Clamp(value, this.min, this.max);
        }

        public float RandomValue(System.Random random)
        {
            return this.min + ((float)random.NextDouble() * (this.max - this.min));
        }
    }

    [Serializable]
    public class PushTEvolutionConfig
    {
        [Header("Search")]
        public int populationSize = 24;
        public int eliteCount = 6;
        public int generationCount = 20;
        public int episodesPerGenome = 10;
        public int randomSeed = 12345;

        [Header("Mutation")]
        [Range(0f, 1f)] public float mutationRate = 0.35f;
        [Range(0f, 1f)] public float mutationStrength = 0.15f;

        [Header("Genome Ranges")]
        public FloatRange widthRange = new FloatRange { min = 0.5f, max = 3.0f };
        public FloatRange heightRange = new FloatRange { min = 0.5f, max = 5.0f };
        public FloatRange depthRange = new FloatRange { min = 0.25f, max = 2.0f };
        public FloatRange massRange = new FloatRange { min = 0.5f, max = 6.0f };
        public FloatRange frictionRange = new FloatRange { min = 0.1f, max = 1.0f };
        public FloatRange dragRange = new FloatRange { min = 0.05f, max = 1.5f };
        public FloatRange bouncinessRange = new FloatRange { min = 0f, max = 0.3f };

        [Header("Fitness Thresholds")]
        public float successRateThreshold = 0.9f;
        public float minUsefulDifficulty = 0.25f;
        public float maxAcceptableGoalError = 2.0f;

        [Header("Time Target Zones (seconds)")]
        public float timeTooFast = 3.0f;
        public float timeEasy = 5.0f;
        public float timeIdeal = 12.0f;
        public float timeHard = 20.0f;
    }
}
