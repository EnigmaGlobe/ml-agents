using System;
using System.Collections.Generic;
using UnityEngine;

namespace PushTEvolutionMvp
{
    [Serializable]
    public class PushTEpisodeResult
    {
        public int episodeIndex;
        public int episodeSeed;
        public bool success;
        public float reward;
        public float progress;
        public float finalGoalError;
        public float timeToGoal;
        public bool timedOut;
        public bool invalidPhysics;
        public string notes = "";

        // Spatial trajectory samples (sampled per decision step or fixed interval)
        public List<Vector3> agentPositions = new List<Vector3>();
        public List<Vector3> blockPositions = new List<Vector3>();
        public Vector3 goalPosition;
        public Vector3 sceneCenter;

        // M4-derived per-episode metrics
        public float blockNetDisplacement;
        public float agentNetDisplacement;
        public float blockRadialSpread;
        public float agentRadialSpread;
        public float taskCentroidCentrality;
        public float sceneCentroidCentrality;
    }

    [Serializable]
    public class PushTGenomeEvaluation
    {
        public int generationIndex;
        public int genomeIndex;
        public PushTBlockGenome genome = new PushTBlockGenome();
        public List<PushTEpisodeResult> episodes = new List<PushTEpisodeResult>();

        // Aggregated statistics (computed after all episodes finish)
        public float successRate;
        public float meanProgress;
        public float iqmProgress;
        public float sdProgress;
        public float iqrProgress;
        public float cvProgress;
        public float meanGoalError;
        public float sdGoalError;
        public float meanTimeToGoal;
        public float meanReward;
        public float difficulty;
        public float tooEasyPenalty;
        public float invalidPenalty;
        public float goalErrorScore;
        public float timeScore;
        public float fitness;

        // M4 spatial behavior profile (aggregated across episodes)
        public PushTSpatialBehaviorProfile spatialProfile = new PushTSpatialBehaviorProfile();

        // Fitness component scores for diagnostics and CSV export
        public float learnabilityScore;
        public float challengeScore;
        public float spatialBehaviorScore;
        public float noveltyScore;
    }

    [Serializable]
    public class PushTEvolutionRunResult
    {
        public List<PushTGenomeEvaluation> bestPerGeneration = new List<PushTGenomeEvaluation>();
        public PushTGenomeEvaluation bestOverall;
        public List<PushTGenomeEvaluation> allEvaluations = new List<PushTGenomeEvaluation>();
        public int totalEpisodesRun;
    }
}
