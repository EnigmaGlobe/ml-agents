using System;
using System.Collections.Generic;

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
