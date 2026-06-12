using System;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Result object returned by PushTCurriculumEC.RunCurriculumECAsync().
    /// Contains enough information for the file-based orchestrator to write
    /// ec_done.json or ec_error.json.
    /// </summary>
    [Serializable]
    public class PushTCurriculumECResult
    {
        public bool success;
        public int selectedCount;
        public int generationId;
        public string poolPath;
        public string summaryCsvPath;
        public string allEvaluationsCsvPath;
        public string episodesCsvPath;
        public string errorMessage;
    }
}
