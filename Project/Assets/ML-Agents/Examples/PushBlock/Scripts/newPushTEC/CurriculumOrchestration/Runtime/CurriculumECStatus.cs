namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// High-level status of the file-based Curriculum EC orchestration pipeline.
    /// </summary>
    public enum CurriculumECStatus
    {
        Idle,
        RequestDetected,
        Validating,
        RunningEC,
        WritingDone,
        WritingError,
        Completed,
        ErrorInvalidRequest,
        ErrorMissingCurriculumEC,
        ErrorMissingOutputPool,
        ErrorMissingFrozenEvaluator,
        ErrorMissingModel,
        ErrorBehaviorTypeMismatch,
        ErrorBehaviorNameMismatch,
        ErrorEcAlreadyRunning,
        ErrorExecutionFailed
    }
}
