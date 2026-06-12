namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// High-level status of the checkpoint swap pipeline.
    /// Prefix "Error" indicates a terminal failure state.
    /// </summary>
    public enum CheckpointSwapStatus
    {
        Idle,
        SearchingForCheckpoint,
        CopyingCheckpoint,
        WritingManifest,
        CheckpointDetected,
        WaitingForStableFile,
        RefreshingAssetDatabase,
        LoadingModel,
        ModelReady,
        ErrorMissingOnnx,
        ErrorImportFailed,
        ErrorMissingFrozenEvaluator,
        ErrorBehaviorMismatch,
        ErrorEcAlreadyRunning
    }
}
