using UnityEngine;

namespace PushTEvolutionMvp.CurriculumOrchestration
{
    /// <summary>
    /// Shared settings for the file-based Curriculum EC orchestration system.
    /// Create via Assets > Create > PushT > Curriculum EC Orchestration Settings.
    /// </summary>
    [CreateAssetMenu(
        fileName = "CurriculumECOrchestrationSettings",
        menuName = "PushT/Curriculum EC Orchestration Settings")]
    public class CurriculumECOrchestrationSettings : ScriptableObject
    {
        [Header("Manifest Paths")]
        [Tooltip("Project-relative path to the pending EC request manifest. Kept outside Assets to avoid .meta files.")]
        public string requestManifestPath = "Temp/FrozenEvaluator/ec_request.json";
        [Tooltip("Project-relative path to the EC done manifest. Kept outside Assets to avoid .meta files.")]
        public string doneManifestPath = "Temp/FrozenEvaluator/ec_done.json";
        [Tooltip("Project-relative path to the EC error manifest. Kept outside Assets to avoid .meta files.")]
        public string errorManifestPath = "Temp/FrozenEvaluator/ec_error.json";

        [Header("Watcher")]
        public float debounceSeconds = 1.0f;

        [Header("Validation")]
        public bool strictBehaviorNameCheck = false;

        [Header("Export")]
        public bool writeProjectRelativePaths = true;
    }
}
