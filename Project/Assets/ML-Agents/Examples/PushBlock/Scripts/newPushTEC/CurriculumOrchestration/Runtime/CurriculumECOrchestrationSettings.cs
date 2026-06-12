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
        public string requestManifestPath = "Assets/Models/FrozenEvaluator/ec_request.json";
        public string doneManifestPath = "Assets/Models/FrozenEvaluator/ec_done.json";
        public string errorManifestPath = "Assets/Models/FrozenEvaluator/ec_error.json";

        [Header("Watcher")]
        public float debounceSeconds = 1.0f;

        [Header("Validation")]
        public bool strictBehaviorNameCheck = false;

        [Header("Export")]
        public bool writeProjectRelativePaths = true;
    }
}
