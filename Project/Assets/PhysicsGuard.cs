using UnityEngine;

/// <summary>
/// Ensures physics simulation is enabled when the scene starts.
/// 
/// This guards against Editor tools (e.g. EC evaluators) that may leave
/// Physics.autoSimulation=false or Time.timeScale!=1 across Play Mode transitions.
/// </summary>
[DefaultExecutionOrder(-1000)]
public class PhysicsGuard : MonoBehaviour
{
    [Tooltip("Target Time.timeScale. Leave at 1 for normal simulation.")]
    public float targetTimeScale = 1f;

    [Tooltip("If true, force Physics.simulationMode back to FixedUpdate on start.")]
    public bool forceFixedUpdateSimulation = true;

    void Awake()
    {
        if (Time.timeScale != targetTimeScale)
        {
            Debug.LogWarning($"[PhysicsGuard] Time.timeScale was {Time.timeScale}. Resetting to {targetTimeScale}.");
            Time.timeScale = targetTimeScale;
        }

        if (!Physics.autoSimulation)
        {
            Debug.LogWarning("[PhysicsGuard] Physics.autoSimulation was false. Re-enabling.");
            Physics.autoSimulation = true;
        }

        if (forceFixedUpdateSimulation && Physics.simulationMode != SimulationMode.FixedUpdate)
        {
            Debug.LogWarning($"[PhysicsGuard] Physics.simulationMode was {Physics.simulationMode}. Setting to FixedUpdate.");
            Physics.simulationMode = SimulationMode.FixedUpdate;
        }
    }
}
