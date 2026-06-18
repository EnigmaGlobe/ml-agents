using UnityEngine;

/// <summary>
/// Temporary diagnostic: logs physics state and fixes common issues that prevent
/// Rigidbodies from moving (simulationMode=Script, autoSimulation=false, timeScale=0).
///
/// Usage: add to the Agent GameObject (or any active GameObject) and enter Play mode.
/// </summary>
public class PhysicsDiagnostic : MonoBehaviour
{
    [Tooltip("If true, reset abnormal Time.timeScale back to 1.")]
    public bool resetTimeScale = true;

    void Start()
    {
        LogPhysicsState("Start");

        // Fix common misconfigurations
        bool fixedSomething = false;

        if (resetTimeScale && Mathf.Abs(Time.timeScale - 1f) > 0.01f)
        {
            Debug.LogWarning($"[PHYSICS-DIAG] Time.timeScale was {Time.timeScale}. Resetting to 1.");
            Time.timeScale = 1f;
            fixedSomething = true;
        }

        if (!Physics.autoSimulation)
        {
            Debug.LogWarning("[PHYSICS-DIAG] Physics.autoSimulation was false. Re-enabling.");
            Physics.autoSimulation = true;
            fixedSomething = true;
        }

        if (Physics.simulationMode == SimulationMode.Script)
        {
            Debug.LogWarning("[PHYSICS-DIAG] Physics.simulationMode was Script. Setting to FixedUpdate.");
            Physics.simulationMode = SimulationMode.FixedUpdate;
            fixedSomething = true;
        }

        if (fixedSomething)
        {
            LogPhysicsState("After fix");
        }
        else
        {
            Debug.Log("[PHYSICS-DIAG] No common physics misconfiguration detected.");
        }
    }

    void LogPhysicsState(string label)
    {
        Debug.Log(
            $"[PHYSICS-DIAG] {label}:\n" +
            $"  Time.timeScale={Time.timeScale}\n" +
            $"  Time.fixedDeltaTime={Time.fixedDeltaTime}\n" +
            $"  Physics.autoSimulation={Physics.autoSimulation}\n" +
            $"  Physics.simulationMode={Physics.simulationMode}\n" +
            $"  Physics.autoSyncTransforms={Physics.autoSyncTransforms}\n" +
            $"  Physics.gravity={Physics.gravity}"
        );

        var rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            Debug.Log(
                $"[PHYSICS-DIAG] Rigidbody on {gameObject.name}:\n" +
                $"  isKinematic={rb.isKinematic}\n" +
                $"  constraints={rb.constraints}\n" +
                $"  position={rb.position}\n" +
                $"  linearVelocity={rb.linearVelocity}\n" +
                $"  isSleeping={rb.IsSleeping()}"
            );
        }
    }
}
