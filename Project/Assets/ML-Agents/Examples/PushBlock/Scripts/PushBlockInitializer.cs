using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class PushBlockInitializer : MonoBehaviour
{
    void Start()
    {
        // First, try to get settings on the same GameObject (current workflow).
        var settings = GetComponent<PushBlockSettings>();
        if (settings == null)
        {
            // Fallback: find a global settings component in the scene.
            settings = FindAnyObjectByType<PushBlockSettings>();
            if (settings == null)
            {
                return;
            }
        }

        // Apply uniform scale based on selected block size.
        float size = settings.SelectedBlockSize;
        transform.localScale = Vector3.one * size;

        // Apply Rigidbody settings.
        var rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = settings.SelectedBlockMass;
            rb.linearDamping = settings.defaultBlockDrag;
        }

        // Apply friction via a runtime PhysicMaterial on the collider.
        var col = GetComponent<Collider>();
        if (col != null)
        {
            var pm = new PhysicsMaterial("BlockPhysic_" + GetInstanceID())
            {
                staticFriction = settings.SelectedStaticFriction,
                dynamicFriction = settings.SelectedDynamicFriction,
                frictionCombine = PhysicsMaterialCombine.Average
            };
            col.material = pm;
        }
    }
}
