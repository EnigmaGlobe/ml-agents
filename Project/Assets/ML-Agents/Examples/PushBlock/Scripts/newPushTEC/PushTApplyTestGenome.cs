using UnityEngine;

namespace PushTEvolutionMvp
{
    public class PushTApplyTestGenome : MonoBehaviour
    {
        [Header("Scene References")]
        public Transform blockTransform;
        public Rigidbody blockRigidbody;
        public PhysicsMaterial groundPhysicMaterial;

        [Header("Test Values")]
        public float blockScale = 1.25f;
        public float blockMass = 2f;
        public float blockDrag = 0.5f;
        public float dynamicFriction = 0.7f;
        public float staticFriction = 0.7f;

        [ContextMenu("Apply Test Genome")]
        public void ApplyTestGenome()
        {
            if (this.blockTransform != null)
            {
                this.blockTransform.localScale = new Vector3(this.blockScale, 0.75f, this.blockScale);
            }

            if (this.blockRigidbody != null)
            {
                this.blockRigidbody.mass = this.blockMass;
                this.blockRigidbody.linearDamping = this.blockDrag;
                this.blockRigidbody.linearVelocity = Vector3.zero;
                this.blockRigidbody.angularVelocity = Vector3.zero;
                this.blockRigidbody.ResetInertiaTensor();
            }

            if (this.groundPhysicMaterial != null)
            {
                this.groundPhysicMaterial.dynamicFriction = this.dynamicFriction;
                this.groundPhysicMaterial.staticFriction = this.staticFriction;
            }

            Debug.Log(
                $"Applied PushT test genome: scale={this.blockScale}, mass={this.blockMass}, " +
                $"drag={this.blockDrag}, dynamicFriction={this.dynamicFriction}, " +
                $"staticFriction={this.staticFriction}");
        }
    }
}
