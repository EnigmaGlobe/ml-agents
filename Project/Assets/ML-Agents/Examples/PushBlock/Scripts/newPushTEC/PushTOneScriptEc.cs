using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace PushTEvolutionMvp
{
    public class PushTOneScriptEc : MonoBehaviour
    {
        [Serializable]
        public class FloatRange
        {
            public float min;
            public float max;

            public float Sample(System.Random random)
            {
                return this.min + ((float)random.NextDouble() * (this.max - this.min));
            }

            public float Clamp(float value)
            {
                return Mathf.Clamp(value, this.min, this.max);
            }
        }

        [Serializable]
        public class Candidate
        {
            public float blockScale = 1f;
            public float blockMass = 1f;
            public float blockDrag = 0.5f;
            public float dynamicFriction = 0.5f;
            public float staticFriction = 0.5f;
            public float fitness;

            public Candidate Clone()
            {
                return new Candidate
                {
                    blockScale = this.blockScale,
                    blockMass = this.blockMass,
                    blockDrag = this.blockDrag,
                    dynamicFriction = this.dynamicFriction,
                    staticFriction = this.staticFriction,
                    fitness = this.fitness
                };
            }
        }

        [Header("Run")]
        public bool runOnPlay = true;
        public int randomSeed = 12345;
        public int populationSize = 4;
        public int eliteCount = 1;
        public int generations = 2;
        public float episodeTimeoutSeconds = 15f;

        [Header("Optional Overrides")]
        public GameObject blockOverride;
        public GameObject groundOverride;

        [Header("Search Ranges")]
        public FloatRange blockScaleRange = new FloatRange { min = 0.75f, max = 1.5f };
        public FloatRange blockMassRange = new FloatRange { min = 0.5f, max = 4f };
        public FloatRange blockDragRange = new FloatRange { min = 0f, max = 1f };
        public FloatRange dynamicFrictionRange = new FloatRange { min = 0.1f, max = 1f };
        public FloatRange staticFrictionRange = new FloatRange { min = 0.1f, max = 1f };

        [Header("Mutation")]
        [Range(0f, 1f)] public float mutationRate = 0.35f;
        [Range(0f, 1f)] public float mutationStrength = 0.15f;

        [Header("Result")]
        public Candidate bestCandidate;
        [TextArea] public string status = "Idle";

        [Header("Validator UI")]
        public bool showValidatorUi = true;
        public Rect validatorRect = new Rect(16f, 16f, 420f, 170f);

        private GameObject _block;
        private GameObject _ground;
        private GameObject _goal;
        private Rigidbody _blockRigidbody;
        private PhysicsMaterial _groundMaterialInstance;
        private MonoBehaviour _agentTarget;
        private System.Random _random;
        private int _currentGeneration;
        private int _currentCandidate;
        private int _totalCandidates;
        private float _lastFitness;
        private bool _lastSuccess;
        private string _lastValidationMessage = "Waiting for Play";

        private void Start()
        {
            if (this.runOnPlay)
            {
                StartCoroutine(this.RunEc());
            }
        }

        [ContextMenu("Run Tiny EC")]
        public void RunTinyEcFromInspector()
        {
            StopAllCoroutines();
            StartCoroutine(this.RunEc());
        }

        private IEnumerator RunEc()
        {
            this.status = "Starting PushT one-script EC";
            this._lastValidationMessage = "Resolving PushT scene references";
            this._random = new System.Random(this.randomSeed);

            if (!this.ResolveSceneReferences())
            {
                yield break;
            }

            // Disable per-frame screenshot capture during EC to avoid disk spam.
            bool originalScreenshotSetting = this.TryGetScreenshotEnabled();
            this.TrySetScreenshotEnabled(false);

            var population = this.CreateInitialPopulation();
            this.bestCandidate = null;
            this._totalCandidates = population.Count;

            for (int generation = 0; generation < this.generations; generation++)
            {
                this._currentGeneration = generation + 1;
                this.status = $"Generation {generation + 1}/{this.generations}";

                for (int i = 0; i < population.Count; i++)
                {
                    yield return this.EvaluateCandidate(population[i], generation, i);
                }

                population.Sort((a, b) => b.fitness.CompareTo(a.fitness));

                if (this.bestCandidate == null || population[0].fitness > this.bestCandidate.fitness)
                {
                    this.bestCandidate = population[0].Clone();
                }

                Debug.Log(
                    $"[PushT EC] generation={generation} bestFitness={population[0].fitness:F3} " +
                    $"scale={population[0].blockScale:F3} mass={population[0].blockMass:F3} " +
                    $"drag={population[0].blockDrag:F3} dynamicFriction={population[0].dynamicFriction:F3} " +
                    $"staticFriction={population[0].staticFriction:F3}");

                population = this.CreateNextPopulation(population);
            }

            this.status = this.bestCandidate == null
                ? "Finished with no candidate"
                : $"Finished. Best fitness {this.bestCandidate.fitness:F3}";
            this._lastValidationMessage = this.status;

            // Restore screenshot setting so non-EC runs are unaffected.
            this.TrySetScreenshotEnabled(originalScreenshotSetting);
        }

        private bool ResolveSceneReferences()
        {
            this._agentTarget = this.FindAgentTarget();
            if (this._agentTarget == null)
            {
                Debug.LogError("[PushT EC] No PushT agent component found on this GameObject.");
                this.status = "Missing agent";
                this._lastValidationMessage = "Missing agent component on this GameObject";
                return false;
            }

            this._block = this.blockOverride != null
                ? this.blockOverride
                : this.GetGameObjectField("block");

            this._ground = this.groundOverride != null
                ? this.groundOverride
                : this.GetGameObjectField("ground");

            this._goal = this.GetGameObjectField("goal");

            if (this._block == null)
            {
                Debug.LogError("[PushT EC] No block found. Assign blockOverride or attach this to PushAgentBasic.");
                this.status = "Missing block";
                this._lastValidationMessage = "Missing block reference";
                return false;
            }

            this._blockRigidbody = this._block.GetComponent<Rigidbody>();
            if (this._blockRigidbody == null)
            {
                Debug.LogError("[PushT EC] Block has no Rigidbody.");
                this.status = "Missing block Rigidbody";
                this._lastValidationMessage = "Missing block Rigidbody";
                return false;
            }

            if (this._ground != null)
            {
                var groundCollider = this._ground.GetComponent<Collider>();
                if (groundCollider != null)
                {
                    this._groundMaterialInstance = groundCollider.material != null
                        ? Instantiate(groundCollider.material)
                        : new PhysicsMaterial("PushT EC Ground Material");
                    groundCollider.material = this._groundMaterialInstance;
                }
            }

            return true;
        }

        private MonoBehaviour FindAgentTarget()
        {
            var components = GetComponents<MonoBehaviour>();
            foreach (var component in components)
            {
                if (component == null || component == this)
                {
                    continue;
                }

                if (this.FindFieldOn(component.GetType(), "block") != null &&
                    this.FindFieldOn(component.GetType(), "ground") != null)
                {
                    return component;
                }
            }

            return null;
        }

        private List<Candidate> CreateInitialPopulation()
        {
            var count = Mathf.Max(1, this.populationSize);
            var population = new List<Candidate>(count);

            for (int i = 0; i < count; i++)
            {
                population.Add(new Candidate
                {
                    blockScale = this.blockScaleRange.Sample(this._random),
                    blockMass = this.blockMassRange.Sample(this._random),
                    blockDrag = this.blockDragRange.Sample(this._random),
                    dynamicFriction = this.dynamicFrictionRange.Sample(this._random),
                    staticFriction = this.staticFrictionRange.Sample(this._random)
                });
            }

            return population;
        }

        private List<Candidate> CreateNextPopulation(List<Candidate> sortedPopulation)
        {
            var eliteTotal = Mathf.Clamp(this.eliteCount, 1, sortedPopulation.Count);
            var next = new List<Candidate>(Mathf.Max(1, this.populationSize));

            for (int i = 0; i < eliteTotal; i++)
            {
                next.Add(sortedPopulation[i].Clone());
            }

            while (next.Count < this.populationSize)
            {
                var parent = sortedPopulation[this._random.Next(eliteTotal)].Clone();
                this.Mutate(parent);
                next.Add(parent);
            }

            return next;
        }

        private IEnumerator EvaluateCandidate(Candidate candidate, int generation, int index)
        {
            this.status = $"Evaluating g{generation + 1} candidate {index + 1}";
            this._currentGeneration = generation + 1;
            this._currentCandidate = index + 1;
            this._lastValidationMessage = "Resetting episode";

            var beforeResetEpisodeId = this.GetIntField("m_episodeId", -1);
            this.TryInvokeAgentMethod("EndEpisode");
            yield return null;

            var resetDeadline = Time.time + 2f;
            while (Time.time < resetDeadline && this.GetIntField("m_episodeId", -1) == beforeResetEpisodeId)
            {
                yield return null;
            }

            var evaluationEpisodeId = this.GetIntField("m_episodeId", beforeResetEpisodeId);
            this.ApplyCandidate(candidate);
            this._lastValidationMessage = "Candidate applied, waiting for episode result";

            var bestObservedReward = this.GetFloatField("m_episodeCumulativeReward", 0f);
            var lastProgress = this.GetFloatField("m_normalizedTaskProgress", 0f);
            var lastFinalError = this.GetLiveGoalError(999f);
            var lastEndReason = "running";
            var deadline = Time.time + this.episodeTimeoutSeconds;

            while (Time.time < deadline && this.GetIntField("m_episodeId", evaluationEpisodeId) == evaluationEpisodeId)
            {
                bestObservedReward = this.GetFloatField("m_episodeCumulativeReward", bestObservedReward);
                lastProgress = this.GetFloatField("m_normalizedTaskProgress", lastProgress);
                lastFinalError = this.GetLiveGoalError(lastFinalError);
                lastEndReason = this.GetStringField("m_episodeEndReason", lastEndReason);
                yield return null;
            }

            var episodeRolledOver = this.GetIntField("m_episodeId", evaluationEpisodeId) != evaluationEpisodeId;
            lastProgress = this.GetFloatField("m_normalizedTaskProgress", lastProgress);
            lastFinalError = episodeRolledOver
                ? this.GetRecordedGoalError(lastFinalError)
                : this.GetLiveGoalError(lastFinalError);
            lastEndReason = this.GetStringField("m_episodeEndReason", lastEndReason);

            var success = lastEndReason == "goal" || lastProgress >= 0.999f;
            candidate.fitness = (success ? 100f : 0f) + bestObservedReward + (25f * lastProgress) - lastFinalError;
            this._lastFitness = candidate.fitness;
            this._lastSuccess = success;
            this._lastValidationMessage = $"Candidate scored: {candidate.fitness:F3}";

            Debug.Log(
                $"[PushT EC] g={generation} i={index} fitness={candidate.fitness:F3} success={success} " +
                $"reward={bestObservedReward:F3} progress={lastProgress:F3} finalError={lastFinalError:F3}");
        }

        private void ApplyCandidate(Candidate candidate)
        {
            this._block.transform.localScale = new Vector3(candidate.blockScale, 0.75f, candidate.blockScale);

            this._blockRigidbody.mass = candidate.blockMass;
            this._blockRigidbody.linearDamping = candidate.blockDrag;
            this._blockRigidbody.linearVelocity = Vector3.zero;
            this._blockRigidbody.angularVelocity = Vector3.zero;
            this._blockRigidbody.ResetInertiaTensor();

            if (this._groundMaterialInstance != null)
            {
                this._groundMaterialInstance.dynamicFriction = candidate.dynamicFriction;
                this._groundMaterialInstance.staticFriction = candidate.staticFriction;
            }
        }

        private void Mutate(Candidate candidate)
        {
            candidate.blockScale = this.MutateValue(candidate.blockScale, this.blockScaleRange);
            candidate.blockMass = this.MutateValue(candidate.blockMass, this.blockMassRange);
            candidate.blockDrag = this.MutateValue(candidate.blockDrag, this.blockDragRange);
            candidate.dynamicFriction = this.MutateValue(candidate.dynamicFriction, this.dynamicFrictionRange);
            candidate.staticFriction = this.MutateValue(candidate.staticFriction, this.staticFrictionRange);
        }

        private float MutateValue(float value, FloatRange range)
        {
            if (this._random.NextDouble() > this.mutationRate)
            {
                return value;
            }

            var span = range.max - range.min;
            var delta = ((float)this._random.NextDouble() * 2f - 1f) * span * this.mutationStrength;
            return range.Clamp(value + delta);
        }

        private GameObject GetGameObjectField(string fieldName)
        {
            var field = this.FindField(fieldName);
            return field == null ? null : field.GetValue(this._agentTarget) as GameObject;
        }

        private int GetIntField(string fieldName, int fallback)
        {
            var field = this.FindField(fieldName);
            return field == null ? fallback : Convert.ToInt32(field.GetValue(this._agentTarget));
        }

        private float GetFloatField(string fieldName, float fallback)
        {
            var field = this.FindField(fieldName);
            return field == null ? fallback : Convert.ToSingle(field.GetValue(this._agentTarget));
        }

        private string GetStringField(string fieldName, string fallback)
        {
            var field = this.FindField(fieldName);
            return field == null ? fallback : field.GetValue(this._agentTarget) as string ?? fallback;
        }

        private float GetRecordedGoalError(float fallback)
        {
            var recorded = this.GetFloatField("m_finalBlockGoalDistance", -1f);
            if (recorded >= 0f)
            {
                return recorded;
            }

            return this.GetLiveGoalError(fallback);
        }

        private float GetLiveGoalError(float fallback)
        {
            if (this._block == null || this._goal == null)
            {
                return fallback;
            }

            return Vector3.Distance(this._block.transform.position, this._goal.transform.position);
        }

        private FieldInfo FindField(string fieldName)
        {
            return this._agentTarget == null ? null : this.FindFieldOn(this._agentTarget.GetType(), fieldName);
        }

        private FieldInfo FindFieldOn(Type targetType, string fieldName)
        {
            var type = targetType;
            while (type != null)
            {
                var field = type.GetField(fieldName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (field != null)
                {
                    return field;
                }

                type = type.BaseType;
            }

            return null;
        }

        private bool TryInvokeAgentMethod(string methodName)
        {
            if (this._agentTarget == null)
            {
                return false;
            }

            var type = this._agentTarget.GetType();
            while (type != null)
            {
                var method = type.GetMethod(methodName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (method != null && method.GetParameters().Length == 0)
                {
                    try
                    {
                        method.Invoke(this._agentTarget, null);
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"[PushT EC] Could not invoke {methodName}: {ex.Message}");
                        return false;
                    }
                }

                type = type.BaseType;
            }

            return false;
        }

        private bool TryGetScreenshotEnabled()
        {
            if (this._agentTarget == null)
            {
                return false;
            }

            var field = this.FindFieldOn(this._agentTarget.GetType(), "enableScreenshotCapture");
            if (field != null && field.FieldType == typeof(bool))
            {
                try
                {
                    return (bool)field.GetValue(this._agentTarget);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[PushT EC] Could not read enableScreenshotCapture: {ex.Message}");
                }
            }

            return false;
        }

        private void TrySetScreenshotEnabled(bool value)
        {
            if (this._agentTarget == null)
            {
                return;
            }

            var field = this.FindFieldOn(this._agentTarget.GetType(), "enableScreenshotCapture");
            if (field != null && field.FieldType == typeof(bool))
            {
                try
                {
                    field.SetValue(this._agentTarget, value);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[PushT EC] Could not set enableScreenshotCapture: {ex.Message}");
                }
            }
        }

        private void OnGUI()
        {
            if (!this.showValidatorUi)
            {
                return;
            }

            GUILayout.BeginArea(this.validatorRect, GUI.skin.box);
            GUILayout.Label("PushT EC Validator");
            GUILayout.Label($"Status: {this.status}");
            GUILayout.Label($"Check: {this._lastValidationMessage}");
            GUILayout.Label($"Generation: {this._currentGeneration}/{this.generations}");
            GUILayout.Label($"Candidate: {this._currentCandidate}/{Mathf.Max(1, this._totalCandidates)}");
            GUILayout.Label($"Last fitness: {this._lastFitness:F3}");
            GUILayout.Label($"Last success: {this._lastSuccess}");

            if (this.bestCandidate != null)
            {
                GUILayout.Label(
                    $"Best: fit={this.bestCandidate.fitness:F3}, scale={this.bestCandidate.blockScale:F2}, " +
                    $"mass={this.bestCandidate.blockMass:F2}");
            }
            else
            {
                GUILayout.Label("Best: waiting");
            }

            GUILayout.EndArea();
        }
    }
}
