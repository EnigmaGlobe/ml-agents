//Put this script on your blue cube.

using System.Collections;
using System.Globalization;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using System.Diagnostics;

public class PushAgentBasic : Agent
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;

    public GameObject area;

    /// <summary>
    /// The area bounds.
    /// </summary>
    [HideInInspector]
    public Bounds areaBounds;

    PushBlockSettings m_PushBlockSettings;

    /// <summary>
    /// The goal to push the block to.
    /// </summary>
    public GameObject goal;

    /// <summary>
    /// The block to be pushed to the goal.
    /// </summary>
    public GameObject block;

    /// <summary>
    /// Detects when the block touches the goal.
    /// </summary>
    [HideInInspector]
    public GoalDetect goalDetect;

    public bool useVectorObs;

    Rigidbody m_BlockRb;  //cached on initialization
    Rigidbody m_AgentRb;  //cached on initialization
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    EnvironmentParameters m_ResetParams;
    // Episode tracking for StatsRecorder
    int m_episodeSteps = 0;
    float m_episodeCumulativeReward = 0f;
    bool m_episodeMetricsRecorded = false;

    [Header("Learning Improvement Logging")]
    public bool enableLearningImprovementLogs = true;
    public string agentIdOverride = "";
    public string learningImprovementLogPrefix = "[learning_improvement]";

    [Header("Agent Efficiency Logging")]
    public bool enableAgentEfficiencyLogs = true;
    public string agentEfficiencyLogPrefix = "[agent_efficiency]";

    [Header("Block Progress Logging")]
    public bool enableBlockProgressLogs = true;
    public string blockProgressLogPrefix = "[block_progress]";

    [Header("Reliability Logging")]
    public bool enableReliabilityLogs = true;
    public string reliabilityLogPrefix = "[reliability]";

    [Header("Control Quality Logging")]
    public bool enableControlQualityLogs = true;
    public string controlQualityLogPrefix = "[control_quality]";

    int m_episodeId = 0;
    Vector3 m_episodeStartAgentPos;
    Vector3 m_episodeStartBlockPos;
    Vector3 m_episodeEndBlockPos;
    Vector3 m_episodeGoalPos;
    float m_startBlockGoalDistance = -1f;
    float m_finalBlockGoalDistance = -1f;
    float m_normalizedBlockProgress = 0f;
    string m_episodeEndReason = "unknown";
    string m_learningImprovementCsvPath;
    string m_agentEfficiencyCsvPath;
    string m_blockProgressCsvPath;
    string m_reliabilityCsvPath;
    string m_controlQualityCsvPath;

    Vector3 m_previousAgentPos;
    Vector3 m_previousBlockPos;
    float m_previousBlockGoalDistance = -1f;
    int m_previousAction = -1;
    bool m_isTouchingBlock = false;
    int m_firstContactStep = -1;
    int m_contactSteps = 0;
    int m_idleSteps = 0;
    int m_actionChanges = 0;
    int m_oppositeActionCount = 0;
    float m_agentPathLength = 0f;
    float m_blockPathLength = 0f;
    float m_usefulBlockDisplacement = 0f;
    float m_pathEfficiency = 0f;
    float m_pushEfficiency = 0f;
    float m_pushRatio = 0f;
    float m_idleRatio = 0f;
    float m_actionSwitchRate = 0f;
    float m_oscillationIndex = 0f;
    float m_startAgentToBlockDistance = 0f;
    float m_progressRate = 0f;

    int[] m_actionCounts = new int[7];
    int m_controlSamples = 0;
    float m_sumGoalVelocity = 0f;
    float m_sumGoalVelocitySq = 0f;
    float m_sumGoalVelocityDeltaSq = 0f;
    float m_prevGoalVelocity = 0f;
    float m_prevGoalVelocityDelta = 0f;
    float m_sumAngularSpeed = 0f;
    float m_sumAngularSpeedSq = 0f;
    int m_sameActionSteps = 0;
    int m_totalActionTransitions = 0;
    float m_totalActionDelta = 0f;
    float m_goalVelocityMean = 0f;
    float m_goalVelocityVariance = 0f;
    float m_goalVelocityAccelerationVariance = 0f;
    float m_goalVelocityJerk = 0f;
    float m_rotationVariance = 0f;
    float m_actionEntropy = 0f;
    float m_repeatedActionRatio = 0f;
    float m_policySmoothness = 0f;

    protected override void Awake()
    {
        base.Awake();
        m_PushBlockSettings = FindAnyObjectByType<PushBlockSettings>();
    }

    public override void Initialize()
    {
        goalDetect = block.GetComponent<GoalDetect>();
        goalDetect.agent = this;

        // Cache the agent rigidbody
        m_AgentRb = GetComponent<Rigidbody>();
        // Cache the block rigidbody
        m_BlockRb = block.GetComponent<Rigidbody>();
        // Get the ground's bounds
        areaBounds = ground.GetComponent<Collider>().bounds;
        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();
        // Keep the CSV alongside the PushBlock example under Assets for easy inspection during editor testing.
        m_learningImprovementCsvPath = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "learning_improvement.csv");
        m_agentEfficiencyCsvPath = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "agent_efficiency.csv");
        m_blockProgressCsvPath = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "block_progress.csv");
        m_reliabilityCsvPath = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "reliability.csv");
        m_controlQualityCsvPath = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "control_quality.csv");
    }

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(2.5f, 0.01f, 2.5f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
    }

    /// <summary>
    /// Called when the agent moves the block into the goal.
    /// </summary>
    public void ScoredAGoal()
    {
        // We use a reward of 5.
        AddReward(5f);
        m_episodeCumulativeReward += 5f;

        CaptureEpisodeEndMetrics(true, "goal");

        // By marking an agent as done AgentReset() will be called automatically.
        EndEpisode();

        // Swap ground material for a bit to indicate we scored.
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));
    }

    /// <summary>
    /// Swap ground material, wait time seconds, then swap back to the regular material.
    /// </summary>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); // Wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        // Track step count per episode
        m_episodeSteps++;

        // Move the agent using the action.
        MoveAgent(actionBuffers.DiscreteActions);

        var currentAction = actionBuffers.DiscreteActions[0];
        UpdateControlQualityMetrics(currentAction);
        UpdateAgentEfficiencyMetrics(currentAction);

        // Penalty given each step to encourage agent to finish task quickly.
        var stepPenalty = -1f / MaxStep;
        AddReward(stepPenalty);
        m_episodeCumulativeReward += stepPenalty;

        // If we reached max steps and haven't recorded metrics yet, record as failure (success=0).
        if (m_episodeSteps >= MaxStep && !m_episodeMetricsRecorded)
        {
            CaptureEpisodeEndMetrics(false, "timeout");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }

    /// <summary>
    /// Resets the block position and velocities.
    /// </summary>
    void ResetBlock()
    {
        // Get a random position for the block.
        block.transform.position = GetRandomSpawnPos();

        // Reset block velocity back to zero.
        m_BlockRb.linearVelocity = Vector3.zero;

        // Reset block angularVelocity back to zero.
        m_BlockRb.angularVelocity = Vector3.zero;
    }

    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        m_episodeId++;

        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        area.transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        ResetBlock();
        transform.position = GetRandomSpawnPos();
        m_AgentRb.linearVelocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;

        // Reset episode-tracking state
        m_episodeSteps = 0;
        m_episodeCumulativeReward = 0f;
        m_episodeMetricsRecorded = false;
        m_episodeEndReason = "unknown";

        CaptureEpisodeStartMetrics();
        CaptureAgentEfficiencyStartMetrics();
        CaptureControlQualityStartMetrics();
        LogReliability(
            "start",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} training_step={2} start_reason=reset",
                GetAgentId(),
                m_episodeId,
                GetTrainingStep()
            )
        );

        SetResetParameters();
    }

    string GetAgentId()
    {
        return string.IsNullOrWhiteSpace(agentIdOverride) ? gameObject.name : agentIdOverride;
    }

    int GetTrainingStep()
    {
        return Academy.Instance != null ? Academy.Instance.StepCount : -1;
    }

    void LogLearningImprovement(string stage, string message)
    {
        if (!enableLearningImprovementLogs)
        {
            return;
        }

        UnityEngine.Debug.Log($"{learningImprovementLogPrefix} {stage} {message}");
    }

    string VectorToCsv(Vector3 v)
    {
        return string.Format(CultureInfo.InvariantCulture, "{0:F4},{1:F4},{2:F4}", v.x, v.y, v.z);
    }

    void CaptureEpisodeStartMetrics()
    {
        m_episodeStartBlockPos = block != null ? block.transform.position : Vector3.zero;
        m_episodeGoalPos = goal != null ? goal.transform.position : Vector3.zero;
        m_startBlockGoalDistance = Vector3.Distance(m_episodeStartBlockPos, m_episodeGoalPos);

        LogLearningImprovement(
            "start",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} training_step={2} start_dist={3:F4} goal_pos=({4}) block_pos=({5})",
                GetAgentId(),
                m_episodeId,
                GetTrainingStep(),
                m_startBlockGoalDistance,
                VectorToCsv(m_episodeGoalPos),
                VectorToCsv(m_episodeStartBlockPos)
            )
        );

        LogBlockProgress(
            "start",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} training_step={2} start_dist={3:F4} goal_pos=({4}) block_pos=({5})",
                GetAgentId(),
                m_episodeId,
                GetTrainingStep(),
                m_startBlockGoalDistance,
                VectorToCsv(m_episodeGoalPos),
                VectorToCsv(m_episodeStartBlockPos)
            )
        );
    }

    void CaptureAgentEfficiencyStartMetrics()
    {
        m_episodeStartAgentPos = transform.position;
        m_previousAgentPos = transform.position;
        m_previousBlockPos = block != null ? block.transform.position : Vector3.zero;
        m_startAgentToBlockDistance = block != null ? Vector3.Distance(m_episodeStartAgentPos, m_previousBlockPos) : 0f;
        m_previousBlockGoalDistance = goal != null && block != null ? Vector3.Distance(m_previousBlockPos, goal.transform.position) : -1f;
        m_previousAction = -1;
        m_isTouchingBlock = false;
        m_firstContactStep = -1;
        m_contactSteps = 0;
        m_idleSteps = 0;
        m_actionChanges = 0;
        m_oppositeActionCount = 0;
        m_agentPathLength = 0f;
        m_blockPathLength = 0f;
        m_usefulBlockDisplacement = 0f;
        m_pathEfficiency = 0f;
        m_pushEfficiency = 0f;
        m_pushRatio = 0f;
        m_idleRatio = 0f;
        m_actionSwitchRate = 0f;
        m_oscillationIndex = 0f;

        LogAgentEfficiency(
            "start",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} training_step={2} agent_start=({3}) block_start=({4})",
                GetAgentId(),
                m_episodeId,
                GetTrainingStep(),
                VectorToCsv(m_episodeStartAgentPos),
                VectorToCsv(m_previousBlockPos)
            )
        );
    }

    void CaptureControlQualityStartMetrics()
    {
        for (var i = 0; i < m_actionCounts.Length; i++)
        {
            m_actionCounts[i] = 0;
        }

        m_controlSamples = 0;
        m_sumGoalVelocity = 0f;
        m_sumGoalVelocitySq = 0f;
        m_sumGoalVelocityDeltaSq = 0f;
        m_prevGoalVelocity = 0f;
        m_prevGoalVelocityDelta = 0f;
        m_sumAngularSpeed = 0f;
        m_sumAngularSpeedSq = 0f;
        m_sameActionSteps = 0;
        m_totalActionTransitions = 0;
        m_totalActionDelta = 0f;
        m_goalVelocityMean = 0f;
        m_goalVelocityVariance = 0f;
        m_goalVelocityAccelerationVariance = 0f;
        m_goalVelocityJerk = 0f;
        m_rotationVariance = 0f;
        m_actionEntropy = 0f;
        m_repeatedActionRatio = 0f;
        m_policySmoothness = 0f;

        LogControlQuality(
            "start",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} training_step={2} start_reason=reset",
                GetAgentId(),
                m_episodeId,
                GetTrainingStep()
            )
        );
    }

    void UpdateAgentEfficiencyMetrics(int currentAction)
    {
        if (block == null || goal == null)
        {
            return;
        }

        var currentAgentPos = transform.position;
        var currentBlockPos = block.transform.position;
        var currentBlockGoalDistance = Vector3.Distance(currentBlockPos, goal.transform.position);

        var agentStepDistance = Vector3.Distance(currentAgentPos, m_previousAgentPos);
        var blockStepDistance = Vector3.Distance(currentBlockPos, m_previousBlockPos);
        var blockGoalDelta = m_previousBlockGoalDistance >= 0f ? m_previousBlockGoalDistance - currentBlockGoalDistance : 0f;

        if (m_episodeSteps > 0 || m_previousAction >= 0)
        {
            m_agentPathLength += agentStepDistance;
            m_blockPathLength += blockStepDistance;

            if (blockGoalDelta > 0f)
            {
                m_usefulBlockDisplacement += blockGoalDelta;
            }

            if (m_isTouchingBlock)
            {
                m_contactSteps++;

                if (m_firstContactStep < 0)
                {
                    m_firstContactStep = m_episodeSteps;
                }
            }

            const float idleEpsilon = 0.001f;
            if (agentStepDistance < idleEpsilon && blockStepDistance < idleEpsilon && Mathf.Abs(blockGoalDelta) < idleEpsilon)
            {
                m_idleSteps++;
            }

            if (m_previousAction >= 0 && currentAction != m_previousAction)
            {
                m_actionChanges++;
            }

            if (IsOppositeAction(m_previousAction, currentAction))
            {
                m_oppositeActionCount++;
            }
        }

        m_previousAgentPos = currentAgentPos;
        m_previousBlockPos = currentBlockPos;
        m_previousBlockGoalDistance = currentBlockGoalDistance;
        m_previousAction = currentAction;
    }

    void UpdateControlQualityMetrics(int currentAction)
    {
        if (block == null || goal == null)
        {
            return;
        }

        var goalDirection = goal.transform.position - block.transform.position;
        var goalDirectionMagnitude = goalDirection.magnitude;
        var normalizedGoalDirection = goalDirectionMagnitude > 0.0001f ? goalDirection / goalDirectionMagnitude : Vector3.zero;

        var goalVelocity = Vector3.Dot(m_BlockRb.linearVelocity, normalizedGoalDirection);
        var angularSpeed = m_AgentRb.angularVelocity.magnitude;
        var goalVelocityDelta = goalVelocity - m_prevGoalVelocity;
        var goalAccelerationDelta = goalVelocityDelta - m_prevGoalVelocityDelta;

        m_controlSamples++;
        m_sumGoalVelocity += goalVelocity;
        m_sumGoalVelocitySq += goalVelocity * goalVelocity;
        m_sumGoalVelocityDeltaSq += goalVelocityDelta * goalVelocityDelta;
        m_sumAngularSpeed += angularSpeed;
        m_sumAngularSpeedSq += angularSpeed * angularSpeed;

        if (m_previousAction >= 0)
        {
            if (currentAction == m_previousAction)
            {
                m_sameActionSteps++;
            }
            else
            {
                m_totalActionTransitions++;
                m_totalActionDelta += Mathf.Abs(currentAction - m_previousAction);
            }
        }

        if (currentAction >= 0 && currentAction < m_actionCounts.Length)
        {
            m_actionCounts[currentAction]++;
        }

        m_goalVelocityJerk += Mathf.Abs(goalAccelerationDelta);
        m_prevGoalVelocityDelta = goalVelocityDelta;
        m_prevGoalVelocity = goalVelocity;
        m_previousAction = currentAction;
    }

    bool IsOppositeAction(int previousAction, int currentAction)
    {
        return (previousAction == 1 && currentAction == 2) ||
               (previousAction == 2 && currentAction == 1) ||
               (previousAction == 3 && currentAction == 4) ||
               (previousAction == 4 && currentAction == 3) ||
               (previousAction == 5 && currentAction == 6) ||
               (previousAction == 6 && currentAction == 5);
    }

    void LogAgentEfficiency(string stage, string message)
    {
        if (!enableAgentEfficiencyLogs)
        {
            return;
        }

        UnityEngine.Debug.Log($"{agentEfficiencyLogPrefix} {stage} {message}");
    }

    void LogBlockProgress(string stage, string message)
    {
        if (!enableBlockProgressLogs)
        {
            return;
        }

        UnityEngine.Debug.Log($"{blockProgressLogPrefix} {stage} {message}");
    }

    void LogControlQuality(string stage, string message)
    {
        if (!enableControlQualityLogs)
        {
            return;
        }

        UnityEngine.Debug.Log($"{controlQualityLogPrefix} {stage} {message}");
    }

    void LogReliability(string stage, string message)
    {
        if (!enableReliabilityLogs)
        {
            return;
        }

        UnityEngine.Debug.Log($"{reliabilityLogPrefix} {stage} {message}");
    }

    void CaptureEpisodeEndMetrics(bool success, string endReason)
    {
        if (m_episodeMetricsRecorded)
        {
            return;
        }

        m_episodeEndReason = endReason;
        m_episodeEndBlockPos = block != null ? block.transform.position : Vector3.zero;
        m_episodeGoalPos = goal != null ? goal.transform.position : Vector3.zero;
        m_finalBlockGoalDistance = Vector3.Distance(m_episodeEndBlockPos, m_episodeGoalPos);

        if (m_startBlockGoalDistance > 0.0001f)
        {
            m_normalizedBlockProgress = (m_startBlockGoalDistance - m_finalBlockGoalDistance) / m_startBlockGoalDistance;
        }
        else
        {
            m_normalizedBlockProgress = 0f;
        }

        var stats = Academy.Instance.StatsRecorder;
        stats.Add("PushBlock/episode_reward", m_episodeCumulativeReward);
        stats.Add("PushBlock/success", success ? 1 : 0);
        stats.Add("PushBlock/episode_length", m_episodeSteps);
        stats.Add("PushBlock/time_to_goal", success ? m_episodeSteps : -1);

        ExportLearningImprovementSummary(success);
        ExportAgentEfficiencySummary(success);
        ExportBlockProgressSummary(success);
        ExportReliabilitySummary(success);
        ExportControlQualitySummary(success);
        m_episodeMetricsRecorded = true;
    }

    void ExportAgentEfficiencySummary(bool success)
    {
        if (!enableAgentEfficiencyLogs)
        {
            return;
        }

        var agentId = GetAgentId();
        var trainingStep = GetTrainingStep();

        if (m_agentPathLength > 0.0001f)
        {
            m_pathEfficiency = m_startAgentToBlockDistance / m_agentPathLength;
            m_pushEfficiency = m_usefulBlockDisplacement / m_agentPathLength;
        }
        else
        {
            m_pathEfficiency = 0f;
            m_pushEfficiency = 0f;
        }

        if (m_episodeSteps > 0)
        {
            m_pushRatio = (float)m_contactSteps / m_episodeSteps;
            m_idleRatio = (float)m_idleSteps / m_episodeSteps;
            m_actionSwitchRate = (float)m_actionChanges / m_episodeSteps;
            m_oscillationIndex = (float)m_oppositeActionCount / m_episodeSteps;
        }

        var contactLatency = m_firstContactStep >= 0 ? m_firstContactStep : -1;

        var row = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6:F4},{7},{8},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4},{15:F4}",
            agentId,
            m_episodeId,
            trainingStep,
            success ? 1 : 0,
            m_episodeEndReason,
            m_agentPathLength,
            m_blockPathLength,
            contactLatency,
            m_contactSteps,
            m_pushRatio,
            m_idleRatio,
            m_actionSwitchRate,
            m_oscillationIndex,
            m_pathEfficiency,
            m_pushEfficiency,
            m_usefulBlockDisplacement
        );

        var dir = Path.GetDirectoryName(m_agentEfficiencyCsvPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var fileExists = File.Exists(m_agentEfficiencyCsvPath);
        if (!fileExists)
        {
            var header = "agent_id,episode_id,training_step,success,end_reason,agent_path_length,block_path_length,contact_latency,contact_steps,push_ratio,idle_ratio,action_switch_rate,oscillation_index,path_efficiency,push_efficiency,useful_block_displacement";
            File.WriteAllText(m_agentEfficiencyCsvPath, header + System.Environment.NewLine);
        }

        File.AppendAllText(m_agentEfficiencyCsvPath, row + System.Environment.NewLine);

        LogAgentEfficiency(
            "end",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} reason={2} steps={3} path_eff={4:F4} push_eff={5:F4} contact_latency={6} push_ratio={7:F4} idle_ratio={8:F4} action_switch_rate={9:F4} oscillation_index={10:F4}",
                agentId,
                m_episodeId,
                m_episodeEndReason,
                m_episodeSteps,
                m_pathEfficiency,
                m_pushEfficiency,
                contactLatency,
                m_pushRatio,
                m_idleRatio,
                m_actionSwitchRate,
                m_oscillationIndex
            )
        );

        LogAgentEfficiency(
            "export",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} recorded=true csv_path={2}",
                agentId,
                m_episodeId,
                m_agentEfficiencyCsvPath
            )
        );
    }

    void ExportBlockProgressSummary(bool success)
    {
        if (!enableBlockProgressLogs)
        {
            return;
        }

        var agentId = GetAgentId();
        var trainingStep = GetTrainingStep();

        if (m_episodeSteps > 0)
        {
            m_progressRate = m_normalizedBlockProgress / m_episodeSteps;
        }
        else
        {
            m_progressRate = 0f;
        }

        var row = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4}",
            agentId,
            m_episodeId,
            trainingStep,
            success ? 1 : 0,
            m_episodeEndReason,
            m_startBlockGoalDistance,
            m_finalBlockGoalDistance,
            m_normalizedBlockProgress,
            m_progressRate,
            m_finalBlockGoalDistance
        );

        var dir = Path.GetDirectoryName(m_blockProgressCsvPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var fileExists = File.Exists(m_blockProgressCsvPath);
        if (!fileExists)
        {
            var header = "agent_id,episode_id,training_step,success,end_reason,start_block_goal_distance,final_block_goal_distance,normalized_block_progress,progress_rate,final_goal_error";
            File.WriteAllText(m_blockProgressCsvPath, header + System.Environment.NewLine);
        }

        File.AppendAllText(m_blockProgressCsvPath, row + System.Environment.NewLine);

        LogBlockProgress(
            "end",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} reason={2} steps={3} start_dist={4:F4} final_dist={5:F4} norm_progress={6:F4} progress_rate={7:F4}",
                agentId,
                m_episodeId,
                m_episodeEndReason,
                m_episodeSteps,
                m_startBlockGoalDistance,
                m_finalBlockGoalDistance,
                m_normalizedBlockProgress,
                m_progressRate
            )
        );

        LogBlockProgress(
            "export",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} recorded=true csv_path={2}",
                agentId,
                m_episodeId,
                m_blockProgressCsvPath
            )
        );
    }

    void ExportReliabilitySummary(bool success)
    {
        if (!enableReliabilityLogs)
        {
            return;
        }

        var agentId = GetAgentId();
        var trainingStep = GetTrainingStep();
        var failureMode = success ? "none" : m_episodeEndReason;

        var row = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5},{6:F4},{7:F4}",
            agentId,
            m_episodeId,
            trainingStep,
            success ? 1 : 0,
            m_episodeEndReason,
            failureMode,
            m_episodeCumulativeReward,
            m_finalBlockGoalDistance
        );

        var dir = Path.GetDirectoryName(m_reliabilityCsvPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var fileExists = File.Exists(m_reliabilityCsvPath);
        if (!fileExists)
        {
            var header = "agent_id,episode_id,training_step,success,end_reason,failure_mode,episode_reward,final_goal_error";
            File.WriteAllText(m_reliabilityCsvPath, header + System.Environment.NewLine);
        }

        File.AppendAllText(m_reliabilityCsvPath, row + System.Environment.NewLine);

        LogReliability(
            "end",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} reason={2} success={3} reward={4:F4} final_goal_error={5:F4}",
                agentId,
                m_episodeId,
                m_episodeEndReason,
                success ? 1 : 0,
                m_episodeCumulativeReward,
                m_finalBlockGoalDistance
            )
        );

        LogReliability(
            "export",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} recorded=true csv_path={2}",
                agentId,
                m_episodeId,
                m_reliabilityCsvPath
            )
        );
    }

    void ExportControlQualitySummary(bool success)
    {
        if (!enableControlQualityLogs)
        {
            return;
        }

        var agentId = GetAgentId();
        var trainingStep = GetTrainingStep();

        if (m_controlSamples > 0)
        {
            m_goalVelocityMean = m_sumGoalVelocity / m_controlSamples;
            m_goalVelocityVariance = (m_sumGoalVelocitySq / m_controlSamples) - (m_goalVelocityMean * m_goalVelocityMean);
            m_goalVelocityAccelerationVariance = (m_sumGoalVelocityDeltaSq / m_controlSamples);
            m_rotationVariance = (m_sumAngularSpeedSq / m_controlSamples) - Mathf.Pow(m_sumAngularSpeed / m_controlSamples, 2f);
            m_policySmoothness = m_totalActionTransitions > 0 ? m_totalActionDelta / m_totalActionTransitions : 0f;

            var entropy = 0f;
            for (var i = 0; i < m_actionCounts.Length; i++)
            {
                if (m_actionCounts[i] <= 0)
                {
                    continue;
                }

                var p = (float)m_actionCounts[i] / m_controlSamples;
                entropy -= p * Mathf.Log(p + 1e-8f);
            }
            m_actionEntropy = entropy;
        }

        m_repeatedActionRatio = m_controlSamples > 0 ? (float)m_sameActionSteps / m_controlSamples : 0f;

        var row = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4}",
            agentId,
            m_episodeId,
            trainingStep,
            success ? 1 : 0,
            m_episodeEndReason,
            m_goalVelocityMean,
            m_goalVelocityVariance,
            m_goalVelocityAccelerationVariance,
            m_goalVelocityJerk / Mathf.Max(1, m_controlSamples),
            m_rotationVariance,
            m_actionEntropy,
            m_repeatedActionRatio,
            m_policySmoothness,
            m_controlSamples,
            m_sumAngularSpeed / Mathf.Max(1, m_controlSamples)
        );

        var dir = Path.GetDirectoryName(m_controlQualityCsvPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var fileExists = File.Exists(m_controlQualityCsvPath);
        if (!fileExists)
        {
            var header = "agent_id,episode_id,training_step,success,end_reason,mean_goal_velocity,goal_velocity_variance,goal_velocity_acceleration_variance,goal_velocity_jerk,rotation_variance,action_entropy,repeated_action_ratio,policy_smoothness,control_samples,mean_angular_speed";
            File.WriteAllText(m_controlQualityCsvPath, header + System.Environment.NewLine);
        }

        File.AppendAllText(m_controlQualityCsvPath, row + System.Environment.NewLine);

        LogControlQuality(
            "end",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} reason={2} mean_goal_velocity={3:F4} goal_velocity_variance={4:F4} action_entropy={5:F4} repeated_action_ratio={6:F4} policy_smoothness={7:F4}",
                agentId,
                m_episodeId,
                m_episodeEndReason,
                m_goalVelocityMean,
                m_goalVelocityVariance,
                m_actionEntropy,
                m_repeatedActionRatio,
                m_policySmoothness
            )
        );

        LogControlQuality(
            "export",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} recorded=true csv_path={2}",
                agentId,
                m_episodeId,
                m_controlQualityCsvPath
            )
        );
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject == block)
        {
            m_isTouchingBlock = true;
            if (m_firstContactStep < 0)
            {
                m_firstContactStep = m_episodeSteps;
            }
        }
    }

    void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject == block)
        {
            m_isTouchingBlock = true;
        }
    }

    void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject == block)
        {
            m_isTouchingBlock = false;
        }
    }

    void ExportLearningImprovementSummary(bool success)
    {
        if (!enableLearningImprovementLogs)
        {
            return;
        }

        var agentId = GetAgentId();
        var trainingStep = GetTrainingStep();
        var finalGoalError = m_finalBlockGoalDistance;

        var row = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6},{7},{8:F4},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4},{15:F4},{16:F4},{17:F4},{18:F4},{19:F4},{20:F4}",
            agentId,
            m_episodeId,
            trainingStep,
            success ? 1 : 0,
            m_episodeEndReason,
            m_episodeCumulativeReward,
            m_episodeSteps,
            success ? m_episodeSteps : -1,
            m_startBlockGoalDistance,
            m_finalBlockGoalDistance,
            m_normalizedBlockProgress,
            finalGoalError,
            m_episodeStartBlockPos.x,
            m_episodeStartBlockPos.y,
            m_episodeStartBlockPos.z,
            m_episodeEndBlockPos.x,
            m_episodeEndBlockPos.y,
            m_episodeEndBlockPos.z,
            m_episodeGoalPos.x,
            m_episodeGoalPos.y,
            m_episodeGoalPos.z
        );

        var dir = Path.GetDirectoryName(m_learningImprovementCsvPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var fileExists = File.Exists(m_learningImprovementCsvPath);
        if (!fileExists)
        {
            var header = "agent_id,episode_id,training_step,success,end_reason,episode_reward,episode_length,time_to_goal,start_block_goal_distance,final_block_goal_distance,normalized_block_progress,final_goal_error,start_block_x,start_block_y,start_block_z,end_block_x,end_block_y,end_block_z,goal_x,goal_y,goal_z";
            File.WriteAllText(m_learningImprovementCsvPath, header + System.Environment.NewLine);
        }

        File.AppendAllText(m_learningImprovementCsvPath, row + System.Environment.NewLine);

        LogLearningImprovement(
            "end",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} reason={2} steps={3} reward={4:F4} start_dist={5:F4} final_dist={6:F4} norm_progress={7:F4}",
                agentId,
                m_episodeId,
                m_episodeEndReason,
                m_episodeSteps,
                m_episodeCumulativeReward,
                m_startBlockGoalDistance,
                m_finalBlockGoalDistance,
                m_normalizedBlockProgress
            )
        );

        LogLearningImprovement(
            "export",
            string.Format(
                CultureInfo.InvariantCulture,
                "agent={0} episode={1} recorded=true csv_path={2}",
                agentId,
                m_episodeId,
                m_learningImprovementCsvPath
            )
        );
    }

    public void SetGroundMaterialFriction()
    {
        var groundCollider = ground.GetComponent<Collider>();

        // Get defaults from PushBlockSettings (selected level), allow env params to override.
        var defaultDynamic = m_PushBlockSettings != null ? m_PushBlockSettings.SelectedDynamicFriction : 0.5f;
        var defaultStatic = m_PushBlockSettings != null ? m_PushBlockSettings.SelectedStaticFriction : 0.5f;

        var dynamicF = m_ResetParams.GetWithDefault("dynamic_friction", defaultDynamic);
        var statF = m_ResetParams.GetWithDefault("static_friction", defaultStatic);

        // Avoid mutating shared PhysicMaterial assets: clone the existing material instance and assign it.
        var oldMat = groundCollider.material;
        if (oldMat != null)
        {
            var instMat = UnityEngine.Object.Instantiate(oldMat);
            instMat.dynamicFriction = dynamicF;
            instMat.staticFriction = statF;
            groundCollider.material = instMat;
        }
        else
        {
            UnityEngine.Debug.LogWarning("PushAgentBasic: ground collider has no PhysicMaterial; skipping friction assignment.");
        }
        UnityEngine.Debug.Log($"PushAgentBasic: Applied ground friction dynamic={dynamicF}, static={statF}");
    }

    public void SetBlockProperties()
    {
        // Use PushBlockSettings selected defaults, but allow EnvironmentParameters to override.
        var defaultScale = m_PushBlockSettings != null ? m_PushBlockSettings.SelectedBlockSize : 1.0f;
        var defaultMass = m_PushBlockSettings != null ? m_PushBlockSettings.SelectedBlockMass : 1.0f;
        var defaultDrag = m_PushBlockSettings != null ? m_PushBlockSettings.defaultBlockDrag : 0.5f;

        var scale = m_ResetParams.GetWithDefault("block_scale", defaultScale);
        // Set the scale of the block (keep consistent height)
        m_BlockRb.transform.localScale = new Vector3(scale, 0.75f, scale);

        // Set the mass (allow env override)
        var mass = m_ResetParams.GetWithDefault("block_mass", defaultMass);
        m_BlockRb.mass = mass;

        // Set the drag of the block
        m_BlockRb.linearDamping = m_ResetParams.GetWithDefault("block_drag", defaultDrag);

        // Debug log applied values for verification
        // Ensure physics tensors update after mass/scale changes
        try
        {
            m_BlockRb.ResetInertiaTensor();
        }
        catch (System.Exception) { }

        var blockCol = block.GetComponent<Collider>();
        if (blockCol != null)
        {
            UnityEngine.Debug.Log($"PushAgentBasic: Applied block properties - mass={m_BlockRb.mass}, scale={m_BlockRb.transform.localScale.x}, linearDamping={m_BlockRb.linearDamping}, colliderBounds={blockCol.bounds}");
        }
        else
        {
            UnityEngine.Debug.Log($"PushAgentBasic: Applied block properties - mass={m_BlockRb.mass}, scale={m_BlockRb.transform.localScale.x}, linearDamping={m_BlockRb.linearDamping}, no collider found on block");
        }
    }

    void SetResetParameters()
    {
        SetGroundMaterialFriction();
        SetBlockProperties();
    }
}
