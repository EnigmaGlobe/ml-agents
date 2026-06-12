using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Simplified training logger that writes two small CSVs:
    /// 1) training_transition_log.csv   – one row per decision/action transition
    /// 2) episode_genome_log.csv        – one row per episode (genome params + actual applied material values)
    ///
    /// No full observation JSON, no ray raw values, no real-time timestamps.
    /// Keeps just enough state to recalculate rewards from actions and observations.
    /// </summary>
    public class PushTTrainingStepLogger : MonoBehaviour
    {
        [Header("Logging Toggles")]
        [Tooltip("Write per-decision transition CSV.")]
        public bool enableTransitionLogging = true;

        [Tooltip("Write episode-genome CSV (genome params, not repeated per transition).")]
        public bool enableEpisodeGenomeLogging = true;

        [Tooltip("Also log the FrozenEvaluator agent.")]
        public bool logFrozenEvaluator = false;

        [Tooltip("Legacy wide-step CSV (disabled by default).")]
        public bool enableStepLogging = false;

        [Header("Intervals")]
        [Tooltip("Flush writers every N Academy steps.")]
        public int flushIntervalSteps = 500;

        [Header("Output")]
        public string outputSubfolder = "training_logs";

        // ------------------------------------------------------------------
        // Internal
        // ------------------------------------------------------------------
        string m_transitionCsvPath;
        string m_episodeGenomeCsvPath;
        StreamWriter m_transitionWriter;
        StreamWriter m_episodeGenomeWriter;
        int m_lastFlushedStep = -1;
        string m_runStamp;
        readonly HashSet<int> m_loggedEpisodeIds = new HashSet<int>();

        void Awake()
        {
            m_runStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        }

        void OnDestroy()
        {
            Flush();
            m_transitionWriter?.Close();
            m_episodeGenomeWriter?.Close();
        }

        // ------------------------------------------------------------------
        // Public API
        // ------------------------------------------------------------------

        public void LogTransition(in TransitionLogData data)
        {
            if (!enableTransitionLogging) return;
            if (!logFrozenEvaluator && data.isFrozenEvaluator) return;

            EnsureTransitionWriter();
            m_transitionWriter.WriteLine(FormatTransitionLine(in data));
        }

        public void LogEpisodeGenome(int episodeId, int academyStepStart, PushTBlockGenome genome,
            float actualFriction, float actualBounciness)
        {
            if (!enableEpisodeGenomeLogging) return;
            if (genome == null) return;
            if (m_loggedEpisodeIds.Contains(episodeId)) return;
            m_loggedEpisodeIds.Add(episodeId);

            EnsureEpisodeGenomeWriter();
            var sb = new StringBuilder(256);
            sb.Append(episodeId); sb.Append(",");
            sb.Append(academyStepStart); sb.Append(",");
            sb.Append(genome.width.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.height.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.depth.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.mass.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.friction.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.bounciness.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(genome.blockDrag.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(actualFriction.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(actualBounciness.ToString("F4", CultureInfo.InvariantCulture));
            m_episodeGenomeWriter.WriteLine(sb.ToString());
        }

        public void MaybeFlush(int academyStep)
        {
            if (m_lastFlushedStep < 0)
                m_lastFlushedStep = academyStep;

            if (academyStep - m_lastFlushedStep >= flushIntervalSteps)
            {
                Flush();
                m_lastFlushedStep = academyStep;
            }
        }

        public void Flush()
        {
            try
            {
                m_transitionWriter?.Flush();
                m_episodeGenomeWriter?.Flush();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TrainingStepLogger] Flush failed: {e.Message}");
            }
        }

        // ------------------------------------------------------------------
        // Formatting
        // ------------------------------------------------------------------

        string FormatTransitionLine(in TransitionLogData d)
        {
            var sb = new StringBuilder(512);

            // Step alignment
            sb.Append(d.academyStepBefore); sb.Append(",");
            sb.Append(d.academyStepAfter); sb.Append(",");
            sb.Append(d.episodeId); sb.Append(",");
            sb.Append(d.episodeStep); sb.Append(",");
            sb.Append(d.decisionStep); sb.Append(",");

            // Action
            sb.Append(d.actionX.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.actionY.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");

            // Time
            sb.Append(d.gameTimeBefore.ToString("F3", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.gameTimeAfter.ToString("F3", CultureInfo.InvariantCulture)); sb.Append(",");

            // Goal position
            sb.Append(d.goalX.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.goalZ.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");

            // Before state
            sb.Append(d.agentPosXBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.agentPosZBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.agentRotYBefore.ToString("F2", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockPosXBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockPosZBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockVelXBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockVelZBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.goalErrorBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.normalizedProgressBefore.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");

            // After state
            sb.Append(d.agentPosXAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.agentPosZAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.agentRotYAfter.ToString("F2", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockPosXAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockPosZAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockVelXAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.blockVelZAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.goalErrorAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.normalizedProgressAfter.ToString("F4", CultureInfo.InvariantCulture)); sb.Append(",");

            // Reward (with audit)
            sb.Append(d.rewardDelta.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.cumulativeReward.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.stepPenalty.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.progressReward.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.goalReward.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.invalidPenalty.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.rewardOther.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.rewardComponentSum.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");
            sb.Append(d.rewardMismatch.ToString("F6", CultureInfo.InvariantCulture)); sb.Append(",");

            // Episode end flags
            sb.Append(d.success ? 1 : 0); sb.Append(",");
            sb.Append(d.episodeDone ? 1 : 0); sb.Append(",");
            sb.Append(EscapeCsv(d.endReason));

            return sb.ToString();
        }

        // ------------------------------------------------------------------
        // Writers
        // ------------------------------------------------------------------

        void EnsureTransitionWriter()
        {
            if (m_transitionWriter != null) return;

            var dir = GetOutputDir();
            Directory.CreateDirectory(dir);
            m_transitionCsvPath = Path.Combine(dir, $"training_transition_log_{m_runStamp}.csv");
            m_transitionWriter = new StreamWriter(m_transitionCsvPath, false, Encoding.UTF8, 65536);
            m_transitionWriter.AutoFlush = false;

            var header =
                "academy_step_before,academy_step_after,episode_id,episode_step,decision_step," +
                "action_x,action_y," +
                "game_time_before,game_time_after," +
                "goal_x,goal_z," +
                "agent_pos_x_before,agent_pos_z_before,agent_rot_y_before," +
                "block_pos_x_before,block_pos_z_before,block_vel_x_before,block_vel_z_before," +
                "goal_error_before,normalized_progress_before," +
                "agent_pos_x_after,agent_pos_z_after,agent_rot_y_after," +
                "block_pos_x_after,block_pos_z_after,block_vel_x_after,block_vel_z_after," +
                "goal_error_after,normalized_progress_after," +
                "reward_delta,cumulative_reward," +
                "step_penalty,progress_reward,goal_reward,invalid_penalty,reward_other," +
                "reward_component_sum,reward_mismatch," +
                "success,episode_done,end_reason";
            m_transitionWriter.WriteLine(header);
        }

        void EnsureEpisodeGenomeWriter()
        {
            if (m_episodeGenomeWriter != null) return;

            var dir = GetOutputDir();
            Directory.CreateDirectory(dir);
            m_episodeGenomeCsvPath = Path.Combine(dir, $"episode_genome_log_{m_runStamp}.csv");
            m_episodeGenomeWriter = new StreamWriter(m_episodeGenomeCsvPath, false, Encoding.UTF8, 65536);
            m_episodeGenomeWriter.AutoFlush = false;

            var header = "episode_id,academy_step_start,genome_width,genome_height,genome_depth,genome_mass,genome_friction,genome_bounciness,genome_blockDrag,actual_friction,actual_bounciness";
            m_episodeGenomeWriter.WriteLine(header);
        }

        string GetOutputDir()
        {
            return Path.Combine(
                Application.dataPath,
                "ML-Agents",
                "Examples",
                "PushBlock",
                "Scripts",
                "newPushTEC",
                outputSubfolder);
        }

        static string EscapeCsv(string s)
        {
            if (string.IsNullOrEmpty(s))
                return "";
            if (s.Contains(",") || s.Contains("\"") || s.Contains("\n"))
            {
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            }
            return s;
        }

        // ------------------------------------------------------------------
        // Legacy API (kept for backward compatibility, disabled by default)
        // ------------------------------------------------------------------

        public void LogStep(in StepLogData data)
        {
            // No-op: legacy wide-step CSV is disabled by default.
            // Transition logging (LogTransition) is the preferred path.
        }

        public void LogEpisodeEnd(int academyStep, float episodeReward, bool success, float progress,
            float goalError, Dictionary<string, float> rewardComponents, float totalStepRewardDelta)
        {
            // No-op: summary CSV is disabled by default.
        }
    }

    // ------------------------------------------------------------------
    // Data structs
    // ------------------------------------------------------------------

    [Serializable]
    public struct StepLogData
    {
        public int academyStep;
        public int episodeId;
        public int episodeStep;
        public string agentName;
        public bool isTrainingAgent;
        public bool isFrozenEvaluator;
        public PushTBlockGenome genome;
        public string actionsJson;
        public int observationCount;
        public float observationMean;
        public float observationMin;
        public float observationMax;
        public string observationsJson;
        public float rewardDelta;
        public float cumulativeReward;
        public string rewardComponentsJson;
        public float normalizedProgress;
        public float goalError;
        public bool success;
        public bool episodeDone;
        public string endReason;
    }

    [Serializable]
    public struct TransitionLogData
    {
        public int academyStepBefore;
        public int academyStepAfter;
        public int episodeId;
        public int episodeStep;
        public int decisionStep;
        public bool isFrozenEvaluator;

        // Action
        public float actionX;
        public float actionY;

        // Time
        public float gameTimeBefore;
        public float gameTimeAfter;

        // Goal position
        public float goalX;
        public float goalZ;

        // Before state
        public float agentPosXBefore;
        public float agentPosZBefore;
        public float agentRotYBefore;
        public float blockPosXBefore;
        public float blockPosZBefore;
        public float blockVelXBefore;
        public float blockVelZBefore;
        public float goalErrorBefore;
        public float normalizedProgressBefore;

        // After state
        public float agentPosXAfter;
        public float agentPosZAfter;
        public float agentRotYAfter;
        public float blockPosXAfter;
        public float blockPosZAfter;
        public float blockVelXAfter;
        public float blockVelZAfter;
        public float goalErrorAfter;
        public float normalizedProgressAfter;

        // Reward components (interval-level, not episode cumulative)
        public float rewardDelta;
        public float cumulativeReward;
        public float stepPenalty;
        public float progressReward;
        public float goalReward;
        public float invalidPenalty;
        public float rewardOther;
        public float rewardComponentSum;
        public float rewardMismatch;

        // Episode end flags
        public bool success;
        public bool episodeDone;
        public string endReason;
    }
}
