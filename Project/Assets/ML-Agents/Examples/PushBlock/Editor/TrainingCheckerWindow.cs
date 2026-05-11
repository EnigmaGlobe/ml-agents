using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

public class TrainingCheckerWindow : EditorWindow
{
    private const string DefaultTensorBoardFolderName = "learning_improvement02";
    private const string RewardFileName = "cumulative reward.csv";
    private const string ExtrinsicRewardFileName = "extrinsic reward.csv";
    private const string LearningRateFileName = "learning rate.csv";
    private const string PolicyLossFileName = "losses policy loss.csv";
    private const string ValueLossFileName = "value loss.csv";
    private const string SummaryOutputFileName = "learning_improvement_summary.csv";

    private string m_episodeCsvPath;
    private string m_tensorBoardFolderPath;
    private string m_summaryOutputPath;
    private int m_analysisStartRow = 2;
    private int m_analysisEndRow;

    private int m_selectedTab;
    private Vector2 m_scrollPosition;
    private Vector2 m_reportTextScrollPosition;
    private LearningImprovementReport m_report;
    private string m_lastError;
    private string m_copyableReportText;
    private static readonly string[] TabLabels =
    {
        "Learning Improvement",
        "Block Progress",
        "Agent Efficiency",
        "Control Quality",
        "Reliability",
        "TensorBoard Comparison"
    };

    [MenuItem("ML-Agents/PushBlock/Training Checker")]
    public static void ShowWindow()
    {
        var window = GetWindow<TrainingCheckerWindow>("PushBlock Diagnostics");
        window.minSize = new Vector2(860f, 560f);
        window.InitializeDefaults();
    }

    private void OnEnable()
    {
        InitializeDefaults();
    }

    private void InitializeDefaults()
    {
        var pushBlockRoot = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock");
        var defaultTensorBoardFolderPath = Path.Combine(pushBlockRoot, "TFModels", DefaultTensorBoardFolderName);

        if (string.IsNullOrWhiteSpace(m_episodeCsvPath))
        {
            m_episodeCsvPath = Path.Combine(pushBlockRoot, "learning_improvement.csv");
        }

        if (string.IsNullOrWhiteSpace(m_tensorBoardFolderPath))
        {
            m_tensorBoardFolderPath = defaultTensorBoardFolderPath;
        }

        if (string.IsNullOrWhiteSpace(m_summaryOutputPath))
        {
            m_summaryOutputPath = Path.Combine(pushBlockRoot, "TFModels", "trianercheckfix", SummaryOutputFileName);
        }
    }

    private void OnGUI()
    {
        InitializeDefaults();

        EditorGUILayout.LabelField("PushBlock Diagnostics", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "Use one window for metric-specific diagnostics. Learning Improvement is active now; the other tabs are reserved for the next metric checkers.",
            MessageType.Info);

        DrawTabToolbar();
        DrawActiveTab();
    }

    private void DrawTabToolbar()
    {
        EditorGUILayout.Space();
        m_selectedTab = GUILayout.Toolbar(m_selectedTab, TabLabels);
        EditorGUILayout.Space();
    }

    private void DrawActiveTab()
    {
        switch (m_selectedTab)
        {
            case 0:
                DrawLearningImprovementTab();
                break;
            case 1:
                DrawPlaceholderTab(
                    "Block Progress",
                    "This tab will host the block-progress diagnostics window. Keep the same pattern: auto-pick CSV inputs, compute aligned summaries, and show results in-window.");
                break;
            case 2:
                DrawPlaceholderTab(
                    "Agent Efficiency",
                    "This tab will host the agent-efficiency checker once the analysis rules are finalized.");
                break;
            case 3:
                DrawPlaceholderTab(
                    "Control Quality",
                    "This tab will host the control-quality checker. This one will likely need heavier step-trace visualization.");
                break;
            case 4:
                DrawPlaceholderTab(
                    "Reliability",
                    "This tab will host the reliability checker with rolling success, consistency, and failure-mode summaries.");
                break;
            case 5:
                DrawPlaceholderTab(
                    "TensorBoard Comparison",
                    "This tab will host the cross-metric comparison layer between Unity behavior CSVs and TensorBoard scalar exports.");
                break;
        }
    }

    private void DrawLearningImprovementTab()
    {
        EditorGUILayout.LabelField("Learning Improvement", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "Compute the four learning-improvement metrics from the Unity episode export, keep TensorBoard as a reference source, and write only the minimal summary output.",
            MessageType.None);

        DrawPathSection();
        DrawDetectedFilesSection();
        DrawActionSection();
        DrawResultsSection();
    }

    private void DrawPlaceholderTab(string title, string description)
    {
        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Benchmark", GUILayout.Width(100f)))
            {
                OpenBenchmarkPopupForTopic(title, description);
            }
        }

        EditorGUILayout.HelpBox(description, MessageType.Info);
        EditorGUILayout.LabelField("Status: not implemented in this window yet.");
    }

    private void DrawPathSection()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Inputs", EditorStyles.boldLabel);

        DrawPathField(
            "Episode CSV",
            ref m_episodeCsvPath,
            true,
            "CSV files|*.csv");

        DrawPathField(
            "TensorBoard Folder",
            ref m_tensorBoardFolderPath,
            false,
            string.Empty);

        DrawPathField(
            "Summary Output CSV",
            ref m_summaryOutputPath,
            true,
            "CSV files|*.csv",
            savePanel: true);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Analysis Filter", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "Use CSV row numbers here. Row numbers follow the actual file lines, so the header is row 1 and the first data row is row 2.",
            MessageType.None);
        m_analysisStartRow = EditorGUILayout.IntField("Start Row", m_analysisStartRow);
        m_analysisEndRow = EditorGUILayout.IntField("End Row", m_analysisEndRow);
    }

    private void DrawPathField(string label, ref string value, bool isFile, string extensionFilter, bool savePanel = false)
    {
        using (new EditorGUILayout.HorizontalScope())
        {
            value = EditorGUILayout.TextField(label, value);

            if (GUILayout.Button("Browse", GUILayout.Width(70f)))
            {
                if (isFile)
                {
                    var directory = Directory.Exists(Path.GetDirectoryName(value))
                        ? Path.GetDirectoryName(value)
                        : Application.dataPath;

                    value = savePanel
                        ? EditorUtility.SaveFilePanel(label, directory, Path.GetFileName(value), "csv")
                        : EditorUtility.OpenFilePanelWithFilters(label, directory, new[] { "CSV files", "csv" });
                }
                else
                {
                    var directory = Directory.Exists(value) ? value : Application.dataPath;
                    value = EditorUtility.OpenFolderPanel(label, directory, string.Empty);
                }
            }
        }
    }

    private void DrawDetectedFilesSection()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Detected TensorBoard Files", EditorStyles.boldLabel);

        if (!Directory.Exists(m_tensorBoardFolderPath))
        {
            EditorGUILayout.HelpBox("TensorBoard folder does not exist.", MessageType.Warning);
            return;
        }

        DrawDetectionStatus(RewardFileName, File.Exists(Path.Combine(m_tensorBoardFolderPath, RewardFileName)), true);
        DrawDetectionStatus(ExtrinsicRewardFileName, File.Exists(Path.Combine(m_tensorBoardFolderPath, ExtrinsicRewardFileName)), false);
        DrawDetectionStatus(LearningRateFileName, File.Exists(Path.Combine(m_tensorBoardFolderPath, LearningRateFileName)), false);
        DrawDetectionStatus(PolicyLossFileName, File.Exists(Path.Combine(m_tensorBoardFolderPath, PolicyLossFileName)), false);
        DrawDetectionStatus(ValueLossFileName, File.Exists(Path.Combine(m_tensorBoardFolderPath, ValueLossFileName)), false);
    }

    private void DrawDetectionStatus(string fileName, bool exists, bool required)
    {
        var prefix = exists ? "Found" : required ? "Missing required" : "Missing optional";
        var messageType = exists ? MessageType.None : required ? MessageType.Error : MessageType.Warning;
        EditorGUILayout.HelpBox($"{prefix}: {fileName}", messageType);
    }

    private void DrawActionSection()
    {
        EditorGUILayout.Space();

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Analyze"))
            {
                Analyze();
            }

            GUI.enabled = m_report != null;

            if (GUILayout.Button("Copy Report"))
            {
                EditorGUIUtility.systemCopyBuffer = m_copyableReportText ?? string.Empty;
            }

            if (GUILayout.Button("Reveal Outputs"))
            {
                EditorUtility.RevealInFinder(Path.GetDirectoryName(m_summaryOutputPath));
            }

            GUI.enabled = true;
        }

        if (!string.IsNullOrEmpty(m_lastError))
        {
            EditorGUILayout.HelpBox(m_lastError, MessageType.Error);
        }
    }

    private void DrawResultsSection()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Results", EditorStyles.boldLabel);

        if (m_report == null)
        {
            EditorGUILayout.HelpBox("Run Analyze to compute the learning-improvement metrics and write the summary CSV.", MessageType.Info);
            return;
        }

        var statusType = m_report.Status == "PASS"
            ? MessageType.Info
            : m_report.Status == "PASS_WITH_WARNING" || m_report.Status == "WARN"
                ? MessageType.Warning
                : MessageType.Error;

        using (new EditorGUILayout.HorizontalScope())
        {
            using (new EditorGUILayout.VerticalScope(GUILayout.ExpandWidth(true)))
            {
                EditorGUILayout.HelpBox(
                    $"Computation Status: {m_report.Status}\n" +
                    $"Training Quality: {m_report.OverallBenchmarkVerdict}\n" +
                    $"Confidence: {m_report.StabilityConfidence}\n" +
                    $"Passes: {m_report.PassedChecks}/{m_report.TotalChecks}\n" +
                    $"Episodes used: {m_report.EpisodeCount}",
                    statusType);
            }

            if (GUILayout.Button("Benchmark", GUILayout.Width(100f), GUILayout.Height(48f)))
            {
                OpenBenchmarkPopup();
            }
        }

        m_scrollPosition = EditorGUILayout.BeginScrollView(m_scrollPosition);

        DrawSummaryTable();
        DrawChecksTable();
        DrawCopyableReportSection();

        EditorGUILayout.EndScrollView();
    }

    private void DrawSummaryTable()
    {
        EditorGUILayout.LabelField("Primary Metrics", EditorStyles.boldLabel);
        DrawMetricRow("Computation status", m_report.Status);
        DrawMetricRow("Training quality", m_report.OverallBenchmarkVerdict);
        DrawMetricRow("AUC raw", m_report.AUC.RawAUC);
        DrawMetricRow("AUC normalized", m_report.AUC.NormalizedAUC);
        DrawMetricRow("AUC verdict", m_report.AucVerdict);
        DrawMetricRow("Final window IQM progress", m_report.FinalWindow_IQMProgress);
        DrawMetricRow("Final window mean progress", m_report.FinalWindow_MeanProgress);
        DrawMetricRow("Final window mean reward", m_report.FinalWindow_MeanReward);
        DrawMetricRow("Final window mean goal error", m_report.FinalWindow_MeanGoalError);
        DrawMetricRow("Final window CI lower", m_report.FinalWindow_CI_Lower);
        DrawMetricRow("Final window CI upper", m_report.FinalWindow_CI_Upper);
        DrawMetricRow("Final performance verdict", m_report.FinalPerformanceVerdict);
        DrawMetricRow("Final window SD progress", m_report.Stability_Progress.StandardDeviation);
        DrawMetricRow("Final window IQR progress", m_report.Stability_Progress.IQR);
        DrawMetricRow("Final window CV progress", m_report.Stability_Progress.CoefficientOfVariation);
        DrawMetricRow("Stability verdict", m_report.StabilityVerdict);
        DrawMetricRow("Confidence", m_report.StabilityConfidence);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Secondary Metric", EditorStyles.boldLabel);
        DrawMetricRow("Learning slope", FormatScientific(m_report.LearningSlopeStats.Slope));
        DrawMetricRow("Slope per 100k steps", FormatScientific(m_report.LearningSlopePer100kSteps));
        DrawMetricRow("Slope intercept", m_report.LearningSlopeStats.Intercept);
        DrawMetricRow("Slope R^2", m_report.LearningSlopeStats.RSquared);
        DrawMetricRow("Slope p-value", m_report.LearningSlopeStats.PValue);
        DrawMetricRow("Slope verdict", m_report.LearningSlopeVerdict);
    }

    private void DrawChecksTable()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Checks", EditorStyles.boldLabel);

        foreach (var check in m_report.Checks)
        {
            var icon = check.Passed ? "PASS" : "FAIL";
            EditorGUILayout.LabelField($"{icon}  {check.Name}: {check.Detail}");
        }
    }

    private void DrawCopyableReportSection()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Copyable Debug Report", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "This report includes the formulas, the windows used, and the exact values behind computation status and training-quality verdicts. Copy this back to the developer when a number looks wrong.",
            MessageType.None);

        m_reportTextScrollPosition = EditorGUILayout.BeginScrollView(m_reportTextScrollPosition, GUILayout.MinHeight(220f));
        EditorGUILayout.SelectableLabel(m_copyableReportText ?? string.Empty, EditorStyles.textArea, GUILayout.ExpandHeight(true));
        EditorGUILayout.EndScrollView();
    }

    private void DrawMetricRow(string label, double value)
    {
        EditorGUILayout.LabelField($"{label}: {FormatDouble(value)}");
    }

    private void DrawMetricRow(string label, string value)
    {
        EditorGUILayout.LabelField($"{label}: {value}");
    }

    private void Analyze()
    {
        m_lastError = null;

        try
        {
            ValidateInputs();

            var episodes = LoadEpisodes(m_episodeCsvPath);
            var scalarBundle = LoadScalarBundle(m_tensorBoardFolderPath);
            var filteredEpisodes = FilterEpisodes(
                episodes,
                out var appliedStartRow,
                out var appliedEndRow,
                out var appliedStartStep,
                out var appliedEndStep);
            var alignedRows = BuildAlignedRows(filteredEpisodes, scalarBundle);
            m_report = BuildReport(filteredEpisodes, alignedRows);
            m_report.OriginalEpisodeCount = episodes.Count;
            m_report.AppliedStartRow = appliedStartRow;
            m_report.AppliedEndRow = appliedEndRow;
            m_report.AppliedStartStep = appliedStartStep;
            m_report.AppliedEndStep = appliedEndStep;
            m_copyableReportText = BuildCopyableReport(m_report);

            WriteSummaryCsv(m_report, m_summaryOutputPath);
            AssetDatabase.Refresh();
            Debug.Log($"[learning_improvement_checker]\n{m_copyableReportText}");
        }
        catch (Exception exception)
        {
            m_report = null;
            m_copyableReportText = null;
            m_lastError = exception.Message;
            Debug.LogError($"[learning_improvement_checker] {exception}");
        }
    }

    private void ValidateInputs()
    {
        if (string.IsNullOrWhiteSpace(m_episodeCsvPath) || !File.Exists(m_episodeCsvPath))
        {
            throw new FileNotFoundException("Episode CSV was not found.", m_episodeCsvPath);
        }

        if (string.IsNullOrWhiteSpace(m_tensorBoardFolderPath) || !Directory.Exists(m_tensorBoardFolderPath))
        {
            throw new DirectoryNotFoundException("TensorBoard folder was not found.");
        }

        if (string.IsNullOrWhiteSpace(m_summaryOutputPath))
        {
            throw new InvalidOperationException("Summary Output CSV path is empty.");
        }

        var rewardPath = Path.Combine(m_tensorBoardFolderPath, RewardFileName);
        if (!File.Exists(rewardPath))
        {
            throw new FileNotFoundException("Required TensorBoard file was not found.", rewardPath);
        }
    }

    private List<EpisodeRow> FilterEpisodes(
        List<EpisodeRow> episodes,
        out int appliedStartRow,
        out int appliedEndRow,
        out long appliedStartStep,
        out long appliedEndStep)
    {
        if (episodes.Count == 0)
        {
            throw new InvalidOperationException("Episode CSV contains zero rows after load.");
        }

        var firstDataRow = 2;
        var lastDataRow = episodes.Max(row => row.SourceRowNumber);
        var effectiveStartRow = Math.Max(firstDataRow, m_analysisStartRow);
        var effectiveEndRow = m_analysisEndRow > 0 ? Math.Min(m_analysisEndRow, lastDataRow) : lastDataRow;

        if (effectiveEndRow < effectiveStartRow)
        {
            throw new InvalidOperationException(
                $"Row filter is invalid. start_row={effectiveStartRow}, end_row={effectiveEndRow}.");
        }

        appliedStartRow = effectiveStartRow;
        appliedEndRow = effectiveEndRow;

        var filtered = episodes
            .Where(row => row.SourceRowNumber >= effectiveStartRow && row.SourceRowNumber <= effectiveEndRow)
            .OrderBy(row => row.SourceRowNumber)
            .ToList();

        if (filtered.Count == 0)
        {
            throw new InvalidOperationException(
                $"No episode rows remain after filtering. start_row={effectiveStartRow}, end_row={effectiveEndRow}.");
        }

        appliedStartStep = filtered.First().TrainingStep;
        appliedEndStep = filtered.Last().TrainingStep;

        return filtered;
    }

    private static List<EpisodeRow> LoadEpisodes(string path)
    {
        var rows = new List<EpisodeRow>();
        var lines = ReadAllLinesShared(path);

        if (lines.Length < 2)
        {
            throw new InvalidOperationException("Episode CSV does not contain any data rows.");
        }

        var headerMap = BuildHeaderMap(lines[0]);

        for (var index = 1; index < lines.Length; index++)
        {
            if (string.IsNullOrWhiteSpace(lines[index]))
            {
                continue;
            }

            var cells = SplitCsvLine(lines[index]);
            rows.Add(new EpisodeRow
            {
                SourceRowNumber = index + 1,
                AgentId = ReadString(cells, headerMap, "agent_id"),
                EpisodeId = ReadLong(cells, headerMap, "episode_id"),
                TrainingStep = ReadLong(cells, headerMap, "training_step"),
                Success = ReadInt(cells, headerMap, "success") == 1,
                EndReason = ReadString(cells, headerMap, "end_reason"),
                EpisodeReward = ReadDouble(cells, headerMap, "episode_reward"),
                EpisodeLength = ReadLong(cells, headerMap, "episode_length"),
                TimeToGoal = ReadNullableDouble(cells, headerMap, "time_to_goal"),
                NormalizedTaskProgress = ReadPreferredDouble(cells, headerMap, "normalized_task_progress", "normalized_block_progress"),
                FinalGoalZoneErrorXZ = ReadPreferredDouble(cells, headerMap, "final_goal_zone_error_xz", "final_goal_error"),
                LegacyNormalizedBlockProgress = ReadOptionalDouble(cells, headerMap, "normalized_block_progress"),
                LegacyFinalGoalError = ReadOptionalDouble(cells, headerMap, "final_goal_error"),
                SuccessGoalConsistency = ReadOptionalInt(cells, headerMap, "success_goal_consistency")
            });
        }

        return rows
            .OrderBy(row => row.SourceRowNumber)
            .ToList();
    }

    private static ScalarBundle LoadScalarBundle(string folderPath)
    {
        var bundle = new ScalarBundle
        {
            RewardPoints = LoadScalarPoints(Path.Combine(folderPath, RewardFileName)),
            ExtrinsicRewardByStep = LoadOptionalScalarMap(Path.Combine(folderPath, ExtrinsicRewardFileName)),
            LearningRateByStep = LoadOptionalScalarMap(Path.Combine(folderPath, LearningRateFileName)),
            PolicyLossByStep = LoadOptionalScalarMap(Path.Combine(folderPath, PolicyLossFileName)),
            ValueLossByStep = LoadOptionalScalarMap(Path.Combine(folderPath, ValueLossFileName))
        };

        if (bundle.RewardPoints.Count == 0)
        {
            throw new InvalidOperationException("Cumulative reward CSV did not contain any scalar rows.");
        }

        return bundle;
    }

    private static List<ScalarPoint> LoadScalarPoints(string path)
    {
        if (!File.Exists(path))
        {
            return new List<ScalarPoint>();
        }

        var lines = ReadAllLinesShared(path);
        if (lines.Length < 2)
        {
            return new List<ScalarPoint>();
        }

        var headerMap = BuildHeaderMap(lines[0]);
        var points = new List<ScalarPoint>();

        for (var index = 1; index < lines.Length; index++)
        {
            if (string.IsNullOrWhiteSpace(lines[index]))
            {
                continue;
            }

            var cells = SplitCsvLine(lines[index]);
            points.Add(new ScalarPoint
            {
                Step = ReadLong(cells, headerMap, "Step"),
                Value = ReadDouble(cells, headerMap, "Value")
            });
        }

        return points.OrderBy(point => point.Step).ToList();
    }

    private static string[] ReadAllLinesShared(string path)
    {
        var lines = new List<string>();

        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var reader = new StreamReader(stream))
        {
            while (!reader.EndOfStream)
            {
                lines.Add(reader.ReadLine());
            }
        }

        return lines.ToArray();
    }

    private static Dictionary<long, double> LoadOptionalScalarMap(string path)
    {
        return LoadScalarPoints(path)
            .GroupBy(point => point.Step)
            .ToDictionary(group => group.Key, group => group.Last().Value);
    }

    private static List<AlignedRow> BuildAlignedRows(List<EpisodeRow> episodes, ScalarBundle bundle)
    {
        var aligned = new List<AlignedRow>();
        var totalEpisodes = episodes.Count;
        var totalRewardPoints = bundle.RewardPoints.Count;

        for (var pointIndex = 0; pointIndex < totalRewardPoints; pointIndex++)
        {
            var rewardPoint = bundle.RewardPoints[pointIndex];
            var startIndex = (int)Math.Floor((double)(pointIndex * totalEpisodes) / totalRewardPoints);
            var endExclusive = (int)Math.Floor((double)((pointIndex + 1) * totalEpisodes) / totalRewardPoints);

            if (endExclusive <= startIndex)
            {
                continue;
            }

            var windowEpisodes = episodes
                .Skip(startIndex)
                .Take(endExclusive - startIndex)
                .ToList();

            if (windowEpisodes.Count == 0)
            {
                continue;
            }

            var successfulTimes = windowEpisodes
                .Where(row => row.Success && row.TimeToGoal.HasValue && row.TimeToGoal.Value >= 0d)
                .Select(row => row.TimeToGoal.Value)
                .ToList();

            aligned.Add(new AlignedRow
            {
                AgentId = windowEpisodes[0].AgentId,
                BucketIndex = pointIndex + 1,
                TensorBoardStep = rewardPoint.Step,
                EpisodeCount = windowEpisodes.Count,
                StartSourceRow = windowEpisodes.First().SourceRowNumber,
                EndSourceRow = windowEpisodes.Last().SourceRowNumber,
                MeanReward = windowEpisodes.Average(row => row.EpisodeReward),
                SuccessRate = windowEpisodes.Average(row => row.Success ? 1d : 0d),
                MeanEpisodeLength = windowEpisodes.Average(row => row.EpisodeLength),
                MeanTimeToGoal = successfulTimes.Count > 0 ? successfulTimes.Average() : double.NaN,
                MeanNormalizedProgress = windowEpisodes.Average(row => row.NormalizedTaskProgress),
                StandardDeviationNormalizedProgress = ComputeStandardDeviation(windowEpisodes.Select(row => row.NormalizedTaskProgress)),
                MeanFinalGoalError = windowEpisodes.Average(row => row.FinalGoalZoneErrorXZ),
                TensorBoardCumulativeReward = rewardPoint.Value,
                TensorBoardExtrinsicReward = TryGetScalar(bundle.ExtrinsicRewardByStep, rewardPoint.Step),
                TensorBoardLearningRate = TryGetScalar(bundle.LearningRateByStep, rewardPoint.Step),
                TensorBoardPolicyLoss = TryGetScalar(bundle.PolicyLossByStep, rewardPoint.Step),
                TensorBoardValueLoss = TryGetScalar(bundle.ValueLossByStep, rewardPoint.Step)
            });
        }

        return aligned;
    }

    private static LearningImprovementReport BuildReport(List<EpisodeRow> episodes, List<AlignedRow> alignedRows)
    {
        var orderedEpisodes = episodes
            .OrderBy(row => row.TrainingStep)
            .ThenBy(row => row.EpisodeId)
            .ToList();

        var report = new LearningImprovementReport
        {
            EpisodeCount = orderedEpisodes.Count,
            AlignedRows = alignedRows ?? new List<AlignedRow>()
        };

        if (orderedEpisodes.Count < 3)
        {
            report.AucVerdict = "FAIL";
            report.FinalPerformanceVerdict = "FAIL";
            report.StabilityVerdict = "UNSTABLE_FAIL";
            report.StabilityConfidence = "LOW";
            report.LearningSlopeVerdict = "FLAT";
            report.OverallBenchmarkVerdict = "FAIL";
            report.Status = "FAIL";
            report.Checks = new List<LearningCheck>
            {
                BuildCheck(
                    "Sufficient episode rows",
                    false,
                    "Need at least 3 episode rows to compute the four learning-improvement metrics.")
            };
            report.TotalChecks = report.Checks.Count;
            report.PassedChecks = 0;
            return report;
        }

        var rollingWindowSize = ComputeRollingWindowSize(orderedEpisodes.Count);
        var rollingSeries = BuildRollingProgressSeries(orderedEpisodes, rollingWindowSize);
        var xs = rollingSeries.Select(point => point.Step).Select(step => (double)step).ToList();
        var ys = rollingSeries.Select(point => point.Value).ToList();

        report.RollingWindowSize = rollingWindowSize;
        report.AUC = ComputeAUC(xs, ys);

        var finalWindowSize = ComputeFinalWindowSize(orderedEpisodes.Count);
        var finalWindow = orderedEpisodes.Skip(Math.Max(0, orderedEpisodes.Count - finalWindowSize)).ToList();
        var finalProgress = finalWindow.Select(row => row.NormalizedTaskProgress).ToList();

        report.FinalWindowSize = finalWindow.Count;
        report.FinalWindow_MeanProgress = finalProgress.Average();
        report.FinalWindow_IQMProgress = ComputeIQM(finalProgress);
        report.FinalWindow_MeanReward = finalWindow.Average(row => row.EpisodeReward);
        report.FinalWindow_MeanGoalError = finalWindow.Average(row => row.FinalGoalZoneErrorXZ);
        report.FinalWindow_SuccessRate = finalWindow.Average(row => row.Success ? 1d : 0d);
        report.FinalWindow_MinProgress = finalProgress.Min();
        report.FinalWindow_MaxProgress = finalProgress.Max();
        report.FinalWindow_MedianProgress = ComputeMedian(finalProgress);
        report.FinalWindow_Q1Progress = ComputePercentile(finalProgress, 0.25d);
        report.FinalWindow_Q3Progress = ComputePercentile(finalProgress, 0.75d);

        var bootstrap = ComputeBootstrapCI(finalProgress);
        report.FinalWindow_CI_Lower = bootstrap.Lower;
        report.FinalWindow_CI_Upper = bootstrap.Upper;

        report.Stability_Progress = ComputeStability(finalProgress);
        report.LearningSlopeStats = ComputeSlopeStats(xs, ys);
        report.LearningSlopePer100kSteps = report.LearningSlopeStats.Slope * 100000d;
        report.AucVerdict = EvaluateAucVerdict(report.AUC.NormalizedAUC);
        report.FinalPerformanceVerdict = EvaluateFinalPerformanceVerdict(report.FinalWindow_SuccessRate, report.FinalWindow_IQMProgress);
        report.StabilityVerdict = EvaluateStabilityVerdict(report.Stability_Progress, out report.StabilityConfidence);
        report.LearningSlopeVerdict = EvaluateSlopeVerdict(report.LearningSlopePer100kSteps);
        report.OverallBenchmarkVerdict = EvaluateOverallBenchmarkVerdict(
            report.AucVerdict,
            report.FinalPerformanceVerdict,
            report.StabilityVerdict,
            report.StabilityConfidence);
        report.Status = report.OverallBenchmarkVerdict;

        report.Checks = new List<LearningCheck>
        {
            BuildCheck(
                "AUC computed",
                !double.IsNaN(report.AUC.NormalizedAUC),
                $"normalized_auc={FormatDouble(report.AUC.NormalizedAUC)}"),
            BuildCheck(
                "Final window computed",
                !double.IsNaN(report.FinalWindow_IQMProgress) && !double.IsNaN(report.FinalWindow_CI_Lower) && !double.IsNaN(report.FinalWindow_CI_Upper),
                $"final_iqm={FormatDouble(report.FinalWindow_IQMProgress)}, ci=[{FormatDouble(report.FinalWindow_CI_Lower)}, {FormatDouble(report.FinalWindow_CI_Upper)}]"),
            BuildCheck(
                "Stability computed",
                !double.IsNaN(report.Stability_Progress.StandardDeviation) && !double.IsNaN(report.Stability_Progress.IQR),
                $"sd={FormatDouble(report.Stability_Progress.StandardDeviation)}, iqr={FormatDouble(report.Stability_Progress.IQR)}, cv={FormatDouble(report.Stability_Progress.CoefficientOfVariation)}"),
            BuildCheck(
                "Learning slope computed",
                !double.IsNaN(report.LearningSlopeStats.Slope),
                $"slope={FormatDouble(report.LearningSlopeStats.Slope)}, r2={FormatDouble(report.LearningSlopeStats.RSquared)}")
        };

        report.TotalChecks = report.Checks.Count;
        report.PassedChecks = report.Checks.Count(check => check.Passed);
        report.Status = report.PassedChecks == report.TotalChecks ? "PASS" : report.PassedChecks >= 2 ? "WARN" : "FAIL";

        return report;
    }

    private static string EvaluateAucVerdict(double normalizedAuc)
    {
        if (normalizedAuc >= 0.60d) return "EXCELLENT";
        if (normalizedAuc >= 0.45d) return "STRONG";
        if (normalizedAuc >= 0.30d) return "PASS";
        if (normalizedAuc >= 0.15d) return "WARN";
        return "FAIL";
    }

    private static string EvaluateFinalPerformanceVerdict(double successRate, double iqmProgress)
    {
        if (successRate >= 0.95d && iqmProgress >= 0.90d) return "EXCELLENT";
        if (successRate >= 0.90d && iqmProgress >= 0.85d) return "STRONG";
        if (successRate >= 0.80d && iqmProgress >= 0.70d) return "PASS";
        if (successRate >= 0.60d || iqmProgress >= 0.50d) return "WARN";
        return "FAIL";
    }

    private static string EvaluateStabilityVerdict(LearningStability stability, out string confidenceLabel)
    {
        if (stability.IQR <= 0.10d && (stability.StandardDeviation > 0.50d || stability.CoefficientOfVariation > 1.00d))
        {
            confidenceLabel = "MEDIUM";
            return "OUTLIER_WARNING";
        }

        if (stability.IQR <= 0.10d && stability.StandardDeviation <= 0.25d)
        {
            confidenceLabel = "HIGH";
            return "HIGH_CONFIDENCE";
        }

        if (stability.IQR <= 0.25d)
        {
            confidenceLabel = "MEDIUM";
            return "MEDIUM_CONFIDENCE";
        }

        if (stability.IQR > 0.50d || stability.CoefficientOfVariation > 1.00d)
        {
            confidenceLabel = "LOW";
            return "LOW_CONFIDENCE";
        }

        confidenceLabel = "LOW";
        return "UNSTABLE_FAIL";
    }

    private static string EvaluateSlopeVerdict(double slopePer100kSteps)
    {
        if (slopePer100kSteps >= 0.05d) return "STRONG";
        if (slopePer100kSteps > 0d) return "WEAK";
        if (Math.Abs(slopePer100kSteps) <= 0.0001d) return "FLAT";
        return "NEGATIVE";
    }

    private static string EvaluateOverallBenchmarkVerdict(string aucVerdict, string finalVerdict, string stabilityVerdict, string stabilityConfidence)
    {
        if (aucVerdict == "FAIL" || finalVerdict == "FAIL")
        {
            return "FAIL";
        }

        if (aucVerdict == "WARN" || finalVerdict == "WARN")
        {
            return "WARN";
        }

        var primaryPass = IsPrimaryPassOrBetter(aucVerdict) && IsPrimaryPassOrBetter(finalVerdict);
        if (!primaryPass)
        {
            return "WARN";
        }

        if (stabilityVerdict == "HIGH_CONFIDENCE" && stabilityConfidence == "HIGH")
        {
            return "PASS";
        }

        return "PASS_WITH_WARNING";
    }

    private static bool IsPrimaryPassOrBetter(string verdict)
    {
        return verdict == "PASS" || verdict == "STRONG" || verdict == "EXCELLENT";
    }

    private static string BuildCopyableReport(LearningImprovementReport report)
    {
        var builder = new StringBuilder();
        builder.AppendLine("Learning Improvement Diagnostic Report");
        builder.AppendLine($"ComputationStatus = {report.Status}");
        builder.AppendLine($"TrainingQuality = {report.OverallBenchmarkVerdict}");
        builder.AppendLine($"AUC Verdict = {report.AucVerdict}");
        builder.AppendLine($"Final Performance Verdict = {report.FinalPerformanceVerdict}");
        builder.AppendLine($"Stability Verdict = {report.StabilityVerdict}");
        builder.AppendLine($"Stability Confidence = {report.StabilityConfidence}");
        builder.AppendLine($"Slope Verdict = {report.LearningSlopeVerdict}");
        builder.AppendLine($"PassedChecks = {report.PassedChecks}/{report.TotalChecks}");
        builder.AppendLine($"OriginalEpisodeCount = {report.OriginalEpisodeCount}");
        builder.AppendLine($"FilteredEpisodeCount = {report.EpisodeCount}");
        builder.AppendLine($"AppliedRowFilter = {report.AppliedStartRow}..{report.AppliedEndRow}");
        builder.AppendLine($"FilteredTrainingStepRange = {report.AppliedStartStep}..{report.AppliedEndStep}");
        builder.AppendLine($"RollingWindowSize = {report.RollingWindowSize}");
        builder.AppendLine($"FinalWindowSize = {report.FinalWindowSize}");
        builder.AppendLine($"LearningSlopePer100kSteps = {FormatScientific(report.LearningSlopePer100kSteps)}");
        builder.AppendLine();

        builder.AppendLine("Formulas");
        builder.AppendLine("AUC = trapezoidal area under rolling normalized_task_progress over training_step");
        builder.AppendLine("Final performance window = last 10% of episodes, with a minimum of 25 when available");
        builder.AppendLine("Final window IQM = interquartile mean of normalized_task_progress in the final window");
        builder.AppendLine("Stability = standard deviation, IQR, and CV in the final window");
        builder.AppendLine("Learning slope = OLS slope of rolling progress versus training_step");
        builder.AppendLine("Learning slope is secondary and diagnostic only.");
        builder.AppendLine();

        builder.AppendLine("Computed Metrics");
        builder.AppendLine($"auc_raw = {FormatDouble(report.AUC.RawAUC)}");
        builder.AppendLine($"auc_normalized = {FormatDouble(report.AUC.NormalizedAUC)}");
        builder.AppendLine($"final_window_mean_progress = {FormatDouble(report.FinalWindow_MeanProgress)}");
        builder.AppendLine($"final_window_iqm_progress = {FormatDouble(report.FinalWindow_IQMProgress)}");
        builder.AppendLine($"final_window_ci_lower = {FormatDouble(report.FinalWindow_CI_Lower)}");
        builder.AppendLine($"final_window_ci_upper = {FormatDouble(report.FinalWindow_CI_Upper)}");
        builder.AppendLine($"final_window_mean_reward = {FormatDouble(report.FinalWindow_MeanReward)}");
        builder.AppendLine($"final_window_mean_goal_error = {FormatDouble(report.FinalWindow_MeanGoalError)}");
        builder.AppendLine($"final_window_success_rate = {FormatDouble(report.FinalWindow_SuccessRate)}");
        builder.AppendLine($"final_window_sd_progress = {FormatDouble(report.Stability_Progress.StandardDeviation)}");
        builder.AppendLine($"final_window_iqr_progress = {FormatDouble(report.Stability_Progress.IQR)}");
        builder.AppendLine($"final_window_cv_progress = {FormatDouble(report.Stability_Progress.CoefficientOfVariation)}");
        builder.AppendLine($"learning_slope = {FormatScientific(report.LearningSlopeStats.Slope)}");
        builder.AppendLine($"learning_slope_per_100k_steps = {FormatScientific(report.LearningSlopePer100kSteps)}");
        builder.AppendLine($"learning_slope_intercept = {FormatDouble(report.LearningSlopeStats.Intercept)}");
        builder.AppendLine($"learning_slope_r2 = {FormatDouble(report.LearningSlopeStats.RSquared)}");
        builder.AppendLine($"learning_slope_pvalue = {FormatDouble(report.LearningSlopeStats.PValue)}");
        builder.AppendLine();

        builder.AppendLine("Checks");
        foreach (var check in report.Checks)
        {
            builder.AppendLine($"{(check.Passed ? "PASS" : "FAIL")} {check.Name}: {check.Detail}");
        }

        return builder.ToString();
    }

    private void OpenBenchmarkPopup()
    {
        var content = BuildBenchmarkPopupContent();
        BenchmarkPopupWindow.Show(content);
    }

    private void OpenBenchmarkPopupForTopic(string title, string description)
    {
        var content = new BenchmarkPopupContent
        {
            TopicName = title,
            TopicSummary = description,
            ComputationStatus = "N/A",
            TrainingQuality = "N/A",
            ConfidenceLabel = "N/A",
            Rows = new List<BenchmarkRow>
            {
                new BenchmarkRow
                {
                    Metric = "Benchmark availability",
                    Thresholds = "Not configured yet.",
                    CurrentValue = "No numeric thresholds available.",
                    Verdict = "N/A",
                    Note = "This topic will use the same popup template once its benchmark thresholds are defined."
                }
            }
        };

        BenchmarkPopupWindow.Show(content);
    }

    private BenchmarkPopupContent BuildBenchmarkPopupContent()
    {
        if (m_report == null)
        {
            return new BenchmarkPopupContent
            {
                TopicName = "Learning Improvement",
                TopicSummary = "No computed report is available yet.",
                ComputationStatus = "N/A",
                TrainingQuality = "N/A",
                ConfidenceLabel = "N/A",
                Rows = new List<BenchmarkRow>()
            };
        }

        var rows = new List<BenchmarkRow>
        {
            new BenchmarkRow
            {
                Metric = "AUC of performance curve",
                Thresholds = "Fail < 0.15 | Warn 0.15-0.30 | Pass 0.30-0.45 | Strong 0.45-0.60 | Excellent >= 0.60",
                CurrentValue = $"normalized_auc = {FormatDouble(m_report.AUC.NormalizedAUC)}",
                Verdict = m_report.AucVerdict,
                Note = "Mutually exclusive tiers. Evaluate from Excellent down to Fail."
            },
            new BenchmarkRow
            {
                Metric = "Final performance window",
                Thresholds = "Fail < 0.60 / < 0.50 | Warn >= 0.60 or >= 0.50 | Pass >= 0.80 and >= 0.70 | Strong >= 0.90 and >= 0.85 | Excellent >= 0.95 and >= 0.90",
                CurrentValue = $"success_rate = {FormatDouble(m_report.FinalWindow_SuccessRate)}, iqm_progress = {FormatDouble(m_report.FinalWindow_IQMProgress)}",
                Verdict = m_report.FinalPerformanceVerdict,
                Note = "Highest matching tier only. Success rate is the main task-completion signal."
            },
            new BenchmarkRow
            {
                Metric = "Learning stability",
                Thresholds = "OUTLIER_WARNING if IQR <= 0.10 and SD > 0.50 or CV > 1.00 | HIGH_CONFIDENCE if IQR <= 0.10 and SD <= 0.25 | MEDIUM_CONFIDENCE if IQR <= 0.25 | LOW_CONFIDENCE if IQR > 0.50 or CV > 1.00",
                CurrentValue = $"IQR = {FormatDouble(m_report.Stability_Progress.IQR)}, SD = {FormatDouble(m_report.Stability_Progress.StandardDeviation)}, CV = {FormatDouble(m_report.Stability_Progress.CoefficientOfVariation)}",
                Verdict = $"{m_report.StabilityVerdict} / Confidence {m_report.StabilityConfidence}",
                Note = "Outlier warning is intentionally separate from low confidence."
            },
            new BenchmarkRow
            {
                Metric = "Learning slope",
                Thresholds = "Negative < 0 | Flat approx 0 | Weak > 0 and < 0.05 per 100k steps | Strong >= 0.05 per 100k steps",
                CurrentValue = $"slope_per_100k_steps = {FormatScientific(m_report.LearningSlopePer100kSteps)}",
                Verdict = m_report.LearningSlopeVerdict,
                Note = "Diagnostic only. Does not override primary verdicts."
            }
        };

        return new BenchmarkPopupContent
        {
            TopicName = "Learning Improvement",
            TopicSummary = "Thresholds are task-specific operational benchmarks calibrated for PushBlock. They are not universal RL constants.",
            ComputationStatus = m_report.Status,
            TrainingQuality = m_report.OverallBenchmarkVerdict,
            ConfidenceLabel = m_report.StabilityConfidence,
            Rows = rows
        };
    }

    private static string FormatAlignedRowForDebug(AlignedRow row)
    {
        return $"bucket={row.BucketIndex}, row_range={row.StartSourceRow}..{row.EndSourceRow}, tb_step={row.TensorBoardStep}, n_episodes={row.EpisodeCount}, success_rate={FormatDouble(row.SuccessRate)}, mean_task_progress={FormatDouble(row.MeanNormalizedProgress)}, mean_goal_zone_error={FormatDouble(row.MeanFinalGoalError)}, tb_reward={FormatDouble(row.TensorBoardCumulativeReward)}, tb_policy_loss={FormatNullable(row.TensorBoardPolicyLoss)}, tb_value_loss={FormatNullable(row.TensorBoardValueLoss)}";
    }

    private static string BuildEpisodeWindowSummary(List<EpisodeRow> rows)
    {
        if (rows == null || rows.Count == 0)
        {
            return "No episode rows.";
        }

        var firstRow = rows.First().SourceRowNumber;
        var lastRow = rows.Last().SourceRowNumber;
        var firstStep = rows.First().TrainingStep;
        var lastStep = rows.Last().TrainingStep;
        var successRate = rows.Average(row => row.Success ? 1d : 0d);
        var meanProgress = rows.Average(row => row.NormalizedTaskProgress);
        var meanGoalError = rows.Average(row => row.FinalGoalZoneErrorXZ);
        var meanReward = rows.Average(row => row.EpisodeReward);

        return $"episodes={rows.Count}, row_range={firstRow}..{lastRow}, step_range={firstStep}..{lastStep}, success_rate={FormatDouble(successRate)}, mean_task_progress={FormatDouble(meanProgress)}, mean_goal_zone_error={FormatDouble(meanGoalError)}, mean_reward={FormatDouble(meanReward)}";
    }

    private static void WriteAlignedCsv(List<AlignedRow> rows, string path)
    {
        EnsureOutputDirectory(path);

        var builder = new StringBuilder();
        builder.AppendLine("agent_id,bucket_index,start_source_row,end_source_row,tb_step,n_episodes,mean_reward,success_rate,mean_episode_length,mean_time_to_goal,mean_normalized_task_progress,sd_normalized_task_progress,mean_final_goal_zone_error_xz,tb_cumulative_reward,tb_extrinsic_reward,tb_learning_rate,tb_policy_loss,tb_value_loss");

        foreach (var row in rows)
        {
            builder.AppendLine(string.Join(",",
                EscapeCsv(row.AgentId),
                row.BucketIndex.ToString(CultureInfo.InvariantCulture),
                row.StartSourceRow.ToString(CultureInfo.InvariantCulture),
                row.EndSourceRow.ToString(CultureInfo.InvariantCulture),
                row.TensorBoardStep.ToString(CultureInfo.InvariantCulture),
                row.EpisodeCount.ToString(CultureInfo.InvariantCulture),
                FormatCsv(row.MeanReward),
                FormatCsv(row.SuccessRate),
                FormatCsv(row.MeanEpisodeLength),
                FormatCsv(row.MeanTimeToGoal),
                FormatCsv(row.MeanNormalizedProgress),
                FormatCsv(row.StandardDeviationNormalizedProgress),
                FormatCsv(row.MeanFinalGoalError),
                FormatCsv(row.TensorBoardCumulativeReward),
                FormatCsv(row.TensorBoardExtrinsicReward),
                FormatCsv(row.TensorBoardLearningRate),
                FormatCsv(row.TensorBoardPolicyLoss),
                FormatCsv(row.TensorBoardValueLoss)));
        }

        File.WriteAllText(path, builder.ToString());
    }

    private static void WriteEpisodeSnapshotCsv(List<EpisodeRow> rows, string path)
    {
        EnsureOutputDirectory(path);

        var builder = new StringBuilder();
        builder.AppendLine("source_row_number,agent_id,episode_id,training_step,success,end_reason,episode_reward,episode_length,time_to_goal,normalized_task_progress,final_goal_zone_error_xz,legacy_normalized_block_progress,legacy_final_goal_error,success_goal_consistency");

        foreach (var row in rows)
        {
            builder.AppendLine(string.Join(",",
                row.SourceRowNumber.ToString(CultureInfo.InvariantCulture),
                EscapeCsv(row.AgentId),
                row.EpisodeId.ToString(CultureInfo.InvariantCulture),
                row.TrainingStep.ToString(CultureInfo.InvariantCulture),
                row.Success ? "1" : "0",
                EscapeCsv(row.EndReason),
                FormatCsv(row.EpisodeReward),
                row.EpisodeLength.ToString(CultureInfo.InvariantCulture),
                FormatCsv(row.TimeToGoal),
                FormatCsv(row.NormalizedTaskProgress),
                FormatCsv(row.FinalGoalZoneErrorXZ),
                FormatCsv(row.LegacyNormalizedBlockProgress),
                FormatCsv(row.LegacyFinalGoalError),
                row.SuccessGoalConsistency.ToString(CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(path, builder.ToString());
    }

    private static void WriteSummaryCsv(LearningImprovementReport report, string path)
    {
        EnsureOutputDirectory(path);

        var builder = new StringBuilder();
        builder.AppendLine("computation_status,training_quality,auc_verdict,final_performance_verdict,stability_verdict,stability_confidence,learning_slope_verdict,passed_checks,total_checks,original_episode_count,filtered_episode_count,applied_start_row,applied_end_row,applied_start_step,applied_end_step,rolling_window_size,final_window_size,auc_raw,auc_normalized,final_window_mean_progress,final_window_iqm_progress,final_window_ci_lower,final_window_ci_upper,final_window_mean_reward,final_window_mean_goal_error,final_window_success_rate,final_window_sd_progress,final_window_iqr_progress,final_window_cv_progress,learning_slope,learning_slope_per_100k_steps,learning_slope_intercept,learning_slope_r2,learning_slope_pvalue");
        builder.AppendLine(string.Join(",",
            EscapeCsv(report.Status),
            EscapeCsv(report.OverallBenchmarkVerdict),
            EscapeCsv(report.AucVerdict),
            EscapeCsv(report.FinalPerformanceVerdict),
            EscapeCsv(report.StabilityVerdict),
            EscapeCsv(report.StabilityConfidence),
            EscapeCsv(report.LearningSlopeVerdict),
            report.PassedChecks.ToString(CultureInfo.InvariantCulture),
            report.TotalChecks.ToString(CultureInfo.InvariantCulture),
            report.OriginalEpisodeCount.ToString(CultureInfo.InvariantCulture),
            report.EpisodeCount.ToString(CultureInfo.InvariantCulture),
            report.AppliedStartRow.ToString(CultureInfo.InvariantCulture),
            report.AppliedEndRow.ToString(CultureInfo.InvariantCulture),
            report.AppliedStartStep.ToString(CultureInfo.InvariantCulture),
            report.AppliedEndStep.ToString(CultureInfo.InvariantCulture),
            report.RollingWindowSize.ToString(CultureInfo.InvariantCulture),
            report.FinalWindowSize.ToString(CultureInfo.InvariantCulture),
            FormatCsv(report.AUC.RawAUC),
            FormatCsv(report.AUC.NormalizedAUC),
            FormatCsv(report.FinalWindow_MeanProgress),
            FormatCsv(report.FinalWindow_IQMProgress),
            FormatCsv(report.FinalWindow_CI_Lower),
            FormatCsv(report.FinalWindow_CI_Upper),
            FormatCsv(report.FinalWindow_MeanReward),
            FormatCsv(report.FinalWindow_MeanGoalError),
            FormatCsv(report.FinalWindow_SuccessRate),
            FormatCsv(report.Stability_Progress.StandardDeviation),
            FormatCsv(report.Stability_Progress.IQR),
            FormatCsv(report.Stability_Progress.CoefficientOfVariation),
            FormatCsv(report.LearningSlopeStats.Slope),
            FormatCsv(report.LearningSlopePer100kSteps),
            FormatCsv(report.LearningSlopeStats.Intercept),
            FormatCsv(report.LearningSlopeStats.RSquared),
            FormatCsv(report.LearningSlopeStats.PValue)));

        builder.AppendLine();
        builder.AppendLine("check_name,passed,detail");

        foreach (var check in report.Checks)
        {
            builder.AppendLine(string.Join(",",
                EscapeCsv(check.Name),
                check.Passed ? "1" : "0",
                EscapeCsv(check.Detail)));
        }

        File.WriteAllText(path, builder.ToString());
    }

    private static Dictionary<string, int> BuildHeaderMap(string headerLine)
    {
        var map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var headers = SplitCsvLine(headerLine);

        for (var index = 0; index < headers.Count; index++)
        {
            map[headers[index].Trim()] = index;
        }

        return map;
    }

    private static List<string> SplitCsvLine(string line)
    {
        var cells = new List<string>();
        var builder = new StringBuilder();
        var inQuotes = false;

        foreach (var character in line)
        {
            if (character == '"')
            {
                inQuotes = !inQuotes;
                continue;
            }

            if (character == ',' && !inQuotes)
            {
                cells.Add(builder.ToString());
                builder.Clear();
                continue;
            }

            builder.Append(character);
        }

        cells.Add(builder.ToString());
        return cells;
    }

    private static string ReadString(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        if (!headerMap.TryGetValue(column, out var index) || index >= cells.Count)
        {
            throw new InvalidOperationException($"Required column '{column}' was not found in the CSV.");
        }

        return cells[index];
    }

    private static long ReadLong(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        return long.Parse(ReadString(cells, headerMap, column), CultureInfo.InvariantCulture);
    }

    private static int ReadInt(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        return int.Parse(ReadString(cells, headerMap, column), CultureInfo.InvariantCulture);
    }

    private static double ReadDouble(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        return double.Parse(ReadString(cells, headerMap, column), CultureInfo.InvariantCulture);
    }

    private static double ReadPreferredDouble(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string primaryColumn, string fallbackColumn)
    {
        if (headerMap.ContainsKey(primaryColumn))
        {
            return ReadDouble(cells, headerMap, primaryColumn);
        }

        return ReadDouble(cells, headerMap, fallbackColumn);
    }

    private static double ReadOptionalDouble(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        if (!headerMap.TryGetValue(column, out var index) || index >= cells.Count || string.IsNullOrWhiteSpace(cells[index]))
        {
            return double.NaN;
        }

        return double.Parse(cells[index], CultureInfo.InvariantCulture);
    }

    private static int ReadOptionalInt(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        if (!headerMap.TryGetValue(column, out var index) || index >= cells.Count || string.IsNullOrWhiteSpace(cells[index]))
        {
            return -1;
        }

        var rawValue = cells[index].Trim();
        if (int.TryParse(rawValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out var intValue))
        {
            return intValue;
        }

        if (double.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out var doubleValue))
        {
            return (int)Math.Round(doubleValue, MidpointRounding.AwayFromZero);
        }

        throw new FormatException($"Unable to parse optional integer value '{rawValue}' for column '{column}'.");
    }

    private static double? ReadNullableDouble(IReadOnlyList<string> cells, IReadOnlyDictionary<string, int> headerMap, string column)
    {
        var value = ReadString(cells, headerMap, column);
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return double.Parse(value, CultureInfo.InvariantCulture);
    }

    private static double ComputeStandardDeviation(IEnumerable<double> values)
    {
        var data = values.Where(value => !double.IsNaN(value)).ToList();
        if (data.Count <= 1)
        {
            return 0d;
        }

        var mean = data.Average();
        var variance = data.Sum(value => Math.Pow(value - mean, 2d)) / data.Count;
        return Math.Sqrt(variance);
    }

    private static int ComputeRollingWindowSize(int episodeCount)
    {
        if (episodeCount <= 0)
        {
            return 0;
        }

        if (episodeCount < 50)
        {
            return Math.Max(5, episodeCount / 10);
        }

        return 50;
    }

    private static int ComputeFinalWindowSize(int episodeCount)
    {
        if (episodeCount <= 0)
        {
            return 0;
        }

        return Math.Max(25, (int)Math.Ceiling(episodeCount * 0.10d));
    }

    private static List<CurvePoint> BuildRollingProgressSeries(List<EpisodeRow> episodes, int rollingWindowSize)
    {
        var curve = new List<CurvePoint>();
        if (episodes.Count == 0)
        {
            return curve;
        }

        var window = Math.Max(1, Math.Min(rollingWindowSize, episodes.Count));
        for (var index = 0; index < episodes.Count; index++)
        {
            var start = Math.Max(0, index - window + 1);
            var slice = episodes.GetRange(start, index - start + 1);
            curve.Add(new CurvePoint
            {
                Step = episodes[index].TrainingStep,
                Value = slice.Average(row => row.NormalizedTaskProgress)
            });
        }

        return curve;
    }

    private static AUCResult ComputeAUC(IReadOnlyList<double> xs, IReadOnlyList<double> ys)
    {
        if (xs == null || ys == null || xs.Count != ys.Count || xs.Count < 2)
        {
            return default;
        }

        var rawAuc = 0d;
        for (var index = 1; index < xs.Count; index++)
        {
            var dx = xs[index] - xs[index - 1];
            var averageY = (ys[index] + ys[index - 1]) / 2d;
            rawAuc += dx * averageY;
        }

        var span = xs[xs.Count - 1] - xs[0];
        var normalizedAuc = Math.Abs(span) > double.Epsilon ? rawAuc / span : 0d;

        return new AUCResult
        {
            RawAUC = rawAuc,
            NormalizedAUC = normalizedAuc,
            Span = span
        };
    }

    private static double ComputeIQM(IReadOnlyList<double> values)
    {
        if (values == null || values.Count == 0)
        {
            return double.NaN;
        }

        var sorted = values.Where(value => !double.IsNaN(value)).OrderBy(value => value).ToList();
        if (sorted.Count == 0)
        {
            return double.NaN;
        }

        if (sorted.Count < 4)
        {
            return sorted.Average();
        }

        var q1Index = (int)Math.Floor(sorted.Count * 0.25d);
        var q3Index = (int)Math.Ceiling(sorted.Count * 0.75d);
        q1Index = Math.Max(0, Math.Min(q1Index, sorted.Count - 1));
        q3Index = Math.Max(q1Index + 1, Math.Min(q3Index, sorted.Count));

        var middle = sorted.Skip(q1Index).Take(q3Index - q1Index).ToList();
        return middle.Count > 0 ? middle.Average() : sorted.Average();
    }

    private static double ComputeMedian(IReadOnlyList<double> values)
    {
        if (values == null || values.Count == 0)
        {
            return double.NaN;
        }

        var sorted = values.Where(value => !double.IsNaN(value)).OrderBy(value => value).ToList();
        if (sorted.Count == 0)
        {
            return double.NaN;
        }

        var middle = sorted.Count / 2;
        if (sorted.Count % 2 == 1)
        {
            return sorted[middle];
        }

        return (sorted[middle - 1] + sorted[middle]) / 2d;
    }

    private static double ComputePercentile(IReadOnlyList<double> values, double percentile)
    {
        if (values == null || values.Count == 0)
        {
            return double.NaN;
        }

        var sorted = values.Where(value => !double.IsNaN(value)).OrderBy(value => value).ToList();
        if (sorted.Count == 0)
        {
            return double.NaN;
        }

        var clamped = Math.Max(0d, Math.Min(1d, percentile));
        var position = (sorted.Count - 1) * clamped;
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        if (lower == upper)
        {
            return sorted[lower];
        }

        var weight = position - lower;
        return sorted[lower] * (1d - weight) + sorted[upper] * weight;
    }

    private static ConfidenceInterval ComputeBootstrapCI(IReadOnlyList<double> values, int iterations = 10000, double confidenceLevel = 0.95d)
    {
        var cleaned = values?.Where(value => !double.IsNaN(value)).ToList() ?? new List<double>();
        if (cleaned.Count == 0)
        {
            return new ConfidenceInterval { Lower = double.NaN, Upper = double.NaN };
        }

        if (cleaned.Count < 3)
        {
            var iqm = ComputeIQM(cleaned);
            return new ConfidenceInterval { Lower = iqm, Upper = iqm };
        }

        var rng = new System.Random(42);
        var bootstrap = new List<double>(iterations);
        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var sample = new double[cleaned.Count];
            for (var index = 0; index < cleaned.Count; index++)
            {
                sample[index] = cleaned[rng.Next(cleaned.Count)];
            }

            bootstrap.Add(ComputeIQM(sample));
        }

        bootstrap.Sort();
        var alpha = 1d - confidenceLevel;
        var lowerIndex = Math.Max(0, (int)Math.Floor(alpha / 2d * (bootstrap.Count - 1)));
        var upperIndex = Math.Min(bootstrap.Count - 1, (int)Math.Ceiling((1d - alpha / 2d) * (bootstrap.Count - 1)));
        return new ConfidenceInterval { Lower = bootstrap[lowerIndex], Upper = bootstrap[upperIndex] };
    }

    private static LearningStability ComputeStability(IReadOnlyList<double> values)
    {
        var cleaned = values?.Where(value => !double.IsNaN(value)).ToList() ?? new List<double>();
        if (cleaned.Count == 0)
        {
            return default;
        }

        var mean = cleaned.Average();
        var sd = ComputeStandardDeviation(cleaned);
        var q1 = ComputePercentile(cleaned, 0.25d);
        var q3 = ComputePercentile(cleaned, 0.75d);
        var median = ComputeMedian(cleaned);

        return new LearningStability
        {
            StandardDeviation = sd,
            IQR = q3 - q1,
            CoefficientOfVariation = Math.Abs(mean) > double.Epsilon ? sd / Math.Abs(mean) : double.NaN,
            Min = cleaned.Min(),
            Max = cleaned.Max(),
            Median = median,
            Q1 = q1,
            Q3 = q3
        };
    }

    private static SlopeStats ComputeSlopeStats(IReadOnlyList<double> xs, IReadOnlyList<double> ys)
    {
        if (xs == null || ys == null || xs.Count != ys.Count || xs.Count < 3)
        {
            return default;
        }

        var n = xs.Count;
        var meanX = xs.Average();
        var meanY = ys.Average();
        var ssXY = 0d;
        var ssXX = 0d;
        var ssYY = 0d;

        for (var index = 0; index < n; index++)
        {
            var dx = xs[index] - meanX;
            var dy = ys[index] - meanY;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (Math.Abs(ssXX) < double.Epsilon)
        {
            return default;
        }

        var slope = ssXY / ssXX;
        var intercept = meanY - slope * meanX;
        var rSquared = ssYY > double.Epsilon ? (ssXY * ssXY) / (ssXX * ssYY) : 0d;
        var residualSumSquares = 0d;
        for (var index = 0; index < n; index++)
        {
            var prediction = intercept + slope * xs[index];
            residualSumSquares += Math.Pow(ys[index] - prediction, 2d);
        }

        var standardError = n > 2 ? Math.Sqrt(residualSumSquares / (n - 2) / ssXX) : double.NaN;
        var tStatistic = !double.IsNaN(standardError) && standardError > double.Epsilon ? slope / standardError : double.NaN;
        var pValue = double.NaN;
        if (!double.IsNaN(tStatistic))
        {
            var absT = Math.Abs(tStatistic);
            pValue = 2d * (1d - NormalCDF(absT));
            pValue = Math.Max(0d, Math.Min(1d, pValue));
        }
        else if (double.IsNaN(standardError) || standardError <= double.Epsilon)
        {
            pValue = Math.Abs(slope) <= double.Epsilon ? 1d : 0d;
        }

        return new SlopeStats
        {
            Slope = slope,
            Intercept = intercept,
            RSquared = rSquared,
            StandardError = standardError,
            PValue = pValue,
            SampleCount = n
        };
    }

    private static double NormalCDF(double x)
    {
        return 0.5d * (1d + Erf(x / Math.Sqrt(2d)));
    }

    private static double Erf(double x)
    {
        var sign = x >= 0d ? 1d : -1d;
        x = Math.Abs(x);
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;
        var t = 1d / (1d + p * x);
        var y = 1d - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double ComputeSlope(IEnumerable<double> xValues, IEnumerable<double> yValues)
    {
        var xs = xValues.ToList();
        var ys = yValues.ToList();

        if (xs.Count != ys.Count || xs.Count < 2)
        {
            return 0d;
        }

        var meanX = xs.Average();
        var meanY = ys.Average();
        var numerator = 0d;
        var denominator = 0d;

        for (var index = 0; index < xs.Count; index++)
        {
            var deltaX = xs[index] - meanX;
            numerator += deltaX * (ys[index] - meanY);
            denominator += deltaX * deltaX;
        }

        return Math.Abs(denominator) < double.Epsilon ? 0d : numerator / denominator;
    }

    private static double ComputeCorrelation(IEnumerable<double> xValues, IEnumerable<double> yValues)
    {
        var xs = xValues.ToList();
        var ys = yValues.ToList();

        if (xs.Count != ys.Count || xs.Count < 3)
        {
            return double.NaN;
        }

        var meanX = xs.Average();
        var meanY = ys.Average();
        var numerator = 0d;
        var left = 0d;
        var right = 0d;

        for (var index = 0; index < xs.Count; index++)
        {
            var deltaX = xs[index] - meanX;
            var deltaY = ys[index] - meanY;
            numerator += deltaX * deltaY;
            left += deltaX * deltaX;
            right += deltaY * deltaY;
        }

        var denominator = Math.Sqrt(left * right);
        return denominator <= double.Epsilon ? double.NaN : numerator / denominator;
    }

    private static double? TryGetScalar(Dictionary<long, double> valuesByStep, long step)
    {
        return valuesByStep.TryGetValue(step, out var value) ? value : null;
    }

    private static LearningCheck BuildCheck(string name, bool passed, string detail)
    {
        return new LearningCheck
        {
            Name = name,
            Passed = passed,
            Detail = detail
        };
    }

    private static string FormatDouble(double value)
    {
        return double.IsNaN(value) ? "NaN" : value.ToString("0.####", CultureInfo.InvariantCulture);
    }

    private static string FormatScientific(double value)
    {
        return double.IsNaN(value) ? "NaN" : value.ToString("0.0000E0", CultureInfo.InvariantCulture);
    }

    private static string FormatNullable(double? value)
    {
        return value.HasValue ? FormatDouble(value.Value) : "NA";
    }

    private static string FormatCsv(double value)
    {
        return double.IsNaN(value) ? string.Empty : value.ToString("G17", CultureInfo.InvariantCulture);
    }

    private static string FormatCsv(double? value)
    {
        return value.HasValue ? FormatCsv(value.Value) : string.Empty;
    }

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        if (value.Contains(",") || value.Contains("\""))
        {
            return $"\"{value.Replace("\"", "\"\"")}\"";
        }

        return value;
    }

    private static void EnsureOutputDirectory(string path)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }

    private sealed class EpisodeRow
    {
        public int SourceRowNumber;
        public string AgentId;
        public long EpisodeId;
        public long TrainingStep;
        public bool Success;
        public string EndReason;
        public double EpisodeReward;
        public long EpisodeLength;
        public double? TimeToGoal;
        public double NormalizedTaskProgress;
        public double FinalGoalZoneErrorXZ;
        public double LegacyNormalizedBlockProgress;
        public double LegacyFinalGoalError;
        public int SuccessGoalConsistency;
    }

    private sealed class ScalarPoint
    {
        public long Step;
        public double Value;
    }

    private sealed class ScalarBundle
    {
        public List<ScalarPoint> RewardPoints;
        public Dictionary<long, double> ExtrinsicRewardByStep;
        public Dictionary<long, double> LearningRateByStep;
        public Dictionary<long, double> PolicyLossByStep;
        public Dictionary<long, double> ValueLossByStep;
    }

    private struct CurvePoint
    {
        public long Step;
        public double Value;
    }

    private struct AUCResult
    {
        public double RawAUC;
        public double NormalizedAUC;
        public double Span;
    }

    private struct LearningStability
    {
        public double StandardDeviation;
        public double IQR;
        public double CoefficientOfVariation;
        public double Min;
        public double Max;
        public double Median;
        public double Q1;
        public double Q3;
    }

    private struct SlopeStats
    {
        public double Slope;
        public double Intercept;
        public double RSquared;
        public double StandardError;
        public double PValue;
        public int SampleCount;
    }

    private struct ConfidenceInterval
    {
        public double Lower;
        public double Upper;
    }

    private sealed class AlignedRow
    {
        public string AgentId;
        public int BucketIndex;
        public long TensorBoardStep;
        public int EpisodeCount;
        public int StartSourceRow;
        public int EndSourceRow;
        public double MeanReward;
        public double SuccessRate;
        public double MeanEpisodeLength;
        public double MeanTimeToGoal;
        public double MeanNormalizedProgress;
        public double StandardDeviationNormalizedProgress;
        public double MeanFinalGoalError;
        public double TensorBoardCumulativeReward;
        public double? TensorBoardExtrinsicReward;
        public double? TensorBoardLearningRate;
        public double? TensorBoardPolicyLoss;
        public double? TensorBoardValueLoss;
    }

    private sealed class LearningCheck
    {
        public string Name;
        public bool Passed;
        public string Detail;
    }

    private sealed class LearningImprovementReport
    {
        public string Status;
        public int OriginalEpisodeCount;
        public int EpisodeCount;
        public int PassedChecks;
        public int TotalChecks;
        public int RollingWindowSize;
        public int FinalWindowSize;
        public int AppliedStartRow;
        public int AppliedEndRow;
        public long AppliedStartStep;
        public long AppliedEndStep;
        public AUCResult AUC;
        public string AucVerdict;
        public double FinalWindow_MeanProgress;
        public double FinalWindow_IQMProgress;
        public double FinalWindow_CI_Lower;
        public double FinalWindow_CI_Upper;
        public double FinalWindow_MeanReward;
        public double FinalWindow_MeanGoalError;
        public double FinalWindow_SuccessRate;
        public double FinalWindow_MinProgress;
        public double FinalWindow_MaxProgress;
        public double FinalWindow_MedianProgress;
        public double FinalWindow_Q1Progress;
        public double FinalWindow_Q3Progress;
        public LearningStability Stability_Progress;
        public string StabilityVerdict;
        public string StabilityConfidence;
        public SlopeStats LearningSlopeStats;
        public double LearningSlopePer100kSteps;
        public string FinalPerformanceVerdict;
        public string LearningSlopeVerdict;
        public string OverallBenchmarkVerdict;
        public List<LearningCheck> Checks;
        public List<AlignedRow> AlignedRows;
    }

    private sealed class BenchmarkRow
    {
        public string Metric;
        public string Thresholds;
        public string CurrentValue;
        public string Verdict;
        public string Note;
    }

    private sealed class BenchmarkPopupContent
    {
        public string TopicName;
        public string TopicSummary;
        public string ComputationStatus;
        public string TrainingQuality;
        public string ConfidenceLabel;
        public List<BenchmarkRow> Rows;
    }

    private sealed class BenchmarkPopupWindow : EditorWindow
    {
        private BenchmarkPopupContent m_content;
        private Vector2 m_scrollPosition;

        public static void Show(BenchmarkPopupContent content)
        {
            var window = CreateInstance<BenchmarkPopupWindow>();
            window.titleContent = new GUIContent("Benchmark");
            window.m_content = content;
            window.minSize = new Vector2(640f, 420f);
            window.ShowUtility();
        }

        private void OnGUI()
        {
            if (m_content == null)
            {
                EditorGUILayout.HelpBox("No benchmark content is available.", MessageType.Warning);
                return;
            }

            EditorGUILayout.LabelField($"{m_content.TopicName} Benchmark", EditorStyles.boldLabel);
            if (!string.IsNullOrWhiteSpace(m_content.TopicSummary))
            {
                EditorGUILayout.HelpBox(m_content.TopicSummary, MessageType.None);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField($"Computation Status: {m_content.ComputationStatus}");
            EditorGUILayout.LabelField($"Training Quality: {m_content.TrainingQuality}");
            if (!string.IsNullOrWhiteSpace(m_content.ConfidenceLabel))
            {
                EditorGUILayout.LabelField($"Confidence: {m_content.ConfidenceLabel}");
            }

            EditorGUILayout.Space();

            m_scrollPosition = EditorGUILayout.BeginScrollView(m_scrollPosition);
            foreach (var row in m_content.Rows ?? new List<BenchmarkRow>())
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.LabelField(row.Metric, EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Thresholds: {row.Thresholds}");
                EditorGUILayout.LabelField($"Current: {row.CurrentValue}");
                EditorGUILayout.LabelField($"Verdict: {row.Verdict}");
                if (!string.IsNullOrWhiteSpace(row.Note))
                {
                    EditorGUILayout.LabelField($"Note: {row.Note}");
                }
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(4f);
            }

            EditorGUILayout.EndScrollView();
        }
    }
}
