using System;
using System.IO;
using System.Linq;
using UnityEngine;

namespace PushTEvolutionMvp.CheckpointSwap
{
    /// <summary>
    /// Safely copies the latest ONNX checkpoint into a fixed Unity project path.
    /// Writes checkpoint_ready.json only after the ONNX copy is verified complete.
    /// </summary>
    public static class CheckpointSafeCopier
    {
        public class CopyResult
        {
            public bool success;
            public string errorMessage;
            public string sourcePath;
            public string destinationPath;
            public CheckpointReadyManifest manifest;
        }

        /// <summary>
        /// Finds the latest .onnx in sourceDirectory (by last-write UTC time), copies it
        /// safely to destinationOnnxPath, optionally archives it in historyDirectory, and
        /// writes the manifest only after verification.
        /// </summary>
        public static CopyResult CopyLatestCheckpoint(
            string sourceDirectory,
            string searchPattern,
            string destinationOnnxPath,
            string manifestPath,
            bool keepHistory = true,
            string historyDirectory = null)
        {
            var result = new CopyResult { success = false };

            try
            {
                if (!Directory.Exists(sourceDirectory))
                {
                    result.errorMessage = $"Source directory does not exist: {sourceDirectory}";
                    return result;
                }

                var onnxFiles = Directory.GetFiles(sourceDirectory, searchPattern, SearchOption.TopDirectoryOnly)
                    .Where(f => f.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                    .ToArray();

                if (onnxFiles.Length == 0)
                {
                    result.errorMessage = $"No .onnx files found in {sourceDirectory} matching '{searchPattern}'.";
                    return result;
                }

                var latestSource = onnxFiles
                    .Select(f => new FileInfo(f))
                    .OrderByDescending(fi => fi.LastWriteTimeUtc)
                    .First()
                    .FullName;

                var sourceInfo = new FileInfo(latestSource);
                if (!sourceInfo.Exists || sourceInfo.Length == 0)
                {
                    result.errorMessage = $"Source ONNX is missing or empty: {latestSource}";
                    return result;
                }

                string destDir = Path.GetDirectoryName(Path.GetFullPath(destinationOnnxPath));
                Directory.CreateDirectory(destDir);

                // Step 1: copy to a temp file so Unity never sees a partially-written .onnx.
                string tempPath = destinationOnnxPath + ".tmp";
                File.Copy(latestSource, tempPath, overwrite: true);

                // Step 2: verify temp file size matches source.
                var tempInfo = new FileInfo(tempPath);
                if (!tempInfo.Exists || tempInfo.Length != sourceInfo.Length)
                {
                    SafeDelete(tempPath);
                    result.errorMessage = "Temporary copy verification failed (size mismatch).";
                    return result;
                }

                // Step 3: atomically replace the destination file.
                if (File.Exists(destinationOnnxPath))
                {
                    string backupPath = destinationOnnxPath + ".prev";
                    File.Replace(tempPath, destinationOnnxPath, backupPath);
                }
                else
                {
                    File.Move(tempPath, destinationOnnxPath);
                }

                // Step 4: optional history archive.
                if (keepHistory && !string.IsNullOrEmpty(historyDirectory))
                {
                    Directory.CreateDirectory(historyDirectory);
                    string stepSuffix = TryExtractCheckpointStep(sourceInfo.Name);
                    string historyName = string.IsNullOrEmpty(stepSuffix)
                        ? $"{Path.GetFileNameWithoutExtension(sourceInfo.Name)}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx"
                        : $"checkpoint_{stepSuffix}.onnx";
                    string historyPath = Path.Combine(historyDirectory, historyName);
                    File.Copy(latestSource, historyPath, overwrite: true);
                }

                // Step 5: write the manifest ONLY after ONNX is fully on disk.
                var manifest = new CheckpointReadyManifest
                {
                    run_id = TryExtractRunId(sourceInfo.FullName),
                    checkpoint_step = TryParseCheckpointStep(sourceInfo.Name),
                    source_onnx_path = latestSource,
                    unity_onnx_path = Path.GetFullPath(destinationOnnxPath),
                    copied_at = DateTime.UtcNow.ToString("O"),
                    status = "ready"
                };

                string manifestDirectory = Path.GetDirectoryName(Path.GetFullPath(manifestPath));
                if (!string.IsNullOrEmpty(manifestDirectory) && !Directory.Exists(manifestDirectory))
                {
                    Directory.CreateDirectory(manifestDirectory);
                }

                CheckpointReadyManifest.Save(manifestPath, manifest);

                result.success = true;
                result.sourcePath = latestSource;
                result.destinationPath = destinationOnnxPath;
                result.manifest = manifest;
            }
            catch (Exception ex)
            {
                result.errorMessage = $"Copy failed: {ex.Message}";
            }

            return result;
        }

        static void SafeDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch
            {
                // Best-effort cleanup.
            }
        }

        static string TryExtractRunId(string fullPath)
        {
            try
            {
                var dir = Path.GetDirectoryName(fullPath);
                var parts = dir.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                for (int i = parts.Length - 1; i >= 0; i--)
                {
                    string part = parts[i];
                    if (!string.IsNullOrWhiteSpace(part) &&
                        !part.Equals("results", StringComparison.OrdinalIgnoreCase) &&
                        !part.Equals("config", StringComparison.OrdinalIgnoreCase))
                    {
                        return part;
                    }
                }
            }
            catch { }
            return "";
        }

        static string TryExtractCheckpointStep(string fileName)
        {
            try
            {
                string name = Path.GetFileNameWithoutExtension(fileName);
                var parts = name.Split('_', '-');
                foreach (var part in parts)
                {
                    if (long.TryParse(part, out var step) && step > 0)
                    {
                        return step.ToString("D9");
                    }
                }
            }
            catch { }
            return "";
        }

        static long TryParseCheckpointStep(string fileName)
        {
            try
            {
                string name = Path.GetFileNameWithoutExtension(fileName);
                var parts = name.Split('_', '-');
                foreach (var part in parts)
                {
                    if (long.TryParse(part, out var step) && step > 0)
                    {
                        return step;
                    }
                }
            }
            catch { }
            return -1;
        }
    }
}
