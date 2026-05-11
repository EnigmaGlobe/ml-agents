using UnityEngine;

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Encoder;
using UnityEditor.Recorder.Input;
#endif

[DisallowMultipleComponent]
[DefaultExecutionOrder(-100)]
public sealed class PushBlockAutoRecorder : MonoBehaviour
{
    [Header("Capture Source")]
    [SerializeField] private Camera[] camerasToRecord;
#pragma warning disable CS0414
    [SerializeField] private int _outputWidth = 1920;
    [SerializeField] private int _outputHeight = 1080;
    [SerializeField] private float _frameRate = 60f;

    [Header("Output")]
    [SerializeField] private string _outputFolderPath = "Recordings/PushBlock";
    [SerializeField] private string _outputFilePrefix = "PushBlock_Record";
    [SerializeField] private bool _captureAudio = false;
    [SerializeField] private bool _captureAlpha = false;
#pragma warning restore CS0414

#if UNITY_EDITOR
    [Serializable]
    private sealed class CameraRecordingState
    {
        public Camera Camera;
        public RenderTexture RenderTexture;
        public RenderTexture OriginalTargetTexture;
        public RecorderController Controller;
        public MovieRecorderSettings MovieSettings;
        public bool RecordingStarted;
    }

    private readonly List<CameraRecordingState> recordingStates = new List<CameraRecordingState>();
#endif

    private void Start()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            return;
        }

        StartRecording();
#endif
    }

    private void OnDisable()
    {
#if UNITY_EDITOR
        StopRecordingAll();
#endif
    }

    private void OnApplicationQuit()
    {
#if UNITY_EDITOR
        StopRecordingAll();
#endif
    }

#if UNITY_EDITOR
    private void StartRecording()
    {
        if (recordingStates.Count > 0)
        {
            return;
        }

        var cameras = GetCamerasToRecord();
        if (cameras.Count == 0)
        {
            Debug.LogWarning("PushBlockAutoRecorder found no cameras to record.");
            return;
        }

        var outputFolder = ResolveOutputFolder(_outputFolderPath);
        Directory.CreateDirectory(outputFolder);

        foreach (var camera in cameras)
        {
            if (camera == null)
            {
                continue;
            }

            var state = new CameraRecordingState
            {
                Camera = camera,
                OriginalTargetTexture = camera.targetTexture
            };

            state.RenderTexture = CreateRenderTexture(_outputWidth, _outputHeight);
            camera.targetTexture = state.RenderTexture;

            var controllerSettings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
            controllerSettings.SetRecordModeToManual();
            controllerSettings.FrameRate = _frameRate;

            state.MovieSettings = ScriptableObject.CreateInstance<MovieRecorderSettings>();
            state.MovieSettings.name = $"PushBlock Auto Recorder - {camera.name}";
            state.MovieSettings.Enabled = true;
            state.MovieSettings.CaptureAudio = _captureAudio;
            state.MovieSettings.CaptureAlpha = _captureAlpha;
            state.MovieSettings.EncoderSettings = new CoreEncoderSettings
            {
                EncodingQuality = CoreEncoderSettings.VideoEncodingQuality.High,
                Codec = CoreEncoderSettings.OutputCodec.MP4
            };
            state.MovieSettings.ImageInputSettings = new RenderTextureInputSettings
            {
                RenderTexture = state.RenderTexture,
                OutputWidth = _outputWidth,
                OutputHeight = _outputHeight,
                FlipFinalOutput = false
            };
            state.MovieSettings.OutputFile = Path.Combine(
                outputFolder,
                $"{_outputFilePrefix}_{SanitizeFileName(camera.name)}_{DateTime.Now:yyyyMMdd_HHmmss}"
            );

            controllerSettings.AddRecorderSettings(state.MovieSettings);
            state.Controller = new RecorderController(controllerSettings);

            RecorderOptions.VerboseMode = false;
            state.Controller.PrepareRecording();

            if (state.Controller.StartRecording())
            {
                state.RecordingStarted = true;
                recordingStates.Add(state);
                Debug.Log($"PushBlockAutoRecorder started: {state.MovieSettings.OutputFile}.mp4");
            }
            else
            {
                RestoreCamera(state);
                Debug.LogWarning($"PushBlockAutoRecorder could not start recording for camera '{camera.name}'.");
            }
        }
    }

    private void StopRecordingAll()
    {
        if (recordingStates.Count == 0)
        {
            return;
        }

        foreach (var state in recordingStates)
        {
            if (state == null)
            {
                continue;
            }

            if (state.Controller != null && state.Controller.IsRecording())
            {
                state.Controller.StopRecording();
            }

            RestoreCamera(state);
        }

        recordingStates.Clear();
    }

    private static string ResolveOutputFolder(string configuredPath)
    {
        if (Path.IsPathRooted(configuredPath))
        {
            return Path.GetFullPath(configuredPath);
        }

        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", configuredPath));
    }

    private List<Camera> GetCamerasToRecord()
    {
        var cameras = new List<Camera>();
        if (camerasToRecord != null)
        {
            foreach (var camera in camerasToRecord)
            {
                if (camera != null && !cameras.Contains(camera))
                {
                    cameras.Add(camera);
                }
            }
        }

        if (cameras.Count == 0 && Camera.main != null)
        {
            cameras.Add(Camera.main);
        }

        return cameras;
    }

    private static RenderTexture CreateRenderTexture(int width, int height)
    {
        var renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32)
        {
            name = $"PushBlockAutoRecorder_{width}x{height}",
            useMipMap = false,
            autoGenerateMips = false
        };
        renderTexture.Create();
        return renderTexture;
    }

    private static void RestoreCamera(CameraRecordingState state)
    {
        if (state == null || state.Camera == null)
        {
            return;
        }

        state.Camera.targetTexture = state.OriginalTargetTexture;

        if (state.RenderTexture != null)
        {
            state.RenderTexture.Release();
            DestroyImmediate(state.RenderTexture);
            state.RenderTexture = null;
        }

        state.Controller = null;
        state.MovieSettings = null;
        state.RecordingStarted = false;
    }

    private static string SanitizeFileName(string value)
    {
        foreach (var invalidChar in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(invalidChar.ToString(), "_");
        }

        return string.IsNullOrWhiteSpace(value) ? "Camera" : value;
    }
#endif
}
