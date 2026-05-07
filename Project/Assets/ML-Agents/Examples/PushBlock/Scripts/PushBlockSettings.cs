using UnityEngine;

public class PushBlockSettings : MonoBehaviour
{
    /// <summary>
    /// The "walking speed" of the agents in the scene.
    /// </summary>
    public float agentRunSpeed;

    /// <summary>
    /// The agent rotation speed.
    /// Every agent will use this setting.
    /// </summary>
    public float agentRotationSpeed;

    /// <summary>
    /// The spawn area margin multiplier.
    /// ex: .9 means 90% of spawn area will be used.
    /// .1 margin will be left (so players don't spawn off of the edge).
    /// The higher this value, the longer training time required.
    /// </summary>
    public float spawnAreaMarginMultiplier;

    /// <summary>
    /// When a goal is scored the ground will switch to this
    /// material for a few seconds.
    /// </summary>
    public Material goalScoredMaterial;

    /// <summary>
    /// When an agent fails, the ground will turn this material for a few seconds.
    /// </summary>
    public Material failMaterial;
    // Parameter levels (Low / Medium / High) exposed in the Inspector.
    public enum Level { Low, Medium, High }

    [Header("Block Mass")]
    public Level blockMassLevel = Level.Medium;
    public float blockMassLow = 1f;
    public float blockMassMedium = 2f;
    public float blockMassHigh = 5f;

    [Header("Block Size (uniform scale)")]
    public Level blockSizeLevel = Level.Medium;
    public float blockSizeLow = 0.75f;
    public float blockSizeMedium = 1.5f;
    public float blockSizeHigh = 2.5f;

    [Header("Surface Friction (static/dynamic)")]
    public Level surfaceFrictionLevel = Level.Medium;
    public float surfaceStaticFrictionLow = 0.2f;
    public float surfaceStaticFrictionMedium = 0.6f;
    public float surfaceStaticFrictionHigh = 1.0f;
    public float surfaceDynamicFrictionLow = 0.2f;
    public float surfaceDynamicFrictionMedium = 0.6f;
    public float surfaceDynamicFrictionHigh = 1.0f;

    [Header("Other Defaults")]
    public float defaultBlockDrag = 0.5f;

    // Read-only accessors to get the numeric value for the selected level.
    public float SelectedBlockMass
    {
        get
        {
            switch (blockMassLevel)
            {
                case Level.Low: return blockMassLow;
                case Level.High: return blockMassHigh;
                default: return blockMassMedium;
            }
        }
    }

    public float SelectedBlockSize
    {
        get
        {
            switch (blockSizeLevel)
            {
                case Level.Low: return blockSizeLow;
                case Level.High: return blockSizeHigh;
                default: return blockSizeMedium;
            }
        }
    }

    public float SelectedStaticFriction
    {
        get
        {
            switch (surfaceFrictionLevel)
            {
                case Level.Low: return surfaceStaticFrictionLow;
                case Level.High: return surfaceStaticFrictionHigh;
                default: return surfaceStaticFrictionMedium;
            }
        }
    }

    public float SelectedDynamicFriction
    {
        get
        {
            switch (surfaceFrictionLevel)
            {
                case Level.Low: return surfaceDynamicFrictionLow;
                case Level.High: return surfaceDynamicFrictionHigh;
                default: return surfaceDynamicFrictionMedium;
            }
        }
    }

}
