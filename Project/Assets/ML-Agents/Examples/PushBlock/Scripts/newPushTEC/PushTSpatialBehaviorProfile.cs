using System;
using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Aggregated spatial-behavior profile for a genome evaluation.
    /// Derived from M4 (Spatial Distribution) in xr-metrics-suite/docs/Spatial/Specs.md,
    /// adapted to PushBlock as agent/block/goal geometry.
    /// </summary>
    [Serializable]
    public class PushTSpatialBehaviorProfile
    {
        // Per-genome means across episodes (block trajectory is primary task signal)
        public float meanBlockNetDisplacement;
        public float meanAgentNetDisplacement;
        public float meanBlockRadialSpread;
        public float meanAgentRadialSpread;
        public float meanTaskCentroidCentrality;
        public float meanSceneCentroidCentrality;

        // Per-genome variability across episodes (behavioral stability)
        public float sdBlockNetDisplacement;
        public float sdBlockRadialSpread;
        public float sdTaskCentroidCentrality;

        public PushTSpatialBehaviorProfile Clone()
        {
            return new PushTSpatialBehaviorProfile
            {
                meanBlockNetDisplacement = this.meanBlockNetDisplacement,
                meanAgentNetDisplacement = this.meanAgentNetDisplacement,
                meanBlockRadialSpread = this.meanBlockRadialSpread,
                meanAgentRadialSpread = this.meanAgentRadialSpread,
                meanTaskCentroidCentrality = this.meanTaskCentroidCentrality,
                meanSceneCentroidCentrality = this.meanSceneCentroidCentrality,
                sdBlockNetDisplacement = this.sdBlockNetDisplacement,
                sdBlockRadialSpread = this.sdBlockRadialSpread,
                sdTaskCentroidCentrality = this.sdTaskCentroidCentrality
            };
        }
    }
}
