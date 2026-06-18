using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PushTEvolutionMvp
{
    /// <summary>
    /// Static helpers for computing M4-style spatial-distribution metrics
    /// from agent/block trajectories during PushBlock episodes.
    /// </summary>
    public static class PushTSpatialMetrics
    {
        /// <summary>
        /// Net displacement magnitude between first and last points.
        /// Corresponds to M4 net_displacement_magnitude.
        /// </summary>
        public static float NetDisplacementMagnitude(List<Vector3> positions)
        {
            if (positions == null || positions.Count < 2) return 0f;
            return Vector3.Distance(positions[0], positions[positions.Count - 1]);
        }

        /// <summary>
        /// Centroid of a point sequence.
        /// </summary>
        public static Vector3 Centroid(List<Vector3> positions)
        {
            if (positions == null || positions.Count == 0) return Vector3.zero;
            Vector3 sum = Vector3.zero;
            foreach (var p in positions) sum += p;
            return sum / positions.Count;
        }

        /// <summary>
        /// Radial spread: standard deviation of distances from a center.
        /// Corresponds to M4 radial_spread_scene_center.
        /// </summary>
        public static float RadialSpread(List<Vector3> positions, Vector3 center)
        {
            if (positions == null || positions.Count < 2) return 0f;
            var distances = positions.Select(p => Vector3.Distance(p, center)).ToList();
            return CalculateStdDev(distances);
        }

        /// <summary>
        /// Task centroid centrality: negative distance between trajectory centroid and goal.
        /// Higher value means the trajectory centroid is closer to the goal.
        /// Corresponds to M4 attention_centroid_centrality, but centered on the goal.
        /// </summary>
        public static float TaskCentroidCentrality(List<Vector3> positions, Vector3 goal)
        {
            Vector3 c = Centroid(positions);
            return -Vector3.Distance(c, goal);
        }

        /// <summary>
        /// Scene centroid centrality: negative distance between trajectory centroid and scene center.
        /// Corresponds to M4 attention_centroid_centrality centered on the scene.
        /// </summary>
        public static float SceneCentroidCentrality(List<Vector3> positions, Vector3 sceneCenter)
        {
            Vector3 c = Centroid(positions);
            return -Vector3.Distance(c, sceneCenter);
        }

        /// <summary>
        /// Standard deviation of a list of floats (population std dev).
        /// </summary>
        public static float CalculateStdDev(List<float> values)
        {
            if (values == null || values.Count == 0) return 0f;
            float mean = values.Average();
            float sumSq = values.Sum(v =>
            {
                float d = v - mean;
                return d * d;
            });
            return Mathf.Sqrt(sumSq / values.Count);
        }
    }
}
