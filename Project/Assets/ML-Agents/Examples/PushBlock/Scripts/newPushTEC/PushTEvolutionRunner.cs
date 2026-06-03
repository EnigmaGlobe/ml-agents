using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace PushTEvolutionMvp
{
    public class PushTEvolutionRunner : MonoBehaviour
    {
        public PushTEvolutionConfig config = new PushTEvolutionConfig();
        public PushTBlockEvaluator evaluator;

        [TextArea] public string status = "Idle";
        public PushTEvolutionRunResult lastRun = new PushTEvolutionRunResult();

        [ContextMenu("Run Evolution")]
        public async void RunEvolution()
        {
            if (this.evaluator == null)
            {
                Debug.LogError("No PushTBlockEvaluator assigned.");
                return;
            }

            this.lastRun = await this.RunEvolutionAsync(this.evaluator);
            ExportAllResultsToCsv();
        }

        [ContextMenu("Export Results to CSV")]
        public void ExportAllResultsToCsv()
        {
            if (this.lastRun == null || this.lastRun.allEvaluations == null || this.lastRun.allEvaluations.Count == 0)
            {
                Debug.LogWarning("[PushT EC] No results to export. Run evolution first.");
                return;
            }

            var outputDir = Path.Combine(Application.dataPath, "ML-Agents", "Examples", "PushBlock", "Scripts", "newPushTEC", "outputs");
            Directory.CreateDirectory(outputDir);
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            ExportSummaryCsv(Path.Combine(outputDir, $"summary_{timestamp}.csv"));
            ExportAllEvaluationsCsv(Path.Combine(outputDir, $"all_evaluations_{timestamp}.csv"));
            ExportEpisodesCsv(Path.Combine(outputDir, $"episodes_{timestamp}.csv"));

            Debug.Log($"[PushT EC] Results exported to {outputDir}");
        }

        private void ExportSummaryCsv(string path)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,fitness,success_rate,mean_progress,iqm_progress,sd_progress,iqr_progress,difficulty,time_score,invalid_penalty,goal_error_score,sd_goal_error,width,height,depth,mass,block_drag,friction,bounciness");

            foreach (var eval in this.lastRun.bestPerGeneration)
            {
                var g = eval.genome;
                lines.Add(
                    $"{eval.generationIndex},{eval.genomeIndex},{eval.fitness:F6},{eval.successRate:F6},{eval.meanProgress:F6},{eval.iqmProgress:F6},{eval.sdProgress:F6},{eval.iqrProgress:F6},{eval.difficulty:F6},{eval.timeScore:F6},{eval.invalidPenalty:F6},{eval.goalErrorScore:F6},{eval.sdGoalError:F6},{g.width:F4},{g.height:F4},{g.depth:F4},{g.mass:F4},{g.blockDrag:F4},{g.friction:F4},{g.bounciness:F4}");
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[PushT EC] Summary CSV: {path}");
        }

        private void ExportAllEvaluationsCsv(string path)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,fitness,success_rate,mean_progress,iqm_progress,sd_progress,iqr_progress,cv_progress,mean_goal_error,sd_goal_error,mean_time_to_goal,mean_reward,difficulty,time_score,goal_error_score,invalid_penalty,block_scale,block_mass,block_drag,friction,width,height,depth,mass,bounciness");

            foreach (var eval in this.lastRun.allEvaluations)
            {
                var g = eval.genome;
                lines.Add(
                    $"{eval.generationIndex},{eval.genomeIndex},{eval.fitness:F6},{eval.successRate:F6},{eval.meanProgress:F6},{eval.iqmProgress:F6},{eval.sdProgress:F6},{eval.iqrProgress:F6},{eval.cvProgress:F6},{eval.meanGoalError:F6},{eval.sdGoalError:F6},{eval.meanTimeToGoal:F6},{eval.meanReward:F6},{eval.difficulty:F6},{eval.timeScore:F6},{eval.goalErrorScore:F6},{eval.invalidPenalty:F6},{g.blockScale:F4},{g.blockMass:F4},{g.blockDrag:F4},{g.friction:F4},{g.width:F4},{g.height:F4},{g.depth:F4},{g.mass:F4},{g.bounciness:F4}");
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[PushT EC] All evaluations CSV: {path}");
        }

        private void ExportEpisodesCsv(string path)
        {
            var lines = new List<string>();
            lines.Add("generation,genome_index,episode_index,episode_seed,success,reward,progress,final_goal_error,time_to_goal,timed_out,invalid_physics");

            foreach (var eval in this.lastRun.allEvaluations)
            {
                foreach (var ep in eval.episodes)
                {
                    lines.Add(
                        $"{eval.generationIndex},{eval.genomeIndex},{ep.episodeIndex},{ep.episodeSeed},{(ep.success ? 1 : 0)},{ep.reward:F6},{ep.progress:F6},{ep.finalGoalError:F6},{ep.timeToGoal:F6},{(ep.timedOut ? 1 : 0)},{(ep.invalidPhysics ? 1 : 0)}");
                }
            }

            File.WriteAllLines(path, lines);
            Debug.Log($"[PushT EC] Episodes CSV: {path}");
        }

        public async Task<PushTEvolutionRunResult> RunEvolutionAsync(PushTBlockEvaluator assignedEvaluator)
        {
            ValidateConfig(this.config);

            var random = new System.Random(this.config.randomSeed);
            var population = CreateInitialPopulation(this.config, random);
            var runResult = new PushTEvolutionRunResult();

            for (int generationIndex = 0; generationIndex < this.config.generationCount; generationIndex++)
            {
                this.status = $"Evaluating generation {generationIndex + 1}/{this.config.generationCount}";

                var evaluations = await EvaluatePopulationAsync(
                    population,
                    assignedEvaluator,
                    this.config,
                    generationIndex,
                    random);

                runResult.allEvaluations.AddRange(evaluations);

                evaluations.Sort((a, b) => b.fitness.CompareTo(a.fitness));
                var bestThisGeneration = evaluations[0];
                runResult.bestPerGeneration.Add(bestThisGeneration);

                if (runResult.bestOverall == null ||
                    bestThisGeneration.fitness > runResult.bestOverall.fitness)
                {
                    runResult.bestOverall = bestThisGeneration;
                }

                population = CreateNextPopulation(evaluations, this.config, random);

                Debug.Log(
                    $"Generation {generationIndex} best fitness: {bestThisGeneration.fitness:F3}, " +
                    $"success rate: {bestThisGeneration.successRate:P0}");
            }

            runResult.totalEpisodesRun = runResult.allEvaluations.Sum(e => e.episodes.Count);

            this.status = runResult.bestOverall == null
                ? "Finished with no result."
                : $"Finished. Best fitness: {runResult.bestOverall.fitness:F3}";

            return runResult;
        }

        private static List<PushTBlockGenome> CreateInitialPopulation(
            PushTEvolutionConfig config,
            System.Random random)
        {
            var population = new List<PushTBlockGenome>(config.populationSize);
            for (int i = 0; i < config.populationSize; i++)
            {
                population.Add(PushTBlockGenome.CreateRandom(config, random));
            }

            return population;
        }

        private static async Task<List<PushTGenomeEvaluation>> EvaluatePopulationAsync(
            IReadOnlyList<PushTBlockGenome> population,
            PushTBlockEvaluator evaluator,
            PushTEvolutionConfig config,
            int generationIndex,
            System.Random random)
        {
            var evaluations = new List<PushTGenomeEvaluation>(population.Count);
            for (int genomeIndex = 0; genomeIndex < population.Count; genomeIndex++)
            {
                var evaluation = await evaluator.EvaluateGenomeAsync(
                    population[genomeIndex],
                    config,
                    generationIndex,
                    genomeIndex,
                    random);
                evaluations.Add(evaluation);
            }

            return evaluations;
        }

        private static List<PushTBlockGenome> CreateNextPopulation(
            IReadOnlyList<PushTGenomeEvaluation> evaluations,
            PushTEvolutionConfig config,
            System.Random random)
        {
            var eliteCount = Mathf.Min(config.eliteCount, evaluations.Count);
            var elites = evaluations
                .Take(eliteCount)
                .Select(x => x.genome.Clone())
                .ToList();

            var nextPopulation = new List<PushTBlockGenome>(config.populationSize);

            foreach (var elite in elites)
            {
                nextPopulation.Add(elite.Clone());
            }

            while (nextPopulation.Count < config.populationSize)
            {
                var parent = elites[random.Next(elites.Count)].Clone();
                parent.Mutate(config, random);
                nextPopulation.Add(parent);
            }

            return nextPopulation;
        }

        private static void ValidateConfig(PushTEvolutionConfig config)
        {
            if (config.populationSize < 2)
            {
                throw new ArgumentException("Population size must be at least 2.");
            }

            if (config.eliteCount < 1 || config.eliteCount > config.populationSize)
            {
                throw new ArgumentException("Elite count must be between 1 and population size.");
            }

            if (config.generationCount < 1)
            {
                throw new ArgumentException("Generation count must be at least 1.");
            }

            if (config.episodesPerGenome < 1)
            {
                throw new ArgumentException("Episodes per genome must be at least 1.");
            }
        }
    }
}
