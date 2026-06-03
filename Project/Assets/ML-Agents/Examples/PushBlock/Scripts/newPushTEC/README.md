# PushT Evolution MVP

This folder is a tiny, transplantable EC starter kit for a Unity + ML-Agents
project like PushT.

It is intentionally much smaller than ABL's GP/planner stack. The goal is to
reuse the useful experiment-loop ideas from ABL without bringing over the
domain-specific simulation, planner, editor UI, or vendored plugins.

## Files

- `PushTBlockGenome.cs`
  Defines the block parameters to evolve.
- `PushTEvolutionConfig.cs`
  Defines search settings and parameter ranges.
- `PushTEvolutionResult.cs`
  Defines episode, genome, and run result records.
- `PushTBlockEvaluator.cs`
  Abstract base class for evaluating one genome over one or more episodes.
- `PushTEvolutionRunner.cs`
  Minimal elite-selection + mutation evolution loop.
- `PushTSceneEvaluatorTemplate.cs`
  A concrete scene evaluator template to adapt to your PushT environment.
- `PushTApplyTestGenome.cs`
  The Phase 0 helper: assign block/ground references and run `Apply Test Genome`
  before touching episode scoring or evolution.
- `PushTOneScriptEc.cs`
  The simplest PushT path: clone the scene, add this one script to the agent
  GameObject, press Play, and it runs a tiny EC smoke test.
- `FRD-PushT-Evolution-MVP.md`
  A short functional requirements document explaining how to integrate this
  folder into a new PushT scene.

## Simplest Path

1. Clone your existing PushT scene.
2. Add `PushTOneScriptEc` to the same GameObject as `PushAgentBasic`.
3. Press Play.

The script tries to find the agent's existing `block` and `ground` references,
applies small candidate parameter sets, waits for episodes, and logs the best
candidate.

You will know it is working if the on-screen `PushT EC Validator` panel appears
and moves through reference resolution, candidate evaluation, fitness, and best
candidate updates.

## How To Use In Your Project

1. Copy this folder into your Unity project, for example:
   `Assets/Scripts/Evolution/`
2. Create a concrete evaluator that extends `PushTBlockEvaluator`.
3. In that evaluator:
   - Apply the genome to the block/environment.
   - Reset your ML-Agents scene.
   - Run one episode.
   - Return a fitness score.
4. Add `PushTEvolutionRunner` and your evaluator to a GameObject.
5. Assign the evaluator reference in the inspector.
6. Use the context menu `Run Evolution` on the runner.

## Notes

- This is a plain parameter-vector EC setup, not tree-based GP.
- The default search is:
  - random initialization
  - evaluate population
  - keep elites
  - refill by mutating elite copies
- Crossover is omitted on purpose to keep the MVP small.

## What To Adapt First

- Add or remove fields in `PushTBlockGenome`.
- Update fitness shaping in your evaluator.
- Tune ranges in `PushTEvolutionConfig`.
- Increase `episodesPerGenome` if you want robustness across seeds.
