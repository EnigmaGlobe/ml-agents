# Latent Dynamics / World-Model Family

## Core Idea

Latent dynamics and world-model methods learn a predictive model of how the environment evolves in a learned latent space. A typical setup is:

- encode observations into latent states
- predict next latent state, reward, termination, or future rollout conditioned on action
- use the learned model for representation learning, planning, auxiliary training, or policy improvement

These methods are broader than curiosity because they are not limited to intrinsic reward. They are often representation-centric or model-centric.

## Why This Family Matters Here

This is one of the strongest research matches for JEPA when JEPA changes what PPO sees internally. If your JEPA encoder produces latents that feed the actor and critic, or if JEPA predicts future latent states conditioned on action, then the project belongs partly in this family.

## Match To The JEPA Plan

This family is a strong match to:

- JEPA as encoder for PPO
- JEPA as encoder plus intrinsic reward
- JEPA as a predictive latent model trained jointly with PPO

It is a weaker match to:

- JEPA as reward only, because a pure reward bonus does not exploit the full world-model framing

## Does The Plan Match Recent Studies?

Yes, strongly. This is the most natural family for your plan once JEPA affects representation learning or latent prediction for control rather than only exploration reward.

Recent studies in visual RL often compare latent-model or self-supervised predictive encoders against standard policy encoders on:

- sample efficiency
- final task return
- robustness or transfer
- representation quality

That aligns well with a JEPA encoder-only or hybrid design.

## How To Position It In A Thesis Or Paper

Use this family when the core claim is that JEPA improves the learned state representation for control. In that framing, curiosity becomes only one baseline, not the central comparison.

Recommended wording:

"Our method is most closely related to latent predictive representation learning for control, with an optional intrinsic reward interpretation when latent prediction error is converted into an exploration bonus."

## Representative Papers

- Ha and Schmidhuber, World Models
- Dreamer and related latent dynamics RL methods
- Action-conditioned predictive latent models for control

## Recommendation

Keep this family in both the literature review and experiment framing. If JEPA is used as an encoder or hybrid system, this should be a primary conceptual anchor.

3. Latent dynamics / world-model family

Very strong match for encoder-only and hybrid JEPA.

Key papers:

Paper	Why it matters for your study
DreamerV3 — Hafner et al., 2023/2024	Learns a world model and improves behavior by imagining future scenarios; it solves many diverse tasks with a single configuration, including pixel/sparse-reward settings. This supports the broader claim that latent predictive models are powerful for control.
TD-MPC2 — Hansen, Su & Wang, ICLR 2024	Performs trajectory optimization in the latent space of a learned implicit, decoder-free world model, improving across 104 online RL tasks and scaling to a 317M parameter multi-task agent. Very relevant because it emphasizes latent dynamics without necessarily reconstructing pixels.
Tang et al., 2023 — “Understanding Self-Predictive Learning for Reinforcement Learning”	Directly studies representations learned by predicting future latent representations, including the collapse problem and why optimization design matters. This is highly relevant to JEPA encoder-only design.
Khetarpal et al., 2025 — “A Unifying Framework for Action-Conditional Self-Predictive Reinforcement Learning”	Very close conceptually: action-conditioned self-predictive RL jointly learns latent representation and dynamics by bootstrapping from future latents. This is one of the strongest references for your action-conditioned JEPA variant.

Positioning:
This is where your project becomes stronger than “curiosity.” Your encoder-only and hybrid conditions ask:

Does predictive latent structure make PPO learn better because the observation representation becomes more control-relevant?

That is a different and more current question than simply adding an intrinsic reward.

Frontier status:
Yes, especially if you emphasize latent predictive representation for online visual control. But you should avoid claiming you are competing with full world-model planners like DreamerV3 or TD-MPC2 unless you actually implement planning/imagination. Your work is better framed as model-free PPO enhanced by JEPA-style latent prediction.