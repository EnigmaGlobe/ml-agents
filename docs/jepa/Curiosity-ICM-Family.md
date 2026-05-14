# Curiosity / ICM Family

## Core Idea

Curiosity methods, especially the Intrinsic Curiosity Module (ICM), learn a latent feature space and use next-state prediction error as an intrinsic reward. A common setup is:

- encode the current observation and next observation into latent embeddings
- use an inverse model to predict the action from the two embeddings
- use a forward model to predict the next embedding from the current embedding and action
- define intrinsic reward from forward prediction error

This makes curiosity an action-conditioned exploration method. It rewards transitions that are hard to predict in a learned latent space.

## Why This Family Matters Here

This is the closest family to the existing ML-Agents curiosity implementation. In this repo, curiosity is already integrated as an intrinsic reward provider for PPO, so it is the strongest baseline for a JEPA reward-only variant.

If your JEPA module produces a scalar intrinsic bonus from latent prediction error, then the comparison to curiosity is direct and defensible.

## Match To The JEPA Plan

This family is a strong match to:

- JEPA as intrinsic reward only
- JEPA as hybrid encoder plus intrinsic reward, if the reward is derived from latent prediction error

It is a weaker match to:

- JEPA as encoder only, because curiosity is primarily an exploration bonus method rather than a representation-learning baseline

## Does The Plan Match Recent Studies?

Yes. Comparing PPO plus curiosity against PPO plus JEPA reward is consistent with how recent sparse-reward RL studies frame intrinsic exploration baselines.

That said, curiosity is no longer the only expected baseline. Reviewers may still accept it as a core baseline because it is:

- widely recognized
- implemented in ML-Agents already
- conceptually close to action-conditioned latent surprise

But it is stronger as part of a baseline set than as the only comparator.

## How To Position It In A Thesis Or Paper

Use curiosity when the claim is about exploration through learned predictive error. This is the cleanest apples-to-apples comparison for a JEPA reward-only design.

Recommended wording:

"We compare a JEPA-inspired latent predictive intrinsic reward against the standard curiosity-style intrinsic reward baseline already supported in ML-Agents."

## Representative Papers

- Pathak et al., Curiosity-driven Exploration by Self-supervised Prediction
- Follow-up intrinsic motivation studies that retain ICM-style action-conditioned prediction

## Recommendation

Keep this family in the main experiment section. It should be one of the primary baselines.


1. Curiosity / ICM family

Strong match. Use as main apples-to-apples baseline.

Key papers:

Paper	Why it matters for your study
Pathak et al., 2017 — “Curiosity-driven Exploration by Self-supervised Prediction”	This is the core ICM paper. It defines intrinsic reward as the error in predicting the consequence of the agent’s own action in learned visual feature space. Very close to your JEPA reward-only condition.
Burda et al., 2018 — “Large-Scale Study of Curiosity-Driven Learning”	Shows curiosity-only learning across many benchmark environments and discusses prediction-error intrinsic rewards at scale. Useful for justifying curiosity as a serious baseline, not a toy comparison.
Kayal, Pignatelli & Toni, 2025 — “The impact of intrinsic rewards on exploration in Reinforcement Learning”	Recent empirical work that still includes ICM as a representative intrinsic reward, showing that ICM remains relevant in current sparse-reward exploration comparisons.
Unity ML-Agents documentation	ML-Agents explicitly says its curiosity reward signal enables ICM, using an inverse model plus forward model where forward prediction loss becomes intrinsic reward. This is directly useful because your comparison is against ML-Agents curiosity.

Positioning:
This is your cleanest baseline family. Your JEPA reward-only model can be framed as a JEPA-inspired alternative to ICM-style learned prediction-error curiosity.

Frontier status:
Not frontier by itself. It is now a standard baseline. The novelty comes from replacing the ICM feature/prediction structure with JEPA-style latent prediction and testing whether that changes exploration quality.