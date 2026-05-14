# JEPA-Style Predictor Family

## Core Idea

JEPA-style architectures learn by predicting target embeddings from context embeddings in latent space. They avoid raw reconstruction and often avoid explicit negative pairs. A common setup is:

- a context encoder processes the visible or current input
- a target encoder produces the target latent for future or masked content
- a predictor maps context latent, and optionally action or temporal context, toward the target latent
- training minimizes disagreement between predicted and target embeddings

This makes JEPA-style systems representation-centric and predictive without forcing pixel-level reconstruction.

## Why This Family Matters Here

This is the direct architectural family for your proposal. If the method is inspired by JEPA, this document defines the closest conceptual home.

The challenge is not conceptual fit. The challenge is empirical precedent in online RL. Most well-known JEPA papers come from representation learning in vision or video rather than standard PPO exploration benchmarks.

## Match To The JEPA Plan

This family is a strong match to:

- JEPA as encoder only
- JEPA as hybrid encoder plus reward
- action-conditioned JEPA-style predictive latent models for control

It is a weaker match to:

- JEPA as reward only, unless the latent prediction error is explicitly converted into intrinsic reward

## Does The Plan Match Recent Studies?

Conceptually yes, but with caution. The plan matches the direction of recent predictive representation work, but you should avoid overstating the maturity of JEPA-specific online RL comparisons.

The safe claim is:

- JEPA-inspired predictive latent learning is relevant to RL representation learning

The risky claim is:

- JEPA is already a standard benchmark family for PPO intrinsic exploration

The second claim would be too strong.

## How To Position It In A Thesis Or Paper

Frame the method as JEPA-inspired or JEPA-style predictive representation learning adapted for control. That is precise and consistent with the literature.

Recommended wording:

"We adapt a JEPA-style latent predictive architecture for visual reinforcement learning and study it as an intrinsic reward mechanism, an encoder, and a hybrid system."

## Representative Papers

- JEPA and I-JEPA style predictive representation papers
- Action-conditioned predictive latent architectures that share JEPA-like structure

## Recommendation

Use this family as the architectural anchor of the proposal, but support it with stronger RL-adjacent comparator families such as curiosity, RND, latent dynamics, and temporal self-supervision.

5. JEPA-style predictor family

Conceptually strongest, but evidence in online RL is thinner. This is your frontier lane.

Key papers:

Paper	Why it matters for your study
I-JEPA — Assran et al., CVPR 2023	Introduces image-based JEPA: predict target-block representations from context-block representations, without pixel reconstruction or hand-crafted augmentations. This is the conceptual base for your visual latent prediction claim.
V-JEPA — Bardes et al., 2024	Extends JEPA to video by predicting masked regions in representation space. This matters because your RL setting is temporal and visual, closer to video than static images.
V-JEPA 2 — Assran et al., 2025	Very strong frontier reference: combines internet-scale video pretraining with small robot interaction data, then uses an action-conditioned latent world model for robotic planning without task-specific reward.
“JEPA for RL” — 2025 preprint	Directly attempts to adapt JEPA to reinforcement learning from images, discussing collapse and showing CartPole image-observation experiments. This is probably the closest “someone is trying this” reference.
TD-JEPA — 2025/2026 preprint	Uses temporal-difference latent predictive representations for unsupervised RL and zero-shot reward optimization. Useful for showing JEPA-like latent prediction is entering RL, but still emerging.

Positioning:
This is your best novelty claim:

JEPA has strong evidence in visual/video representation learning and emerging evidence in RL/robotics, but it is not yet a standard PPO intrinsic-reward benchmark. Our work tests whether JEPA-style latent prediction helps online visual RL as reward, representation, or both.

Frontier status:
Yes — this is the most frontier part. The important caution is wording. Say “JEPA-inspired predictive representation for RL”, not “JEPA is already a standard intrinsic-reward baseline.”