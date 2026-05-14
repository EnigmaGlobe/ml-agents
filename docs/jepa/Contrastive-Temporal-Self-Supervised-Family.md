# Contrastive Predictive / Temporal Self-Supervised Family

## Core Idea

Contrastive and temporal self-supervised methods learn useful representations by making temporally related observations similar in latent space while separating unrelated observations. Common setups include:

- encode current and future observations
- maximize agreement for positive temporal pairs
- use negatives, augmentations, or other discrimination signals
- transfer the learned representation into RL or train it jointly with RL

These methods are primarily about representation quality rather than intrinsic reward.

## Why This Family Matters Here

If your JEPA design is intended to improve visual perception or temporal abstraction for PPO, then this family is highly relevant. It gives you a literature frame for encoder-only JEPA even if JEPA itself is not contrastive.

The similarity is not in the exact loss function. The similarity is in the goal: learn better latent state representations from temporal structure.

## Match To The JEPA Plan

This family is a strong match to:

- JEPA as encoder only
- JEPA as hybrid encoder plus auxiliary predictive objective

It is a weak match to:

- JEPA as reward only

## Does The Plan Match Recent Studies?

Yes. This family matches recent studies when the claim is about better perception, state abstraction, or visual representation for downstream control.

It is especially relevant if the study measures:

- task performance
- sample efficiency
- robustness across seeds
- generalization or transfer

That is more consistent with encoder-only JEPA than with reward-only JEPA.

## How To Position It In A Thesis Or Paper

Use this family to justify why JEPA belongs in RL even if it is not itself an intrinsic-reward mechanism. It supports the argument that temporally structured self-supervision can improve control through better latent features.

Recommended wording:

"Our encoder-only JEPA condition is positioned as a temporally predictive self-supervised representation learner for RL rather than as a pure exploration bonus method."

## Representative Papers

- Oord et al., Contrastive Predictive Coding
- CURL and related contrastive visual RL methods
- Temporal contrastive RL methods

## Recommendation

Keep this family in the main literature review. It is especially important if one of your experiment conditions uses JEPA as an encoder without intrinsic reward.

4. Contrastive predictive / temporal self-supervised family

Strong match for encoder-only JEPA, but not identical.

Key papers:

Paper	Why it matters for your study
van den Oord et al., 2018 — Contrastive Predictive Coding	Foundational paper for predicting future latent representations using a contrastive loss. It explicitly evaluates learned representations across domains including reinforcement learning in 3D environments.
CURL — Laskin, Srinivas & Abbeel, ICML 2020	Uses contrastive representation learning from raw pixels for RL, improving sample efficiency in DeepMind Control Suite and Atari. This is a key baseline/reference for “better visual representation improves control.”
TACO — Zheng et al., NeurIPS 2023	Very relevant because it is temporal, action-driven, contrastive representation learning for visual RL. It learns state and action representations by matching current state/action-sequence representations to future-state representations, improving visual continuous-control sample efficiency.
MOOSS, 2024	A more recent temporal contrastive visual RL method that tries to model smooth state evolution, useful as a related-work example if you discuss temporal representation learning rather than only intrinsic reward.

Positioning:
This family supports your encoder-only condition. The shared claim is:

Better self-supervised visual representations can improve downstream RL performance, sample efficiency, and robustness.

Frontier status:
Moderately frontier. Contrastive visual RL is active, but not as novel as JEPA-style latent prediction. Your advantage is that JEPA is non-contrastive / predictive in embedding space, so you can present it as an alternative to contrastive auxiliary learning.

4. Contrastive predictive / temporal self-supervised family

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv. https://doi.org/10.48550/arXiv.1807.03748

Laskin, M., Srinivas, A., & Abbeel, P. (2020). CURL: Contrastive unsupervised representations for reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning (Vol. 119, pp. 5639–5650). PMLR. https://proceedings.mlr.press/v119/laskin20a.html

Zheng, R., Wang, X., Sun, Y., Ma, S., Zhao, J., Xu, H., Daumé III, H., & Huang, F. (2023). TACO: Temporal latent action-driven contrastive loss for visual reinforcement learning. In Advances in Neural Information Processing Systems, 36. https://papers.nips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html

Sun, J., Akcal, M. U., Chowdhary, G., & Zhang, W. (2025). MOOSS: Mask-enhanced temporal contrastive learning for smooth state evolution in visual reinforcement learning. In Proceedings of the Winter Conference on Applications of Computer Vision (pp. 6719–6729). https://doi.org/10.1109/WACV61041.2025.00654