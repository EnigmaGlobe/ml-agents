# Empowerment / Information-Gain / Uncertainty-Exploration Family

## Core Idea

This family rewards the agent for visiting states that are informative, uncertain, or highly controllable. Different variants estimate:

- epistemic uncertainty
- information gain
- controllable influence over future states
- disagreement across model ensembles

The mathematical motivation differs from curiosity and RND, even though all of them target better exploration.

## Why This Family Matters Here

This family matters mostly as contrastive related work. It helps show that exploration can be defined in more than one way:

- predictive surprise
- novelty
- uncertainty reduction
- controllability or empowerment

That broadens the literature review and makes the positioning of JEPA more precise.

## Match To The JEPA Plan

This family is a weak match to:

- JEPA as intrinsic reward only, unless the reward is explicitly tied to uncertainty or information gain

It is sometimes related to:

- JEPA as a broader predictive model that could support uncertainty-aware exploration

But that is not the same as the current plan.

## Does The Plan Match Recent Studies?

Only weakly as a main experiment. Your current plan does not appear to implement uncertainty estimation, mutual information objectives, or empowerment explicitly.

So this family is better suited to the related-work section than to the main experiment matrix.

## How To Position It In A Thesis Or Paper

Use this family to clarify what your method is not. That helps prevent overclaiming.

Recommended wording:

"Although our method targets exploration through predictive latent modeling, it does not explicitly optimize information gain, epistemic uncertainty, or empowerment. We therefore discuss these methods as related but distinct exploration families."

## Representative Papers

- Empowerment-based exploration methods
- Information-gain exploration methods such as VIME-style approaches
- Ensemble disagreement or uncertainty-driven exploration methods

## Recommendation

Include this family in the literature review, not as a required experimental baseline unless the project later adds explicit uncertainty or information-gain objectives.

6. Empowerment / information-gain / uncertainty exploration

Good related work, weak main-experiment match unless you implement it.

Key papers:

Paper	Why it matters for your study
VIME — Houthooft et al., 2016	Classic information-gain exploration method. It rewards actions that change the agent’s belief about environment dynamics. This is mathematically different from plain prediction error.
Empowerment through Causal Learning — Cao et al., ICLR 2025	Recent frontier work connecting empowerment, causal dynamics models, and model-based RL. It defines empowerment around controllability through mutual information between actions and future states.
Kayal et al., 2025 intrinsic-reward comparison	Useful because it groups intrinsic rewards into knowledge-based and skill/behavioral diversity families, helping you place empowerment/information-gain methods as adjacent but not identical to your JEPA plan.

Positioning:
Mention this family in related work, but do not include it in the main experiment unless you add an explicit uncertainty, information-gain, or controllability objective.

Frontier status:
The family is frontier in some areas, especially causal/empowerment exploration, but your current JEPA plan is not directly testing that hypothesis.

6. Empowerment / information-gain / uncertainty exploration

Good related work, weak main-experiment match unless you implement it.

Key papers:

Paper	Why it matters for your study
VIME — Houthooft et al., 2016	Classic information-gain exploration method. It rewards actions that change the agent’s belief about environment dynamics. This is mathematically different from plain prediction error.
Empowerment through Causal Learning — Cao et al., ICLR 2025	Recent frontier work connecting empowerment, causal dynamics models, and model-based RL. It defines empowerment around controllability through mutual information between actions and future states.
Kayal et al., 2025 intrinsic-reward comparison	Useful because it groups intrinsic rewards into knowledge-based and skill/behavioral diversity families, helping you place empowerment/information-gain methods as adjacent but not identical to your JEPA plan.

Positioning:
Mention this family in related work, but do not include it in the main experiment unless you add an explicit uncertainty, information-gain, or controllability objective.

Frontier status:
The family is frontier in some areas, especially causal/empowerment exploration, but your current JEPA plan is not directly testing that hypothesis.

6. Empowerment / information-gain / uncertainty exploration

Houthooft, R., Chen, X., Duan, Y., Schulman, J., De Turck, F., & Abbeel, P. (2016). VIME: Variational information maximizing exploration. In Advances in Neural Information Processing Systems, 29 (pp. 1109–1117). https://papers.nips.cc/paper/6591-vime-variational-information-maximizing-exploration

Cao, H., Feng, F., Fang, M., Dong, S., Yang, T., Huo, J., & Gao, Y. (2025). Towards empowerment gain through causal structure learning in model-based reinforcement learning. International Conference on Learning Representations. https://openreview.net/forum?id=c745bfa5b50544882938ff4f89ff26ac

Kayal, A., Pignatelli, E., & Toni, L. (2025). The impact of intrinsic rewards on exploration in reinforcement learning. Neural Computing and Applications, 37, 16269–16303. https://doi.org/10.1007/s00521-025-11340-0