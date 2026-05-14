# RND Family

## Core Idea

Random Network Distillation (RND) measures novelty by comparing the output of a trainable predictor network against a fixed random target network. The standard setup is:

- pass the state through a fixed random target network
- train a predictor network to match the target output on visited states
- use prediction error as intrinsic reward

Unlike curiosity, RND does not model action-conditioned dynamics. It measures how novel a state is, not how hard it is to predict under agent-controlled transitions.

## Why This Family Matters Here

RND is one of the standard intrinsic reward baselines in exploration-heavy RL. Even though it is not as close to JEPA as ICM is, it tests an important competing explanation:

- curiosity-like methods reward controllable predictive surprise
- RND-like methods reward state novelty

If JEPA reward beats curiosity but not RND, that changes the interpretation of the result.

## Match To The JEPA Plan

This family is a partial match to:

- JEPA as intrinsic reward only

It is a weak match to:

- JEPA as encoder only
- JEPA as full predictive latent model

The connection is strongest when your JEPA module is used to emit a novelty bonus rather than to improve the policy representation.

## Does The Plan Match Recent Studies?

Yes, partially. Including RND in the experiment matrix would make the plan more consistent with recent exploration-benchmark practice.

If you only compare JEPA reward against curiosity, reviewers may ask why you excluded the other standard intrinsic reward baseline. RND is often the first missing comparator they will notice.

## How To Position It In A Thesis Or Paper

Use RND as a complementary baseline, not as the conceptual center of the study. It helps separate two questions:

- does JEPA help because of general novelty detection
- or does it help because of structured latent prediction over environment transitions

Recommended wording:

"We include RND as a state-novelty baseline to distinguish improvements due to predictive latent modeling from improvements due to generic novelty bonuses."

## Representative Papers

- Burda et al., Exploration by Random Network Distillation

## Recommendation

Include this family in the main experiment section if scope allows. If scope is tight, include it at minimum in the literature review and justify explicitly if it is omitted from experiments.


2. RND family

Partial match, but include it. Reviewers may expect it.

Key papers:

Paper	Why it matters for your study
Burda et al., 2018 — “Exploration by Random Network Distillation”	Defines RND as prediction error against a fixed random network; the intrinsic reward is high for novel observations and decreases with repeated visits.
Unity ML-Agents documentation	ML-Agents includes RND as a sparse-reward intrinsic reward option, separate from curiosity. That means it is a very fair baseline in your actual tool ecosystem.
RLeXplore, 2024 — intrinsic-reward toolkit	Shows that modern intrinsic-reward research still treats methods like ICM/RND as representative baselines and evaluates them across sparse-reward environments such as Procgen, MiniGrid, Ant-UMaze, and ALE hard-exploration games.

Positioning:
RND tests state novelty. Your JEPA reward-only tests something closer to predictive surprise in learned semantic/latent space, especially if action-conditioned.

Frontier status:
RND is not frontier, but it is a necessary credibility baseline. Including RND makes your comparison stronger because you can say JEPA is not merely beating “old curiosity,” but is being compared against both major learned intrinsic-reward families: controllable prediction error and state novelty.

2. RND family

Partial match, but include it. Reviewers may expect it.

Key papers:

Paper	Why it matters for your study
Burda et al., 2018 — “Exploration by Random Network Distillation”	Defines RND as prediction error against a fixed random network; the intrinsic reward is high for novel observations and decreases with repeated visits.
Unity ML-Agents documentation	ML-Agents includes RND as a sparse-reward intrinsic reward option, separate from curiosity. That means it is a very fair baseline in your actual tool ecosystem.
RLeXplore, 2024 — intrinsic-reward toolkit	Shows that modern intrinsic-reward research still treats methods like ICM/RND as representative baselines and evaluates them across sparse-reward environments such as Procgen, MiniGrid, Ant-UMaze, and ALE hard-exploration games.

Positioning:
RND tests state novelty. Your JEPA reward-only tests something closer to predictive surprise in learned semantic/latent space, especially if action-conditioned.

Frontier status:
RND is not frontier, but it is a necessary credibility baseline. Including RND makes your comparison stronger because you can say JEPA is not merely beating “old curiosity,” but is being compared against both major learned intrinsic-reward families: controllable prediction error and state novelty.