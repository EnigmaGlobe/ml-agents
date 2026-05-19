## c-JEPA Intrinsic Reward - Functional Requirements Document (FRD)

Document purpose
-----------------
This FRD describes a minimal, testable design to integrate the C-JEPA masked-slot predictor as an intrinsic-reward module inside the ML-Agents trainer pipeline. It focuses on a drop-in "reward provider" style integration (the least invasive path) and documents API contracts, data shapes, config, failure modes, tests, and exact repo locations to edit.

Scope
-----
- Implement `JepaRewardProvider` that turns C-JEPA prediction error into a scalar intrinsic reward per environment transition.
- Support two operational modes:
  - frozen pretrained JEPA: load checkpoint, no updates (default for experiments where JEPA was pretrained offline)
  - online JEPA fine-tune: keep predictor optimizable and update it during PPO updates (optional)
- Minimal changes to ML-Agents core: add provider, register it, add settings block. No trainer algorithm rewrites.

Non-goals
---------
- Replacing the policy/critic encoder with JEPA latents (encoder integration is outside this FRD).
- Full multi-GPU distributed training for the JEPA module (keep simple optimizer hooks compatible with ML-Agents patterns).

Success criteria
----------------
- A working `JepaRewardProvider` Python module that:
  - loads a checkpoint (or constructs the predictor)
  - given one batch of transitions (obs, next_obs, action) returns a tensor of intrinsic rewards with shape (B, T)
  - when configured, exposes an `update(mini_batch)` method invoked by the trainer that runs one optimizer step on JEPA losses
- Unit tests: two small tests (evaluate-only, and evaluate+update mock) passing locally.

Contract / API
--------------
The provider implements the same interface as existing reward providers in this repo (follow curiosity_reward_provider.py). Minimal contract:

- class JepaRewardProvider:
  - __init__(self, settings, device): constructs the predictor, loads checkpoint if provided, builds optimizer(s) if online training enabled.
  - evaluate(self, trajectory_batch) -> intrinsic_rewards: computes intrinsic reward per transition (tensor shaped [B, T])
  - update(self, mini_batch) -> loss_dict: optional; performs a single optimization step on the predictor and returns training info.
  - save(self, path) / load(self, path): save/load predictor state_dict (for trainer checkpointing interoperability).

Data shapes
-----------
- Input observation expected formats (two alternatives supported):
  1. Pre-extracted slots (recommended & simplest): trajectory_batch contains `pixels_embed` shaped (B, T, S, D).
  2. Raw visual observations: provider must declare it needs a `jepa_encoder` helper or fail. (Out of scope for first iteration.)
- Action, proprio: optional. If JEPA checkpoint expects concatenated action/proprio in slot embedding, the provider must accept `action_embed` and `proprio_embed` in the batch and concatenate appropriately.
- Output intrinsic reward: Tensor shaped (B, T) or (B, T, 1) with float values. Scales controlled by config multiplier `strength`.

Computation details
-------------------
Important note (masked vs inference):

- The C-JEPA predictor supports two distinct usage modes which are not interchangeable:
  1. inference() mode: a non-masked forward used for rollouts / prediction when the full history is visible. This builds a future-token input (mask tokens + anchor + timePE) internally and returns predicted future slots without relying on masked-token training semantics.
  2. masked prediction / training mode (predictor(...) call): constructs the query grid with mask tokens and computes both history-reconstruction (masked slots in history) and future-prediction outputs plus mask indices. This path is used during JEPA training and returns mask information needed for selective losses.

The reward provider must choose which mode to use depending on the data available and the desired signal:
- If the trainer provides pre-extracted slot embeddings for both history and future frames (preferred), use the predictor.inference(history) flow and compute the prediction error between predicted future slots and the true future slots. Inference mode does not require Hungarian matching for the predictor's internal ordering if the predictor maintains a stable slot ordering, but in practice slot permutations can still occur and should be guarded with a matching step if the dataset/environment is not slot-consistent.
- If you want to compute an intrinsic signal based on the JEPA training objective (i.e., masked-slot reconstruction error), call the masked predictor path. This requires access to mask indices and mask-aware loss computation (and typically the trainer-provided masked targets). When computing intrinsic reward from masked losses you should restrict the error to the masked slots only.

Hungarian matching (slot reordering)
-----------------------------------

Slot permutation across frames breaks naive MSE comparisons. The CausalWM wrapper implements Hungarian reordering in its `rollout()`/`predict()` flows to maintain slot identity across time; if the provider computes MSE without alignment, the signal will be noisy or meaningless. The FRD therefore mandates one of these choices:

- Use the full `CausalWM` model for evaluation: call `CausalWM.encode()` to get an embedding (or feed pre-extracted slots into CausalWM.encode if you want reordering), then call `CausalWM.predict()` or `CausalWM.rollout()` which performs internal Hungarian reordering as configured. Read the `use_hungarian` / `hungarian_cost_type` flags from the predictor config to control matching.
- Or, if you operate purely on pre-extracted slots and do not want to use CausalWM, run an explicit Hungarian matching step (same cost metric used in training, e.g., MSE on pixel-slot part) between predicted slots and target slots before computing MSE.

Evaluation flow (frozen mode) — recommended
-----------------------------------------
1. From `trajectory_batch`, extract pre-extracted slots keyed `pixels_embed` shaped (B, T, S, D). If only raw pixels are available, the provider must either fail with a clear error or use a configured encoder helper.
2. Build history windows of length `history_size` per sample. If insufficient frames are present, either skip / return zeros or pad using the earliest available frames (configurable).
3. Call `causal_wm.predict(history)` with `use_inference_function=True` (or `causal_wm.inference(history)`), which returns predicted future slots. If using the raw predictor, call `predictor.inference(history)`.
4. If using the raw predictor and slot ordering is not guaranteed, run Hungarian matching between `predicted_future` and `target_future` (same pixel/proto subset and cost metric as training) to reorder predictions to best-match targets.
5. Compute per-step scalar error: reduce per-slot, per-dim MSE into a single scalar per sample/time (configurable: mean|max|sum), optionally average across `num_preds` frames.
6. Optionally standardize or clip the intrinsic signal and multiply by `strength` before returning an array shaped (B, T') aligned with the trainer’s transition timing.

Update flow (online training mode)
---------------------------------
1. During trainer update, the provider receives mini-batches (the same AgentBuffer structure used by other reward providers).
2. Use the masked prediction path: call `predictor(history_with_masking)` which returns `preds, mask_indices` or call the `CausalWM` wrapper which internally runs masking logic compatible with the checkpoint format.
3. Compute JEPA training losses (masked history reconstruction + future prediction). If `use_hungarian_matching` is enabled in config, apply matching in the future-loss branch exactly as the original training code.
4. Run optimizer step on JEPA predictor parameters and return diagnostics (loss scalars) for logging.

Config (example)
----------------
Add a `jepa` block in the `reward_signals` config-under `settings.py` schema. Example YAML snippet to include in a trainer config:

```yaml
reward_signals:
  jepa:
    strength: 0.01          # multiplier applied to intrinsic reward
    gamma: 0.99
    checkpoint_path: null   # or path to predictor state_dict (.pth)
    freeze: true            # if true: do not optimize the predictor
    history_size: 5         # frames used by predictor
    num_preds: 1
    reduction: mean         # how to reduce slot errors -> scalar: mean|max|sum
    standardize: true       # optionally z-score normalize the intrinsic signal
    scale_by_feature_dim: true # divide MSE by feature dimension to keep scale consistent
    optimizer:
      lr: 5e-4
      type: Adam
```

Files to change (exact locations)
---------------------------------
- Add new provider module: `ml-agents/ml-agents/mlagents/trainers/torch_entities/components/reward_providers/jepa_reward_provider.py`
- Register provider in factory: `ml-agents/ml-agents/mlagents/trainers/torch_entities/components/reward_providers/reward_provider_factory.py`
- Add config parsing: `ml-agents/ml-agents/mlagents/trainers/settings.py` (define JepaSettings or similar)
- Trainer update ordering already supports reward-provider updates; verify `ml-agents/ml-agents/mlagents/trainers/on_policy_trainer.py` calls provider.update after PPO policy updates (mirror curiosity behavior).

Pseudocode (evaluate + update)
--------------------------------
Below are two clearer flows. The FRD recommends implementing both and choosing the appropriate one at runtime via config (`mode: frozen_preextracted | frozen_causalwm | online_training`).

Evaluate (frozen, pre-extracted slots — recommended):
```python
def evaluate(self, trajectory_batch):
  # trajectory_batch['pixels_embed'] -> (B, T, S, D)
  slots = trajectory_batch.get('pixels_embed')
  if slots is None:
    # No slots available: fail gracefully or return zeros
    return np.zeros((batch_size, 1), dtype=np.float32)

  history = extract_last_windows(slots, self.cfg.history_size)  # (B, H, S, D)

  # Use CausalWM wrapper if available for Hungarian-safe predictions
  if self.use_causal_wm:
    pred_future = self.causal_wm.predict(history, use_inference_function=True)
  else:
    pred_future = self.predictor.inference(history)

  # Target frames are taken from the same trajectory batch at a fixed offset
  # matching the predictor's `history_size` and `num_preds`.
  # Equivalent to: embedding[:, history_size:history_size+num_preds, :, :]
  target_future = slots[:, self.cfg.history_size : self.cfg.history_size + self.cfg.num_preds, :, :]

  if not self.use_causal_wm and self.cfg.use_hungarian_matching:
    pred_future = hungarian_reorder(pred_future, target_future, cost_type=self.cfg.hungarian_cost_type)

  per_sample_err = ((pred_future - target_future)**2).mean(axis=(2,3))  # (B, num_preds)
  scalar_err = reduce(per_sample_err, axis=1, op=self.cfg.reduction)    # (B,)
  reward = scalar_err * self.cfg.strength
  return reward
```

Update (online training mode, masked prediction + optional Hungarian in future loss):
```python
def update(self, mini_batch):
  if self.cfg.freeze:
    return {}

  # mini_batch contains history windows and targets in the same shape used by training
  # Use the CausalWM or predictor masked path which returns mask_indices
  if self.use_causal_wm:
    preds, mask_indices = self.causal_wm.predict(mini_batch['history'], return_mask=True)
  else:
    preds, mask_indices = self.predictor(mini_batch['history_with_masking'])

  # compute masked history loss
  loss_masked = mse_masked(preds['history'], mini_batch['history_targets'], mask_indices)

  # compute future loss (with Hungarian if enabled)
  future_pred = preds['future']
  future_target = mini_batch['future_targets']
  if self.cfg.use_hungarian_matching:
    reorder = hungarian_reorder(future_pred, future_target, cost_type=self.cfg.hungarian_cost_type)
    future_pred = reorder

  loss_future = mse(future_pred, future_target)
  loss = loss_masked + loss_future

  self.opt.zero_grad(); loss.backward(); self.opt.step()
  return {'jepa_masked_loss': loss_masked.item(), 'jepa_future_loss': loss_future.item()}
```

Edge cases and failure modes
---------------------------
- Missing slot embeddings in trainer inputs. Mitigation: provider must return zeros and log a warning. Recommend experiments use pre-extracted slots or add a small adapter that builds slots from observations offline.
- Mismatch of embedding dims between predictor checkpoint and runtime: provider should validate `predictor.slot_dim == incoming_slot_dim` and raise a clear error if mismatched.
- Predictor is large and increases trainer memory: document memory impact; default `freeze: true` to avoid optimizer state if pretrained is used.

C-JEPA Hungarian utilities (from your codebase)
-----------------------------------------------

You provided a set of Hungarian-matching helper functions in the C-JEPA codebase which the provider can reuse to guarantee slot alignment. Key functions and how to use them:

- `hungarian_matching_loss_AP(pred, target, cost_type='mse', reduction='mean') -> dict`
  - Inputs: `pred, target` shaped (B, T, N, D).
  - Returns: dict with `pixels_loss` (torch scalar). Use this when you want a full matched loss computed exactly like in your trainer.

- `hungarian_matching_loss_with_proprio(pred, target, pixels_dim, proprio_dim=0, cost_type='mse', reduction='mean') -> dict`
  - Matches slots using the pixels portion only and computes both pixels and proprio losses (returns `pixels_loss`, `proprio_loss`, `total_loss`). Useful when embeddings contain concatenated proprio/action dims.

- `hungarian_cost(preds, goal, cost_type='mse', pixels_dim=None) -> Tensor(B, N)`
  - Computes matched costs used by planning and matching diagnostics. Useful if you need a matched cost per candidate.

- `reorder_slots_to_match(pred, reference, cost_type='mse', pixels_dim=None) -> reordered_pred`
  - Reorders predicted slots to align with a `reference` ordering (applies same permutation across time). This is handy in the reward provider when you want to reorder predicted slots to match target slots before MSE.

Implementation note:

- The utility functions use the CPU Hungarian solver (`scipy.optimize.linear_sum_assignment`) on detached tensors to compute permutations and then compute differentiable losses w.r.t. `pred` using the matched targets. They are a direct fit for the FRD's recommended approaches.
- Where to import from: your C-JEPA training code imports these as `from src.custom_codes.hungarian import ...`. The reward provider can import them the same way if `src/` is on PYTHONPATH in training; otherwise copy the minimal helper functions into the new provider module.

Example call (inside `evaluate()` when not using `CausalWM`):

```python
from src.custom_codes.hungarian import reorder_slots_to_match, hungarian_matching_loss_with_proprio

pred_future = predictor.inference(history)  # (B, num_preds, S, D)
# Target frames are taken from the same trajectory batch at a fixed offset
# matching the predictor's `history_size` and `num_preds`:
target_future = slots[:, cfg.history_size : cfg.history_size + cfg.num_preds, :, :]

if cfg.use_hungarian_matching:
    pred_future = reorder_slots_to_match(pred_future, reference=slots[:, -1], cost_type=cfg.hungarian_cost_type, pixels_dim=cfg.pixels_dim)

loss_dict = hungarian_matching_loss_with_proprio(pred_future, target_future, pixels_dim=cfg.pixels_dim, proprio_dim=cfg.proprio_dim)
reward = loss_dict['pixels_loss'].detach().cpu().numpy() * cfg.strength
```

Tests (minimal set)
--------------------
1. Unit test: `test_jepa_evaluate_preextracted_slots` — create small random slots tensor matching predictor dims, run evaluate(), assert shape and finite values.
2. Unit test: `test_jepa_update_step` — create dummy mini_batch, set `freeze=false`, run update() and assert loss decreases for two steps or optimizer state changed.
3. Integration smoke test: hook provider into a tiny instantiation of TrainerFactory with a tiny PPO config and single-step environment (optional; can be an integration test marked slow).

Quality gates and verification
-----------------------------
- Lint and style: run repo flake8/black (match project style) on new file.
- Type checks: add type hints and run mypy if project uses it.
- Unit tests: run the two unit tests above via pytest.
- Manual smoke: run a single-step training with `freeze: true` and precomputed slots to ensure no runtime crashes and intrinsic rewards are returned.

Deliverables
------------
- `jepa_reward_provider.py` implementing the provider contract.
- `settings.py` change to accept `jepa` reward block.
- `reward_provider_factory.py` registration entry.
- Two unit tests under `ml-agents/ml-agents/tests` or `ml-agents/ml-agents/mlagents/trainers/tests`.
- `docs/jepa/USAGE.md` (optional) describing how to export predictor checkpoint suited for the provider.

Phased implementation plan
-------------------------
This FRD uses a phased rollout so we can implement and verify the JEPA intrinsic reward provider step-by-step. Each phase has small substeps (1.1, 1.2, ...) so you can approve or stop at any checkpoint.

Phase 1 — Minimal, low-risk integration (get a working baseline)
  1.1 Scaffold `jepa_reward_provider.py` implementing:
      - constructor that loads a predictor checkpoint or CausalWM wrapper (config-driven)
      - `evaluate()` using pre-extracted slots and exact target extraction:
        `target = slots[:, history_size:history_size+num_preds, :, :]`
      - optional Hungarian reorder via `reorder_slots_to_match` helper
      - return intrinsic reward shaped (B, T') scaled by `strength`
  1.2 Add unit tests:
      - `test_jepa_evaluate_preextracted_slots` (random tensor, check shape and finite values)
      - `test_jepa_update_step` (if `freeze=false`, run one update step and assert optimizer state changed)
  1.3 Add `docs/jepa/USAGE.md` with a short example config and how to precompute slots.

  Deliverable: runnable provider module + 2 unit tests + a short usage doc. Low repo impact.

Phase 2 — Trainer wiring and config
  2.1 Add `JepaSettings` to `mlagents/trainers/settings.py` and document available fields.
  2.2 Register provider in `reward_provider_factory.py` so it can be enabled via `reward_signals.jepa`.
  2.3 Ensure checkpoint save/load compatibility (provider exposes `save`/`load` hooks or returns modules to trainer checkpointing).
  2.4 Smoke test: run a single-step training with `freeze: true` and precomputed slots to verify end-to-end behavior.

  Deliverable: provider usable through ML‑Agents config and verified by a smoke test.

Phase 3 — Enhanced features and experiments
  3.1 Add CausalWM-based evaluation mode (encode→predict with internal Hungarian matching) and a `mode` config flag: `frozen_preextracted|frozen_causalwm|online_training`.
  3.2 Improve diagnostics and logging: `jepa_masked_loss`, `jepa_future_loss`, reward mean/std per update.
  3.3 Provide an experiments cookbook: step-by-step for comparing PPO baseline, PPO+curiosity, PPO+JEPA (frozen vs online), including recommended seeds and budgets.

  Deliverable: production-ready feature set with experiment templates and monitoring.

Edge tasks (optional)
  - Adapter to compute slots online from raw pixels (for encoder integration experiments).
  - Multi-GPU optimizer plumbing for large-scale online JEPA training.

Estimated effort (rough)
  - Phase 1: 2–4 hours (scaffold + tests)
  - Phase 2: 1–3 hours (wiring + smoke)
  - Phase 3: 4–8+ hours (features, docs, experiments)

Immediate proposal
------------------
Start with Phase 1.1 (scaffold `jepa_reward_provider.py`) and the two unit tests. If you confirm, I'll implement Phase 1.1 now and report back with the created files and test results.

Document history
----------------
Created: 2026-05-14 by automated FRD generator based on user-provided C-JEPA code and repo inspection.
