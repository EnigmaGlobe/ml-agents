# c-JEPA Intrinsic Reward — Implementation Checklist

Goal: implement the JEPA intrinsic-reward provider described in `c-jepa-intrinsic-reward-FRD.md` in three clear phases so we can test and iterate safely.

How to use this file
- Work top-down through phases. Tick checkboxes as you complete items. Keep commits small and reference checklist items in PR descriptions.

Phase 1 — Minimal, low-risk integration (Baseline)
- [x] 1.1 Create `c-jepa` reward provider module (provider scaffolded)
  - Files created:
    - `C:\new\ml-agents\ml-agents\mlagents\trainers\torch_entities\components\reward_providers\c_jepa_reward_provider.py`
  - Goal: evaluate() using pre-extracted slots; safe default (frozen)
- [x] 1.2 Add unit tests (scaffolded)
  - File created:
    - `C:\new\ml-agents\ml-agents\mlagents\trainers\torch_entities\components\reward_providers\test_c_jepa_reward_provider.py`
  - Tests: `test_jepa_evaluate_preextracted_slots`, `test_jepa_update_stub_noop`
- [ ] 1.3 Add short usage doc
  - File: `docs/jepa/USAGE.md` (or update README) — how to precompute slots and set config
 - [x] 1.3 Add short usage doc
  - File: `docs/jepa/USAGE.md` (created)

Phase 1 verification
- [ ] Run unit tests for the reward-provider only:
```powershell
# from workspace root
pytest -q ml-agents/ml-agents/mlagents/trainers/torch_entities/components/reward_providers/test_jepa_reward_provider.py
```
- [ ] If tests fail due to missing PYTHONPATH for `src/`, run tests with PYTHONPATH set so `src` is importable.
```powershell
$env:PYTHONPATH = "${PWD}\src;${env:PYTHONPATH}"; pytest -q path\to\test_jepa_reward_provider.py
```

- Phase 2 — Trainer wiring and config
- [ ] 2.1 Add `JepaSettings` to `mlagents/trainers/settings.py` (a small dataclass or attr compatible with existing settings types)
- [ ] 2.2 Register provider in `reward_provider_factory.py` so `reward_signals.jepa` is supported
- [ ] 2.3 Ensure provider exposes modules via `get_modules()` for trainer checkpointing
- [ ] 2.4 Smoke test: run a single-step training with `freeze: true` and precomputed slots
 - [ ] 2.4 Smoke test: run a single-step training with `freeze: true` and precomputed slots
   - Script added: `tools/smoke_cjepa.py` (call with the workspace venv python)
 - [x] 2.1 Add `JepaSettings` to `mlagents/trainers/settings.py` (a small dataclass or attr compatible with existing settings types)
 - [x] 2.2 Register provider in `reward_provider_factory.py` so `reward_signals.jepa` is supported
 - [x] 2.3 Ensure provider exposes modules via `get_modules()` for trainer checkpointing
 - [ ] 2.4 Smoke test: run a single-step training with `freeze: true` and precomputed slots
 - [x] 2.4 Smoke test: run a single-step training with `freeze: true` and precomputed slots

Additional recent verifications
- [x] Add unit test asserting JEPA update loss decreases across 2 steps (`test_c_jepa_update_loss_decrease.py`) — PASS
- [x] Persist optimizer state in provider save/load (save/load stores `optimizer_state` and restores it) — validated by `tools/smoke_jepa_save_load.py`
- [x] Vendor fallback Hungarian helpers included in `c_jepa_reward_provider.py` for CI/unit-test robustness
- [x] Small RLTrainer regression test added to ensure `_write_summary()` invokes `report_metrics()` (`mlagents/trainers/trainer/test_rltrainer_report_metrics.py`) — PASS

Tests added:
- `test_c_jepa_update_step.py` — a unit test for the conservative update() that is skipped if torch not installed

Phase 2 verification
- [ ] Create a minimal trainer YAML that enables `reward_signals.jepa` and points to the precomputed slot file.
- [ ] Run a short training job (1-2 steps) and verify no crashes and that intrinsic rewards appear in logs.

Phase 3 — Enhanced features and experiments
- [x] 3.1 Implement `update()` online training path (masked predictor + optimizer)
  - Files changed/added (Phase 3):
    - `mlagents/trainers/torch_entities/components/reward_providers/c_jepa_reward_provider.py` — added masked-update logic, optional Hungarian-matching branch, and optimizer wiring.
    - `mlagents/trainers/torch_entities/components/reward_providers/test_c_jepa_masked_update.py` — unit tests exercising masked-update and Hungarian path.
- [ ] 3.2 Add CausalWM evaluation mode (encode→predict using CausalWM.rollout/predict)
- [ ] 3.3 Add config mode flag: `mode: frozen_preextracted|frozen_causalwm|online_training` and
 - [x] 3.2 Add CausalWM evaluation mode (encode→predict using CausalWM.rollout/predict)
 - [x] 3.3 Add config mode flag: `mode: frozen_preextracted|frozen_causalwm|online_training`
 - Note: provider now records simple metrics (reward mean/std and last loss) and enforces mode gating for updates.
- [ ] 3.4 Add logging + metrics (loss scalars, reward mean/std)
- [ ] 3.5 Add experiment scripts / cookbook for PPO vs PPO+curiosity vs PPO+JEPA

 - [x] 3.4 Add logging + metrics (loss scalars, reward mean/std)
   - Notes: Provider records `JEPA/reward_mean`, `JEPA/reward_std`, and `JEPA/last_loss` and `RLTrainer` calls `report_metrics()` before stats writes so these appear in StatsReporter outputs.
 - [x] 3.5 Add experiment scripts / cookbook for PPO vs PPO+curiosity vs PPO+JEPA
   - Files added: `docs/jepa/experiments.md` (basic cookbook and commands)

Phase 3 verification
- [ ] Run a full small-scale experiment with multiple seeds and plot learning curves.

- Files you will likely edit
- Note: we are using the project name `c-jepa` in documentation and configs; the actual provider file currently uses `jepa_` prefix to avoid breaking imports—if you want, we can rename files later.
- `C:\new\ml-agents\ml-agents\mlagents\trainers\torch_entities\components\reward_providers\jepa_reward_provider.py` (new provider)
- `mlagents/ml-agents/mlagents/trainers/torch_entities/components/reward_providers/reward_provider_factory.py` (register provider)
- `mlagents/ml-agents/mlagents/trainers/settings.py` (add settings schema)
- Tests: `.../reward_providers/test_jepa_reward_provider.py`
- Docs: `docs/jepa/c-jepa-intrinsic-reward-FRD.md`, `docs/jepa/USAGE.md`, `docs/jepa/implementation_checklist.md`

Quick tips
- Use pre-extracted slot embeddings for Phase 1 to avoid changing the policy encoder.
- If you import `src.custom_codes.hungarian`, ensure the test/runner has `src/` on PYTHONPATH or copy the helper locally.
- Keep `freeze: true` as default to avoid optimizer state in early experiments.

Sign-off
- When a phase is complete, create a PR referencing the checklist items and the FRD. Attach test outputs or trainer logs showing smoke runs.

If you want, I can now:
- implement the missing `update()` with optimizer wiring (Phase 1.2), or
- wire the provider into `reward_provider_factory.py` and `settings.py` (Phase 2.1/2.2), or
- add the `docs/jepa/USAGE.md` precompute guide (Phase 1.3).
