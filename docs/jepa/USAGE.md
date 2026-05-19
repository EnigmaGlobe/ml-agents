# c-JEPA Usage (quick start)

This short guide explains the minimal steps to generate pre-extracted slot embeddings and enable the
`c-jepa` intrinsic reward provider in ML-Agents (Phase 1 workflow).

1) Precompute slot embeddings (recommended)

- Run your C-JEPA encoder (or the CausalWM encoder) on your environment recording pipeline and
  save slot tensors shaped (B, T, S, D) where:
  - B = batch / number of trajectories
  - T = number of frames per trajectory
  - S = number of slots per frame
  - D = embedding dimension per slot
- Save files as numpy `.npy` arrays or as a serialized dataset your trainer can load. The provider expects
  the trainer's `AgentBuffer` to contain the slot tensors at `BufferKey.OBSERVATIONS[0]` or under key
  `pixels_embed`.

2) Trainer config (enable `reward_signals.jepa`)

Add a `jepa` block under `reward_signals` in your trainer YAML. Minimal example:

```yaml
reward_signals:
  jepa:
    strength: 0.01
    gamma: 0.99
    freeze: true          # set false to enable online updates (not implemented in Phase 1)
    checkpoint_path: null
    history_size: 5
    num_preds: 1
    use_hungarian_matching: false
```

Notes:
- The provider currently expects precomputed slot embeddings. If you cannot precompute slots, add an
  adapter that populates `pixels_embed` in the `AgentBuffer` before calling the provider.
- `freeze: true` (default) avoids creating optimizer state. If you set `freeze: false`, the provider's
  `update()` is a placeholder in this Phase 1 implementation.

3) Running the unit tests (local dev)

Ensure your environment has dev dependencies installed (`pytest`, `numpy`, etc.). Example PowerShell commands from repo root:

```powershell
python -m pytest -q ml-agents/ml-agents/mlagents/trainers/torch_entities/components/reward_providers/test_c_jepa_reward_provider.py
```

4) Next steps
- Phase 2: register config and verify trainer wiring (already done in this branch).
- Phase 3: implement `update()` (online masked-training / Hungarian matching) and a more complete
  adapter for raw visual inputs.

If you want, I can now implement a conservative `update()` optimizer step (future-MSE) and the
corresponding unit test. Ask me to proceed if you'd like that next.
