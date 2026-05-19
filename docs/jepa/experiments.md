# JEPA Experiments Cookbook

This short cookbook shows how to run small comparisons between PPO baseline and PPO + JEPA intrinsic reward.

1) Prepare a minimal trainer YAML with JEPA enabled
- Create `configs/jepa_smoke.yaml` and enable `reward_signals.jepa` with `freeze: true` and `strength: 0.01`.

2) Run a short training job
- Use the workspace venv python and a local Unity executable or a headless environment.
- Example (adjust paths/behavior):

```powershell
c:/new/ml-agents/.venv/Scripts/python.exe -m mlagents.trainers.run --run-id=jepa_smoke --train --env=Path\To\Your\Env.exe --config=configs/jepa_smoke.yaml
```

3) Compare runs
- Run the baseline (no JEPA) and a JEPA-enabled run for a few hundred steps.
- Use TensorBoard to compare `JEPA/` scalars and PPO reward curves.

```powershell
tensorboard --logdir results; Start TensorBoard in a separate shell and open http://localhost:6006
```

Notes and tips
- Keep `freeze: true` for smoke experiments to avoid online training changes.
- To enable online update, set `mode: online_training` and provide a torch predictor module via `reward_provider.get_modules()` or attach one programmatically.
- The provider will report `JEPA/reward_mean`, `JEPA/reward_std`, and `JEPA/last_loss` into the StatsReporter automatically when summaries are written.

Troubleshooting
- If metrics don't appear, ensure the trainer is calling `RLTrainer._write_summary()` (it does on summary intervals). Also verify `StatsWriter` backends (TensorBoard) are enabled in the trainer settings.

