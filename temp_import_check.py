import importlib
try:
    import mlagents.trainers.trainer.rl_trainer as rt
    print('RLTrainer imported okay')
except Exception as e:
    print('RLTrainer IMPORT-ERROR', type(e).__name__, e)
try:
    import mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider as cj
    print('CJepa provider module imported okay')
except Exception as e:
    print('CJepa IMPORT-ERROR', type(e).__name__, e)
