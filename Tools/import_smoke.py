import importlib

try:
    s = importlib.import_module('mlagents.trainers.settings')
    print('JepaSettings in settings:', hasattr(s, 'JepaSettings'))
except Exception as e:
    print('settings import failed:', repr(e))

try:
    rp = importlib.import_module('mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider')
    print('CJepaRewardProvider in provider:', hasattr(rp, 'CJepaRewardProvider'))
except Exception as e:
    print('provider import failed:', repr(e))
