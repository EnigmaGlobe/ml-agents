import sys
out_path = 'tools/smoke_registration.out'
lines = []

try:
    lines.append('python=' + sys.executable)
    # Try importing the reward provider factory
    try:
        from mlagents.trainers.torch_entities.components.reward_providers import (
            reward_provider_factory,
        )
        lines.append('import: reward_provider_factory OK')
        # dump some attributes
        attrs = [a for a in dir(reward_provider_factory) if not a.startswith('_')]
        lines.append('factory_attrs: ' + ','.join(attrs[:20]))
    except Exception as e:
        lines.append('import: reward_provider_factory FAILED: ' + repr(e))

    # Try import provider class
    try:
        from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import (
            CJepaRewardProvider,
        )
        lines.append('import: CJepaRewardProvider OK')
    except Exception as e:
        lines.append('import: CJepaRewardProvider FAILED: ' + repr(e))

    # Try settings
    try:
        from mlagents.trainers import settings
        if hasattr(settings, 'JepaSettings'):
            lines.append('settings: JepaSettings FOUND')
        else:
            lines.append('settings: JepaSettings MISSING')
    except Exception as e:
        lines.append('import: settings FAILED: ' + repr(e))

except Exception as e:
    lines.append('unexpected failure: ' + repr(e))

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('\n'.join(lines))
