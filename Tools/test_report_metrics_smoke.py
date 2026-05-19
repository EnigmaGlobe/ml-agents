"""Smoke test to assert that `report_metrics()` calls into a StatsReporter-like object.
This doesn't run the trainer; it creates a CJepaRewardProvider, populates metrics, and ensures
report_metrics uses `add_stat` / `set_stat` appropriately.
"""
from collections import defaultdict
from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import CJepaRewardProvider

class FakeStatsReporter:
    def __init__(self):
        self.added = defaultdict(list)
        self.set = {}
    def add_stat(self, key, value, **kwargs):
        self.added[key].append(value)
    def set_stat(self, key, value):
        self.set[key] = value


def main():
    cfg = type('C', (), {'history_size':3, 'num_preds':1, 'freeze':True, 'strength':0.01, 'gamma':0.99, 'learning_rate':1e-4})
    prov = CJepaRewardProvider(None, cfg)
    # seed some metrics
    prov._metrics['reward_mean'] = 0.123
    prov._metrics['reward_std'] = 0.045
    prov._metrics['last_loss'] = 0.987
    fake = FakeStatsReporter()
    prov.report_metrics(fake)
    print('added keys:', list(fake.added.keys()))
    print('set keys:', list(fake.set.keys()))
    assert 'JEPA/reward_mean' in fake.added or 'JEPA/reward_mean' in fake.set
    assert 'JEPA/last_loss' in fake.added or 'JEPA/last_loss' in fake.set
    print('report_metrics smoke: OK')

if __name__ == '__main__':
    main()
