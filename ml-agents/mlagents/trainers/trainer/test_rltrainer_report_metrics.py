import types
from mlagents.trainers.trainer.rl_trainer import RLTrainer


class FakeStatsReporter:
    def __init__(self):
        self.added = []
        self.set = []
        self.written = False

    def add_stat(self, key, value, **kwargs):
        self.added.append((key, value))

    def set_stat(self, key, value):
        self.set.append((key, value))

    def write_stats(self, step):
        self.written = True


class FakeRewardProvider:
    def __init__(self):
        self.reported = False

    def report_metrics(self, stats_reporter):
        self.reported = True
        stats_reporter.add_stat('JEPA/test', 0.5)


def test_rltrainer_calls_report_metrics():
    # Create a minimal concrete RLTrainer subclass to instantiate
    class ConcreteRLTrainer(RLTrainer):
        def _is_ready_update(self):
            return False

        def _process_trajectory(self, trajectory):
            return None

        def _update_policy(self):
            return False

        def create_optimizer(self):
            return None

        def add_policy(self, *args, **kwargs):
            return None

        def create_policy(self, *args, **kwargs):
            return None

    t = ConcreteRLTrainer.__new__(ConcreteRLTrainer)
    # minimal initialization expected by _write_summary
    # RLTrainer expects an internal _stats_reporter; set both to be safe
    t._stats_reporter = FakeStatsReporter()
    try:
        t.stats_reporter = t._stats_reporter
    except Exception:
        # property may be read-only; that's fine
        pass
    fake_optimizer = types.SimpleNamespace()
    fake_optimizer.reward_signals = {'jepa': FakeRewardProvider()}
    t.optimizer = fake_optimizer

    # Minimal training flags expected by should_still_train property
    t.is_training = False
    t._step = 0
    # trainer_settings must have max_steps attr
    t.trainer_settings = type('TS', (), {'max_steps': 100, 'threaded': False, 'summary_freq': 100, 'as_dict': lambda self: {}})()

    # Call _write_summary and ensure report_metrics was used and write_stats called
    t._write_summary(1)
    assert fake_optimizer.reward_signals['jepa'].reported is True
    assert t.stats_reporter.written is True


if __name__ == '__main__':
    test_rltrainer_calls_report_metrics()
    print('RLTRAINER-REPORT-METRICS-TEST: OK')
