import numpy as np
import torch
from mlagents.trainers.buffer import AgentBuffer, BufferKey

from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import (
    CJepaRewardProvider,
)


def make_dummy_buffer(batch_size=2, T=6, S=4, D=8):
    buf = AgentBuffer()
    # create random slot embeddings
    slots = np.random.randn(batch_size, T, S, D).astype(np.float32)
    # put as observation 0 path (mimic curiosity provider way)
    from mlagents.trainers.buffer import ObservationKeyPrefix
    buf[(ObservationKeyPrefix.OBSERVATION, 0)] = [slots]
    # agent ids length
    buf["agent_ids"] = list(range(batch_size))
    return buf


def test_c_jepa_evaluate_preextracted_slots():
    buf = make_dummy_buffer()
    # Create provider with a minimal config-like object
    class Cfg:
        history_size = 3
        num_preds = 1
        freeze = True
    # minimal attrs expected by BaseRewardProvider
    Cfg.gamma = 0.99
    Cfg.strength = 1.0
    Cfg.learning_rate = 1e-4
    cfg = Cfg()

    provider = CJepaRewardProvider(None, cfg)

    # Attach a dummy predictor with inference() that returns zeros of correct shape
    class DummyPred:
        def inference(self, history):
            B, H, S, D = history.shape
            return torch.zeros((B, cfg.num_preds, S, D))

    provider.load_predictor(DummyPred())

    rewards = provider.evaluate(buf)
    assert rewards.shape[0] == 2
    assert (rewards >= 0).all()


def test_c_jepa_update_stub_noop():
    buf = make_dummy_buffer()
    class Cfg:
        history_size = 3
        num_preds = 1
        freeze = True
    Cfg.gamma = 0.99
    Cfg.strength = 1.0
    Cfg.learning_rate = 1e-4
    cfg = Cfg()
    provider = CJepaRewardProvider(None, cfg)
    out = provider.update(buf)
    assert isinstance(out, dict)
