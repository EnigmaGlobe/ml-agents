import numpy as np
import pytest

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import (
    CJepaRewardProvider,
)


def make_dummy_buffer(batch_size=2, T=6, S=4, D=8):
    buf = AgentBuffer()
    slots = np.random.randn(batch_size, T, S, D).astype(np.float32)
    from mlagents.trainers.buffer import ObservationKeyPrefix
    buf[(ObservationKeyPrefix.OBSERVATION, 0)] = [slots]
    buf["agent_ids"] = list(range(batch_size))
    return buf


def test_update_step_or_skip():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        pytest.skip("torch not installed; skipping CJepa update test")

    cfg = type("C", (), {"history_size": 3, "num_preds": 1, "freeze": False, "learning_rate": 1e-3, "gamma": 0.99, "strength": 1.0})
    provider = CJepaRewardProvider(None, cfg)

    # simple predictor: a tiny nn module with an inference(hist) wrapper
    class TinyPred(nn.Module):
        def __init__(self, S, D, num_preds=1):
            super().__init__()
            self.net = nn.Linear(S * D * 3, S * D * num_preds)

        def inference(self, hist):
            # hist shape B,H,S,D
            B, H, S, D = hist.shape
            x = hist.reshape(B, -1)
            out = self.net(x)
            out = out.view(B, 1, S, D)
            return out

    buf = make_dummy_buffer()
    pred = TinyPred(4, 8)
    provider.load_predictor(pred)
    stats = provider.update(buf)
    assert isinstance(stats, dict)
    assert "jepa_future_loss" in stats
