import numpy as np
from mlagents.trainers.settings import JepaSettings
from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import CJepaRewardProvider


class DummyCausalWM:
    def __init__(self, out_val=0.0):
        self.out_val = out_val

    def predict(self, hist, use_inference_function=True):
        # hist is numpy; return zeros shaped (B, num_preds, S, D)
        import numpy as _np
        B, H, S, D = hist.shape
        num_preds = 1
        return _np.zeros((B, num_preds, S, D), dtype=_np.float32) + self.out_val


def test_frozen_causalwm_mode_uses_causalwm():
    cfg = JepaSettings()
    cfg.history_size = 1
    cfg.num_preds = 1
    cfg.strength = 1.0
    cfg.mode = "frozen_causalwm"

    # Prepare dummy buffer with pixels_embed: shape (B, T, S, D)
    B, T, S, D = 2, 3, 4, 8
    pixels = np.zeros((B, T, S, D), dtype=np.float32)
    mb = {"pixels_embed": pixels, "agent_ids": ["a", "b"]}

    provider = CJepaRewardProvider(None, cfg)
    provider.load_predictor(None, causal_wm=DummyCausalWM(out_val=2.0))

    rewards = provider.evaluate(mb)
    assert rewards.shape[0] == B
    # Since causal wm returns constant 2.0 and target is zeros, MSE should be 4.0 per element averaged -> >0
    assert (rewards > 0).all()
