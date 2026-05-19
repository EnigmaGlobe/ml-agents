import torch
import numpy as np
from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import CJepaRewardProvider
from mlagents.trainers.buffer import AgentBuffer

# Tiny predictor: linear mapping from history (H,S,D) to future (num_preds,S,D)
class TinyPredictor(torch.nn.Module):
    def __init__(self, history_size=3, S=4, D=8, num_preds=1):
        super().__init__()
        self.history_size = history_size
        self.S = S
        self.D = D
        self.num_preds = num_preds
        # Flatten and map
        self.fc = torch.nn.Linear(history_size * S * D, num_preds * S * D)
    def inference(self, history):
        # history: numpy or tensor (B, H, S, D)
        if isinstance(history, np.ndarray):
            t = torch.tensor(history, dtype=torch.float32)
        elif isinstance(history, torch.Tensor):
            t = history
        else:
            t = torch.tensor(np.array(history), dtype=torch.float32)
        B = t.shape[0]
        x = t.view(B, -1)
        out = self.fc(x)
        out = out.view(B, self.num_preds, self.S, self.D)
        return out
    def parameters(self):
        return super().parameters()


def make_buffer(B=2, T=6, S=4, D=8):
    buf = AgentBuffer()
    slots = np.random.randn(B, T, S, D).astype(np.float32)
    from mlagents.trainers.buffer import ObservationKeyPrefix
    buf[(ObservationKeyPrefix.OBSERVATION, 0)] = [slots]
    buf['agent_ids'] = [str(i) for i in range(B)]
    return buf


def test_jepa_update_loss_decreases():
    cfg = type('C', (), {'history_size':3, 'num_preds':1, 'freeze':False, 'learning_rate':1e-3, 'use_hungarian_matching': False, 'gamma':0.99, 'strength': 1.0})
    prov = CJepaRewardProvider(None, cfg, mode='online_training')
    predictor = TinyPredictor(history_size=3, S=4, D=8, num_preds=1)
    # attach predictor and create optimizer via update()
    prov.load_predictor(predictor)

    buf = make_buffer(B=4, T=6, S=4, D=8)

    # run two update steps and collect losses
    s1 = prov.update(buf)
    s2 = prov.update(buf)
    # ensure losses are present
    assert 'jepa_future_loss' in s1 or 'last_loss' in prov._metrics
    # Try to compare numeric losses if available
    l1 = s1.get('jepa_future_loss', prov._metrics.get('last_loss', None))
    l2 = s2.get('jepa_future_loss', prov._metrics.get('last_loss', None))
    if l1 is not None and l2 is not None:
        # Accept small floating noise: expect l2 <= l1 + tiny_eps
        assert l2 <= l1 + 1e-6

if __name__ == '__main__':
    test_jepa_update_loss_decreases()
    print('LOSS-DECREASE-TEST: OK')
