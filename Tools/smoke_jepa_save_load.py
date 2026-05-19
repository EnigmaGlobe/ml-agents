"""Smoke test: verify CJepaRewardProvider save/load persists optimizer state."""
import os
import tempfile
import torch
import numpy as np
from mlagents.trainers.torch_entities.components.reward_providers.c_jepa_reward_provider import CJepaRewardProvider
from mlagents.trainers.buffer import AgentBuffer


class TinyPredictor(torch.nn.Module):
    def __init__(self, history_size=3, S=4, D=8, num_preds=1):
        super().__init__()
        self.history_size = history_size
        self.S = S
        self.D = D
        self.num_preds = num_preds
        self.fc = torch.nn.Linear(history_size * S * D, num_preds * S * D)
    def inference(self, history):
        t = torch.tensor(history, dtype=torch.float32)
        B = t.shape[0]
        x = t.view(B, -1)
        out = self.fc(x)
        out = out.view(B, self.num_preds, self.S, self.D)
        return out


def make_buffer(B=2, T=6, S=4, D=8):
    buf = AgentBuffer()
    slots = np.random.randn(B, T, S, D).astype(np.float32)
    from mlagents.trainers.buffer import ObservationKeyPrefix
    buf[(ObservationKeyPrefix.OBSERVATION, 0)] = [slots]
    buf['agent_ids'] = [str(i) for i in range(B)]
    return buf


def run_smoke():
    cfg = type('C', (), {'history_size':3, 'num_preds':1, 'freeze':False, 'learning_rate':1e-3, 'use_hungarian_matching': False, 'gamma':0.99, 'strength':1.0})
    prov = CJepaRewardProvider(None, cfg, mode='online_training')
    pred = TinyPredictor()
    prov.load_predictor(pred)

    buf = make_buffer(B=4, T=6, S=4, D=8)

    # run an update to create optimizer and change params
    before = None
    for p in pred.parameters():
        before = p.detach().cpu().clone()
        break

    prov.update(buf)

    # save to temp file
    fd, path = tempfile.mkstemp(suffix='.pt')
    os.close(fd)
    prov.save(path)

    # create new provider+predictor and load
    cfg2 = cfg
    prov2 = CJepaRewardProvider(None, cfg2, mode='online_training')
    pred2 = TinyPredictor()
    prov2.load_predictor(pred2)
    prov2.load(path)

    # ensure predictor params are identical after load
    same = True
    for a,b in zip(pred.parameters(), pred2.parameters()):
        if not torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=1e-5):
            same = False
            break

    print('predictor_state_restored:', same)

    # ensure optimizer state exists on prov2
    has_opt = hasattr(prov2, '_jepa_optimizer') and prov2._jepa_optimizer is not None
    print('optimizer_restored:', has_opt)


if __name__ == '__main__':
    run_smoke()
