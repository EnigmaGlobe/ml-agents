"""Small-scale JEPA experiment runner.

Runs the CJepaRewardProvider online update for multiple random seeds and
plots the per-step training loss curves to a PNG under the workspace.

Usage: run with the workspace venv python
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def run_seed(seed, steps=50):
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = type('C', (), {'history_size':3, 'num_preds':1, 'freeze':False, 'learning_rate':1e-3, 'use_hungarian_matching': False, 'gamma':0.99, 'strength':1.0})
    prov = CJepaRewardProvider(None, cfg, mode='online_training')
    pred = TinyPredictor()
    prov.load_predictor(pred)

    buf = make_buffer(B=8, T=6, S=4, D=8)

    losses = []
    for i in range(steps):
        stats = prov.update(buf)
        l = stats.get('jepa_future_loss', prov._metrics.get('last_loss', None))
        losses.append(l if l is not None else float('nan'))
    return losses


def main():
    seeds = [0, 1, 2]
    steps = 100
    all_losses = []
    for s in seeds:
        losses = run_seed(s, steps=steps)
        all_losses.append(losses)

    out_dir = os.getcwd()
    png = os.path.join(out_dir, 'jepa_smallscale_learning_curves.png')
    plt.figure(figsize=(6,4))
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label=f'seed_{seeds[i]}')
    plt.xlabel('update step')
    plt.ylabel('jepa loss')
    plt.legend()
    plt.title('JEPA small-scale learning curves (tiny predictor)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png)
    print('Saved learning curves to', png)


if __name__ == '__main__':
    main()
