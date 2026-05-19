"""Lightweight smoke script to run a single evaluate() call on CJepaRewardProvider.

This does NOT run the full trainer. It creates a minimal AgentBuffer with random
pre-extracted slot embeddings and prints the intrinsic rewards returned by the provider.
"""
import numpy as np
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--D", type=int, default=8)
    args = parser.parse_args()

    cfg = type("C", (), {"history_size": 3, "num_preds": 1, "freeze": True, "strength": 0.01, "gamma": 0.99, "learning_rate": 1e-4})
    provider = CJepaRewardProvider(None, cfg)

    # Attach a dummy predictor that returns zeros of proper shape
    class DummyPred:
        def inference(self, history):
            B, H, S, D = history.shape
            return np.zeros((B, cfg.num_preds, S, D), dtype=np.float32)

    provider.load_predictor(DummyPred())

    buf = make_dummy_buffer(args.batch, args.T, args.S, args.D)
    rewards = provider.evaluate(buf)
    print("intrinsic rewards:", rewards)


if __name__ == "__main__":
    main()
