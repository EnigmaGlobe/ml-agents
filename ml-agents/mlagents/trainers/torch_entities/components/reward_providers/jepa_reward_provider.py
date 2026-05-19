import numpy as np
from typing import Dict, Optional
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_entities.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import CuriositySettings

from src.custom_codes.hungarian import (
    reorder_slots_to_match,
    hungarian_matching_loss_with_proprio,
)

logger = __import__("logging").getLogger(__name__)


class JepaRewardProvider(BaseRewardProvider):
    """Minimal JEPA-based intrinsic reward provider.

    Modes supported (config-driven):
    - frozen_preextracted: expects `pixels_embed` in AgentBuffer (B, T, S, D)
    - online_training: will run masked predictor updates (not implemented fully here)
    """

    def __init__(self, specs, settings: CuriositySettings, mode: str = "frozen_preextracted"):
        super().__init__(specs, settings)
        self._device = default_device()
        self.cfg = settings
        self.mode = mode
        self._has_updated_once = False

        # Placeholder: predictor / causal_wm will be set via load_checkpoint or external injection
        self.predictor = None
        self.causal_wm = None

        # Hungarian matching defaults
        self.use_hungarian_matching = getattr(settings, "use_hungarian_matching", True)
        self.hungarian_cost_type = getattr(settings, "hungarian_cost_type", "mse")

    def load_predictor(self, predictor_obj, causal_wm: Optional[object] = None):
        """Attach an already-constructed predictor or causal_wm object (no import here)."""
        self.predictor = predictor_obj
        self.causal_wm = causal_wm
        if self.predictor is not None:
            self.predictor.to(self._device)

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        # Expect pre-extracted slots keyed 'pixels_embed' (B, T, S, D)
        try:
            slots = mini_batch[BufferKey.OBSERVATIONS][0]
        except Exception:
            # Fallback: try explicit key
            slots = mini_batch.get("pixels_embed", None)

        if slots is None:
            logger.warning("JEPA provider: no slots found in mini_batch; returning zeros")
            batch_size = len(mini_batch[BufferKey.AGENT_IDS]) if BufferKey.AGENT_IDS in mini_batch else 1
            return np.zeros((batch_size,))

        # slots is numpy array or tensor-like; ensure torch
        if isinstance(slots, np.ndarray):
            slots_t = torch.from_numpy(slots).to(self._device)
        else:
            slots_t = slots.to(self._device)

        B, T, S, D = slots_t.shape

        # Build history windows
        hist = slots_t[:, : self.cfg.history_size, :, :]

        # Predict
        if self.causal_wm is not None:
            pred = self.causal_wm.predict(hist, use_inference_function=True)
        elif self.predictor is not None:
            pred = self.predictor.inference(hist)
        else:
            logger.warning("JEPA provider: no predictor attached; returning zeros")
            return np.zeros((B,))

        # Extract targets from same batch: embedding[:, history_size:history_size+num_preds, :, :]
        tgt = slots_t[:, self.cfg.history_size : self.cfg.history_size + self.cfg.num_preds, :, :]

        # Optionally reorder predictions using Hungarian
        if self.use_hungarian_matching and self.causal_wm is None:
            pred = reorder_slots_to_match(pred, reference=slots_t[:, -1, :, :], cost_type=self.hungarian_cost_type)

        # Compute MSE per sample across slots and dims, average over pred frames
        err = ((pred - tgt) ** 2).mean(dim=[2, 3])  # (B, num_preds)
        scalar_err = err.mean(dim=1)  # (B,)

        reward = scalar_err.detach().cpu().numpy() * self.strength
        return reward

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        # Minimal stub: real online training requires predictor optimizer and mask handling
        if self.cfg.freeze:
            return {}
        # Not implemented fully here
        return {}

    def get_modules(self):
        # Expose predictor module for checkpoint saving if present
        mods = {}
        if self.predictor is not None:
            mods[f"Module:{self.name}"] = self.predictor
        return mods
