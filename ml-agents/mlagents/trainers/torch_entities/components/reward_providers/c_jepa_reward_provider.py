import numpy as np
from typing import Dict, Optional
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_entities.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)

from mlagents.trainers.settings import JepaSettings

try:
    from src.custom_codes.hungarian import (
        reorder_slots_to_match,
        hungarian_matching_loss_with_proprio,
    )
except Exception:
    # In case src is not on PYTHONPATH during unit tests, provide no-op fallbacks
    # Minimal fallback implementations. These do NOT perform true Hungarian matching
    # but provide sensible fallbacks for unit tests and CI environments where
    # the full `src.custom_codes.hungarian` helpers are not available.
    def reorder_slots_to_match(pred, reference, cost_type="mse"):
        # Attempt a simple nearest-neighbor reorder on CPU using numpy if possible,
        # otherwise return predictions unchanged.
        try:
            import numpy as _np

            p = pred.detach().cpu().numpy() if hasattr(pred, "detach") else _np.array(pred)
            r = reference.detach().cpu().numpy() if hasattr(reference, "detach") else _np.array(reference)
            # p: (B, num_preds, S, D), r: (B, S, D) -> compute simple per-slot distances to reorder each sample
            B = p.shape[0]
            out = _np.zeros_like(p)
            for i in range(B):
                # use last time pred (num_preds may be 1)
                cand = p[i, -1]
                ref = r[i]
                # compute L2 distance matrix SxS
                d = _np.linalg.norm(cand[:, None, :] - ref[None, :, :], axis=-1)
                # greedy matching
                perm = _np.argmin(d, axis=1)
                out[i] = p[i, :, perm, :]
            # return same shape as pred
            return torch.tensor(out, device=getattr(pred, 'device', None)) if hasattr(pred, 'device') else out
        except Exception:
            return pred

    def hungarian_matching_loss_with_proprio(pred, target, pixels_dim, proprio_dim=0, cost_type="mse"):
        # Fallback: simple MSE over pixels portion
        try:
            if hasattr(pred, 'detach'):
                pred_t = pred
            else:
                pred_t = torch.tensor(pred)
            if not isinstance(target, torch.Tensor):
                target_t = torch.tensor(target, device=pred_t.device)
            else:
                target_t = target.to(pred_t.device)
            loss = ((pred_t - target_t) ** 2).mean()
            return {"pixels_loss": loss}
        except Exception:
            return {"pixels_loss": torch.tensor(0.0, device=getattr(pred, 'device', 'cpu'))}

logger = __import__("logging").getLogger(__name__)


class CJepaRewardProvider(BaseRewardProvider):
    """Minimal C-JEPA-based intrinsic reward provider.

    Modes supported (config-driven):
    - frozen_preextracted: expects `pixels_embed` in AgentBuffer (B, T, S, D)
    - frozen_causalwm: prefer using an attached causal_wm.predict(hist) when present
    - online_training: will run masked predictor updates (requires torch predictor)
    """

    def __init__(self, specs, settings: JepaSettings, mode: str = "frozen_preextracted"):
        super().__init__(specs, settings)
        self._device = default_device()
        self.cfg = settings
        # prefer mode from settings if present
        self.mode = getattr(self.cfg, "mode", mode)
        self._has_updated_once = False

        # Placeholder: predictor / causal_wm will be set via load_checkpoint or external injection
        self.predictor = None
        self.causal_wm = None

        # Hungarian matching defaults
        self.use_hungarian_matching = getattr(self.cfg, "use_hungarian_matching", True)
        self.hungarian_cost_type = getattr(self.cfg, "hungarian_cost_type", "mse")
        # lightweight metrics store (mean/std rewards, last loss)
        self._metrics = {}

    def load_predictor(self, predictor_obj, causal_wm: Optional[object] = None):
        """Attach an already-constructed predictor or causal_wm object (no heavy imports).
        The predictor should implement an `inference(history)` method returning
        shape (B, num_preds, S, D) either as numpy array or torch tensor.
        """
        # Simple assignment; the heavy logic is in evaluate/update implemented below
        self.predictor = predictor_obj
        self.causal_wm = causal_wm

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        """Compute intrinsic reward from pre-extracted slots in the mini_batch.

        Expects the mini_batch to contain slot embeddings at
        (ObservationKeyPrefix.OBSERVATION, 0) or 'pixels_embed' shaped (B, T, S, D).
        """
        # Import locally to avoid heavy deps at module import time
        from mlagents.trainers.buffer import ObservationKeyPrefix

        slots = None
        try:
            slots = mini_batch[(ObservationKeyPrefix.OBSERVATION, 0)][0]
        except Exception:
            slots = mini_batch.get("pixels_embed", None)

        if slots is None:
            logger.debug("C-JEPA provider: no slots found; returning zeros")
            batch_size = len(mini_batch.get("agent_ids", [])) or 1
            return np.zeros((batch_size,), dtype=np.float32)

        # Work with numpy arrays when possible to avoid importing torch here.
        try:
            if hasattr(slots, "numpy"):
                slots_np = slots.numpy()
            else:
                slots_np = np.array(slots)
        except Exception:
            slots_np = np.array(slots)

        B, T, S, D = slots_np.shape
        history_size = getattr(self.cfg, "history_size", min(3, T))
        num_preds = getattr(self.cfg, "num_preds", 1)

        if T < history_size + num_preds:
            return np.zeros((B,), dtype=np.float32)

        hist = slots_np[:, :history_size, :, :]

        # Predict using attached predictor (if available)
        pred = None
        # frozen_causalwm mode: prefer causal_wm.predict/rollout
        if self.mode == "frozen_causalwm" and self.causal_wm is not None:
            if hasattr(self.causal_wm, "predict"):
                pred = self.causal_wm.predict(hist, use_inference_function=True)
            elif hasattr(self.causal_wm, "rollout"):
                pred = self.causal_wm.rollout(hist)
        # fallback: predictor inference
        if pred is None:
            if self.predictor is not None and hasattr(self.predictor, "inference"):
                pred = self.predictor.inference(hist)
            else:
                logger.debug("C-JEPA provider: no predictor attached; returning zeros")
                return np.zeros((B,), dtype=np.float32)

        # Convert pred to numpy if needed
        try:
            if hasattr(pred, "numpy"):
                pred_np = pred.numpy()
            else:
                pred_np = np.array(pred)
        except Exception:
            pred_np = np.array(pred)

        tgt = slots_np[:, history_size : history_size + num_preds, :, :]

        # Compute simple MSE reward across slots/dims, averaged over preds
        err = np.mean((pred_np - tgt) ** 2, axis=(2, 3))  # (B, num_preds)
        scalar_err = np.mean(err, axis=1)  # (B,)
        strength = getattr(self.cfg, "strength", 1.0)
        reward = scalar_err * float(strength)
        # record simple metrics
        try:
            self._metrics["reward_mean"] = float(np.mean(reward))
            self._metrics["reward_std"] = float(np.std(reward))
        except Exception:
            self._metrics["reward_mean"] = float(np.mean(np.array(reward)))
            self._metrics["reward_std"] = float(np.std(np.array(reward)))
        logger.debug("C-JEPA reward mean=%.6f std=%.6f", self._metrics.get("reward_mean", 0.0), self._metrics.get("reward_std", 0.0))
        return reward

    def update(self, mini_batch: AgentBuffer) -> Dict[str, float]:
        # Conservative online update: compute simple future-MSE loss and run one optimizer step
        if getattr(self.cfg, "freeze", True):
            return {}

        # Only run gradient updates in explicit online_training mode
        if getattr(self, "mode", "frozen_preextracted") != "online_training":
            logger.debug("CJepa update skipped because mode=%s (not online_training)", getattr(self, "mode", None))
            return {}

        # Require a torch predictor to perform gradient-based updates
        try:
            import torch
            from torch import optim
            import torch.nn.functional as F
        except Exception:
            logger.warning("Torch not available; skipping CJepa update()")
            return {}

        if self.predictor is None or not hasattr(self.predictor, "parameters"):
            logger.warning("No torch predictor attached for CJepa update(); skipping")
            return {}

        # Build a tiny optimizer if not present
        opt = getattr(self, "_jepa_optimizer", None)
        if opt is None:
            opt = optim.Adam(self.predictor.parameters(), lr=getattr(self.cfg, "learning_rate", 1e-4))
            self._jepa_optimizer = opt

        # Extract slots and targets as torch tensors
        from mlagents.trainers.buffer import ObservationKeyPrefix
        try:
            slots = mini_batch[(ObservationKeyPrefix.OBSERVATION, 0)][0]
        except Exception:
            slots = mini_batch.get("pixels_embed", None)
        if slots is None:
            return {}

        # Convert to torch tensor if needed
        if not hasattr(slots, "shape") or not hasattr(slots, "dtype"):
            slots = torch.from_numpy(np.array(slots)).float()
        elif not isinstance(slots, torch.Tensor):
            slots = torch.tensor(slots).float()
        slots = slots.to(next(self.predictor.parameters()).device)

        B, T, S, D = slots.shape
        history_size = getattr(self.cfg, "history_size", min(3, T))
        num_preds = getattr(self.cfg, "num_preds", 1)
        if T < history_size + num_preds:
            return {}

        hist = slots[:, :history_size, :, :]
        tgt = slots[:, history_size: history_size + num_preds, :, :]

        # Determine masks for loss (if provided in buffer)
        # Expect BufferKey.MASKS to contain a list/tensor shaped (B, T) or similar; fallback to no mask
        loss_mask = None
        try:
            from mlagents.trainers.buffer import BufferKey

            if BufferKey.MASKS in mini_batch:
                masks_entry = mini_batch[BufferKey.MASKS]
                # masks_entry may be a list-like; convert to tensor
                if hasattr(masks_entry, "numpy"):
                    loss_mask = torch.tensor(masks_entry).to(slots.device)
                else:
                    loss_mask = torch.tensor(np.array(masks_entry)).to(slots.device)
                # Expect mask shape B x T; reduce to target positions
                # Target frames start at history_size and go for num_preds
                loss_mask = loss_mask[:, history_size : history_size + num_preds]
        except Exception:
            loss_mask = None

        # forward
        self.predictor.train()
        pred = self.predictor.inference(hist)
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(np.array(pred), device=slots.device)

        # Optionally apply Hungarian matching to reorder pred to best-match targets per sample
        if getattr(self.cfg, "use_hungarian_matching", False):
            try:
                # try to use provided helper; returns a dict of losses (pixels_loss, etc.)
                res = hungarian_matching_loss_with_proprio(pred, tgt, getattr(self.cfg, "pixels_dim", D), getattr(self.cfg, "proprio_dim", 0), cost_type=getattr(self.cfg, "hungarian_cost_type", "mse"))
                # If helper returns a pixels_loss tensor, use it
                pixels_loss = res.get("pixels_loss", None) if isinstance(res, dict) else None
                if pixels_loss is not None:
                    loss = pixels_loss.mean()
                else:
                    # Fallback to simple mse
                    loss = F.mse_loss(pred, tgt)
            except Exception:
                logger.exception("Hungarian matching failed; falling back to plain MSE")
                loss = F.mse_loss(pred, tgt)
        else:
            # If mask provided, compute masked MSE across pred/target dims
            if loss_mask is not None:
                # loss_mask shape: (B, num_preds) -> expand to match (B, num_preds, S, D)
                mask = loss_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, D).float()
                per_elem_loss = (pred - tgt) ** 2
                masked_loss = (per_elem_loss * mask).sum() / (mask.sum().clamp(min=1.0))
                loss = masked_loss
            else:
                loss = F.mse_loss(pred, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        final_loss = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
        # record metric
        self._metrics["last_loss"] = final_loss
        logger.info("CJepa update step completed; loss=%.6f", final_loss)
        return {"jepa_future_loss": final_loss}

    def get_metrics(self) -> Dict[str, float]:
        """Return collected metrics (non-exhaustive)."""
        return dict(self._metrics)

    def report_metrics(self, stats_reporter) -> None:
        """Report collected metrics to a StatsReporter instance.

        This expects `stats_reporter` to implement `add_stat(key, value)` and/or `set_stat(key, value)`.
        We namespace the keys under 'JEPA/'.
        """
        if stats_reporter is None:
            return
        try:
            mm = self.get_metrics()
            for k, v in mm.items():
                # prefer add_stat so it's aggregated in summaries
                try:
                    stats_reporter.add_stat(f"JEPA/{k}", float(v))
                except Exception:
                    try:
                        stats_reporter.set_stat(f"JEPA/{k}", float(v))
                    except Exception:
                        logger.debug("Failed to report JEPA metric %s", k)
        except Exception:
            logger.exception("Failed to report CJepa metrics")

    def get_modules(self) -> Dict[str, object]:
        mods: Dict[str, object] = {}
        if self.predictor is not None:
            mods[f"Module:{self.name}"] = self.predictor
        return mods

    def save(self, path: str) -> None:
        # If predictor exposes state_dict, try to save it; otherwise noop
        try:
            state = getattr(self.predictor, "state_dict", None)
            if callable(state):
                import torch

                data = {"predictor_state": self.predictor.state_dict()}
                # Save optimizer state if present
                try:
                    opt = getattr(self, '_jepa_optimizer', None)
                    if opt is not None:
                        data['optimizer_state'] = opt.state_dict()
                except Exception:
                    logger.debug('Could not save JEPA optimizer state')
                torch.save(data, path)
        except Exception:
            logger.exception("Failed to save CJepa predictor state")

    def load(self, path: str) -> None:
        # If predictor attached and has load_state_dict, attempt to load
        try:
            loader = getattr(self.predictor, "load_state_dict", None)
            if callable(loader):
                import torch

                state = torch.load(path, map_location="cpu")
                # Backward-compatible: support either raw state_dict or our saved dict
                if isinstance(state, dict) and 'predictor_state' in state:
                    self.predictor.load_state_dict(state['predictor_state'])
                    # restore optimizer if available and predictor parameters present
                    try:
                        opt_state = state.get('optimizer_state', None)
                        if opt_state is not None:
                            import torch.optim as _optim
                            # create optimizer if missing
                            opt = getattr(self, '_jepa_optimizer', None)
                            if opt is None:
                                opt = _optim.Adam(self.predictor.parameters(), lr=getattr(self.cfg, 'learning_rate', 1e-4))
                                self._jepa_optimizer = opt
                            opt.load_state_dict(opt_state)
                    except Exception:
                        logger.debug('Could not restore JEPA optimizer state')
                else:
                    # assume it's a raw predictor state_dict
                    self.predictor.load_state_dict(state)
        except Exception:
            logger.exception("Failed to load CJepa predictor state")
