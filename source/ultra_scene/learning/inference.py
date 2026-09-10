"""Simulator-independent checkpoint loading and observation preprocessing."""

import numpy as np
import torch

from .models import ACTConfig, ACTPolicy
from .training.data import Normalizer


class ACTInference:
    """Turn an environment observation into absolute 22-target action chunks."""

    def __init__(self, model, normalizer, state_keys, camera_names, device="cpu"):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.state_keys = tuple(state_keys)
        self.camera_names = tuple(camera_names)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        model = ACTPolicy(ACTConfig.from_dict(payload["model_config"]))
        model.load_state_dict(payload["model"])
        dataset_config = payload["manifest"]["config"]
        return cls(
            model, Normalizer.from_dict(payload["normalizer"]),
            dataset_config["state_keys"], dataset_config["camera_names"], device,
        )

    @torch.inference_mode()
    def predict(self, observation):
        state = np.concatenate([
            np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in self.state_keys
        ])
        state = torch.from_numpy(self.normalizer.normalize_state(state).astype(np.float32))[None].to(self.device)
        images = {
            name: torch.from_numpy(np.asarray(observation[name], dtype=np.uint8).copy())
            .permute(2, 0, 1)[None].float().div_(255).to(self.device)
            for name in self.camera_names
        }
        normalized = self.model.predict(state, images)[0].cpu().numpy()
        return self.normalizer.denormalize_action(normalized)
