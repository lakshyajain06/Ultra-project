"""Simulator-independent checkpoint loading and observation preprocessing."""

import numpy as np
import torch

from .models import ACTConfig, ACTPolicy
from .training.data import Normalizer


class ACTInference:
    """Turn an environment observation into absolute 22-target action chunks."""

    def __init__(self, model, normalizer, state_keys, camera_names, controlled_joint_names, device="cpu"):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.state_keys = tuple(state_keys)
        self.camera_names = tuple(camera_names)
        self.controlled_joint_names = tuple(controlled_joint_names)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        model = ACTPolicy(ACTConfig.from_dict(payload["model_config"]))
        model.load_state_dict(payload["model"])
        dataset_config = payload["manifest"]["config"]
        return cls(
            model, Normalizer.from_dict(payload["normalizer"]),
            dataset_config["state_keys"], dataset_config["camera_names"],
            payload["manifest"]["controlled_joint_names"], device,
        )

    @torch.inference_mode()
    def predict(self, observation, observation_joint_names=None):
        """Predict for one manager environment or one named legacy observation."""
        if self.state_keys == ("joint_pos", "joint_vel") and "proprio" in observation:
            state = np.asarray(observation["proprio"], dtype=np.float32)
            if state.ndim == 2:
                if state.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; proprio batch is {state.shape[0]}")
                state = state[0]
            if state.ndim != 1 or state.size != 2 * len(self.controlled_joint_names):
                raise ValueError(f"proprio must have shape [44] or [1,44], got {state.shape}")
        else:
            values = []
            for key in self.state_keys:
                value = np.asarray(observation[key], dtype=np.float32)
                if key in ("joint_pos", "joint_vel") and value.ndim == 2:
                    if value.shape[0] != 1:
                        raise ValueError(f"ACTInference supports one environment; {key} batch is {value.shape[0]}")
                    value = value[0]
                value = value.reshape(-1)
                if key in ("joint_pos", "joint_vel"):
                    if observation_joint_names is None and value.size != len(self.controlled_joint_names):
                        raise ValueError(
                            f"{key} has {value.size} values; provide observation_joint_names to select controlled joints"
                        )
                    if observation_joint_names is not None:
                        names = tuple(observation_joint_names)
                        if len(names) != value.size or len(set(names)) != len(names):
                            raise ValueError("observation_joint_names must be unique and match the joint state length")
                        missing = set(self.controlled_joint_names).difference(names)
                        if missing:
                            raise ValueError(f"observation_joint_names is missing controlled joints: {sorted(missing)}")
                        value = value[[names.index(name) for name in self.controlled_joint_names]]
                values.append(value)
            state = np.concatenate(values)
        state = torch.from_numpy(self.normalizer.normalize_state(state).astype(np.float32))[None].to(self.device)
        images = {}
        for name in self.camera_names:
            image = np.asarray(observation[name], dtype=np.uint8)
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; {name} batch is {image.shape[0]}")
                image = image[0]
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [H,W,3] or [1,H,W,3], got {image.shape}")
            images[name] = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div_(255).to(self.device)
        normalized = self.model.predict(state, images)[0].cpu().numpy()
        return self.normalizer.denormalize_action(normalized)
