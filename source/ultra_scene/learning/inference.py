"""Simulator-independent checkpoint loading and observation preprocessing."""

import numpy as np
import torch

from .models import ACTConfig, ACTPolicy
from .training.data import Normalizer, compose_task_state


def _as_numpy(value, dtype):
    """Convert NumPy-like or device-backed torch observations safely."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


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
        if self.state_keys == ("proprio",) and "proprio" in observation:
            state = _as_numpy(observation["proprio"], np.float32)
            if state.ndim == 2:
                if state.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; proprio batch is {state.shape[0]}")
                state = state[0]
            if state.ndim != 1 or state.size != self.model.config.state_dim:
                raise ValueError(
                    f"proprio must have shape [{self.model.config.state_dim}] or "
                    f"[1,{self.model.config.state_dim}], got {state.shape}"
                )
        elif self.state_keys == ("proprio",):
            joint_pos = _as_numpy(observation["joint_pos"], np.float32)
            if joint_pos.ndim == 2:
                if joint_pos.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; joint_pos batch is {joint_pos.shape[0]}")
                joint_pos = joint_pos[0]
            joint_pos = joint_pos.reshape(-1)
            eef_pose = _as_numpy(observation["eef_pose_body"], np.float32)
            if eef_pose.ndim == 3:
                if eef_pose.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; eef_pose_body batch is {eef_pose.shape[0]}")
                eef_pose = eef_pose[0]
            if observation_joint_names is None:
                if joint_pos.size != len(self.controlled_joint_names):
                    raise ValueError(
                        f"joint_pos has {joint_pos.size} values; provide observation_joint_names to map legacy state"
                    )
                controlled_indices = tuple(range(len(self.controlled_joint_names)))
            else:
                names = tuple(observation_joint_names)
                if len(names) != joint_pos.size or len(set(names)) != len(names):
                    raise ValueError("observation_joint_names must be unique and match the joint state length")
                missing = set(self.controlled_joint_names).difference(names)
                if missing:
                    raise ValueError(f"observation_joint_names is missing controlled joints: {sorted(missing)}")
                controlled_indices = tuple(names.index(name) for name in self.controlled_joint_names)
            state = compose_task_state(joint_pos, eef_pose, controlled_indices)
        else:
            values = []
            for key in self.state_keys:
                value = _as_numpy(observation[key], np.float32)
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
            image = _as_numpy(observation[name], np.uint8)
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise ValueError(f"ACTInference supports one environment; {name} batch is {image.shape[0]}")
                image = image[0]
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [H,W,3] or [1,H,W,3], got {image.shape}")
            images[name] = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div_(255).to(self.device)
        normalized = self.model.predict(state, images)[0].cpu().numpy()
        return self.normalizer.denormalize_action(normalized)
