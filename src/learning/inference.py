"""Simulator-independent policy checkpoint loading and preprocessing."""

import numpy as np
import torch

from .models import load_model
from data.datasets import Normalizer, compose_task_state


def _as_numpy(value, dtype):
    """Convert NumPy-like or device-backed torch observations safely."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class PolicyInference:
    """Turn an environment observation into absolute 22-target action chunks."""

    def __init__(
        self, model, normalizer, state_keys, camera_names, controlled_joint_names, model_name, device="cpu"
    ):
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.normalizer = normalizer
        self.state_keys = tuple(state_keys)
        self.camera_names = tuple(camera_names)
        self.controlled_joint_names = tuple(controlled_joint_names)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        model_name = payload["model_name"]
        model = load_model(model_name, payload["model_config"], payload["model"])
        dataset_config = payload["manifest"]["config"]
        return cls(
            model, Normalizer.from_dict(payload["normalizer"]),
            dataset_config["state_keys"], dataset_config["camera_names"],
            payload["manifest"]["controlled_joint_names"], model_name, device,
        )

    @torch.inference_mode()
    def predict(self, observation, observation_joint_names=None):
        """Predict action chunks for an unbatched or batched observation."""
        batched = False
        if self.state_keys == ("proprio",) and "proprio" in observation:
            state = _as_numpy(observation["proprio"], np.float32)
            batched = state.ndim == 2
            if state.ndim == 1:
                state = state[None]
            if state.ndim != 2 or state.shape[1] != self.model.config["state_dim"]:
                raise ValueError(
                    f"proprio must have shape [{self.model.config['state_dim']}] or "
                    f"[B,{self.model.config['state_dim']}], got {state.shape}"
                )
        elif self.state_keys == ("proprio",):
            joint_pos = _as_numpy(observation["joint_pos"], np.float32)
            eef_pose = _as_numpy(observation["eef_pose_body"], np.float32)
            batched = joint_pos.ndim == 2 or eef_pose.ndim == 3
            if joint_pos.ndim == 1:
                joint_pos = joint_pos[None]
            if eef_pose.ndim == 2:
                eef_pose = eef_pose[None]
            if joint_pos.ndim != 2 or eef_pose.ndim != 3 or joint_pos.shape[0] != eef_pose.shape[0]:
                raise ValueError(
                    "joint_pos and eef_pose_body must have matching [B,J] and [B,2,7] batches"
                )
            if observation_joint_names is None:
                if joint_pos.shape[1] != len(self.controlled_joint_names):
                    raise ValueError(
                        f"joint_pos has {joint_pos.shape[1]} values; provide observation_joint_names to map state"
                    )
                controlled_indices = tuple(range(len(self.controlled_joint_names)))
            else:
                names = tuple(observation_joint_names)
                if len(names) != joint_pos.shape[1] or len(set(names)) != len(names):
                    raise ValueError("observation_joint_names must be unique and match the joint state length")
                missing = set(self.controlled_joint_names).difference(names)
                if missing:
                    raise ValueError(f"observation_joint_names is missing controlled joints: {sorted(missing)}")
                controlled_indices = tuple(names.index(name) for name in self.controlled_joint_names)
            state = np.stack([
                compose_task_state(joint_pos[index], eef_pose[index], controlled_indices)
                for index in range(joint_pos.shape[0])
            ])
        else:
            batch_size = None
            for key in self.state_keys:
                value = _as_numpy(observation[key], np.float32)
                if key in ("joint_pos", "joint_vel") and value.ndim == 2:
                    batch_size = value.shape[0]
                    break
            if batch_size is None:
                for name in self.camera_names:
                    image = _as_numpy(observation[name], np.uint8)
                    if image.ndim == 4:
                        batch_size = image.shape[0]
                        break
            batched = batch_size is not None
            batch_size = batch_size or 1
            values = []
            for key in self.state_keys:
                value = _as_numpy(observation[key], np.float32)
                if not batched:
                    value = value.reshape(1, -1)
                elif value.shape[0] == batch_size:
                    value = value.reshape(batch_size, -1)
                else:
                    raise ValueError(f"{key} batch does not match batch size {batch_size}: {value.shape}")
                if key in ("joint_pos", "joint_vel"):
                    if observation_joint_names is None and value.shape[1] != len(self.controlled_joint_names):
                        raise ValueError(
                            f"{key} has {value.shape[1]} values; provide observation_joint_names "
                            "to select controlled joints"
                        )
                    if observation_joint_names is not None:
                        names = tuple(observation_joint_names)
                        if len(names) != value.shape[1] or len(set(names)) != len(names):
                            raise ValueError("observation_joint_names must be unique and match the joint state length")
                        missing = set(self.controlled_joint_names).difference(names)
                        if missing:
                            raise ValueError(f"observation_joint_names is missing controlled joints: {sorted(missing)}")
                        value = value[:, [names.index(name) for name in self.controlled_joint_names]]
                values.append(value)
            state = np.concatenate(values, axis=1)
        state = torch.from_numpy(self.normalizer.normalize_state(state).astype(np.float32)).to(self.device)
        images = {}
        for name in self.camera_names:
            image = _as_numpy(observation[name], np.uint8)
            if image.ndim == 3:
                image = image[None]
            if image.ndim != 4 or image.shape[0] != state.shape[0] or image.shape[-1] != 3:
                raise ValueError(
                    f"{name} must have shape [H,W,3] or [B,H,W,3] with B={state.shape[0]}, got {image.shape}"
                )
            images[name] = torch.from_numpy(image.copy()).permute(0, 3, 1, 2).float().div_(255).to(self.device)
        normalized = self.model.predict(state, images).cpu().numpy()
        actions = self.normalizer.denormalize_action(normalized)
        return actions if batched else actions[0]
