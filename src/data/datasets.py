"""Episode-level splitting and transition windows for Ultra HDF5 data."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    DUAL_TASK_STATE_NAMES,
    DUAL_ULTRA_ACTION_JOINT_NAMES,
    TASK_STATE_NAMES,
    ULTRA_ACTION_JOINT_NAMES,
)


@dataclass(frozen=True)
class Episode:
    path: str
    key: str
    length: int
    status: str
    controlled_joint_indices: tuple[int, ...]
    controlled_joint_names: tuple[str, ...]
    proprio_layout: tuple[str, ...]

    @property
    def identity(self):
        return f"{Path(self.path).resolve()}::{self.key}"


class Normalizer:
    """Per-coordinate standardization fitted only on training episodes."""

    def __init__(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)

    def normalize_state(self, value):
        return (value - self.state_mean) / self.state_std

    def normalize_action(self, value):
        return (value - self.action_mean) / self.action_std

    def denormalize_action(self, value):
        return value * self.action_std + self.action_mean

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in (
            "state_mean", "state_std", "action_mean", "action_std"
        )}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def discover_episodes(config):
    episodes = []
    seen_statuses = set()
    seen_paths = set()
    requested_keys = {key.removeprefix("data/") for key in config["episode_keys"]}
    matched_keys = {key: [] for key in requested_keys}
    expected_shapes = None
    expected_contract = None
    for filename in config["paths"]:
        path = str(Path(filename).expanduser().resolve())
        if path in seen_paths:
            raise ValueError(f"Duplicate dataset path resolves to {path}; refusing train/validation leakage")
        seen_paths.add(path)
        with h5py.File(path, "r") as handle:
            schema_version = int(handle.attrs.get("schema_version", -1))
            if schema_version not in (4, 5, 6, 7):
                raise ValueError(f"{path} uses unsupported schema_version={schema_version}; expected 4 through 7")
            try:
                metadata = json.loads(handle.attrs["metadata"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path} has missing or invalid JSON metadata") from error
            action_names = tuple(metadata.get("action_joint_names", ()))
            expected_action_names = (
                DUAL_ULTRA_ACTION_JOINT_NAMES if schema_version == 7 else ULTRA_ACTION_JOINT_NAMES
            )
            if action_names != expected_action_names:
                raise ValueError(f"{path} action_joint_names do not match its canonical target order")
            proprio_layout = tuple(metadata.get("proprio_layout", ()))
            expected_proprio_layout = DUAL_TASK_STATE_NAMES if schema_version == 7 else TASK_STATE_NAMES
            if proprio_layout and proprio_layout != expected_proprio_layout:
                raise ValueError(f"{path} proprio_layout does not match its canonical state order")
            proprio_layout = proprio_layout or expected_proprio_layout
            contract = (action_names, proprio_layout)
            if expected_contract is None:
                expected_contract = contract
            elif contract != expected_contract:
                raise ValueError("Selected dataset files mix incompatible single- and dual-robot contracts")
            observation_names = tuple(metadata.get("observation_joint_names", ()))
            if not observation_names:
                raise ValueError(f"{path} metadata has no observation_joint_names")
            if len(set(observation_names)) != len(observation_names):
                raise ValueError(f"{path} observation_joint_names contains duplicates")
            missing_controlled = set(action_names).difference(observation_names)
            if missing_controlled:
                raise ValueError(f"{path} observation_joint_names is missing {sorted(missing_controlled)}")
            controlled_indices = tuple(observation_names.index(name) for name in action_names)
            if "data" not in handle:
                raise ValueError(f"{path} has no /data group")
            for key, demo in handle["data"].items():
                if key in matched_keys:
                    matched_keys[key].append(path)
                if requested_keys and key not in requested_keys:
                    continue
                status = str(demo.attrs.get("status", "unknown"))
                seen_statuses.add(status)
                length = len(demo.get("actions", ()))
                if status in config["statuses"] and length:
                    if demo["actions"].ndim != 2 or demo["actions"].shape[1] != len(action_names):
                        raise ValueError(
                            f"{path}::{key}/actions must be [T,{len(action_names)}] absolute targets; "
                            f"got {demo['actions'].shape}"
                        )
                    state_datasets = []
                    for name in config["state_keys"]:
                        if name == "proprio":
                            state_datasets.extend(
                                ("proprio",) if "obs/proprio" in demo else ("joint_pos", "eef_pose_body")
                            )
                        else:
                            state_datasets.append(name)
                    required = [f"obs/{k}" for k in dict.fromkeys((*state_datasets, *config["camera_names"]))]
                    missing = [name for name in required if name not in demo]
                    if missing:
                        raise KeyError(f"{path}::{key} is missing {missing}")
                    for name in required:
                        if len(demo[name]) != length:
                            raise ValueError(f"{path}::{key}/{name} is not aligned with actions")
                    for name in set(config["state_keys"]).intersection(("joint_pos", "joint_vel")):
                        shape = demo[f"obs/{name}"].shape
                        if len(shape) != 2 or shape[1] != len(observation_names):
                            raise ValueError(
                                f"{path}::{key}/obs/{name} must match {len(observation_names)} "
                                f"observation_joint_names; got {shape}"
                            )
                    if "proprio" in config["state_keys"]:
                        if "obs/proprio" in demo:
                            proprio_shape = demo["obs/proprio"].shape
                            if proprio_shape != (length, len(proprio_layout)):
                                raise ValueError(
                                    f"{path}::{key}/obs/proprio must be [T,{len(proprio_layout)}]; "
                                    f"got {proprio_shape}"
                                )
                            if not np.isfinite(demo["obs/proprio"][...]).all():
                                raise ValueError(f"{path}::{key}/obs/proprio contains non-finite values")
                        else:
                            if schema_version == 7:
                                raise KeyError(f"{path}::{key} schema-7 data must contain obs/proprio")
                            joint_shape = demo["obs/joint_pos"].shape
                            eef_shape = demo["obs/eef_pose_body"].shape
                            if len(joint_shape) != 2 or joint_shape[1] != len(observation_names):
                                raise ValueError(
                                    f"{path}::{key}/obs/joint_pos must match {len(observation_names)} "
                                    f"observation_joint_names; got {joint_shape}"
                                )
                            if eef_shape != (length, 2, 7):
                                raise ValueError(
                                    f"{path}::{key}/obs/eef_pose_body must be [T,2,7]; got {eef_shape}"
                                )
                    for name in config["camera_names"]:
                        shape = demo[f"obs/{name}"].shape
                        if len(shape) != 4 or shape[-1] != 3 or demo[f"obs/{name}"].dtype != np.uint8:
                            raise ValueError(f"{path}::{key}/obs/{name} must be uint8 [T,H,W,3]; got {shape}")
                    shapes = tuple(
                        (len(proprio_layout),) if name == "proprio"
                        else (len(action_names),) if name in ("joint_pos", "joint_vel")
                        else demo[f"obs/{name}"].shape[1:]
                        for name in (*config["state_keys"], *config["camera_names"])
                    )
                    if expected_shapes is None:
                        expected_shapes = shapes
                    elif shapes != expected_shapes:
                        raise ValueError(f"{path}::{key} observation shapes differ from other selected episodes")
                    episodes.append(Episode(
                        path, f"data/{key}", length, status, controlled_indices,
                        action_names, proprio_layout,
                    ))
    missing_keys = sorted(key for key, paths in matched_keys.items() if not paths)
    if missing_keys:
        raise ValueError(f"Requested episode_keys were not found: {missing_keys}")
    ambiguous_keys = {key: paths for key, paths in matched_keys.items() if len(paths) > 1}
    if ambiguous_keys:
        details = ", ".join(f"{key} in {paths}" for key, paths in sorted(ambiguous_keys.items()))
        raise ValueError(f"episode_keys must identify one episode across dataset paths; ambiguous: {details}")
    if not episodes:
        raise ValueError(
            f"No non-empty episodes match statuses={config['statuses']}; found statuses={sorted(seen_statuses)}. "
            "Override dataset.statuses to deliberately include other labels."
        )
    return episodes


def split_episodes(episodes, validation_fraction, seed):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    order = np.random.default_rng(seed).permutation(len(episodes))
    count = 0 if validation_fraction == 0 or len(episodes) < 2 else max(1, round(len(episodes) * validation_fraction))
    count = min(count, len(episodes) - 1)
    validation_ids = set(order[:count].tolist())
    train = [episode for index, episode in enumerate(episodes) if index not in validation_ids]
    validation = [episode for index, episode in enumerate(episodes) if index in validation_ids]
    return train, validation


def compose_task_state(joint_pos, eef_pose_body, controlled_joint_indices):
    """Build the canonical 22D task state from a named articulation state."""
    joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    eef_pose_body = np.asarray(eef_pose_body, dtype=np.float32)
    if eef_pose_body.shape != (2, 7):
        raise ValueError(f"eef_pose_body must have shape [2,7], got {eef_pose_body.shape}")
    indices = tuple(controlled_joint_indices)
    if len(indices) != 22 or len(set(indices)) != 22 or max(indices) >= joint_pos.size:
        raise ValueError("controlled_joint_indices must map all 22 actions into joint_pos")
    controlled = joint_pos[list(indices)]
    state = np.concatenate((controlled[:6], eef_pose_body[0], controlled[13:14],
                            eef_pose_body[1], controlled[21:22]))
    if state.shape != (22,) or not np.isfinite(state).all():
        raise ValueError("Task proprioception must contain 22 finite values")
    return state


def _state(demo, state_keys, index, controlled_joint_indices):
    values = []
    for key in state_keys:
        if key == "proprio":
            value = (
                np.asarray(demo["obs/proprio"][index], dtype=np.float32)
                if "obs/proprio" in demo
                else compose_task_state(
                    demo["obs/joint_pos"][index], demo["obs/eef_pose_body"][index], controlled_joint_indices,
                )
            )
        else:
            value = np.asarray(demo[f"obs/{key}"][index], dtype=np.float32).reshape(-1)
        if key in ("joint_pos", "joint_vel"):
            value = value[list(controlled_joint_indices)]
        values.append(value)
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def fit_normalizer(episodes, state_keys, epsilon=1e-6):
    state_sum = state_sq = action_sum = action_sq = None
    count = 0
    for episode in episodes:
        with h5py.File(episode.path, "r") as handle:
            demo = handle[episode.key]
            state_parts = []
            for key in state_keys:
                if key == "proprio":
                    if "obs/proprio" in demo:
                        value = np.asarray(demo["obs/proprio"], dtype=np.float64)
                    else:
                        value = np.stack([
                            compose_task_state(
                                demo["obs/joint_pos"][index], demo["obs/eef_pose_body"][index],
                                episode.controlled_joint_indices,
                            )
                            for index in range(episode.length)
                        ]).astype(np.float64)
                else:
                    value = np.asarray(demo[f"obs/{key}"], dtype=np.float64).reshape(episode.length, -1)
                if key in ("joint_pos", "joint_vel"):
                    value = value[:, list(episode.controlled_joint_indices)]
                state_parts.append(value)
            states = (
                np.concatenate(state_parts, axis=1)
                if state_parts else np.empty((episode.length, 0), dtype=np.float64)
            )
            actions = np.asarray(demo["actions"], dtype=np.float64)
        state_sum = states.sum(0) if state_sum is None else state_sum + states.sum(0)
        state_sq = np.square(states).sum(0) if state_sq is None else state_sq + np.square(states).sum(0)
        action_sum = actions.sum(0) if action_sum is None else action_sum + actions.sum(0)
        action_sq = np.square(actions).sum(0) if action_sq is None else action_sq + np.square(actions).sum(0)
        count += episode.length
    state_mean, action_mean = state_sum / count, action_sum / count
    state_std = np.sqrt(np.maximum(state_sq / count - state_mean**2, epsilon**2))
    action_std = np.sqrt(np.maximum(action_sq / count - action_mean**2, epsilon**2))
    return Normalizer(state_mean, state_std, action_mean, action_std)


class HDF5ACTDataset(Dataset):
    def __init__(self, episodes, config, normalizer):
        self.episodes = list(episodes)
        self.config = config
        self.normalizer = normalizer
        self.samples = [(episode, step) for episode in self.episodes for step in range(episode.length)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, step = self.samples[index]
        with h5py.File(episode.path, "r") as handle:
            demo = handle[episode.key]
            state = _state(demo, self.config["state_keys"], step, episode.controlled_joint_indices)
            end = min(step + self.config["chunk_size"], episode.length)
            actions = np.asarray(demo["actions"][step:end], dtype=np.float32)
            images = {
                name: np.asarray(demo[f"obs/{name}"][step], dtype=np.uint8)
                for name in self.config["camera_names"]
            }
        valid = len(actions)
        padded = np.empty((self.config["chunk_size"], actions.shape[-1]), dtype=np.float32)
        padded[:valid] = actions
        padded[valid:] = actions[-1]  # stable hold target, ignored by the loss
        is_pad = np.arange(self.config["chunk_size"]) >= valid
        image_tensors = {
            name: torch.from_numpy(value.copy()).permute(2, 0, 1).float().div_(255.0)
            for name, value in images.items()
        }
        return {
            "state": torch.from_numpy(self.normalizer.normalize_state(state).astype(np.float32)),
            "images": image_tensors,
            "actions": torch.from_numpy(self.normalizer.normalize_action(padded).astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
        }


def build_datasets(config):
    required = {
        "paths", "episode_keys", "chunk_size", "state_keys", "camera_names",
        "statuses", "validation_fraction", "seed",
    }
    missing = required.difference(config)
    unexpected = set(config).difference(required)
    if missing or unexpected:
        raise ValueError(f"Invalid dataset config; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if config["chunk_size"] <= 0:
        raise ValueError("dataset.chunk_size must be positive")
    if not config["state_keys"] and not config["camera_names"]:
        raise ValueError("At least one state key or camera is required")
    episodes = discover_episodes(config)
    train_episodes, validation_episodes = split_episodes(
        episodes, config["validation_fraction"], config["seed"],
    )
    normalizer = fit_normalizer(train_episodes, config["state_keys"])
    return (
        HDF5ACTDataset(train_episodes, config, normalizer),
        HDF5ACTDataset(validation_episodes, config, normalizer),
        normalizer,
    )


def dataset_manifest(config, train, validation):
    fingerprints = {}
    for path in sorted({episode.path for episode in (*train.episodes, *validation.episodes)}):
        stat = Path(path).stat()
        fingerprints[path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    all_episodes = (*train.episodes, *validation.episodes)
    contract_episode = all_episodes[0]
    return {
        "config": {key: list(value) if isinstance(value, tuple) else value for key, value in config.items()},
        "train_episodes": [episode.identity for episode in train.episodes],
        "validation_episodes": [episode.identity for episode in validation.episodes],
        "controlled_joint_names": list(contract_episode.controlled_joint_names),
        "state_layout": (
            list(contract_episode.proprio_layout)
            if tuple(config["state_keys"]) == ("proprio",) else None
        ),
        "file_fingerprints": fingerprints,
    }
