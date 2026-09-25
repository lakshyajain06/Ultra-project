#!/usr/bin/env python3
"""Evaluate a dual-Ultra policy using one declarative YAML configuration."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from learning.evaluation_config import compose_bimanual_evaluation_config
from learning.inference import PolicyInference, ReplayInference


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config-name", default="bimanual_eval", help="Hydra config name")
parser.add_argument("overrides", nargs="*", help="Hydra overrides such as rollout.episodes=5")
args = parser.parse_args()
config = compose_bimanual_evaluation_config(args.overrides, args.config_name)

policy_config = config["policy"]
if policy_config["type"] == "checkpoint":
    print(f"Loading checkpoint: {policy_config['checkpoint']}", flush=True)
    policy = PolicyInference.from_checkpoint(policy_config["checkpoint"], device="cpu")
else:
    print(
        f"Loading replay: {policy_config['dataset']}::{policy_config['episode']}",
        flush=True,
    )
    policy = ReplayInference(policy_config["dataset"], policy_config["episode"])
args.device = config["simulation"]["device"]
args.enable_cameras = bool(policy.camera_names or config["recording"]["enabled"])
args.viz = "kit" if config["simulation"]["visualization"] else "none"
launcher = AppLauncher(args)
app = launcher.app

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from data import DUAL_ULTRA_ACTION_JOINT_NAMES
from learning.inference import TemporalEnsembler
from sim.envs import DualUltraEnvCfg, FacingDualUltraEnvCfg
from sim.envs.dual_ultra import CONTROLLED_ARMS, DUAL_CAMERA_PATHS
from sim.envs.facing_dual_ultra import FACING_EVALUATION_CAMERA_STREAMS


ENVIRONMENTS = {
    "facing": (FacingDualUltraEnvCfg, FACING_EVALUATION_CAMERA_STREAMS),
    "side_by_side": (DualUltraEnvCfg, tuple(DUAL_CAMERA_PATHS)),
}


def _tensor(value):
    return getattr(value, "torch", value)


def _rgb(value, env_index):
    value = _tensor(value)
    if hasattr(value, "warp"):
        value = value.warp
    if isinstance(value, torch.Tensor):
        value = value[env_index].detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[env_index]
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"Expected an HWC RGB frame, got {value.shape}")
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def _mosaic(observations, camera_names, env_index):
    frames = {name: _rgb(observations["policy"][name], env_index) for name in camera_names}
    height, width = next(iter(frames.values())).shape[:2]
    if tuple(camera_names) == FACING_EVALUATION_CAMERA_STREAMS:
        image = np.zeros((2 * height, 4 * width, 3), dtype=np.uint8)
        placements = {
            "task_rgb": (0, width + width // 2),
            "robot_left_head_rgb": (0, width // 2),
            "robot_left_left_wrist_rgb": (height, 0),
            "robot_left_right_wrist_rgb": (height, width),
            "robot_right_head_rgb": (0, 2 * width + width // 2),
            "robot_right_left_wrist_rgb": (height, 2 * width),
            "robot_right_right_wrist_rgb": (height, 3 * width),
        }
    elif len(frames) == 3:
        image = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
        placements = {
            camera_names[0]: (0, width // 2),
            camera_names[1]: (height, 0),
            camera_names[2]: (height, width),
        }
    else:
        image = np.zeros((height, len(frames) * width, 3), dtype=np.uint8)
        placements = {name: (0, index * width) for index, name in enumerate(camera_names)}
    for name, (top, left) in placements.items():
        image[top : top + height, left : left + width] = frames[name]
    return image


class EpisodeRecorder:
    def __init__(self, directory, fps, camera_names):
        self.directory = directory
        self.fps = fps
        self.camera_names = camera_names
        self.writers = {}
        self.paths = {}

    def start(self, env_index, episode_id, observations):
        path = self.directory / f"episode_{episode_id:04d}.mp4"
        self.paths[env_index] = path
        self.writers[env_index] = imageio.get_writer(path, fps=self.fps, codec="libx264")
        self.append(env_index, observations)

    def append(self, env_index, observations):
        self.writers[env_index].append_data(_mosaic(observations, self.camera_names, env_index))

    def close(self, env_index=None):
        indices = list(self.writers) if env_index is None else [env_index]
        for index in indices:
            writer = self.writers.pop(index, None)
            if writer is not None:
                writer.close()
                print(f"Recorded {self.paths.pop(index).resolve()}", flush=True)


def _output_directory():
    configured = config["recording"]["directory"]
    if configured:
        directory = Path(configured)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Recording directory is not empty: {directory}")
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
        directory = Path("outputs/evaluation/bimanual") / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _resolved_action_names(env):
    names = []
    for robot_name, arm in CONTROLLED_ARMS:
        robot = env.scene[robot_name]
        _, joint_names = robot.find_joints(
            [*(f"{arm}_j{i}" for i in range(1, 8)), f"{arm}_gripper_joint"],
            preserve_order=True,
        )
        names.extend(f"{robot_name}/{joint_name}" for joint_name in joint_names)
    return tuple(names)


def _default_action(env):
    values = []
    for robot_name, arm in CONTROLLED_ARMS:
        robot = env.scene[robot_name]
        joint_ids, _ = robot.find_joints(
            [*(f"{arm}_j{i}" for i in range(1, 8)), f"{arm}_gripper_joint"],
            preserve_order=True,
        )
        values.append(_tensor(robot.data.default_joint_pos)[:, joint_ids])
    return torch.cat(values, dim=-1)


def _outcome(env, env_index, terminated, truncated):
    if bool(env.termination_manager.get_term("success")[env_index].item()):
        return "success"
    if bool(env.termination_manager.get_term("cube_dropped")[env_index].item()):
        return "cube_dropped"
    if bool(truncated[env_index].item()):
        return "timeout"
    if bool(terminated[env_index].item()):
        return "terminated"
    raise RuntimeError("Cannot classify an unfinished episode")


def main():
    environment_name = config["environment"]
    simulation = config["simulation"]
    rollout = config["rollout"]
    inference = config["inference"]
    cfg_type, all_cameras = ENVIRONMENTS[environment_name]
    if tuple(policy.controlled_joint_names) != DUAL_ULTRA_ACTION_JOINT_NAMES:
        raise ValueError("Policy is not a canonical 16-action dual-Ultra policy")
    unknown_cameras = set(policy.camera_names).difference(all_cameras)
    if unknown_cameras:
        raise ValueError(
            f"Policy cameras are unavailable in {environment_name}: {sorted(unknown_cameras)}"
        )
    chunk_size = policy.model.config["chunk_size"]
    method = inference["method"]
    execution_steps = (inference["steps"] or chunk_size) if method == "chunk" else 1
    if execution_steps > chunk_size:
        raise ValueError(f"Inference steps {execution_steps} exceed model chunk size {chunk_size}")

    recording_directory = _output_directory() if config["recording"]["enabled"] else None
    enabled_cameras = all_cameras if recording_directory else policy.camera_names
    cfg = cfg_type(
        enabled_cameras=enabled_cameras,
        camera_width=simulation["camera_width"],
        camera_height=simulation["camera_height"],
    )
    cfg.scene.num_envs = simulation["num_envs"]
    cfg.scene.env_spacing = simulation["env_spacing"]
    cfg.sim.device = simulation["device"]
    cfg.episode_length_s = rollout["max_steps"] * cfg.decimation * cfg.sim.dt

    policy.device = torch.device(config["policy_device"])
    policy.model.to(policy.device).eval()
    ensembler = (
        TemporalEnsembler(simulation["num_envs"], chunk_size, inference["decay"])
        if method == "temporal_ensemble"
        else None
    )
    recorder = (
        EpisodeRecorder(recording_directory, config["recording"]["fps"], all_cameras)
        if recording_directory
        else None
    )
    if recording_directory:
        (recording_directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    env = ManagerBasedRLEnv(cfg=cfg)
    results = []
    try:
        observations, _ = env.reset(seed=rollout["seed"])
        env.step(_default_action(env))
        observations, _ = env.reset(seed=rollout["seed"])
        if _resolved_action_names(env) != tuple(policy.controlled_joint_names):
            raise RuntimeError("Manager environment action order does not match the policy")

        episode_ids = np.full(simulation["num_envs"], -1, dtype=np.int64)
        initial_count = min(simulation["num_envs"], rollout["episodes"])
        episode_ids[:initial_count] = np.arange(initial_count)
        next_episode_id = initial_count
        returns = np.zeros(simulation["num_envs"], dtype=np.float64)
        lengths = np.zeros(simulation["num_envs"], dtype=np.int64)
        if recorder:
            for env_index in range(initial_count):
                recorder.start(env_index, int(episode_ids[env_index]), observations)

        print(
            f"Evaluating {rollout['episodes']} episode(s) in {environment_name} "
            f"with {simulation['num_envs']} environment(s), inference={method}",
            flush=True,
        )
        while len(results) < rollout["episodes"] and app.is_running():
            chunks = policy.predict(observations["policy"])
            if chunks.shape != (simulation["num_envs"], chunk_size, 16):
                raise RuntimeError(f"Policy produced unexpected action chunks: {chunks.shape}")
            if not np.isfinite(chunks).all():
                raise RuntimeError("Policy produced non-finite actions")
            if ensembler:
                chunks = ensembler.add_and_aggregate(chunks)[:, None, :]

            for chunk_index in range(execution_steps):
                actions = torch.as_tensor(
                    chunks[:, chunk_index], device=env.device, dtype=torch.float32
                )
                observations, reward, terminated, truncated, _ = env.step(actions)
                returns += reward.detach().cpu().numpy()
                lengths += 1
                done = (terminated | truncated).detach().cpu().numpy()
                if recorder:
                    for env_index in np.flatnonzero((episode_ids >= 0) & ~done):
                        recorder.append(int(env_index), observations)

                for env_index in np.flatnonzero(done):
                    env_index = int(env_index)
                    if hasattr(policy, "reset_env"):
                        policy.reset_env(env_index)
                    if ensembler:
                        ensembler.reset_env(env_index)
                    episode_id = int(episode_ids[env_index])
                    if episode_id >= 0:
                        if recorder:
                            recorder.close(env_index)
                        result = {
                            "episode": episode_id,
                            "environment": env_index,
                            "outcome": _outcome(env, env_index, terminated, truncated),
                            "return": float(returns[env_index]),
                            "length": int(lengths[env_index]),
                        }
                        results.append(result)
                        print(json.dumps(result), flush=True)
                    returns[env_index] = 0.0
                    lengths[env_index] = 0
                    if next_episode_id < rollout["episodes"]:
                        episode_ids[env_index] = next_episode_id
                        if recorder:
                            recorder.start(env_index, next_episode_id, observations)
                        next_episode_id += 1
                    else:
                        episode_ids[env_index] = -1
                if np.any(done):
                    break

        if len(results) != rollout["episodes"]:
            raise RuntimeError(f"Simulation stopped after {len(results)}/{rollout['episodes']} episodes")
        results.sort(key=lambda item: item["episode"])
        successes = sum(item["outcome"] == "success" for item in results)
        summary = {
            "episodes": len(results),
            "successes": successes,
            "success_rate": successes / len(results),
            "mean_return": float(np.mean([item["return"] for item in results])),
            "mean_length": float(np.mean([item["length"] for item in results])),
            "environment": environment_name,
            "policy": {
                **policy_config,
                "checkpoint": (
                    str(Path(policy_config["checkpoint"]).resolve())
                    if policy_config["type"] == "checkpoint" else None
                ),
                "dataset": (
                    str(Path(policy_config["dataset"]).resolve())
                    if policy_config["type"] == "replay" else None
                ),
            },
            "recording_directory": str(recording_directory.resolve()) if recording_directory else None,
        }
        print("EVALUATION " + json.dumps(summary), flush=True)
    finally:
        if recorder:
            recorder.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
