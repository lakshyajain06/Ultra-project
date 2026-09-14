#!/usr/bin/env python3
"""Evaluate a policy checkpoint in the manager-based Ultra environment."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="Number of complete episodes to evaluate")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to evaluate in parallel")
parser.add_argument("--seed", type=int, default=0, help="Manager environment reset seed")
parser.add_argument(
    "--policy-device",
    help="Torch device for policy inference (default: the simulation --device)",
)
parser.add_argument(
    "--chunk-steps",
    type=int,
    default=1,
    help="Actions to execute from each predicted chunk (1 is receding-horizon control)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    help="Optional episode time limit in policy steps (default: environment's 120 seconds)",
)
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=240)
parser.add_argument(
    "--record-dir",
    type=Path,
    help="Write one synchronized 2x2 MP4 per episode to this directory",
)
parser.add_argument("--video-fps", type=int, default=25, help="Frame rate for evaluation videos")
AppLauncher.add_app_launcher_args(parser)
# Add required positionals after AppLauncher: it probes the partially-built
# parser while installing its own arguments.
parser.add_argument("checkpoint", type=Path, help="Policy checkpoint produced by train_policy.py")
args = parser.parse_args()

if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
if min(args.episodes, args.num_envs, args.chunk_steps) <= 0:
    parser.error("--episodes, --num-envs, and --chunk-steps must be positive")
if args.max_steps is not None and args.max_steps <= 0:
    parser.error("--max-steps must be positive")
if min(args.camera_width, args.camera_height) <= 0:
    parser.error("camera dimensions must be positive")
if args.video_fps <= 0:
    parser.error("--video-fps must be positive")
if args.record_dir is not None:
    existing = [args.record_dir / f"episode_{index:04d}.mp4" for index in range(args.episodes)]
    existing = [path for path in existing if path.exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing evaluation video: {existing[0]}")

# Inspect the checkpoint on CPU before launching Kit because the selected
# manager environment determines whether RTX camera extensions are required.
import torch

from learning.inference import PolicyInference

policy = PolicyInference.from_checkpoint(args.checkpoint, device="cpu")
args.enable_cameras = bool(policy.camera_names or args.record_dir)
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg

from sim import ULTRA_CONTROLLED_JOINT_NAMES
from sim.envs import CAMERA_STREAMS, UltraCubePlateEnvCfg, UltraCubePlateVisionEnvCfg


THIRD_PERSON_EYE = (3.5, 5.0, 2.4)
THIRD_PERSON_TARGET = (0.0, 0.65, 0.80)


def _tensor(value):
    """Return the torch view exposed by Isaac Lab tensor wrappers."""
    return getattr(value, "torch", value)


def _policy_observation(observations, env, joint_ids):
    """Add manager critic terms and canonical joint state for legacy checkpoints."""
    observation = dict(observations["policy"])
    observation.update(observations.get("critic", {}))
    robot = env.scene["robot"]
    observation["joint_pos"] = _tensor(robot.data.joint_pos)[:, joint_ids]
    observation["joint_vel"] = _tensor(robot.data.joint_vel)[:, joint_ids]
    return observation


def _outcome(env, env_index, terminated, truncated):
    """Read the individual manager terms before the next environment step."""
    success = bool(env.termination_manager.get_term("success")[env_index].item())
    dropped = bool(env.termination_manager.get_term("cube_dropped")[env_index].item())
    if success:
        return "success"
    if dropped:
        return "cube_dropped"
    if bool(truncated[env_index].item()):
        return "timeout"
    if bool(terminated[env_index].item()):
        return "terminated"
    raise RuntimeError("Cannot assign an outcome to an unfinished episode")


def _rgb(value, env_index):
    """Convert one environment's HWC RGB output to uint8."""
    value = _tensor(value)
    if hasattr(value, "warp"):
        value = value.warp
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            value = value[env_index]
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[env_index]
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"Camera frame must be HWC RGB or batched HWC RGB, got {value.shape}")
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def _video_frame(observations, env, env_index):
    """Compose third-person, head, left wrist, and right wrist into a 2x2 frame."""
    policy_observation = observations["policy"]
    frames = [
        _rgb(env.scene["camera"].data.output["rgb"], env_index),
        _rgb(policy_observation["head_rgb"], env_index),
        _rgb(policy_observation["left_wrist_rgb"], env_index),
        _rgb(policy_observation["right_wrist_rgb"], env_index),
    ]
    expected = (args.camera_height, args.camera_width, 3)
    if any(frame.shape != expected for frame in frames):
        shapes = [frame.shape for frame in frames]
        raise RuntimeError(f"Evaluation cameras must all have shape {expected}; got {shapes}")
    return np.concatenate(
        (np.concatenate(frames[:2], axis=1), np.concatenate(frames[2:], axis=1)),
        axis=0,
    )


class EpisodeVideoRecorder:
    """Write synchronized camera mosaics without retaining episodes in memory."""

    def __init__(self, directory, fps):
        self.directory = directory
        self.fps = fps
        self.writers = {}
        self.paths = {}

    def start(self, env_index, episode, observations, env):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"episode_{episode:04d}.mp4"
        self.paths[env_index] = path
        self.writers[env_index] = imageio.get_writer(path, fps=self.fps, codec="libx264")
        self.append(env_index, observations, env)

    def append(self, env_index, observations, env):
        self.writers[env_index].append_data(_video_frame(observations, env, env_index))

    def close(self, env_index=None):
        env_indices = list(self.writers) if env_index is None else [env_index]
        for index in env_indices:
            writer = self.writers.pop(index, None)
            if writer is not None:
                writer.close()
                print(f"Recorded {self.paths.pop(index).resolve()}", flush=True)


def main():
    policy_device = args.policy_device or args.device
    policy.device = torch.device(policy_device)
    policy.model.to(policy.device).eval()
    if tuple(policy.controlled_joint_names) != ULTRA_CONTROLLED_JOINT_NAMES:
        raise ValueError("Checkpoint action joint order does not match the manager environment")
    unknown_cameras = set(policy.camera_names).difference(CAMERA_STREAMS)
    if unknown_cameras:
        raise ValueError(f"Checkpoint requests unsupported cameras: {sorted(unknown_cameras)}")
    if args.chunk_steps > policy.model.config.chunk_size:
        raise ValueError(
            f"--chunk-steps={args.chunk_steps} exceeds checkpoint chunk size "
            f"{policy.model.config.chunk_size}"
        )

    recording_cameras = CAMERA_STREAMS if args.record_dir else policy.camera_names
    if recording_cameras:
        cfg = UltraCubePlateVisionEnvCfg(
            enabled_cameras=recording_cameras,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
    else:
        cfg = UltraCubePlateEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.record_dir:
        cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/CaptureCamera",
            update_period=0.0,
            height=args.camera_height,
            width=args.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=4.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
    if args.max_steps is not None:
        cfg.episode_length_s = args.max_steps * cfg.decimation * cfg.sim.dt

    env = ManagerBasedRLEnv(cfg=cfg)
    results = []
    recorder = EpisodeVideoRecorder(args.record_dir, args.video_fps) if args.record_dir else None
    try:
        if recorder:
            origins = _tensor(env.scene.env_origins).detach().cpu().numpy()
            env.scene["camera"].set_world_poses_from_view(
                origins + np.asarray(THIRD_PERSON_EYE, dtype=np.float32),
                origins + np.asarray(THIRD_PERSON_TARGET, dtype=np.float32),
            )
        observations, _ = env.reset(seed=args.seed)
        robot = env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True)
        if tuple(joint_names) != ULTRA_CONTROLLED_JOINT_NAMES:
            raise RuntimeError(f"Environment action joint order mismatch: {joint_names}")

        episode_returns = np.zeros(args.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
        episode_ids = np.full(args.num_envs, -1, dtype=np.int64)
        initial_count = min(args.num_envs, args.episodes)
        episode_ids[:initial_count] = np.arange(initial_count)
        next_episode_id = initial_count
        if recorder:
            for env_index in range(initial_count):
                recorder.start(env_index, int(episode_ids[env_index]), observations, env)
        while len(results) < args.episodes and app.is_running():
            chunks = policy.predict(_policy_observation(observations, env, joint_ids))
            if chunks.shape[0] != args.num_envs or not np.isfinite(chunks).all():
                raise RuntimeError("Policy emitted a non-finite action")
            for chunk_index in range(args.chunk_steps):
                action_tensor = torch.as_tensor(
                    chunks[:, chunk_index], dtype=torch.float32, device=env.device,
                )
                observations, reward, terminated, truncated, _ = env.step(action_tensor)
                episode_returns += reward.detach().cpu().numpy()
                episode_lengths += 1
                done = (terminated | truncated).detach().cpu().numpy()

                if recorder:
                    for env_index in np.flatnonzero((episode_ids >= 0) & ~done):
                        recorder.append(int(env_index), observations, env)

                for env_index in np.flatnonzero(done):
                    env_index = int(env_index)
                    episode_id = int(episode_ids[env_index])
                    if episode_id >= 0:
                        if recorder:
                            # Returned observations already belong to the
                            # manager's automatically reset next episode.
                            recorder.close(env_index)
                        result = {
                            "episode": episode_id,
                            "environment": env_index,
                            "return": float(episode_returns[env_index]),
                            "length": int(episode_lengths[env_index]),
                            "outcome": _outcome(env, env_index, terminated, truncated),
                        }
                        results.append(result)
                        print(json.dumps(result), flush=True)
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

                    if next_episode_id < args.episodes:
                        episode_ids[env_index] = next_episode_id
                        if recorder:
                            recorder.start(env_index, next_episode_id, observations, env)
                        next_episode_id += 1
                    else:
                        episode_ids[env_index] = -1

                if np.any(done):
                    # At least one environment auto-reset. Re-plan the entire
                    # batch so no reset environment executes a stale chunk.
                    break

        if len(results) != args.episodes:
            raise RuntimeError(f"Simulation stopped after {len(results)}/{args.episodes} complete episodes")
        results.sort(key=lambda result: result["episode"])
        counts = {name: sum(result["outcome"] == name for result in results) for name in (
            "success", "cube_dropped", "timeout", "terminated"
        )}
        summary = {
            "episodes": len(results),
            "successes": counts["success"],
            "success_rate": counts["success"] / len(results),
            "cube_dropped": counts["cube_dropped"],
            "timeouts": counts["timeout"],
            "other_terminations": counts["terminated"],
            "mean_return": float(np.mean([result["return"] for result in results])),
            "mean_length": float(np.mean([result["length"] for result in results])),
            "checkpoint": str(args.checkpoint.resolve()),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "chunk_steps": args.chunk_steps,
            "record_dir": str(args.record_dir.resolve()) if args.record_dir else None,
        }
        print("EVALUATION " + json.dumps(summary), flush=True)
    finally:
        if recorder:
            recorder.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
