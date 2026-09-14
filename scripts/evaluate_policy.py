#!/usr/bin/env python3
"""Evaluate a policy checkpoint in the manager-based Ultra environment."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="Number of complete episodes to evaluate")
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
if args.episodes <= 0 or args.chunk_steps <= 0:
    parser.error("--episodes and --chunk-steps must be positive")
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


def _outcome(env, terminated, truncated):
    """Read the individual manager terms before the next environment step."""
    success = bool(env.termination_manager.get_term("success")[0].item())
    dropped = bool(env.termination_manager.get_term("cube_dropped")[0].item())
    if success:
        return "success"
    if dropped:
        return "cube_dropped"
    if bool(truncated[0].item()):
        return "timeout"
    if bool(terminated[0].item()):
        return "terminated"
    raise RuntimeError("Cannot assign an outcome to an unfinished episode")


def _rgb(value):
    """Convert environment zero's HWC RGB output to uint8."""
    value = _tensor(value)
    if hasattr(value, "warp"):
        value = value.warp
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"Camera frame must be HWC RGB or batched HWC RGB, got {value.shape}")
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def _video_frame(observations, env):
    """Compose third-person, head, left wrist, and right wrist into a 2x2 frame."""
    policy_observation = observations["policy"]
    frames = [
        _rgb(env.scene["camera"].data.output["rgb"]),
        _rgb(policy_observation["head_rgb"]),
        _rgb(policy_observation["left_wrist_rgb"]),
        _rgb(policy_observation["right_wrist_rgb"]),
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
        self.writer = None
        self.path = None

    def start(self, episode, observations, env):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"episode_{episode:04d}.mp4"
        self.writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264")
        self.append(observations, env)

    def append(self, observations, env):
        self.writer.append_data(_video_frame(observations, env))

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"Recorded {self.path.resolve()}", flush=True)
            self.writer = None


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
    cfg.scene.num_envs = 1
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

        episode_return = 0.0
        episode_length = 0
        if recorder:
            recorder.start(0, observations, env)
        while len(results) < args.episodes and app.is_running():
            chunk = policy.predict(_policy_observation(observations, env, joint_ids))
            if not np.isfinite(chunk).all():
                raise RuntimeError("Policy emitted a non-finite action")
            for action in chunk[: args.chunk_steps]:
                action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
                observations, reward, terminated, truncated, _ = env.step(action_tensor)
                episode_return += float(reward[0].item())
                episode_length += 1

                if bool((terminated | truncated)[0].item()):
                    if recorder:
                        # Returned observations already belong to the manager's
                        # automatically reset next episode.
                        recorder.close()
                    result = {
                        "episode": len(results),
                        "return": episode_return,
                        "length": episode_length,
                        "outcome": _outcome(env, terminated, truncated),
                    }
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    episode_return = 0.0
                    episode_length = 0
                    if recorder and len(results) < args.episodes:
                        recorder.start(len(results), observations, env)
                    # ManagerBasedRLEnv has already reset this environment.
                    # Discard the remaining old-state actions and re-plan.
                    break

        if len(results) != args.episodes:
            raise RuntimeError(f"Simulation stopped after {len(results)}/{args.episodes} complete episodes")
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
