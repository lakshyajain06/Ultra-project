#!/usr/bin/env python3
"""Evaluate a policy checkpoint in the manager-based Ultra environment."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="Number of complete episodes to evaluate")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to evaluate in parallel")
parser.add_argument("--seed", type=int, default=0, help="Manager environment reset seed")
parser.add_argument(
    "--randomize-cube-position", "--randomize_cube_position", action="store_true",
    help="Sample the cube's XY position on every reset, matching randomized teleop demonstrations",
)
parser.add_argument(
    "--cube-position-range", "--cube_position_range", nargs=2, type=float, metavar=("X", "Y"),
    help="Independent cube XY half-ranges in metres (default: 0.08 0.08)",
)
parser.add_argument(
    "--policy-device",
    help="Torch device for policy inference (default: the simulation --device)",
)
parser.add_argument(
    "--chunk-steps",
    type=int,
    help="Override inference.chunk.steps; values above 1 select chunk mode for legacy commands",
)
parser.add_argument(
    "--inference", choices=("receding", "chunk", "temporal_ensemble"),
    help="Hydra inference group (default: checkpoint setting, then training/conf/config.yaml)",
)
parser.add_argument(
    "--temporal-decay", type=float,
    help="Override inference.temporal_ensemble.decay (larger values favor newer chunks)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=200,
    help="Episode time limit in policy steps (default: 200, or 8 seconds at 25 Hz)",
)
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=240)
parser.add_argument(
    "--env-spacing",
    type=float,
    default=25.0,
    help="Distance between parallel environments in metres (default: 25, to isolate camera views)",
)
parser.add_argument(
    "--record-dir",
    type=Path,
    help="Override the timestamped MP4 recording directory",
)
parser.add_argument("--no-record", action="store_true", help="Disable evaluation video recording")
parser.add_argument("--video-fps", type=int, default=25, help="Frame rate for evaluation videos")
parser.add_argument(
    "--replay-dataset", type=Path,
    help="Use recorded episode actions as the policy, keeping this evaluator and checkpoint's environment setup",
)
parser.add_argument("--replay-episode", default="0", help="Episode number or name in --replay-dataset")
AppLauncher.add_app_launcher_args(parser)
# Add required positionals after AppLauncher: it probes the partially-built
# parser while installing its own arguments.
parser.add_argument("checkpoint", type=Path, help="Policy checkpoint produced by train_policy.py")
args = parser.parse_args()

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir = Path(__file__).resolve().parents[1] / "src/learning/training/conf"
selection = args.inference
if selection is None and args.chunk_steps is not None:
    selection = "chunk" if args.chunk_steps > 1 else "receding"
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    config = compose(config_name="config", overrides=[f"inference={selection}"] if selection else [])
hydra_inference = OmegaConf.to_container(config.inference, resolve=True)

if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
if args.replay_dataset is not None and not args.replay_dataset.is_file():
    parser.error(f"Replay dataset does not exist: {args.replay_dataset}")
if args.cube_position_range is not None and not args.randomize_cube_position:
    parser.error("--cube-position-range requires --randomize-cube-position")
if args.cube_position_range is None:
    args.cube_position_range = (0.08, 0.08)
if any(not math.isfinite(value) or value < 0 for value in args.cube_position_range):
    parser.error("--cube-position-range values must be finite and nonnegative")
if min(args.episodes, args.num_envs) <= 0 or (args.chunk_steps is not None and args.chunk_steps <= 0):
    parser.error("--episodes, --num-envs, and --chunk-steps must be positive")
if args.max_steps <= 0:
    parser.error("--max-steps must be positive")
if min(args.camera_width, args.camera_height, args.env_spacing) <= 0:
    parser.error("camera dimensions and --env-spacing must be positive")
if args.video_fps <= 0:
    parser.error("--video-fps must be positive")
if args.no_record and args.record_dir is not None:
    parser.error("--no-record cannot be combined with --record-dir")
if args.record_dir is None:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    run_dir = Path("outputs/evaluation") / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = Path("outputs/evaluation") / f"{timestamp}_{suffix:02d}"
        suffix += 1
else:
    run_dir = args.record_dir
if args.record_dir is not None:
    existing = [args.record_dir / f"episode_{index:04d}.mp4" for index in range(args.episodes)]
    existing.append(args.record_dir / "isaac.log")
    existing = [path for path in existing if path.exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing evaluation output: {existing[0]}")
if not args.no_record:
    args.record_dir = run_dir
run_dir.mkdir(parents=True, exist_ok=True)
isaac_log = run_dir / "isaac.log"


@contextmanager
def capture_isaac_output():
    """Redirect process-level stdout/stderr so native Kit logs are captured too."""
    if args.verbose or args.info:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(isaac_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    stdout_fd = os.dup(sys.stdout.fileno())
    stderr_fd = os.dup(sys.stderr.fileno())
    try:
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(log_fd)

# Inspect the checkpoint on CPU before launching Kit because the selected
# manager environment determines whether RTX camera extensions are required.
import torch

from learning.inference import PolicyInference, TemporalEnsembler

print(f"Loading checkpoint: {args.checkpoint}", flush=True)
policy = PolicyInference.from_checkpoint(args.checkpoint, device="cpu")
inference_settings = (
    policy.inference_config
    if selection is None and policy.inference_config is not None
    else hydra_inference
)
inference_method = inference_settings["method"]
if inference_method not in ("receding", "chunk", "temporal_ensemble"):
    parser.error(f"Unknown checkpoint inference method: {inference_method!r}")
if inference_method == "chunk":
    args.chunk_steps = args.chunk_steps if args.chunk_steps is not None else int(inference_settings["steps"])
    if args.chunk_steps <= 0:
        parser.error("inference.chunk.steps must be positive")
else:
    if args.chunk_steps not in (None, 1):
        parser.error("--chunk-steps above 1 requires --inference chunk")
    args.chunk_steps = 1
if inference_method == "temporal_ensemble":
    temporal_decay = (
        args.temporal_decay if args.temporal_decay is not None else float(inference_settings["decay"])
    )
    if not math.isfinite(temporal_decay) or temporal_decay < 0:
        parser.error("--temporal-decay must be finite and nonnegative")
elif args.temporal_decay is not None:
    parser.error("--temporal-decay requires --inference temporal_ensemble")
else:
    temporal_decay = None
if args.replay_dataset is not None:
    import h5py
    import numpy as np

    replay_episode = (
        f"demo_{int(args.replay_episode):06d}" if args.replay_episode.isdigit() else args.replay_episode
    )
    with h5py.File(args.replay_dataset, "r") as dataset:
        if f"data/{replay_episode}" not in dataset:
            parser.error(f"Episode {replay_episode!r} is not in {args.replay_dataset}")
        replay_joint_names = tuple(json.loads(dataset.attrs.get("metadata", "{}"))["action_joint_names"])
        replay_actions = np.asarray(dataset[f"data/{replay_episode}/actions"], dtype=np.float32)
        replay_initial_images = {
            name: np.asarray(dataset[f"data/{replay_episode}/obs/{name}"][0], dtype=np.uint8)
            for name in policy.camera_names
        }
    if replay_actions.ndim != 2 or replay_actions.shape[1] != len(policy.controlled_joint_names):
        parser.error(f"Recorded actions must be [steps, {len(policy.controlled_joint_names)}]")
    if len(replay_actions) == 0 or not np.isfinite(replay_actions).all():
        parser.error("Recorded actions must be nonempty and finite")
    print(
        f"Using {replay_episode} as a recorded-action policy ({len(replay_actions)} actions); "
        "observations will be ignored.", flush=True,
    )
args.enable_cameras = bool(policy.camera_names or args.record_dir)
if args.verbose or args.info:
    print("Isaac logs: console", flush=True)
else:
    print(f"Isaac log: {isaac_log.resolve()}", flush=True)
print("Starting Isaac Sim...", flush=True)
with capture_isaac_output():
    launcher = AppLauncher(args)
app = launcher.app
print("Isaac Sim started.", flush=True)

with capture_isaac_output():
    import numpy as np
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import CameraCfg

    from sim import ULTRA_CONTROLLED_JOINT_NAMES
    from sim.envs import CAMERA_STREAMS, UltraCubePlateEnvCfg


THIRD_PERSON_EYE = (3.5, 5.0, 2.4)
THIRD_PERSON_TARGET = (0.0, 0.65, 0.80)


class RecordedActionPolicy:
    """Drop-in predict() replacement using an episode's absolute joint targets."""

    def __init__(self, checkpoint_policy, actions, num_envs):
        self.model = checkpoint_policy.model
        self.device = checkpoint_policy.device
        self.camera_names = checkpoint_policy.camera_names
        self.controlled_joint_names = checkpoint_policy.controlled_joint_names
        self.actions = actions
        self.positions = np.zeros(num_envs, dtype=np.int64)

    def predict(self, _observation):
        offsets = np.arange(self.model.config["chunk_size"], dtype=np.int64)
        indices = np.minimum(self.positions[:, None] + offsets[None, :], len(self.actions) - 1)
        return self.actions[indices]

    def advance(self):
        self.positions += 1

    def reset_env(self, env_index):
        self.positions[env_index] = 0


def _tensor(value):
    """Return the torch view exposed by Isaac Lab tensor wrappers."""
    return getattr(value, "torch", value)


def _policy_observation(observations):
    """Return the policy-facing manager observations."""
    return observations["policy"]


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
    """Write camera mosaics without retaining episodes in memory."""

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
    global policy
    if args.replay_dataset is not None:
        if replay_joint_names != tuple(policy.controlled_joint_names):
            raise ValueError("Recorded action joint order does not match the checkpoint")
        policy = RecordedActionPolicy(policy, replay_actions, args.num_envs)
    policy_device = args.policy_device or args.device
    policy.device = torch.device(policy_device)
    policy.model.to(policy.device).eval()
    if tuple(policy.controlled_joint_names) != ULTRA_CONTROLLED_JOINT_NAMES:
        raise ValueError("Checkpoint action joint order does not match the manager environment")
    unknown_cameras = set(policy.camera_names).difference(CAMERA_STREAMS)
    if unknown_cameras:
        raise ValueError(f"Checkpoint requests unsupported cameras: {sorted(unknown_cameras)}")
    if args.chunk_steps > policy.model.config["chunk_size"]:
        raise ValueError(
            f"--chunk-steps={args.chunk_steps} exceeds checkpoint chunk size "
            f"{policy.model.config['chunk_size']}"
        )
    ensembler = (
        TemporalEnsembler(args.num_envs, policy.model.config["chunk_size"], temporal_decay)
        if inference_method == "temporal_ensemble" else None
    )

    recording_cameras = CAMERA_STREAMS if args.record_dir else policy.camera_names
    cfg = UltraCubePlateEnvCfg(
        enabled_cameras=recording_cameras,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        randomize_cube_position=args.randomize_cube_position,
        cube_position_range_xy=tuple(args.cube_position_range),
    )
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
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
                clipping_range=(0.1, 10.0),
            ),
        )
    cfg.episode_length_s = args.max_steps * cfg.decimation * cfg.sim.dt

    env = None
    results = []
    recorder = EpisodeVideoRecorder(args.record_dir, args.video_fps) if args.record_dir else None
    try:
        camera_count = args.num_envs * (4 if recorder else len(policy.camera_names))
        print(
            f"Creating {args.num_envs} manager environment(s) with {camera_count} camera stream(s)...",
            flush=True,
        )
        with capture_isaac_output():
            env = ManagerBasedRLEnv(cfg=cfg)
            if recorder:
                origins = _tensor(env.scene.env_origins).detach().cpu().numpy()
                env.scene["camera"].set_world_poses_from_view(
                    origins + np.asarray(THIRD_PERSON_EYE, dtype=np.float32),
                    origins + np.asarray(THIRD_PERSON_TARGET, dtype=np.float32),
                )
            observations, _ = env.reset(seed=args.seed)
            if recording_cameras:
                # On a cold Isaac Sim startup, the first camera observation can
                # show the asset's authored pose rather than the reset joint
                # state. Advance physics once, then reset before episode 0.
                warmup_robot = env.scene["robot"]
                warmup_ids, _ = warmup_robot.find_joints(ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True)
                warmup_action = _tensor(warmup_robot.data.default_joint_pos)[:, warmup_ids].clone()
                env.step(warmup_action)
                observations, _ = env.reset(seed=args.seed)
        print(
            f"Environment ready. Running evaluation with {inference_method} inference"
            + (f" (decay={temporal_decay:g})" if ensembler else f" ({args.chunk_steps} action(s) per chunk)")
            + "...", flush=True,
        )
        if args.replay_dataset is not None and policy.camera_names:
            image_mae = {
                name: float(np.mean(np.abs(
                    _rgb(observations["policy"][name], 0).astype(np.float32)
                    - replay_initial_images[name].astype(np.float32)
                )))
                for name in policy.camera_names
            }
            print("RESET_IMAGE_MAE " + json.dumps(image_mae), flush=True)
        robot = env.scene["robot"]
        _, joint_names = robot.find_joints(ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True)
        if tuple(joint_names) != ULTRA_CONTROLLED_JOINT_NAMES:
            raise RuntimeError(f"Environment action joint order mismatch: {joint_names}")

        episode_returns = np.zeros(args.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
        episode_ids = np.full(args.num_envs, -1, dtype=np.int64)
        cube_start_xy = (
            _tensor(env.scene["cube"].data.root_pos_w)[:, :2]
            - _tensor(env.scene.env_origins)[:, :2]
        ).detach().cpu().numpy().copy()
        initial_count = min(args.num_envs, args.episodes)
        episode_ids[:initial_count] = np.arange(initial_count)
        next_episode_id = initial_count
        if recorder:
            for env_index in range(initial_count):
                recorder.start(env_index, int(episode_ids[env_index]), observations, env)
        while len(results) < args.episodes and app.is_running():
            chunks = policy.predict(_policy_observation(observations))
            if chunks.shape[0] != args.num_envs or not np.isfinite(chunks).all():
                raise RuntimeError("Policy emitted a non-finite action")
            if ensembler is not None:
                chunks = ensembler.add_and_aggregate(chunks)[:, None, :]
            for chunk_index in range(args.chunk_steps):
                action_tensor = torch.as_tensor(
                    chunks[:, chunk_index], dtype=torch.float32, device=env.device,
                )
                observations, reward, terminated, truncated, _ = env.step(action_tensor)
                if isinstance(policy, RecordedActionPolicy):
                    policy.advance()
                episode_returns += reward.detach().cpu().numpy()
                episode_lengths += 1
                done = (terminated | truncated).detach().cpu().numpy()

                if recorder:
                    for env_index in np.flatnonzero((episode_ids >= 0) & ~done):
                        recorder.append(int(env_index), observations, env)

                for env_index in np.flatnonzero(done):
                    env_index = int(env_index)
                    if isinstance(policy, RecordedActionPolicy):
                        policy.reset_env(env_index)
                    if ensembler is not None:
                        ensembler.reset_env(env_index)
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
                        if args.randomize_cube_position:
                            result["cube_start_xy"] = cube_start_xy[env_index].tolist()
                        results.append(result)
                        print(json.dumps(result), flush=True)
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

                    if next_episode_id < args.episodes:
                        episode_ids[env_index] = next_episode_id
                        cube_start_xy[env_index] = (
                            _tensor(env.scene["cube"].data.root_pos_w)[env_index, :2]
                            - _tensor(env.scene.env_origins)[env_index, :2]
                        ).detach().cpu().numpy()
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
            "replay_dataset": str(args.replay_dataset.resolve()) if args.replay_dataset else None,
            "replay_episode": replay_episode if args.replay_dataset else None,
            "seed": args.seed,
            "randomize_cube_position": args.randomize_cube_position,
            "cube_position_range_xy": list(args.cube_position_range) if args.randomize_cube_position else None,
            "num_envs": args.num_envs,
            "chunk_steps": args.chunk_steps,
            "inference_method": inference_method,
            "temporal_decay": temporal_decay,
            "record_dir": str(args.record_dir.resolve()) if args.record_dir else None,
        }
        print("EVALUATION " + json.dumps(summary), flush=True)
    finally:
        if recorder:
            recorder.close()
        if env is not None:
            with capture_isaac_output():
                env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        with capture_isaac_output():
            app.close()
