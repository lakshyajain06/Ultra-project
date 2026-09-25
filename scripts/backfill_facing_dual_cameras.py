"""Backfill facing-dual wrist cameras by replaying episodes in parallel."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import h5py
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Existing schema-7 facing-dual dataset")
parser.add_argument("output", type=Path, help="New upgraded dataset; must not already exist")
parser.add_argument(
    "--num-envs",
    type=int,
    default=32,
    help="Number of episodes replayed simultaneously (default: 32)",
)
parser.add_argument(
    "--write-block-size",
    type=int,
    default=16,
    help="Rendered timesteps buffered between HDF5 writes (default: 16)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args.write_block_size <= 0:
    parser.error("--write-block-size must be positive")
if not args.source.is_file():
    parser.error(f"Source dataset does not exist: {args.source}")
if args.output.exists():
    parser.error(f"Output already exists: {args.output}")
partial = args.output.with_name(args.output.name + ".partial")
if partial.exists():
    parser.error(f"Partial output already exists: {partial}")

with h5py.File(args.source, "r") as source:
    if int(source.attrs.get("schema_version", -1)) != 7:
        parser.error("Camera backfill requires schema_version=7")
    metadata = json.loads(source.attrs.get("metadata", "{}"))
    if metadata.get("layout") != "facing":
        parser.error("Camera backfill requires a facing-layout dual-Ultra dataset")
    episode_names = tuple(source.get("data", ()))
    if not episode_names:
        parser.error("Source dataset has no episodes")
    if len(episode_names) > args.num_envs:
        parser.error(
            f"Dataset has {len(episode_names)} episodes but --num-envs is {args.num_envs}; "
            "use at least one environment per episode"
        )
    width = int(metadata.get("camera_width", 320))
    height = int(metadata.get("camera_height", 240))

args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from sim import UltraJointPositionController
from sim.envs.facing_dual_ultra import (
    FACING_CAMERA_PATHS,
    FACING_CONTROLLED_ARMS,
    FacingDualUltraSceneCfg,
    facing_robot_camera_cfg,
)
from sim.robots.ultra import as_torch


WRIST_CAMERAS = tuple(name for name in FACING_CAMERA_PATHS if "wrist" in name)
ROBOT_NAMES = tuple(name for name, _ in FACING_CONTROLLED_ARMS)
OBJECT_NAMES = ("cube", "plate")


def _create_video_dataset(group, name, length):
    return group.create_dataset(
        name,
        shape=(length, height, width, 3),
        dtype="uint8",
        chunks=(1, height, width, 3),
        compression="lzf",
        shuffle=True,
    )


class EpisodeParallelReplay:
    """Replay one dataset episode in each cloned environment."""

    def __init__(self):
        self.sim = SimulationContext(
            sim_utils.SimulationCfg(dt=0.02, render_interval=1, device=args.device)
        )
        cfg = FacingDualUltraSceneCfg(num_envs=args.num_envs, env_spacing=4.0)
        for stream in WRIST_CAMERAS:
            sensor_name = stream.removesuffix("_rgb") + "_camera"
            setattr(cfg, sensor_name, facing_robot_camera_cfg(stream, width, height))
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.controllers = {
            robot_name: UltraJointPositionController(self.scene[robot_name])
            for robot_name in ROBOT_NAMES
        }
        self.device = self.scene[ROBOT_NAMES[0]].device
        self.dtype = torch.float32
        self.episode_count = len(episode_names)
        self.episode_ids = torch.arange(self.episode_count, device=self.device, dtype=torch.long)

    @property
    def observation_joint_names(self):
        return tuple(
            f"{robot_name}/{joint_name}"
            for robot_name in ROBOT_NAMES
            for joint_name in self.scene[robot_name].joint_names
        )

    def restore_initial_states(self, demos):
        for robot_name in ROBOT_NAMES:
            robot = self.scene[robot_name]
            positions = torch.as_tensor(
                np.stack([demo[f"initial_state/{robot_name}_joint_pos"][...] for demo in demos]),
                device=self.device,
                dtype=self.dtype,
            )
            velocities = torch.as_tensor(
                np.stack([demo[f"initial_state/{robot_name}_joint_vel"][...] for demo in demos]),
                device=self.device,
                dtype=self.dtype,
            )
            robot.write_joint_state_to_sim_index(
                position=positions, velocity=velocities, env_ids=self.episode_ids
            )
            controller = self.controllers[robot_name]
            controller.target[self.episode_ids] = positions[:, controller.joint_ids]

        for object_name in OBJECT_NAMES:
            poses = torch.as_tensor(
                np.stack([demo[f"initial_state/{object_name}_pose"][...] for demo in demos]),
                device=self.device,
                dtype=self.dtype,
            ).clone()
            poses[:, :3] += self.scene.env_origins[self.episode_ids]
            velocities = torch.as_tensor(
                np.stack([demo[f"initial_state/{object_name}_velocity"][...] for demo in demos]),
                device=self.device,
                dtype=self.dtype,
            )
            obj = self.scene[object_name]
            obj.write_root_pose_to_sim_index(root_pose=poses, env_ids=self.episode_ids)
            obj.write_root_velocity_to_sim_index(
                root_velocity=velocities, env_ids=self.episode_ids
            )
        self.scene.reset(self.episode_ids)
        return self.capture()

    def apply_actions(self, actions, active):
        active_ids = torch.as_tensor(np.flatnonzero(active), device=self.device, dtype=torch.long)
        active_actions = torch.as_tensor(actions[active], device=self.device, dtype=self.dtype)
        for robot_index, (robot_name, arm) in enumerate(FACING_CONTROLLED_ARMS):
            controller = self.controllers[robot_name]
            selected = active_actions[:, 8 * robot_index : 8 * (robot_index + 1)]
            action_start = 6 if arm == "la" else 14
            gripper_index = 13 if arm == "la" else 21
            controller.target[active_ids, action_start : action_start + 7] = selected[:, :7]
            controller.target[active_ids, gripper_index] = selected[:, 7]

        for _ in range(2):
            for controller in self.controllers.values():
                controller.apply()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(0.02)

    def capture(self):
        self.sim.forward()
        self.sim.render()
        images = {}
        for stream in WRIST_CAMERAS:
            camera = self.scene[stream.removesuffix("_rgb") + "_camera"]
            camera.update(0.0, force_recompute=True)
            rgb = camera.data.output["rgb"][: self.episode_count, ..., :3]
            images[stream] = rgb.detach().cpu().numpy().astype(np.uint8, copy=False)
        return images


def _prepare_outputs(demos):
    outputs = []
    for episode_name, demo in zip(episode_names, demos):
        for camera in WRIST_CAMERAS:
            for group_name in ("initial_state", "obs", "next_obs"):
                path = f"{group_name}/{camera}"
                if path in demo:
                    raise ValueError(f"{episode_name} already contains {path}")
        length = len(demo["actions"])
        outputs.append(
            {
                "obs": {
                    camera: _create_video_dataset(demo["obs"], camera, length)
                    for camera in WRIST_CAMERAS
                },
                "next_obs": {
                    camera: _create_video_dataset(demo["next_obs"], camera, length)
                    for camera in WRIST_CAMERAS
                },
            }
        )
    return outputs


def _flush_block(outputs, buffered, start, lengths):
    block_length = next(iter(buffered.values())).shape[0]
    for episode_index, length in enumerate(lengths):
        next_count = min(block_length, max(0, length - start))
        obs_count = min(block_length, max(0, length - start - 1))
        for camera in WRIST_CAMERAS:
            if next_count:
                outputs[episode_index]["next_obs"][camera][start : start + next_count] = (
                    buffered[camera][:next_count, episode_index]
                )
            if obs_count:
                outputs[episode_index]["obs"][camera][start + 1 : start + 1 + obs_count] = (
                    buffered[camera][:obs_count, episode_index]
                )


def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(args.source, partial)
        replay = EpisodeParallelReplay()
        expected_joint_names = tuple(metadata.get("observation_joint_names", ()))
        if expected_joint_names and replay.observation_joint_names != expected_joint_names:
            raise ValueError("Dataset joint ordering does not match the current facing-dual scene")

        with h5py.File(partial, "r+") as dataset:
            demos = [dataset[f"data/{name}"] for name in episode_names]
            lengths = np.asarray([len(demo["actions"]) for demo in demos], dtype=np.int64)
            max_length = int(lengths.max())
            actions = np.zeros((max_length, len(demos), 16), dtype=np.float32)
            for episode_index, demo in enumerate(demos):
                actions[: lengths[episode_index], episode_index] = demo["actions"][...]

            upgraded_metadata = dict(metadata)
            upgraded_metadata["camera_streams"] = list(FACING_CAMERA_PATHS)
            upgraded_metadata["camera_backfill"] = {
                "source": str(args.source.resolve()),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "method": "episode-parallel deterministic action replay",
                "num_envs": args.num_envs,
                "parallel_episodes": len(demos),
                "regenerated_streams": list(WRIST_CAMERAS),
            }
            dataset.attrs["metadata"] = json.dumps(upgraded_metadata)
            outputs = _prepare_outputs(demos)

            initial_images = replay.restore_initial_states(demos)
            for episode_index, demo in enumerate(demos):
                for camera in WRIST_CAMERAS:
                    frame = initial_images[camera][episode_index]
                    demo.create_dataset(
                        f"initial_state/{camera}", data=frame, compression="lzf", shuffle=True
                    )
                    if lengths[episode_index]:
                        outputs[episode_index]["obs"][camera][0] = frame

            print(
                f"Replaying {len(demos)} episodes across {args.num_envs} environments "
                f"for {max_length} parallel timesteps",
                flush=True,
            )
            for block_start in range(0, max_length, args.write_block_size):
                block_stop = min(block_start + args.write_block_size, max_length)
                captured = {camera: [] for camera in WRIST_CAMERAS}
                for timestep in range(block_start, block_stop):
                    replay.apply_actions(actions[timestep], timestep < lengths)
                    frames = replay.capture()
                    for camera in WRIST_CAMERAS:
                        captured[camera].append(frames[camera])
                buffered = {
                    camera: np.stack(frames, axis=0) for camera, frames in captured.items()
                }
                _flush_block(outputs, buffered, block_start, lengths)
                dataset.flush()
                print(f"Parallel replay: {block_stop}/{max_length} timesteps", flush=True)

        os.replace(partial, args.output)
        print(f"Wrote upgraded dataset: {args.output}", flush=True)
    except BaseException as exc:
        print(f"Camera backfill failed: {type(exc).__name__}: {exc}", flush=True)
        if partial.exists():
            partial.unlink()
        raise


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
