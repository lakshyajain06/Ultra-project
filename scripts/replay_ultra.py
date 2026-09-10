"""List or replay an Ultra HDF5 demonstration in the original scene."""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path)
parser.add_argument("--episode", default="0", help="Episode number or name, e.g. 3 or demo_000003")
parser.add_argument("--list", action="store_true", help="List episodes without launching Isaac Sim")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--start-step", type=int, default=0)
parser.add_argument("--end-step", type=int, help="Exclusive end step")
parser.add_argument("--hold-seconds", type=float, default=5.0, help="Keep simulating the final pose after playback")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def episode_name(value):
    return f"demo_{int(value):06d}" if value.isdigit() else value


def read_dataset(path, selected=None):
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise ValueError(f"{path} has no data group")
        metadata = json.loads(file.attrs.get("metadata", "{}"))
        rows = []
        for name, demo in file["data"].items():
            samples = demo["actions"].shape[0] if "actions" in demo else 0
            rows.append((name, str(demo.attrs.get("status", "unknown")),
                         bool(demo.attrs.get("success", False)), samples))
        if selected is None:
            return metadata, rows, None
        if selected not in file["data"]:
            raise KeyError(f"Episode {selected!r} not found; use --list")
        demo = file[f"data/{selected}"]
        initial = {key: dataset[...] for key, dataset in demo["initial_state"].items()}
        payload = {
            "actions": demo["actions"][...],
            "initial": initial,
            "status": str(demo.attrs.get("status", "unknown")),
            "success": bool(demo.attrs.get("success", False)),
            "next_joint_pos": demo["next_obs/joint_pos"][...] if "next_obs/joint_pos" in demo else None,
            "next_cube_pose": demo["next_obs/cube_pose"][...] if "next_obs/cube_pose" in demo else None,
        }
        return metadata, rows, payload


if not args.dataset.is_file():
    parser.error(f"Dataset does not exist: {args.dataset}")
try:
    metadata, episodes, payload = read_dataset(args.dataset, None if args.list else episode_name(args.episode))
except (OSError, KeyError, ValueError) as exc:
    parser.error(str(exc))

if args.list:
    print(f"{args.dataset} | schema inspected | task={metadata.get('task', 'unknown')!r}")
    control_hz = float(metadata.get("control_hz", 25.0))
    for name, status, success, samples in episodes:
        print(f"{name}: status={status}, success={success}, samples={samples}, seconds={samples / control_hz:.2f}")
    raise SystemExit(0)

if args.speed <= 0 or args.start_step < 0 or args.hold_seconds < 0:
    parser.error("--speed must be positive; --start-step and --hold-seconds must be nonnegative")
actions = payload["actions"]
end = len(actions) if args.end_step is None else min(args.end_step, len(actions))
if end <= args.start_step:
    parser.error(f"Empty replay range [{args.start_step}, {end}) for {len(actions)} samples")

launcher = AppLauncher(args)
app = launcher.app

from ultra_scene.teleop.env import UltraTeleopEnv


def main():
    env = UltraTeleopEnv(args.device)
    env.restore_initial_state(payload["initial"])
    control_hz = float(metadata.get("control_hz", 25.0))
    period = 1.0 / (control_hz * args.speed)
    selected = episode_name(args.episode)
    print(
        f"Replaying {selected} [{args.start_step}:{end}] at {args.speed:g}x "
        f"({payload['status']}, success={payload['success']})",
        flush=True,
    )
    # Reconstruct the state at --start-step rather than applying that action
    # directly to the episode's initial state.
    for index in range(args.start_step):
        if not app.is_running():
            return
        env.step(actions[index])
    last = args.start_step - 1
    for index in range(args.start_step, end):
        if not app.is_running():
            break
        started = time.monotonic()
        observation, _, _, _, _ = env.step(actions[index])
        last = index
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    if last >= args.start_step:
        details = []
        if payload["next_joint_pos"] is not None:
            details.append(
                f"joint max error={np.max(np.abs(observation['joint_pos'] - payload['next_joint_pos'][last])):.5f} rad"
            )
        if payload["next_cube_pose"] is not None:
            details.append(
                f"cube position error={np.linalg.norm(observation['cube_pose'][:3] - payload['next_cube_pose'][last, :3]):.5f} m"
            )
        print(f"Reached sample {last}. " + ", ".join(details), flush=True)
        deadline = time.monotonic() + args.hold_seconds
        while app.is_running() and time.monotonic() < deadline:
            env.step(actions[last])
            time.sleep(env.control_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
