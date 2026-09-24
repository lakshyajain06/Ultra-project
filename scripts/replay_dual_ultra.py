"""List or replay a schema-7 dual-Ultra HDF5 demonstration."""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path)
parser.add_argument("--episode", default="0", help="Episode number or name")
parser.add_argument("--list", action="store_true")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--start-step", type=int, default=0)
parser.add_argument("--end-step", type=int)
parser.add_argument("--hold-seconds", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def episode_name(value):
    return f"demo_{int(value):06d}" if value.isdigit() else value


if not args.dataset.is_file():
    parser.error(f"Dataset does not exist: {args.dataset}")
with h5py.File(args.dataset, "r") as handle:
    if int(handle.attrs.get("schema_version", -1)) != 7:
        parser.error("Dual replay requires schema_version=7")
    metadata = json.loads(handle.attrs.get("metadata", "{}"))
    if args.list:
        print(f"{args.dataset} | task={metadata.get('task', 'unknown')!r}")
        control_hz = float(metadata.get("control_hz", 25.0))
        for name, demo in handle["data"].items():
            samples = len(demo.get("actions", ()))
            print(
                f"{name}: status={demo.attrs.get('status', 'unknown')}, samples={samples}, "
                f"seconds={samples / control_hz:.2f}"
            )
        raise SystemExit(0)
    selected = episode_name(args.episode)
    if selected not in handle["data"]:
        parser.error(f"Episode {selected!r} not found; use --list")
    demo = handle[f"data/{selected}"]
    actions = demo["actions"][...]
    initial = {key: dataset[...] for key, dataset in demo["initial_state"].items()}
    status = str(demo.attrs.get("status", "unknown"))

if args.speed <= 0 or args.start_step < 0 or args.hold_seconds < 0:
    parser.error("--speed must be positive; step and hold values must be nonnegative")
end = len(actions) if args.end_step is None else min(args.end_step, len(actions))
if end <= args.start_step:
    parser.error(f"Empty replay range [{args.start_step}, {end}) for {len(actions)} samples")

launcher = AppLauncher(args)
app = launcher.app

from sim.teleop.dual_env import DualUltraTeleopEnv


def main():
    layout = metadata.get("layout", "facing" if metadata.get("environment") == "dual_ultra_facing" else "side_by_side")
    env = DualUltraTeleopEnv(args.device, layout=layout)
    env.restore_initial_state(initial)
    period = 1.0 / (float(metadata.get("control_hz", 25.0)) * args.speed)
    print(
        f"Replaying {selected} [{args.start_step}:{end}] at {args.speed:g}x ({status})",
        flush=True,
    )
    for index in range(args.start_step):
        if not app.is_running():
            return
        env.step(actions[index])
    observation = None
    for index in range(args.start_step, end):
        if not app.is_running():
            break
        started = time.monotonic()
        observation, _, _, _, _ = env.step(actions[index])
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    if observation is not None:
        print(
            f"Reached sample {index}; finite joints={bool(np.isfinite(observation['joint_pos']).all())}",
            flush=True,
        )
        deadline = time.monotonic() + args.hold_seconds
        while app.is_running() and time.monotonic() < deadline:
            env.step(actions[index])
            time.sleep(env.control_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
