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
parser.add_argument("--no-recorded-images", action="store_true", help="Do not show saved camera frames")
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
            "image_keys": tuple(
                key for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
                if f"obs/{key}" in demo and f"next_obs/{key}" in demo
            ),
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


class RecordedCameraViewer:
    """Kit window backed by the RGB arrays stored in the demonstration."""

    LABELS = {
        "head_rgb": "Head",
        "left_wrist_rgb": "Left wrist",
        "right_wrist_rgb": "Right wrist",
    }

    def __init__(self, image_keys):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("Recorded camera frames", width=1020, height=330)
        self.providers = {}
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.status = ui.Label("Recorded sample", height=22)
                with ui.HStack(spacing=4):
                    for key in image_keys:
                        with ui.VStack(spacing=2):
                            ui.Label(self.LABELS[key], alignment=ui.Alignment.CENTER, height=20)
                            provider = ui.ByteImageProvider()
                            self.providers[key] = provider
                            ui.ImageWithProvider(
                                provider,
                                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                            )

    def update(self, images, index):
        self.status.text = f"Recorded sample {index}"
        for key, rgb in images.items():
            rgb = np.asarray(rgb, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise ValueError(f"Recorded {key} must be HWC RGB/RGBA, got {rgb.shape}")
            if rgb.shape[2] == 3:
                rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
                rgba[..., :3] = rgb
                rgba[..., 3] = 255
            else:
                rgba = np.ascontiguousarray(rgb)
            self.providers[key].set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])

    def close(self):
        self.window.destroy()


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
    image_keys = () if args.no_recorded_images else payload["image_keys"]
    if not args.no_recorded_images and not image_keys:
        print("This episode has no recorded camera frames; replaying simulation only.", flush=True)
    viewer = RecordedCameraViewer(image_keys) if image_keys else None
    try:
        with h5py.File(args.dataset, "r") as file:
            demo = file[f"data/{selected}"]
            if viewer:
                viewer.update({key: demo[f"obs/{key}"][args.start_step] for key in image_keys}, args.start_step)
            # Reconstruct the state at --start-step rather than applying that
            # action directly to the episode's initial state.
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
                if viewer:
                    viewer.update({key: demo[f"next_obs/{key}"][index] for key in image_keys}, index)
                time.sleep(max(0.0, period - (time.monotonic() - started)))
            if last >= args.start_step:
                details = []
                if payload["next_joint_pos"] is not None:
                    details.append(
                        "joint max error="
                        f"{np.max(np.abs(observation['joint_pos'] - payload['next_joint_pos'][last])):.5f} rad"
                    )
                if payload["next_cube_pose"] is not None:
                    details.append(
                        "cube position error="
                        f"{np.linalg.norm(observation['cube_pose'][:3] - payload['next_cube_pose'][last, :3]):.5f} m"
                    )
                print(f"Reached sample {last}. " + ", ".join(details), flush=True)
                deadline = time.monotonic() + args.hold_seconds
                while app.is_running() and time.monotonic() < deadline:
                    env.step(actions[last])
                    time.sleep(env.control_dt)
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
