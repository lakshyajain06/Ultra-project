"""Quest 3 teleoperation and HDF5 collection; run --smoke without a headset."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, help="New HDF5 path; omit to control without recording")
parser.add_argument("--task", default="Place the cube on the plate", help="Instruction recorded with the dataset")
parser.add_argument("--smoke", action="store_true", help="Exercise both IK chains and recording without XR")
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps; smoke defaults to 50")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--scale", type=float, default=1.0, help="Controller-to-wrist translation scale")
parser.add_argument(
    "--tool_rotation_offset", nargs=3, type=float, default=(0.0, 0.0, 90.0), metavar=("X", "Y", "Z"),
    help="Fixed controller-local XYZ tool rotation in degrees (default: 0 0 90)",
)
parser.add_argument("--episode_seconds", type=float, default=120.0)
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--anchor_pos", nargs=3, type=float, default=(0.0, 1.8, 0.0))
parser.add_argument("--anchor_yaw", type=float, default=0.0, help="Rotate the XR view about world Z, in degrees")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.scale, args.episode_seconds) <= 0 or args.steps < 0:
    parser.error("scale and episode_seconds must be positive; steps must be nonnegative")
if min(args.camera_width, args.camera_height) <= 0:
    parser.error("camera dimensions must be positive")
if args.dataset and args.dataset.exists():
    parser.error(f"Dataset already exists: {args.dataset}; choose a new path")
if not args.smoke:
    args.xr = True
    version("isaacteleop")  # Fail before launching Kit when --extra teleop was omitted.
elif not args.steps:
    args.steps = 50
args.enable_cameras = bool(args.dataset or args.smoke)
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
from scipy.spatial.transform import Rotation
from ultra_scene.teleop.control import BodyTargetMapper, ClutchMapper
from ultra_scene.teleop.env import UltraTeleopEnv, numpy
from ultra_scene.teleop.recording import EpisodeRecorder
from ultra_scene.robots.ultra.ultra_cfg import ULTRA_USD, REPO_ROOT


def main():
    env = UltraTeleopEnv(
        args.device, enable_cameras=args.enable_cameras,
        camera_width=args.camera_width, camera_height=args.camera_height,
    )
    obs, _ = env.reset(seed=args.seed)
    mapper = ClutchMapper(scale=args.scale, rotation_offset_deg=args.tool_rotation_offset)
    body_mapper = BodyTargetMapper()
    recorder = None
    metadata = {
        "task": args.task, "seed": args.seed, "control_hz": 25, "physics_hz": 50,
        "action_joint_names": env.controller.joint_names,
        "observation_joint_names": env.robot.joint_names,
        "action_units": "torso and arms: radians; grippers: metres (0 closed, 0.045 open)",
        "pose_convention": (
            "Quest: simulation world; eef_targets: position in shared-body frame, "
            "absolute orientation in simulation world; xyz + quaternion xyzw"
        ),
        "quest_columns": ["x", "y", "z", "qx", "qy", "qz", "qw", "trigger", "squeeze", "valid",
                          "primary", "secondary", "stick_click", "stick_x", "stick_y"],
        "versions": {p: version(p) for p in ("isaaclab", "isaacsim", "h5py")},
        "asset_sha256": hashlib.sha256(ULTRA_USD.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scale": args.scale,
        "tool_rotation_offset_xyz_deg": args.tool_rotation_offset,
        "camera_streams": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        "camera_encoding": "uint8 RGB, HWC, lossless HDF5 LZF",
        "camera_width": args.camera_width, "camera_height": args.camera_height,
        "anchor_pos": args.anchor_pos, "anchor_yaw_deg": args.anchor_yaw, "synthetic": args.smoke,
        "success_label": "operator supplied; synthetic smoke episodes are never successes",
    }
    if not args.smoke:
        metadata["versions"]["isaacteleop"] = version("isaacteleop")
    previous_buttons = np.zeros((2, 3), dtype=bool)
    paused = False
    recording = False
    step = 0
    start_pose = obs["eef_pose"].copy()
    max_motion = np.zeros(2)
    max_rotation = np.zeros(2)
    with ExitStack() as stack:
        if args.dataset:
            recorder = EpisodeRecorder(args.dataset, metadata)
            stack.callback(recorder.close)
        if not args.smoke:
            from isaaclab_teleop import IsaacTeleopCfg, XrCfg, CLOUDXR_JS_ENV, create_isaac_teleop_device
            from isaacteleop.teleop_session_manager import RetargetingExecutionConfig
            from ultra_scene.teleop.input import build_pipeline

            cfg = IsaacTeleopCfg(
                pipeline_builder=build_pipeline, sim_device=args.device,
                xr_cfg=XrCfg(
                    anchor_pos=tuple(args.anchor_pos),
                    anchor_rot=tuple(Rotation.from_euler("z", args.anchor_yaw, degrees=True).as_quat()),
                ),
                retargeting_execution=RetargetingExecutionConfig(mode="sync"),
            )
            teleop = stack.enter_context(create_isaac_teleop_device(cfg, cloudxr_env_file=CLOUDXR_JS_ENV))
            camera_status = (
                "Head and both wrist RGB cameras are recorded."
                if args.dataset else "Camera recording is disabled without --dataset."
            )
            print("Quest: open the release-1.4.x CloudXR client; enter host IP and Connect.\n"
                  "Release grips, then hold a grip to move that arm. Triggers control jaws.\n"
                  "Left stick moves the shared body in the table plane; right stick controls body yaw/height.\n"
                  "X: save success/reset | Y: abort/reset | B: pause | right stick: recalibrate.\n"
                  f"A is reserved by Isaac Lab. {camera_status}", flush=True)
        while app.is_running() and (not args.steps or step < args.steps):
            started = time.monotonic()
            active = True
            reset_requested = False
            if args.smoke:
                packet = np.zeros((2, 15), dtype=np.float32)
                packet[:, :7] = start_pose
                packet[:, 9] = 1
                packet[:, 8] = float(step > 1)
                packet[:, 2] += min(max(step - 2, 0) * 0.001, 0.025)
                angle = min(max(step - 2, 0) * 0.005, 0.15)
                packet[:, 3:7] = (
                    Rotation.from_euler("z", angle) * Rotation.from_quat(start_pose[:, 3:7])
                ).as_quat()
                packet[0, 14] = 0.5 if step > 15 else 0.0
            else:
                sample = teleop.advance()
                events = teleop.last_control_events
                active = sample is not None and events.is_active is True
                reset_requested = events.should_reset
                packet = numpy(sample).reshape(2, 15) if sample is not None else np.zeros((2, 15))
            buttons = packet[:, 10:13] > 0.5
            rising = buttons & ~previous_buttons
            previous_buttons = buttons.copy()
            if rising[1, 1]:
                paused = not paused
                print(f"Paused: {paused}", flush=True)
            if rising[1, 2]:
                mapper.reset(grippers=numpy(env.controller.target)[0, [13, 21]])
            timed_out = recording and env.elapsed >= args.episode_seconds
            if rising[0, 0] or rising[0, 1] or reset_requested or timed_out:
                status = ("aborted" if rising[0, 1] or reset_requested else
                          "timeout" if timed_out else "success")
                if recorder:
                    recorder.finish(status)
                print(f"Episode: {status}", flush=True)
                obs, _ = env.reset(seed=args.seed)
                mapper.reset()
                body_mapper.reset()
                recording = False
                step += 1
                continue
            enabled = active and not paused
            targets, grippers = mapper.update(
                packet, obs["eef_pose_body"], env.control_dt, enabled, reference_pose=obs["body_pose"],
            )
            body_target, body_active = body_mapper.update(packet, obs["body_pose"], env.control_dt, enabled)
            arm_active = enabled and any(a is not None for a in mapper.anchor)
            engaged = arm_active or body_active
            if engaged and not recording:
                if recorder:
                    recorder.begin(env.initial_state())
                recording = True
                env.elapsed = 0.0
            # When paused/disconnected/clutched out, keep last motor targets exactly.
            action = env.ik_action(targets, grippers, body_target, body_active) if engaged else numpy(env.controller.target)[0]
            # A released arm holds its joint targets, so shared-body motion
            # naturally carries the whole arm instead of Cartesian IK undoing it.
            for arm, start in enumerate((6, 14)):
                if mapper.anchor[arm] is None or not enabled:
                    action[start:start + 8] = numpy(env.controller.target)[0, start:start + 8]
            next_obs, _, _, _, _ = env.step(action)
            if recording and recorder:
                recorder.append(obs, action, next_obs, packet, targets, time.time(), env.elapsed)
            obs = next_obs
            max_motion = np.maximum(max_motion, np.linalg.norm(obs["eef_pose"][:, :3] - start_pose[:, :3], axis=1))
            rotation = (
                Rotation.from_quat(obs["eef_pose"][:, 3:7])
                * Rotation.from_quat(start_pose[:, 3:7]).inv()
            ).magnitude()
            max_rotation = np.maximum(max_rotation, rotation)
            step += 1
            if not args.smoke:
                time.sleep(max(0.0, env.control_dt - (time.monotonic() - started)))
        if args.smoke:
            if recorder:
                recorder.finish("synthetic")
            if not np.isfinite(obs["joint_pos"]).all():
                raise RuntimeError("Nonfinite joint state in smoke test")
            for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                frame = obs[key]
                expected = (args.camera_height, args.camera_width, 3)
                if frame.shape != expected or frame.dtype != np.uint8 or np.ptp(frame) < 10:
                    raise RuntimeError(
                        f"Invalid {key} frame: shape={frame.shape}, dtype={frame.dtype}, range={np.ptp(frame)}"
                    )
            if args.steps >= 20 and np.any(max_motion < 0.001):
                raise RuntimeError(f"Both wrists must move in smoke test: {max_motion}")
            if args.steps >= 20 and np.any(max_rotation < 0.01):
                raise RuntimeError(f"Both wrists must rotate in smoke test: {max_rotation}")
            print(
                f"SMOKE PASS: {step} steps, wrist motion metres={max_motion.tolist()}, "
                f"wrist rotation radians={max_rotation.tolist()}", flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
