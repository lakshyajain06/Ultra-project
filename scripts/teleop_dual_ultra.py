"""Quest teleoperation for two Ultras; each hand drives one inner arm."""

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
parser.add_argument("--task", default="Cooperatively place the cube on the plate")
parser.add_argument(
    "--layout",
    choices=("side_by_side", "facing"),
    default="side_by_side",
    help="Robot arrangement; facing records both robots' head cameras",
)
parser.add_argument("--smoke", action="store_true", help="Exercise both controlled arms without XR")
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps; smoke defaults to 50")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--tool_rotation_offset", nargs=3, type=float, metavar=("X", "Y", "Z"))
parser.add_argument("--episode_seconds", type=float, default=120.0)
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--anchor_pos", nargs=3, type=float, default=(0.0, 0.0, 0.0))
parser.add_argument("--anchor_yaw", type=float, default=0.0)
parser.add_argument("--headset_height", type=float, default=1.65)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.scale, args.episode_seconds) <= 0 or args.steps < 0:
    parser.error("scale and episode_seconds must be positive; steps must be nonnegative")
if min(args.camera_width, args.camera_height) <= 0 or args.headset_height < 0:
    parser.error("camera dimensions must be positive and headset_height must be nonnegative")
if args.dataset and args.dataset.exists():
    parser.error(f"Dataset already exists: {args.dataset}; choose a new path")
if not args.smoke:
    args.xr = True
    version("isaacteleop")
elif not args.steps:
    args.steps = 50
# The central camera prim is also the live XR anchor, so it must exist even
# during control-only sessions. Keeping all three streams active makes live and
# recorded camera geometry identical.
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
from scipy.spatial.transform import Rotation

from data import DUAL_TASK_STATE_NAMES
from data.recording import EpisodeRecorder
from sim.envs.dual_ultra import CONTROLLED_ARMS, DUAL_CAMERA_PATHS
from sim.envs.facing_dual_ultra import (
    FACING_CAMERA_PATHS,
    FACING_CONTROLLED_ARMS,
    FACING_TASK_CAMERA_PATH,
)
from sim.robots.ultra.ultra_cfg import REPO_ROOT, ULTRA_USD
from sim.teleop.control import ClutchMapper
from sim.teleop.dual_env import DualUltraTeleopEnv
from sim.teleop.env import numpy


def task_camera_anchor_rotation(yaw_offset_degrees):
    offset = Rotation.from_euler("z", yaw_offset_degrees, degrees=True)

    def rotation(_head_pose, camera_pose):
        return (Rotation.from_quat(camera_pose[3:7]) * offset).as_quat()

    return rotation


def main():
    controlled_arms = FACING_CONTROLLED_ARMS if args.layout == "facing" else CONTROLLED_ARMS
    camera_paths = FACING_CAMERA_PATHS if args.layout == "facing" else DUAL_CAMERA_PATHS
    # Rows consumed by the controller must remain in canonical robot/action
    # order. In the side view of the facing layout, robot_left appears on the
    # operator's right, so swap the physical Quest hands for spatially natural
    # control without changing the recorded action schema.
    robot_to_controller = np.asarray((1, 0) if args.layout == "facing" else (0, 1))
    controller_to_robot = np.argsort(robot_to_controller)
    bindings = tuple(controlled_arms[index] for index in controller_to_robot)
    task_camera_path = FACING_TASK_CAMERA_PATH if args.layout == "facing" else camera_paths["task_rgb"]
    task_camera_prim_path = task_camera_path.replace(
        "{ENV_REGEX_NS}", "/World/envs/env_0"
    )
    env = DualUltraTeleopEnv(
        args.device,
        enable_cameras=args.enable_cameras,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        layout=args.layout,
    )
    obs, _ = env.reset(seed=args.seed)
    mapper = ClutchMapper(scale=args.scale, rotation_offset_deg=args.tool_rotation_offset)
    metadata = {
        "task": args.task,
        "environment": f"dual_ultra_{args.layout}",
        "layout": args.layout,
        "seed": args.seed,
        "control_hz": 25,
        "physics_hz": 50,
        "action_joint_names": env.action_joint_names,
        "observation_joint_names": env.observation_joint_names,
        "proprio_layout": list(DUAL_TASK_STATE_NAMES),
        "controller_bindings": {
            "left": f"{bindings[0][0]}/{bindings[0][1]}",
            "right": f"{bindings[1][0]}/{bindings[1][1]}",
        },
        "action_units": "two selected seven-joint arms: radians; two grippers: metres",
        "pose_convention": "selected wrist positions in each robot body frame; orientations in world; XYZW",
        "quest_columns": [
            "x", "y", "z", "qx", "qy", "qz", "qw", "trigger", "squeeze", "valid",
            "primary", "secondary", "stick_click", "stick_x", "stick_y",
        ],
        "camera_streams": list(camera_paths),
        "camera_encoding": "uint8 RGB, HWC, lossless HDF5 LZF",
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "versions": {p: version(p) for p in ("isaaclab", "isaacsim", "h5py")},
        "asset_sha256": hashlib.sha256(ULTRA_USD.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scale": args.scale,
        "tool_rotation_offset_xyz_deg": args.tool_rotation_offset,
        "orientation_calibration": "fixed per controller; initial tracking or right-stick recalibration",
        "anchor_pos": args.anchor_pos,
        "anchor_yaw_deg": args.anchor_yaw,
        "headset_height": args.headset_height,
        "synthetic": args.smoke,
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
        recorder = None
        if args.dataset:
            recorder = EpisodeRecorder(args.dataset, metadata, schema_version=7)
            stack.callback(recorder.close)
        if not args.smoke:
            from isaaclab_teleop import (
                CLOUDXR_JS_ENV,
                IsaacTeleopCfg,
                XrAnchorRotationMode,
                XrCfg,
                create_isaac_teleop_device,
            )
            from isaacteleop.teleop_session_manager import RetargetingExecutionConfig
            from sim.teleop.input import build_pipeline

            cfg = IsaacTeleopCfg(
                pipeline_builder=build_pipeline,
                sim_device=args.device,
                xr_cfg=XrCfg(
                    anchor_pos=(args.anchor_pos[0], args.anchor_pos[1], args.anchor_pos[2] - args.headset_height),
                    anchor_prim_path=task_camera_prim_path,
                    anchor_rotation_mode=XrAnchorRotationMode.CUSTOM,
                    anchor_rotation_custom_func=task_camera_anchor_rotation(args.anchor_yaw),
                    fixed_anchor_height=False,
                ),
                retargeting_execution=RetargetingExecutionConfig(mode="sync"),
            )
            teleop = stack.enter_context(create_isaac_teleop_device(cfg, cloudxr_env_file=CLOUDXR_JS_ENV))
            print(
                f"Left hand: {bindings[0][0]} {bindings[0][1]} arm | "
                f"Right hand: {bindings[1][0]} {bindings[1][1]} arm.\n"
                "Grip clutches an arm; trigger closes its jaw. Thumbsticks are reserved.\n"
                "X: success/reset | Y: abort/reset | B: pause | right stick click: recalibrate.",
                flush=True,
            )

        while app.is_running() and (not args.steps or step < args.steps):
            started = time.monotonic()
            active = True
            reset_requested = False
            if args.smoke:
                control_packet = np.zeros((2, 15), dtype=np.float32)
                control_packet[:, :7] = start_pose
                control_packet[:, 9] = 1
                control_packet[:, 8] = float(step > 1)
                distance = min(max(step - 2, 0) * 0.001, 0.025)
                control_packet[0, 0] += distance
                control_packet[1, 0] -= distance
                angle = min(max(step - 2, 0) * 0.005, 0.15)
                control_packet[0, 3:7] = (
                    Rotation.from_euler("z", angle) * Rotation.from_quat(start_pose[0, 3:7])
                ).as_quat()
                control_packet[1, 3:7] = (
                    Rotation.from_euler("z", -angle) * Rotation.from_quat(start_pose[1, 3:7])
                ).as_quat()
                packet = np.empty_like(control_packet)
                packet[robot_to_controller] = control_packet
            else:
                sample = teleop.advance()
                events = teleop.last_control_events
                active = sample is not None and events.is_active is True
                reset_requested = events.should_reset
                packet = numpy(sample).reshape(2, 15) if sample is not None else np.zeros((2, 15))
                control_packet = packet[robot_to_controller]

            buttons = packet[:, 10:13] > 0.5
            rising = buttons & ~previous_buttons
            previous_buttons = buttons.copy()
            if rising[1, 1]:
                paused = not paused
                print(f"Paused: {paused}", flush=True)
            if rising[1, 2]:
                current = env.current_action()
                mapper.reset(grippers=current[[7, 15]], recalibrate_orientation=True)
            timed_out = recording and env.elapsed >= args.episode_seconds
            if rising[0, 0] or rising[0, 1] or reset_requested or timed_out:
                status = (
                    "aborted" if rising[0, 1] or reset_requested else "timeout" if timed_out else "success"
                )
                if recorder:
                    recorder.finish(status)
                print(f"Episode: {status}", flush=True)
                obs, _ = env.reset()
                mapper.reset()
                recording = False
                step += 1
                continue

            enabled = active and not paused
            targets, grippers = mapper.update(
                control_packet,
                obs["eef_pose_body"],
                env.control_dt,
                enabled,
                reference_pose=obs["body_pose"],
            )
            active_arms = np.asarray([anchor is not None for anchor in mapper.anchor]) & enabled
            engaged = bool(np.any(active_arms))
            if engaged and not recording:
                if recorder:
                    initial_state = env.initial_state()
                    initial_state["controller_rotation_offsets"] = mapper.rotation_offsets_xyzw()
                    recorder.begin(initial_state)
                recording = True
                env.elapsed = 0.0
            action = (
                env.ik_action(targets, grippers, active_arms)
                if engaged
                else env.current_action()
            )
            next_obs, _, _, _, _ = env.step(action)
            if recording and recorder:
                recorder.append(
                    obs,
                    action,
                    next_obs,
                    packet,
                    targets,
                    time.time(),
                    env.elapsed,
                    mapper.rotation_offsets_xyzw(),
                )
            obs = next_obs
            max_motion = np.maximum(
                max_motion, np.linalg.norm(obs["eef_pose"][:, :3] - start_pose[:, :3], axis=1)
            )
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
                raise RuntimeError("Nonfinite joint state in dual-Ultra smoke test")
            for key in camera_paths:
                frame = obs[key]
                expected = (args.camera_height, args.camera_width, 3)
                if frame.shape != expected or frame.dtype != np.uint8 or np.ptp(frame) < 10:
                    raise RuntimeError(
                        f"Invalid {key} frame: shape={frame.shape}, dtype={frame.dtype}, range={np.ptp(frame)}"
                    )
            if args.steps >= 20 and (np.any(max_motion < 0.001) or np.any(max_rotation < 0.01)):
                raise RuntimeError(
                    f"Both controlled wrists must move and rotate: motion={max_motion}, rotation={max_rotation}"
                )
            print(
                f"DUAL SMOKE PASS: {step} steps, wrist motion={max_motion.tolist()}, "
                f"rotation={max_rotation.tolist()}",
                flush=True,
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
