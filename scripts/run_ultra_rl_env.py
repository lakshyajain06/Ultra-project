"""Smoke-test the manager-based Ultra cube-on-plate environment."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--vision", action="store_true", help="Enable the three dataset RGB cameras")
parser.add_argument(
    "--cameras",
    nargs="*",
    choices=("head_rgb", "left_wrist_rgb", "right_wrist_rgb"),
    help="Vision streams to enable (default: all; pass an empty list to disable all)",
)
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=240)
parser.add_argument(
    "--check-success",
    action="store_true",
    help="Place the cube on the plate and verify that stable placement terminates",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.steps < 0 or min(args.camera_width, args.camera_height) <= 0:
    parser.error("--num_envs and camera dimensions must be positive; --steps must be nonnegative")
if args.cameras is not None and not args.vision:
    parser.error("--cameras requires --vision")
if args.check_success and args.steps < 5:
    parser.error("--check-success requires --steps 5 or greater")
if args.vision:
    args.enable_cameras = True

launcher = AppLauncher(args)
app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from ultra_scene import ULTRA_CONTROLLED_JOINT_NAMES
from ultra_scene.rl import CAMERA_STREAMS, UltraCubePlateEnvCfg, UltraCubePlateVisionEnvCfg


def main():
    if args.vision:
        cfg = UltraCubePlateVisionEnvCfg(
            enabled_cameras=tuple(CAMERA_STREAMS if args.cameras is None else args.cameras),
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
    else:
        cfg = UltraCubePlateEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=0)
        robot = env.scene["robot"]
        joint_ids, names = robot.find_joints(ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True)
        if tuple(names) != ULTRA_CONTROLLED_JOINT_NAMES:
            raise RuntimeError(f"Action joint order mismatch: {names}")
        actions = robot.data.default_joint_pos.torch[:, joint_ids].clone()
        if args.check_success:
            plate = env.scene["plate"]
            cube = env.scene["cube"]
            placed_pose = plate.data.root_pose_w.torch.clone()
            placed_pose[:, 2] += 0.05
            cube.write_root_pose_to_sim_index(root_pose=placed_pose)
            cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(cube.data.root_vel_w.torch))
        saw_success = False
        for _ in range(args.steps):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            saw_success |= bool(torch.any(terminated).item())
        if args.check_success and not saw_success:
            raise RuntimeError("Stable cube-on-plate placement did not terminate within the smoke run")
        policy = observations["policy"]
        expected = {
            "proprio": (args.num_envs, 44),
        }
        if args.vision:
            enabled = CAMERA_STREAMS if args.cameras is None else args.cameras
            expected.update({key: (args.num_envs, args.camera_height, args.camera_width, 3) for key in enabled})
        else:
            expected.update(
                {
                    "eef_pose_body": (args.num_envs, 2, 7),
                    "cube_pose": (args.num_envs, 7),
                    "plate_pose": (args.num_envs, 7),
                }
            )
        for key, shape in expected.items():
            if tuple(policy[key].shape) != shape or not torch.isfinite(policy[key]).all():
                raise RuntimeError(f"Invalid {key}: shape={tuple(policy[key].shape)}")
        print(
            f"RL ENV SMOKE PASS: envs={args.num_envs}, steps={args.steps}, "
            f"action_shape={tuple(actions.shape)}, reward_mean={rewards.mean().item():.4f}, "
            f"terminated={terminated.sum().item()}, truncated={truncated.sum().item()}, "
            f"success_seen={saw_success}",
            flush=True,
        )
    finally:
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
