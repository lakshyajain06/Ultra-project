"""Smoke-test the manager-based dual-Ultra shared-workspace environment."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--layout", choices=("side_by_side", "facing"), default="side_by_side")
parser.add_argument("--vision", action="store_true")
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=240)
parser.add_argument("--dump-dir", type=Path, help="Write the final enabled RGB frames as PNG files")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.steps < 0 or min(args.camera_width, args.camera_height) <= 0:
    parser.error("--num_envs and camera dimensions must be positive; --steps must be nonnegative")
if args.vision:
    args.enable_cameras = True

launcher = AppLauncher(args)
app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from sim.envs import DualUltraEnvCfg, FacingDualUltraEnvCfg
from sim.envs.dual_ultra import DUAL_CAMERA_PATHS
from sim.envs.facing_dual_ultra import FACING_CAMERA_PATHS


def main():
    cfg_type = FacingDualUltraEnvCfg if args.layout == "facing" else DualUltraEnvCfg
    camera_paths = FACING_CAMERA_PATHS if args.layout == "facing" else DUAL_CAMERA_PATHS
    cfg = cfg_type(
        enabled_cameras=tuple(camera_paths) if args.vision else (),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=0)
        actions = []
        for name, arm in (("robot_left", "ra"), ("robot_right", "la")):
            robot = env.scene[name]
            expected_names = tuple([*(f"{arm}_j{i}" for i in range(1, 8)), f"{arm}_gripper_joint"])
            joint_ids, names = robot.find_joints(expected_names, preserve_order=True)
            if tuple(names) != expected_names:
                raise RuntimeError(f"{name} action joint order mismatch: {names}")
            actions.append(robot.data.default_joint_pos.torch[:, joint_ids])
        actions = torch.cat(actions, dim=-1)
        for _ in range(args.steps):
            observations, rewards, terminated, truncated, _ = env.step(actions)
        policy = observations["policy"]
        expected = {"proprio": (args.num_envs, 16)}
        if args.vision:
            expected.update({
                key: (args.num_envs, args.camera_height, args.camera_width, 3)
                for key in camera_paths
            })
        for key, shape in expected.items():
            if tuple(policy[key].shape) != shape or not torch.isfinite(policy[key]).all():
                raise RuntimeError(f"Invalid {key}: shape={tuple(policy[key].shape)}")
            if key.endswith("_rgb") and (policy[key].amax() - policy[key].amin()).item() < 10:
                raise RuntimeError(f"Rendered {key} has insufficient dynamic range")
        if args.dump_dir:
            from PIL import Image

            args.dump_dir.mkdir(parents=True, exist_ok=True)
            for key in camera_paths:
                image = policy[key][0].detach().cpu().numpy().clip(0, 255).astype("uint8")
                Image.fromarray(image).save(args.dump_dir / f"{args.layout}_{key}.png")
        print(
            f"DUAL RL ENV SMOKE PASS: layout={args.layout}, envs={args.num_envs}, steps={args.steps}, "
            f"action_shape={tuple(actions.shape)}, reward_mean={rewards.mean().item():.4f}, "
            f"terminated={terminated.sum().item()}, truncated={truncated.sum().item()}",
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
