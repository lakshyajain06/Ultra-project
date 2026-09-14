"""Launch the simple Ultra tabletop scene in Isaac Lab."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned scenes")
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs until the app closes")
parser.add_argument("--screenshot", type=Path, help="Save the reset-pose RGB frame to this PNG path")
parser.add_argument("--video", type=Path, help="Record rendered frames to this MP4 path")
parser.add_argument("--video_fps", type=int, default=10, help="Frame rate for --video (default: 10)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.screenshot or args_cli.video:
    # Select Isaac Lab's headless rendering experience automatically. This is
    # required for RTX camera sensors when no desktop display is available.
    args_cli.enable_cameras = True
    if args_cli.steps == 0:
        args_cli.steps = 150 if args_cli.video else 5

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / Isaac Sim imports must happen after AppLauncher starts Kit.
import torch
import numpy as np
import imageio.v2 as imageio

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from sim import UltraJointPositionController, UltraTabletopSceneCfg  # noqa: E402
from sim.robots.ultra import as_torch  # noqa: E402


def camera_rgb(scene):
    """Copy environment zero's RGB camera result into a NumPy array."""
    pixels = scene["camera"].data.output["rgb"]
    if hasattr(pixels, "warp"):
        pixels = pixels.warp
    if hasattr(pixels, "numpy"):
        pixels = pixels.numpy()
    elif isinstance(pixels, torch.Tensor):
        pixels = pixels.detach().cpu().numpy()
    return np.asarray(pixels[0], dtype=np.uint8)


def reset_scene(scene, controller):
    """Restore the configured robot and object poses in every environment."""
    # Ultra is fixed to the world by its authored root joint. Its base transform
    # is established when the USD is spawned and must not be teleported at reset.
    controller.reset()

    for name in ("cube", "plate"):
        obj = scene[name]
        root_pose = as_torch(obj.data.default_root_pose).clone()
        root_pose[:, :3] += scene.env_origins
        root_velocity = as_torch(obj.data.default_root_vel).clone()
        obj.write_root_pose_to_sim_index(root_pose=root_pose)
        obj.write_root_velocity_to_sim_index(root_velocity=root_velocity)
    scene.reset()


def main():
    # Match real2sim2real/vec/vec_env.py's SIM_HZ exactly.
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 50.0, render_interval=5, device=args_cli.device)
    )
    camera_eye = (3.5, 5.0, 2.4)
    camera_target = (0.0, 0.65, 0.80)
    sim.set_camera_view(eye=camera_eye, target=camera_target)
    scene_cfg = UltraTabletopSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    if not (args_cli.screenshot or args_cli.video):
        scene_cfg.camera = None
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    controller = UltraJointPositionController(robot)
    reset_scene(scene, controller)
    if args_cli.screenshot or args_cli.video:
        env_origins = as_torch(scene.env_origins).detach().cpu().numpy()
        scene["camera"].set_world_poses_from_view(
            env_origins + np.asarray(camera_eye, dtype=np.float32),
            env_origins + np.asarray(camera_target, dtype=np.float32),
        )
    reset_error = torch.max(controller.error()).item()
    initial_cube_z = as_torch(scene["cube"].data.root_pos_w)[:, 2].clone()
    step = 0
    video_writer = None
    try:
        if args_cli.video:
            args_cli.video.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(args_cli.video, fps=args_cli.video_fps, codec="libx264")

        if args_cli.screenshot:
            # Capture the authored reset pose before physics advances. This
            # gives a reliable scene preview even while tuning the controller.
            sim.render()
            scene["camera"].update(0.0, force_recompute=True)
            args_cli.screenshot.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args_cli.screenshot, camera_rgb(scene))
            print(f"[INFO] Screenshot saved to {args_cli.screenshot.resolve()}", flush=True)

        while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
            controller.apply()
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            step += 1
            if video_writer is not None and step % 5 == 0:
                video_writer.append_data(camera_rgb(scene))

    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Video saved to {args_cli.video.resolve()}", flush=True)

    error = controller.error()
    joint_error = torch.max(error).item()
    error_by_joint = error.amax(dim=0)
    worst_joint_index = int(torch.argmax(error_by_joint).item())
    worst_joint = controller.joint_names[worst_joint_index]
    cube_drop = torch.max(initial_cube_z - as_torch(scene["cube"].data.root_pos_w)[:, 2]).item()
    print(
        f"[INFO] Ultra tabletop scene completed {step} steps: "
        f"{args_cli.num_envs} env(s), {robot.num_joints} DOF / "
        f"{len(controller.joint_ids)} controlled, "
        f"reset error={reset_error:.5f} rad, "
        f"max joint error={joint_error:.5f} rad ({worst_joint}), cube drop={cube_drop:.5f} m",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit shutdown can terminate the interpreter, so print the original
        # Python failure before closing the application.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
