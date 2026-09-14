"""Single-instance Gymnasium environment using the existing Ultra scene."""

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from sim import UltraJointPositionController
from sim.envs.cube_plate_pick_place.ultra_tabletop_scene_cfg import (
    ROBOT_CAMERA_PATHS,
    UltraTabletopSceneCfg,
    robot_camera_cfg,
)
from sim.robots.ultra import as_torch
from .control import build_task_proprioception, compose_pose, relative_pose, solve_ik


def numpy(value):
    return as_torch(value).detach().cpu().numpy().copy()


class UltraTeleopEnv(gym.Env):
    """22 joint position actions: radians for torso/arms, metres for jaws.

    Two 50 Hz physics steps per 25 Hz action. Episode labels are operator supplied.
    Camera observations are optional; there is no autonomous success predicate.
    """

    metadata = {"render_modes": []}
    control_dt = 0.04

    def __init__(
        self,
        device="cuda:0",
        enable_cameras=False,
        camera_width=320,
        camera_height=240,
        randomize_cube_position=False,
        cube_position_range=(0.08, 0.08),
    ):
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02, render_interval=1, device=device))
        self.sim.set_camera_view(eye=(3.5, 5.0, 2.4), target=(0.0, 0.65, 0.80))
        cfg = UltraTabletopSceneCfg(num_envs=1, env_spacing=3.0)
        cfg.camera = None
        self.camera_names = ()
        if enable_cameras:
            if camera_width <= 0 or camera_height <= 0:
                raise ValueError("Camera width and height must be positive")
            for stream in ROBOT_CAMERA_PATHS:
                name = stream.removesuffix("_rgb") + "_camera"
                setattr(cfg, name, robot_camera_cfg(stream, camera_width, camera_height))
            self.camera_names = tuple(name.removesuffix("_rgb") + "_camera" for name in ROBOT_CAMERA_PATHS)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.controller = UltraJointPositionController(self.robot)
        self.eef_ids, names = self.robot.find_bodies(["la_gripper", "ra_gripper"], preserve_order=True)
        if names != ["la_gripper", "ra_gripper"]:
            raise RuntimeError(f"Unexpected end effectors: {names}")
        self.arm_ids = [self.robot.find_joints([f"{prefix}_j{i}" for i in range(1, 8)],
                                             preserve_order=True)[0] for prefix in ("la", "ra")]
        self.torso_ids = self.robot.find_joints([f"torso_j{i}" for i in range(1, 7)], preserve_order=True)[0]
        self.gripper_ids = [int(self.controller.joint_ids[13]), int(self.controller.joint_ids[21])]
        body_ids, body_names = self.robot.find_bodies(["fr30_6"], preserve_order=True)
        if body_names != ["fr30_6"]:
            raise RuntimeError(f"Unexpected shared body link: {body_names}")
        self.body_id = body_ids[0]
        self.limits = as_torch(self.robot.data.joint_pos_limits)[0]
        self.velocity_limits = as_torch(self.robot.data.joint_vel_limits)[0]
        action_limits = numpy(self.limits[self.controller.joint_ids])
        self.action_space = gym.spaces.Box(action_limits[:, 0], action_limits[:, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "proprio": gym.spaces.Box(-np.inf, np.inf, (22,), np.float32),
            "joint_pos": gym.spaces.Box(-np.inf, np.inf, (24,), np.float32),
            "joint_vel": gym.spaces.Box(-np.inf, np.inf, (24,), np.float32),
            "joint_acc": gym.spaces.Box(-np.inf, np.inf, (24,), np.float32),
            "joint_torque": gym.spaces.Box(-np.inf, np.inf, (24,), np.float32),
            "joint_target": gym.spaces.Box(-np.inf, np.inf, (22,), np.float32),
            "body_link_pose_w": gym.spaces.Box(
                -np.inf, np.inf, (*self.robot.data.body_link_pose_w.shape[1:],), np.float32,
            ),
            "body_link_vel_w": gym.spaces.Box(
                -np.inf, np.inf, (*self.robot.data.body_link_vel_w.shape[1:],), np.float32,
            ),
            "eef_pose": gym.spaces.Box(-np.inf, np.inf, (2, 7), np.float32),
            "eef_pose_body": gym.spaces.Box(-np.inf, np.inf, (2, 7), np.float32),
            "body_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "cube_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "plate_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        })
        for name in self.camera_names:
            self.observation_space[name.removesuffix("_camera") + "_rgb"] = gym.spaces.Box(
                0, 255, (camera_height, camera_width, 3), np.uint8,
            )
        self.elapsed = 0.0
        self.randomize_cube_position = bool(randomize_cube_position)
        self.cube_position_range = np.asarray(cube_position_range, dtype=np.float32)
        if self.cube_position_range.shape != (2,) or np.any(self.cube_position_range < 0):
            raise ValueError("cube_position_range must contain two nonnegative XY extents")

    def observe(self):
        body_pose = numpy(self.robot.data.body_link_pose_w)[0, self.body_id]
        eef_pose = numpy(self.robot.data.body_link_pose_w)[0, self.eef_ids]
        joint_pos = numpy(self.robot.data.joint_pos)[0]
        joint_vel = numpy(self.robot.data.joint_vel)[0]
        joint_acc = numpy(self.robot.data.joint_acc)[0]
        joint_torque = numpy(self.robot.data.applied_torque)[0]
        body_link_pose_w = numpy(self.robot.data.body_link_pose_w)[0]
        body_link_vel_w = numpy(self.robot.data.body_link_vel_w)[0]
        eef_pose_body = np.stack([relative_pose(pose, body_pose) for pose in eef_pose])
        observation = {
            "proprio": build_task_proprioception(joint_pos, eef_pose_body, self.torso_ids, self.gripper_ids),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "joint_torque": joint_torque,
            "joint_target": numpy(self.controller.target)[0],
            "body_link_pose_w": body_link_pose_w,
            "body_link_vel_w": body_link_vel_w,
            "eef_pose": eef_pose,
            "eef_pose_body": eef_pose_body,
            "body_pose": body_pose,
            "cube_pose": numpy(self.scene["cube"].data.root_pose_w)[0],
            "plate_pose": numpy(self.scene["plate"].data.root_pose_w)[0],
        }
        for name in self.camera_names:
            rgb = numpy(self.scene[name].data.output["rgb"])[0]
            observation[name.removesuffix("_camera") + "_rgb"] = np.asarray(rgb[..., :3], dtype=np.uint8)
        return observation

    def _refresh_cameras(self):
        """Render reset state before exposing a new episode observation."""
        if not self.camera_names:
            return
        self.sim.forward()
        self.sim.render()
        for name in self.camera_names:
            self.scene[name].update(0.0, force_recompute=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Restore followers too, so an episode reset is independent of the last grasp.
        self.robot.write_joint_state_to_sim_index(
            position=as_torch(self.robot.data.default_joint_pos).clone(),
            velocity=as_torch(self.robot.data.default_joint_vel).clone(),
        )
        self.controller.reset()
        for name in ("cube", "plate"):
            obj = self.scene[name]
            pose = as_torch(obj.data.default_root_pose).clone()
            pose[:, :3] += as_torch(self.scene.env_origins)
            if name == "cube" and self.randomize_cube_position:
                offset = self.np_random.uniform(-self.cube_position_range, self.cube_position_range)
                pose[:, :2] += torch.as_tensor(offset, device=pose.device, dtype=pose.dtype)
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=as_torch(obj.data.default_root_vel).clone())
        self.scene.reset()
        self.elapsed = 0.0
        self._refresh_cameras()
        return self.observe(), {}

    def restore_initial_state(self, state):
        """Restore the complete recorded state used to begin a demonstration."""
        required = {"joint_pos", "joint_vel", "cube_pose", "cube_velocity", "plate_pose", "plate_velocity"}
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Recorded initial state is missing: {sorted(missing)}")
        device = self.controller.target.device
        joint_pos = torch.as_tensor(state["joint_pos"], device=device, dtype=self.controller.target.dtype)[None]
        joint_vel = torch.as_tensor(state["joint_vel"], device=device, dtype=self.controller.target.dtype)[None]
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
        self.controller.set_target(joint_pos[:, self.controller.joint_ids])
        for name in ("cube", "plate"):
            obj = self.scene[name]
            pose = torch.as_tensor(state[f"{name}_pose"], device=device, dtype=joint_pos.dtype)[None]
            velocity = torch.as_tensor(state[f"{name}_velocity"], device=device, dtype=joint_pos.dtype)[None]
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=velocity)
        self.scene.reset()
        self.elapsed = 0.0
        self._refresh_cameras()
        return self.observe()

    def ik_action(self, targets_control, grippers, body_target, move_body=False):
        action = self.controller.target.clone()
        observation = self.observe()
        measured = observation["eef_pose"]
        body_pose = observation["body_pose"]
        # Target position is shared-body-relative, while orientation follows
        # the Quest controller absolutely in world space.
        targets = np.stack([compose_pose(body_pose, target) for target in targets_control])
        targets[:, 3:7] = targets_control[:, 3:7]
        jacobians = as_torch(self.robot.data.body_link_jacobian_w)
        joints = as_torch(self.robot.data.joint_pos)[0]
        # Fixed-base PhysX Jacobians exclude the root body, with no floating columns.
        if move_body:
            body_j = jacobians[0, self.body_id - 1][:, self.torso_ids]
            action[0, :6] = solve_ik(
                body_j, body_pose, body_target, joints[self.torso_ids], self.limits[self.torso_ids],
                self.control_dt, max_speed=self.velocity_limits[self.torso_ids],
            )
        for i, (body, ids, start) in enumerate(zip(self.eef_ids, self.arm_ids, (6, 14))):
            arm_j = jacobians[0, body - 1][:, ids]
            action[0, start:start + 7] = solve_ik(
                arm_j, measured[i], targets[i], joints[ids], self.limits[ids], self.control_dt,
                max_speed=self.velocity_limits[ids],
            )
        action[0, 13] = float(grippers[0])
        action[0, 21] = float(grippers[1])
        return numpy(action)[0]

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise ValueError("Ultra actions must be 22 finite joint positions")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.controller.set_target(torch.as_tensor(action[None], device=self.controller.target.device))
        for _ in range(2):
            self.controller.apply()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(0.02)
        self.elapsed += self.control_dt
        return self.observe(), 0.0, False, False, {}

    def initial_state(self):
        state = self.observe()
        for name in ("cube", "plate"):
            state[f"{name}_velocity"] = numpy(self.scene[name].data.root_vel_w)[0]
        return state
