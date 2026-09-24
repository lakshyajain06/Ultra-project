"""Single-instance two-Ultra teleoperation environment."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from sim import UltraJointPositionController
from sim.envs.dual_ultra import CONTROLLED_ARMS, DUAL_CAMERA_PATHS, DualUltraSceneCfg, dual_robot_camera_cfg
from sim.envs.facing_dual_ultra import (
    FACING_CAMERA_PATHS,
    FACING_CONTROLLED_ARMS,
    FacingDualUltraSceneCfg,
    facing_robot_camera_cfg,
)
from sim.robots.ultra import as_torch
from .control import compose_pose, relative_pose, solve_ik
from .env import numpy


@dataclass
class _RobotHandles:
    name: str
    arm: str
    robot: object
    controller: UltraJointPositionController
    body_id: int
    selected_eef_id: int
    selected_arm_ids: list[int]
    all_eef_ids: list[int]
    torso_ids: list[int]
    gripper_ids: list[int]
    limits: torch.Tensor
    velocity_limits: torch.Tensor


class DualUltraTeleopEnv(gym.Env):
    """Two single-arm Ultras; Quest hands operate the two inner arms only."""

    metadata = {"render_modes": []}
    control_dt = 0.04
    action_dim = 16
    proprio_dim = 16

    def __init__(
        self,
        device="cuda:0",
        enable_cameras=False,
        camera_width=320,
        camera_height=240,
        layout="side_by_side",
    ):
        if layout == "side_by_side":
            scene_type = DualUltraSceneCfg
            controlled_arms = CONTROLLED_ARMS
            camera_paths = DUAL_CAMERA_PATHS
            camera_factory = dual_robot_camera_cfg
            viewer_eye = (0.0, 3.8, 2.5)
            viewer_target = (0.0, 0.2, 0.85)
        elif layout == "facing":
            scene_type = FacingDualUltraSceneCfg
            controlled_arms = FACING_CONTROLLED_ARMS
            camera_paths = FACING_CAMERA_PATHS
            camera_factory = facing_robot_camera_cfg
            viewer_eye = (1.9, 0.2, 2.2)
            viewer_target = (0.0, 0.2, 0.85)
        else:
            raise ValueError(f"Unknown dual-Ultra layout: {layout}")
        self.layout = layout
        self.camera_paths = camera_paths
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02, render_interval=1, device=device))
        self.sim.set_camera_view(eye=viewer_eye, target=viewer_target)
        cfg = scene_type(num_envs=1, env_spacing=4.0)
        self.camera_names = ()
        if enable_cameras:
            if camera_width <= 0 or camera_height <= 0:
                raise ValueError("Camera width and height must be positive")
            for stream in camera_paths:
                name = stream.removesuffix("_rgb") + "_camera"
                setattr(cfg, name, camera_factory(stream, camera_width, camera_height))
            if layout == "facing":
                # The task camera anchors XR but is deliberately not part of
                # the six recorded/policy streams.
                cfg.task_camera = camera_factory("task_rgb", camera_width, camera_height)
            self.camera_names = tuple(stream.removesuffix("_rgb") + "_camera" for stream in camera_paths)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        if enable_cameras:
            task_eye = (1.75, 0.20, 1.75) if layout == "facing" else (0.0, 2.4, 1.75)
            self.scene["task_camera"].set_world_poses_from_view(
                np.asarray([task_eye], dtype=np.float32),
                np.asarray([[0.0, 0.15, 0.86]], dtype=np.float32),
            )

        self.robots = tuple(self._resolve_robot(name, arm) for name, arm in controlled_arms)
        action_limits = np.concatenate([
            numpy(handles.limits[[*handles.selected_arm_ids, handles.gripper_ids[0 if handles.arm == "la" else 1]]])
            for handles in self.robots
        ])
        self.action_space = gym.spaces.Box(action_limits[:, 0], action_limits[:, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "proprio": gym.spaces.Box(-np.inf, np.inf, (16,), np.float32),
            "joint_pos": gym.spaces.Box(-np.inf, np.inf, (48,), np.float32),
            "joint_vel": gym.spaces.Box(-np.inf, np.inf, (48,), np.float32),
            "joint_target": gym.spaces.Box(-np.inf, np.inf, (44,), np.float32),
            "eef_pose": gym.spaces.Box(-np.inf, np.inf, (2, 7), np.float32),
            "eef_pose_body": gym.spaces.Box(-np.inf, np.inf, (2, 7), np.float32),
            "body_pose": gym.spaces.Box(-np.inf, np.inf, (2, 7), np.float32),
            "cube_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "plate_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        })
        for name in self.camera_names:
            self.observation_space[name.removesuffix("_camera") + "_rgb"] = gym.spaces.Box(
                0, 255, (camera_height, camera_width, 3), np.uint8
            )
        self.elapsed = 0.0

    def _resolve_robot(self, name, arm):
        robot = self.scene[name]
        controller = UltraJointPositionController(robot)
        body_ids, body_names = robot.find_bodies(["fr30_6"], preserve_order=True)
        eef_ids, eef_names = robot.find_bodies(["la_gripper", "ra_gripper"], preserve_order=True)
        if body_names != ["fr30_6"] or eef_names != ["la_gripper", "ra_gripper"]:
            raise RuntimeError(f"Unexpected links for {name}: body={body_names}, eef={eef_names}")
        arm_ids = robot.find_joints([f"{arm}_j{i}" for i in range(1, 8)], preserve_order=True)[0]
        torso_ids = robot.find_joints([f"torso_j{i}" for i in range(1, 7)], preserve_order=True)[0]
        gripper_ids = [int(controller.joint_ids[13]), int(controller.joint_ids[21])]
        return _RobotHandles(
            name=name,
            arm=arm,
            robot=robot,
            controller=controller,
            body_id=body_ids[0],
            selected_eef_id=eef_ids[0 if arm == "la" else 1],
            selected_arm_ids=arm_ids,
            all_eef_ids=eef_ids,
            torso_ids=torso_ids,
            gripper_ids=gripper_ids,
            limits=as_torch(robot.data.joint_pos_limits)[0],
            velocity_limits=as_torch(robot.data.joint_vel_limits)[0],
        )

    @property
    def action_joint_names(self):
        return tuple(
            f"{handles.name}/{joint}"
            for handles in self.robots
            for joint in (
                *(f"{handles.arm}_j{i}" for i in range(1, 8)),
                f"{handles.arm}_gripper_joint",
            )
        )

    @property
    def observation_joint_names(self):
        return tuple(f"{handles.name}/{joint}" for handles in self.robots for joint in handles.robot.joint_names)

    def current_action(self):
        """Return the two selected arm-and-gripper targets in policy order."""
        values = []
        for handles in self.robots:
            target = numpy(handles.controller.target)[0]
            action_start = 6 if handles.arm == "la" else 14
            gripper_index = 13 if handles.arm == "la" else 21
            values.append(np.concatenate((
                target[action_start:action_start + 7], target[gripper_index:gripper_index + 1]
            )))
        return np.concatenate(values)

    def observe(self):
        per_robot = []
        selected_world = []
        selected_body = []
        body_poses = []
        joint_positions = []
        joint_velocities = []
        targets = []
        observation = {}
        for handles in self.robots:
            robot = handles.robot
            body_pose = numpy(robot.data.body_link_pose_w)[0, handles.body_id]
            all_eef = numpy(robot.data.body_link_pose_w)[0, handles.all_eef_ids]
            all_eef_body = np.stack([relative_pose(pose, body_pose) for pose in all_eef])
            joint_pos = numpy(robot.data.joint_pos)[0]
            joint_vel = numpy(robot.data.joint_vel)[0]
            selected_index = 0 if handles.arm == "la" else 1
            selected_world.append(all_eef[selected_index])
            selected_body.append(all_eef_body[selected_index])
            per_robot.append(np.concatenate((
                all_eef_body[selected_index],
                joint_pos[handles.gripper_ids[selected_index:selected_index + 1]],
            )))
            body_poses.append(body_pose)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)
            targets.append(numpy(handles.controller.target)[0])
            prefix = handles.name
            observation[f"{prefix}_joint_pos"] = joint_pos
            observation[f"{prefix}_joint_vel"] = joint_vel
            observation[f"{prefix}_eef_pose_body"] = all_eef_body
        observation.update({
            "proprio": np.concatenate(per_robot),
            "joint_pos": np.concatenate(joint_positions),
            "joint_vel": np.concatenate(joint_velocities),
            "joint_target": np.concatenate(targets),
            "eef_pose": np.stack(selected_world),
            "eef_pose_body": np.stack(selected_body),
            "body_pose": np.stack(body_poses),
            "cube_pose": numpy(self.scene["cube"].data.root_pose_w)[0],
            "plate_pose": numpy(self.scene["plate"].data.root_pose_w)[0],
        })
        for name in self.camera_names:
            rgb = numpy(self.scene[name].data.output["rgb"])[0]
            observation[name.removesuffix("_camera") + "_rgb"] = np.asarray(rgb[..., :3], dtype=np.uint8)
        return observation

    def _refresh_cameras(self):
        if not self.camera_names:
            return
        self.sim.forward()
        self.sim.render()
        for name in self.camera_names:
            self.scene[name].update(0.0, force_recompute=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for handles in self.robots:
            handles.controller.reset()
        for name in ("cube", "plate"):
            obj = self.scene[name]
            pose = as_torch(obj.data.default_root_pose).clone()
            pose[:, :3] += as_torch(self.scene.env_origins)
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=as_torch(obj.data.default_root_vel).clone())
        self.scene.reset()
        self.elapsed = 0.0
        self._refresh_cameras()
        return self.observe(), {}

    def restore_initial_state(self, state):
        """Restore a complete schema-7 dual-robot episode start state."""
        required = {"cube_pose", "cube_velocity", "plate_pose", "plate_velocity"}
        for handles in self.robots:
            required.update({f"{handles.name}_joint_pos", f"{handles.name}_joint_vel"})
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Recorded dual initial state is missing: {sorted(missing)}")
        for handles in self.robots:
            device = handles.controller.target.device
            dtype = handles.controller.target.dtype
            joint_pos = torch.as_tensor(
                state[f"{handles.name}_joint_pos"], device=device, dtype=dtype
            )[None]
            joint_vel = torch.as_tensor(
                state[f"{handles.name}_joint_vel"], device=device, dtype=dtype
            )[None]
            handles.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
            handles.controller.set_target(joint_pos[:, handles.controller.joint_ids])
        for name in ("cube", "plate"):
            obj = self.scene[name]
            device = self.robots[0].controller.target.device
            dtype = self.robots[0].controller.target.dtype
            pose = torch.as_tensor(state[f"{name}_pose"], device=device, dtype=dtype)[None]
            velocity = torch.as_tensor(state[f"{name}_velocity"], device=device, dtype=dtype)[None]
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=velocity)
        self.scene.reset()
        self.elapsed = 0.0
        self._refresh_cameras()
        return self.observe()

    def ik_action(self, targets_control, grippers, active_arms):
        actions = [handles.controller.target.clone() for handles in self.robots]
        observation = self.observe()
        for index, handles in enumerate(self.robots):
            if not active_arms[index]:
                continue
            target_world = compose_pose(observation["body_pose"][index], targets_control[index])
            target_world[3:7] = targets_control[index, 3:7]
            jacobians = as_torch(handles.robot.data.body_link_jacobian_w)
            joints = as_torch(handles.robot.data.joint_pos)[0]
            arm_j = jacobians[0, handles.selected_eef_id - 1][:, handles.selected_arm_ids]
            action_start = 6 if handles.arm == "la" else 14
            actions[index][0, action_start:action_start + 7] = solve_ik(
                arm_j,
                observation["eef_pose"][index],
                target_world,
                joints[handles.selected_arm_ids],
                handles.limits[handles.selected_arm_ids],
                self.control_dt,
                max_speed=handles.velocity_limits[handles.selected_arm_ids],
            )
            gripper_index = 13 if handles.arm == "la" else 21
            actions[index][0, gripper_index] = float(grippers[index])
        selected_actions = []
        for action, handles in zip(actions, self.robots):
            action_start = 6 if handles.arm == "la" else 14
            gripper_index = 13 if handles.arm == "la" else 21
            selected_actions.append(np.concatenate((
                numpy(action)[0, action_start:action_start + 7],
                numpy(action)[0, gripper_index:gripper_index + 1],
            )))
        return np.concatenate(selected_actions)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Dual Ultra actions must be {self.action_dim} finite joint positions")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for index, handles in enumerate(self.robots):
            selected = action[8 * index:8 * (index + 1)]
            target = handles.controller.target.clone()
            action_start = 6 if handles.arm == "la" else 14
            gripper_index = 13 if handles.arm == "la" else 21
            target[0, action_start:action_start + 7] = torch.as_tensor(
                selected[:7], device=target.device, dtype=target.dtype
            )
            target[0, gripper_index] = float(selected[7])
            handles.controller.set_target(target)
        for _ in range(2):
            for handles in self.robots:
                handles.controller.apply()
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
