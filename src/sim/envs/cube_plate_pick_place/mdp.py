"""Manager terms for the Ultra cube-on-plate task."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_unique, subtract_frame_transforms

from sim import ULTRA_CONTROLLED_JOINT_NAMES


def _tensor(value):
    """Return the torch view exposed by Isaac Lab's tensor wrappers."""
    return getattr(value, "torch", value)


class BoundedJointPositionAction(JointPositionAction):
    """Absolute joint targets clipped to the articulation's authored limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = _tensor(self._asset.data.joint_pos_limits)[:, self._joint_ids]
        self._processed_actions = torch.clamp(self._processed_actions, min=limits[..., 0], max=limits[..., 1])


def proprioception(
    env: ManagerBasedEnv,
    torso_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[f"torso_j{i}" for i in range(1, 7)], preserve_order=True
    ),
    eef_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["fr30_6"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["la_gripper_joint", "ra_gripper_joint"], preserve_order=True
    ),
) -> torch.Tensor:
    """Return the canonical 22-D deployable Ultra state.

    Order is ``[torso_q(6), left_pose_body(7), left_gripper(1),
    right_pose_body(7), right_gripper(1)]``. Poses are XYZ plus canonical XYZW
    quaternion (non-negative W), expressed relative to ``fr30_6``. All values
    are measured or computed from measurable robot state; no object state is
    included.
    """
    robot = env.scene[torso_cfg.name]
    joint_pos = _tensor(robot.data.joint_pos)
    body_poses = _tensor(robot.data.body_link_pose_w)
    reference = body_poses[:, body_cfg.body_ids[0]]
    eef_poses = []
    for body_id in eef_cfg.body_ids:
        target = body_poses[:, body_id]
        position, quaternion = subtract_frame_transforms(
            reference[:, :3], reference[:, 3:7], target[:, :3], target[:, 3:7]
        )
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
            torch.finfo(quaternion.dtype).eps
        )
        eef_poses.append(torch.cat((position, quat_unique(quaternion)), dim=-1))
    return torch.cat(
        (
            joint_pos[:, torso_cfg.joint_ids],
            eef_poses[0],
            joint_pos[:, gripper_cfg.joint_ids[0:1]],
            eef_poses[1],
            joint_pos[:, gripper_cfg.joint_ids[1:2]],
        ),
        dim=-1,
    )


def camera_rgb(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return HWC RGB values as float32 in the dataset's 0..255 range."""
    rgb = _tensor(env.scene[sensor_cfg.name].data.output["rgb"])[..., :3]
    return rgb.to(dtype=torch.float32)


def reset_to_recording_defaults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    randomize_cube_position: bool = False,
    cube_position_range_xy: tuple[float, float] = (0.08, 0.08),
) -> None:
    """Match the teleop reset, optionally sampling the cube's XY offset."""
    robot = env.scene["robot"]
    joint_pos = _tensor(robot.data.default_joint_pos)[env_ids].clone()
    joint_vel = _tensor(robot.data.default_joint_vel)[env_ids].clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
    controlled_ids, _ = robot.find_joints(ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True)
    robot.set_joint_position_target_index(
        target=joint_pos[:, controlled_ids], joint_ids=controlled_ids, env_ids=env_ids
    )

    for name in ("cube", "plate"):
        obj = env.scene[name]
        root_pose = _tensor(obj.data.default_root_pose)[env_ids].clone()
        root_pose[:, :3] += env.scene.env_origins[env_ids]
        if name == "cube" and randomize_cube_position:
            extent = torch.as_tensor(cube_position_range_xy, device=root_pose.device, dtype=root_pose.dtype)
            root_pose[:, :2] += (2.0 * torch.rand((len(env_ids), 2), device=root_pose.device) - 1.0) * extent
        root_velocity = _tensor(obj.data.default_root_vel)[env_ids].clone()
        obj.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def cube_plate_distance(env: ManagerBasedRLEnv, std: float = 0.15) -> torch.Tensor:
    """Smooth reward for moving the cube center to its resting pose on the plate."""
    cube = _tensor(env.scene["cube"].data.root_pos_w)
    plate = _tensor(env.scene["plate"].data.root_pos_w)
    target = plate.clone()
    target[:, 2] += 0.0125 + 0.0375
    distance = torch.linalg.vector_norm(cube - target, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def cube_lifted(env: ManagerBasedRLEnv, minimum_height: float = 0.84) -> torch.Tensor:
    return (_tensor(env.scene["cube"].data.root_pos_w)[:, 2] > minimum_height).float()


def cube_on_plate(
    env: ManagerBasedRLEnv,
    plate_radius: float = 0.14,
    plate_half_height: float = 0.0125,
    cube_half_extent: float = 0.0375,
    edge_margin: float = 0.003,
    height_tolerance: float = 0.015,
    minimum_plate_up_dot: float = 0.9,
    maximum_relative_linear_speed: float = 0.04,
    maximum_relative_angular_speed: float = 0.5,
    release_position: float = 0.035,
    release_distance: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["la_gripper_joint", "ra_gripper_joint"], preserve_order=True
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True),
) -> torch.Tensor:
    """Return environments where a released cube is stably supported by the plate.

    This is a deterministic contact proxy: the cube center must be above the
    usable circular support area in the *plate frame*, at the face-on-face
    resting height, while cube/plate relative motion is small. Requiring the
    full cube footprint to fit prevents edge overlaps from counting. A gripper
    is released when its jaw is open or its palm is clear of the cube, so an
    unrelated closed hand does not block success.
    """
    cube_data = env.scene["cube"].data
    plate_data = env.scene["plate"].data
    cube_pos = _tensor(cube_data.root_pos_w)
    plate_pos = _tensor(plate_data.root_pos_w)
    plate_quat = _tensor(plate_data.root_quat_w)

    offset_w = cube_pos - plate_pos
    offset_plate = quat_apply_inverse(plate_quat, offset_w)
    usable_radius = plate_radius - 2.0**0.5 * cube_half_extent - edge_margin
    footprint_ok = torch.linalg.vector_norm(offset_plate[:, :2], dim=-1) <= usable_radius
    resting_height = plate_half_height + cube_half_extent
    height_ok = torch.abs(offset_plate[:, 2] - resting_height) <= height_tolerance

    up_w = quat_apply(plate_quat, torch.tensor([0.0, 0.0, 1.0], device=plate_quat.device).expand_as(offset_w))
    orientation_ok = up_w[:, 2] >= minimum_plate_up_dot

    cube_linear = _tensor(cube_data.root_lin_vel_w)
    plate_linear = _tensor(plate_data.root_lin_vel_w)
    plate_angular = _tensor(plate_data.root_ang_vel_w)
    support_point_velocity = plate_linear + torch.cross(plate_angular, offset_w, dim=-1)
    linear_stable = (
        torch.linalg.vector_norm(cube_linear - support_point_velocity, dim=-1) <= maximum_relative_linear_speed
    )
    angular_stable = (
        torch.linalg.vector_norm(_tensor(cube_data.root_ang_vel_w) - plate_angular, dim=-1)
        <= maximum_relative_angular_speed
    )

    robot = env.scene[robot_cfg.name]
    jaws_open = _tensor(robot.data.joint_pos)[:, robot_cfg.joint_ids] >= release_position
    gripper_pos = _tensor(robot.data.body_link_pose_w)[:, gripper_cfg.body_ids, :3]
    grippers_clear = torch.linalg.vector_norm(gripper_pos - cube_pos[:, None, :], dim=-1) >= release_distance
    released = torch.all(jaws_open | grippers_clear, dim=-1)
    return footprint_ok & height_ok & orientation_ok & linear_stable & angular_stable & released


class SustainedCubeOnPlate(ManagerTermBase):
    """Require the stable placement predicate for consecutive control steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._stable_steps.zero_()
        else:
            self._stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        hold_steps: int = 5,
        plate_radius: float = 0.14,
        plate_half_height: float = 0.0125,
        cube_half_extent: float = 0.0375,
        edge_margin: float = 0.003,
        height_tolerance: float = 0.015,
        minimum_plate_up_dot: float = 0.9,
        maximum_relative_linear_speed: float = 0.04,
        maximum_relative_angular_speed: float = 0.5,
        release_position: float = 0.035,
        release_distance: float = 0.10,
        robot_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", joint_names=["la_gripper_joint", "ra_gripper_joint"], preserve_order=True
        ),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True
        ),
    ) -> torch.Tensor:
        if hold_steps < 1:
            raise ValueError("hold_steps must be at least one")
        stable = cube_on_plate(
            env,
            plate_radius,
            plate_half_height,
            cube_half_extent,
            edge_margin,
            height_tolerance,
            minimum_plate_up_dot,
            maximum_relative_linear_speed,
            maximum_relative_angular_speed,
            release_position,
            release_distance,
            robot_cfg,
            gripper_cfg,
        )
        self._stable_steps = torch.where(stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        return self._stable_steps >= hold_steps


def cube_dropped(env: ManagerBasedRLEnv, minimum_height: float = 0.70) -> torch.Tensor:
    return _tensor(env.scene["cube"].data.root_pos_w)[:, 2] < minimum_height
