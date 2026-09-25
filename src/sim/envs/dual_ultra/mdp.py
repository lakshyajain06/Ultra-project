"""MDP terms for single-arm pickup, handover, and placement."""

import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_unique, subtract_frame_transforms

from sim import ULTRA_CONTROLLED_JOINT_NAMES
from ..cube_plate_pick_place.mdp import camera_rgb


def _tensor(value):
    return getattr(value, "torch", value)


def selected_arm_proprioception(
    env: ManagerBasedEnv,
    left_eef_cfg: SceneEntityCfg,
    left_body_cfg: SceneEntityCfg,
    left_gripper_cfg: SceneEntityCfg,
    right_eef_cfg: SceneEntityCfg,
    right_body_cfg: SceneEntityCfg,
    right_gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return two body-relative wrist poses and gripper openings (16 values)."""
    values = []
    for eef_cfg, body_cfg, gripper_cfg in (
        (left_eef_cfg, left_body_cfg, left_gripper_cfg),
        (right_eef_cfg, right_body_cfg, right_gripper_cfg),
    ):
        robot = env.scene[eef_cfg.name]
        body_poses = _tensor(robot.data.body_link_pose_w)
        reference = body_poses[:, body_cfg.body_ids[0]]
        target = body_poses[:, eef_cfg.body_ids[0]]
        position, quaternion = subtract_frame_transforms(
            reference[:, :3], reference[:, 3:7], target[:, :3], target[:, 3:7]
        )
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
            torch.finfo(quaternion.dtype).eps
        )
        gripper = _tensor(robot.data.joint_pos)[:, gripper_cfg.joint_ids[0:1]]
        values.append(torch.cat((position, quat_unique(quaternion), gripper), dim=-1))
    return torch.cat(values, dim=-1)


def reset_dual_to_defaults(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Reset both articulations and shared task objects atomically."""
    for name in ("robot_left", "robot_right"):
        robot = env.scene[name]
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
        root_velocity = _tensor(obj.data.default_root_vel)[env_ids].clone()
        obj.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def _eef_cube_distance(env, eef_cfg):
    eef = _tensor(env.scene[eef_cfg.name].data.body_link_pose_w)[:, eef_cfg.body_ids[0], :3]
    cube = _tensor(env.scene["cube"].data.root_pos_w)
    return torch.linalg.vector_norm(eef - cube, dim=-1)


def gripper_cube_proximity(env: ManagerBasedRLEnv, eef_cfg: SceneEntityCfg, std: float = 0.20):
    return 1.0 - torch.tanh(_eef_cube_distance(env, eef_cfg) / std)


def cube_lifted(env: ManagerBasedRLEnv, minimum_height: float = 0.86):
    return (_tensor(env.scene["cube"].data.root_pos_w)[:, 2] > minimum_height).float()


def receiver_cube_proximity(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
    std: float = 0.20,
    minimum_height: float = 0.86,
):
    return gripper_cube_proximity(env, eef_cfg, std) * cube_lifted(env, minimum_height)


def _closed_near_cube(env, gripper_cfg, eef_cfg, close_position=0.02, grasp_distance=0.10):
    robot = env.scene[gripper_cfg.name]
    closed = _tensor(robot.data.joint_pos)[:, gripper_cfg.joint_ids[0]] <= close_position
    return closed & (_eef_cube_distance(env, eef_cfg) <= grasp_distance)


def _giver_released(env, gripper_cfg, eef_cfg, release_position=0.035, release_distance=0.12):
    robot = env.scene[gripper_cfg.name]
    opened = _tensor(robot.data.joint_pos)[:, gripper_cfg.joint_ids[0]] >= release_position
    return opened | (_eef_cube_distance(env, eef_cfg) >= release_distance)


def handover_complete(
    env: ManagerBasedRLEnv,
    giver_eef_cfg: SceneEntityCfg,
    giver_gripper_cfg: SceneEntityCfg,
    receiver_eef_cfg: SceneEntityCfg,
    receiver_gripper_cfg: SceneEntityCfg,
    minimum_height: float = 0.86,
):
    lifted = _tensor(env.scene["cube"].data.root_pos_w)[:, 2] > minimum_height
    received = _closed_near_cube(env, receiver_gripper_cfg, receiver_eef_cfg)
    released = _giver_released(env, giver_gripper_cfg, giver_eef_cfg)
    return (lifted & received & released).float()


def cube_plate_progress(
    env: ManagerBasedRLEnv,
    giver_eef_cfg: SceneEntityCfg,
    giver_gripper_cfg: SceneEntityCfg,
    receiver_eef_cfg: SceneEntityCfg,
    receiver_gripper_cfg: SceneEntityCfg,
    std: float = 0.25,
):
    cube = _tensor(env.scene["cube"].data.root_pos_w)
    plate = _tensor(env.scene["plate"].data.root_pos_w)
    target = plate.clone()
    target[:, 2] += 0.0375
    reward = 1.0 - torch.tanh(torch.linalg.vector_norm(cube - target, dim=-1) / std)
    return reward * handover_complete(
        env, giver_eef_cfg, giver_gripper_cfg, receiver_eef_cfg, receiver_gripper_cfg
    )


def cube_on_plate(
    env: ManagerBasedRLEnv,
    giver_eef_cfg: SceneEntityCfg,
    giver_gripper_cfg: SceneEntityCfg,
    receiver_eef_cfg: SceneEntityCfg,
    receiver_gripper_cfg: SceneEntityCfg,
    plate_radius: float = 0.14,
    plate_half_height: float = 0.0125,
    cube_half_extent: float = 0.025,
    edge_margin: float = 0.003,
    height_tolerance: float = 0.015,
    maximum_relative_linear_speed: float = 0.04,
):
    cube_data = env.scene["cube"].data
    plate_data = env.scene["plate"].data
    cube_pos = _tensor(cube_data.root_pos_w)
    plate_pos = _tensor(plate_data.root_pos_w)
    plate_quat = _tensor(plate_data.root_quat_w)
    offset_w = cube_pos - plate_pos
    offset_plate = quat_apply_inverse(plate_quat, offset_w)
    usable_radius = plate_radius - 2.0**0.5 * cube_half_extent - edge_margin
    footprint_ok = torch.linalg.vector_norm(offset_plate[:, :2], dim=-1) <= usable_radius
    height_ok = torch.abs(offset_plate[:, 2] - plate_half_height - cube_half_extent) <= height_tolerance
    up_w = quat_apply(
        plate_quat, torch.tensor([0.0, 0.0, 1.0], device=plate_quat.device).expand_as(offset_w)
    )
    stable = torch.linalg.vector_norm(
        _tensor(cube_data.root_lin_vel_w) - _tensor(plate_data.root_lin_vel_w), dim=-1
    ) <= maximum_relative_linear_speed
    giver_released = _giver_released(env, giver_gripper_cfg, giver_eef_cfg)
    receiver_released = _giver_released(env, receiver_gripper_cfg, receiver_eef_cfg)
    return footprint_ok & height_ok & (up_w[:, 2] >= 0.9) & stable & giver_released & receiver_released


class SustainedHandoverPlacement(ManagerTermBase):
    """Require only a stable released placement, independent of grasp history."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        targets = slice(None) if env_ids is None else env_ids
        self._stable_steps[targets] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        giver_eef_cfg: SceneEntityCfg,
        giver_gripper_cfg: SceneEntityCfg,
        receiver_eef_cfg: SceneEntityCfg,
        receiver_gripper_cfg: SceneEntityCfg,
        hold_steps: int = 5,
        minimum_height: float = 0.86,
    ):
        if hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        del minimum_height  # Retained in the call signature for configuration compatibility.
        placed = cube_on_plate(
            env, giver_eef_cfg, giver_gripper_cfg, receiver_eef_cfg, receiver_gripper_cfg
        )
        self._stable_steps = torch.where(
            placed,
            self._stable_steps + 1,
            torch.zeros_like(self._stable_steps),
        )
        return self._stable_steps >= hold_steps


def cube_dropped(env: ManagerBasedRLEnv, minimum_height: float = 0.70):
    return _tensor(env.scene["cube"].data.root_pos_w)[:, 2] < minimum_height


__all__ = [
    "SustainedHandoverPlacement",
    "camera_rgb",
    "cube_dropped",
    "cube_lifted",
    "cube_on_plate",
    "cube_plate_progress",
    "gripper_cube_proximity",
    "handover_complete",
    "receiver_cube_proximity",
    "reset_dual_to_defaults",
    "selected_arm_proprioception",
]
