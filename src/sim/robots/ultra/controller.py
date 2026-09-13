"""Joint-position controller matching Ultra's source control interface."""

import torch


# Ordered exactly as configs/robots/ultra.yaml in the source environment. The
# two ``*_jaw_l_joint`` DOFs are followers and are intentionally not commanded.
ULTRA_CONTROLLED_JOINT_NAMES = (
    *(f"torso_j{i}" for i in range(1, 7)),
    *(f"la_j{i}" for i in range(1, 8)),
    "la_gripper_joint",
    *(f"ra_j{i}" for i in range(1, 8)),
    "ra_gripper_joint",
)


def as_torch(value):
    """Return a torch tensor from either standard or wrapped Isaac Lab data."""
    return getattr(value, "torch", value)


class UltraJointPositionController:
    """Resolve and command Ultra's 22 independently controlled joints."""

    def __init__(self, robot):
        self.robot = robot
        self.joint_ids, names = robot.find_joints(
            ULTRA_CONTROLLED_JOINT_NAMES, preserve_order=True
        )
        if tuple(names) != ULTRA_CONTROLLED_JOINT_NAMES:
            raise RuntimeError(f"Ultra control contract mismatch: {names}")
        self.joint_names = tuple(names)
        self.target = as_torch(robot.data.default_joint_pos)[:, self.joint_ids].clone()

    def reset(self):
        """Restore the controlled joint state and target to the configured pose."""
        positions = as_torch(self.robot.data.default_joint_pos)[:, self.joint_ids].clone()
        velocities = as_torch(self.robot.data.default_joint_vel)[:, self.joint_ids].clone()
        self.robot.write_joint_state_to_sim_index(
            position=positions,
            velocity=velocities,
            joint_ids=self.joint_ids,
        )
        self.target.copy_(positions)

    def set_target(self, target):
        """Set a batched target in the source environment's joint order."""
        if target.shape != self.target.shape:
            raise ValueError(f"Expected Ultra target shape {self.target.shape}, got {target.shape}")
        self.target.copy_(target)

    def apply(self):
        """Write the current target into Isaac Lab's command buffer."""
        self.robot.set_joint_position_target_index(
            target=self.target,
            joint_ids=self.joint_ids,
        )

    def error(self):
        """Return absolute position error for the controlled joints."""
        positions = as_torch(self.robot.data.joint_pos)[:, self.joint_ids]
        return torch.abs(positions - self.target)
