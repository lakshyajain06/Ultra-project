"""Canonical observation and action contracts shared across workflows."""

ULTRA_ACTION_JOINT_NAMES = (
    *(f"torso_j{i}" for i in range(1, 7)),
    *(f"la_j{i}" for i in range(1, 8)),
    "la_gripper_joint",
    *(f"ra_j{i}" for i in range(1, 8)),
    "ra_gripper_joint",
)

TASK_STATE_NAMES = (
    *(f"torso_j{i}" for i in range(1, 7)),
    *(f"left_eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "left_gripper_opening",
    *(f"right_eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "right_gripper_opening",
)

DUAL_ULTRA_ACTION_JOINT_NAMES = (
    *(f"robot_left/ra_j{i}" for i in range(1, 8)),
    "robot_left/ra_gripper_joint",
    *(f"robot_right/la_j{i}" for i in range(1, 8)),
    "robot_right/la_gripper_joint",
)

DUAL_TASK_STATE_NAMES = (
    *(f"robot_left/eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "robot_left/gripper_opening",
    *(f"robot_right/eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "robot_right/gripper_opening",
)
