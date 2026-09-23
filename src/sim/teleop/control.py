"""Relative clutch mapping and damped least-squares IK; no simulator imports."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def build_task_proprioception(joint_pos, eef_pose_body, torso_ids, gripper_ids):
    """Build the canonical 22-D task state stored in demonstrations."""
    joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    eef_pose_body = np.asarray(eef_pose_body, dtype=np.float32).copy()
    if eef_pose_body.shape != (2, 7):
        raise ValueError(f"eef_pose_body must have shape (2, 7), got {eef_pose_body.shape}")
    torso_ids = np.asarray(torso_ids, dtype=np.int64)
    gripper_ids = np.asarray(gripper_ids, dtype=np.int64)
    if torso_ids.shape != (6,) or gripper_ids.shape != (2,):
        raise ValueError("torso_ids must contain 6 joints and gripper_ids must contain 2 joints")
    all_ids = np.concatenate((torso_ids, gripper_ids))
    if np.any(all_ids < 0) or np.max(all_ids) >= joint_pos.size:
        raise ValueError("joint ids are outside the articulation state")
    quaternions = eef_pose_body[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < np.finfo(np.float32).eps):
        raise ValueError("eef_pose_body contains a zero-length quaternion")
    quaternions /= norms
    eef_pose_body[:, 3:7] = np.where(quaternions[:, 3:4] < 0.0, -quaternions, quaternions)
    state = np.concatenate((
        joint_pos[torso_ids],
        eef_pose_body[0], joint_pos[gripper_ids[0:1]],
        eef_pose_body[1], joint_pos[gripper_ids[1:2]],
    ))
    if state.shape != (22,) or not np.isfinite(state).all():
        raise ValueError("task proprioception must contain 22 finite values")
    return state


class ClutchMapper:
    def __init__(self, scale=1.0, rotation_offset_deg=None):
        self.scale = scale
        # A supplied offset is fixed in controller-local XYZ. Otherwise each
        # hand calibrates once against the current robot wrist orientation.
        configured = None if rotation_offset_deg is None else Rotation.from_euler(
            "xyz", rotation_offset_deg, degrees=True,
        )
        self.rotation_offsets = [configured, configured]
        self.reset()

    def reset(self, grippers=None, recalibrate_orientation=False):
        self.anchor = [None, None]
        self.target = None
        self.grippers = np.full(2, 0.045) if grippers is None else np.asarray(grippers).copy()
        if recalibrate_orientation:
            self.rotation_offsets = [None, None]

    def rotation_offsets_xyzw(self):
        """Return per-controller calibration quaternions, or NaNs until valid."""
        values = np.full((2, 4), np.nan, dtype=np.float32)
        for i, offset in enumerate(self.rotation_offsets):
            if offset is not None:
                values[i] = offset.as_quat()
        return values

    def update(self, packet, measured, dt, active=True, reference_pose=None):
        """Re-anchor on every engagement; releasing or losing tracking holds position.

        Require release after invalid tracking / pause before moving again.
        Grippers also require the corresponding clutch to prevent unintended drops.
        """
        if reference_pose is None:
            reference_poses = None
        else:
            reference_poses = np.asarray(reference_pose)
            if reference_poses.shape == (7,):
                reference_poses = np.repeat(reference_poses[None], len(packet), axis=0)
            if reference_poses.shape != (len(packet), 7):
                raise ValueError(
                    f"reference_pose must have shape (7,) or ({len(packet)}, 7), got {reference_poses.shape}"
                )
        if self.target is None:
            self.target = measured.copy()
            self.armed = [False, False]
        for i, row in enumerate(packet):
            valid = bool(row[9]) and np.isfinite(row).all() and np.linalg.norm(row[3:7]) > 0.5
            if not active or not valid:
                self.anchor[i] = None
                self.armed[i] = False
                continue
            if self.rotation_offsets[i] is None:
                reference_rot = (
                    Rotation.identity() if reference_poses is None
                    else Rotation.from_quat(reference_poses[i, 3:7])
                )
                measured_world = reference_rot * Rotation.from_quat(measured[i, 3:7])
                self.rotation_offsets[i] = Rotation.from_quat(row[3:7]).inv() * measured_world
            if row[8] < 0.5:
                self.anchor[i] = None
                self.armed[i] = True
                continue
            if not self.armed[i]:
                continue
            if self.anchor[i] is None:
                # Use held target, so re-clutching cannot snap to a different wrist pose.
                reference_rot = (
                    Rotation.identity() if reference_poses is None
                    else Rotation.from_quat(reference_poses[i, 3:7])
                )
                self.anchor[i] = (row[:7].copy(), self.target[i].copy(), reference_rot)
            hand0, wrist0, reference_rot = self.anchor[i]
            # Translation is clutched and body-relative. Orientation is not
            # re-anchored: the controller's world orientation is the absolute
            # wrist orientation, so re-gripping does not redefine rotational zero.
            hand_delta = reference_rot.inv().apply(row[:3] - hand0[:3])
            self.target[i, :3] = wrist0[:3] + self.scale * hand_delta
            self.target[i, 3:7] = (
                Rotation.from_quat(row[3:7]) * self.rotation_offsets[i]
            ).as_quat()
            desired_grip = 0.045 * (1.0 - np.clip(row[7], 0.0, 1.0))
            self.grippers[i] += np.clip(desired_grip - self.grippers[i], -0.08 * dt, 0.08 * dt)
        return self.target.copy(), self.grippers.copy()


class BodyTargetMapper:
    """Integrate Quest thumbsticks into a shared-body Cartesian pose target."""

    def __init__(self, planar_speed=0.12, vertical_speed=0.08, yaw_speed=0.5, deadzone=0.15):
        self.planar_speed = planar_speed
        self.vertical_speed = vertical_speed
        self.yaw_speed = yaw_speed
        self.deadzone = deadzone
        self.target = None

    def reset(self):
        self.target = None

    def _axis(self, value):
        value = float(value)
        return 0.0 if abs(value) < self.deadzone else value

    def update(self, packet, measured, dt, active=True):
        if self.target is None:
            self.target = measured.copy()
        axes = np.array([self._axis(packet[i, j]) for i, j in ((0, 13), (0, 14), (1, 13), (1, 14))])
        if active:
            # Left stick: support left/right and forward/back. Ultra faces -Y.
            self.target[:3] += dt * np.array([
                self.planar_speed * axes[0], -self.planar_speed * axes[1], self.vertical_speed * axes[3]
            ])
            yaw = -self.yaw_speed * axes[2] * dt
            self.target[3:7] = (Rotation.from_euler("z", yaw) * Rotation.from_quat(self.target[3:7])).as_quat()
        # Keep solving after the stick is released until the body reaches the
        # integrated pose target.  Otherwise the torso can permanently lag a
        # fast stick command by one or more IK increments.
        position_error = np.linalg.norm(self.target[:3] - measured[:3])
        rotation_error = (
            Rotation.from_quat(self.target[3:7]) * Rotation.from_quat(measured[3:7]).inv()
        ).magnitude()
        tracking = position_error > 1e-3 or rotation_error > 1e-2
        return self.target.copy(), bool(active and (np.any(axes != 0) or tracking))


def pose_error(measured_pose, target_pose):
    """Bounded world-frame position and orientation error."""
    error = np.concatenate((
        target_pose[:3] - measured_pose[:3],
        (Rotation.from_quat(target_pose[3:]) * Rotation.from_quat(measured_pose[3:]).inv()).as_rotvec(),
    ))
    # Bound errors so unreachable Cartesian targets cannot produce large commands.
    for part, bound in ((slice(0, 3), 0.025), (slice(3, 6), 0.10)):
        error[part] *= min(1.0, bound / max(np.linalg.norm(error[part]), 1e-9))
    return error


def solve_ik(jacobian, measured_pose, target_pose, joint_pos, limits, dt, max_speed=0.6):
    """One world-frame arm-only DLS update (kept as a focused/testable primitive)."""
    error = pose_error(measured_pose, target_pose)
    e = torch.as_tensor(error, device=jacobian.device, dtype=jacobian.dtype)
    dq = jacobian.T @ torch.linalg.solve(
        jacobian @ jacobian.T + 0.05**2 * torch.eye(6, device=jacobian.device), e,
    )
    max_step = torch.as_tensor(max_speed, device=dq.device, dtype=dq.dtype) * dt
    dq = torch.maximum(-max_step, torch.minimum(max_step, dq))
    result = joint_pos + dq
    return torch.maximum(limits[:, 0], torch.minimum(limits[:, 1], result))


def relative_pose(world_pose, reference_pose):
    """Express an XYZW pose in a reference pose's coordinate frame."""
    reference_rot = Rotation.from_quat(reference_pose[3:7])
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = reference_rot.inv().apply(world_pose[:3] - reference_pose[:3])
    pose[3:7] = (reference_rot.inv() * Rotation.from_quat(world_pose[3:7])).as_quat()
    return pose


def compose_pose(reference_pose, relative):
    """Transform a reference-relative XYZW pose into world coordinates."""
    reference_rot = Rotation.from_quat(reference_pose[3:7])
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = reference_pose[:3] + reference_rot.apply(relative[:3])
    pose[3:7] = (reference_rot * Rotation.from_quat(relative[3:7])).as_quat()
    return pose
