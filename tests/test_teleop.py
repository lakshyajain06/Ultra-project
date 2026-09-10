"""Controller and dataset regressions; no Kit window or headset needed."""

import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ultra_scene.teleop.control import BodyTargetMapper, ClutchMapper, compose_pose, relative_pose, solve_ik
from ultra_scene.teleop.recording import EpisodeRecorder


class TeleopTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.zeros((2, 7), dtype=np.float32)
        self.pose[:, 6] = 1
        self.packet = np.zeros((2, 15), dtype=np.float32)
        self.packet[:, :7] = self.pose
        self.packet[:, 9] = 1
        self.mapper = ClutchMapper(scale=1)

    def engage(self):
        self.mapper.update(self.packet, self.pose, 0.04)
        self.packet[:, 8] = 1
        return self.mapper.update(self.packet, self.pose, 0.04)

    def test_clutch_no_jump_and_direct_position_target(self):
        self.packet[:, 0] = 10  # Physical hand origin must not teleport wrist.
        target, _ = self.engage()
        np.testing.assert_allclose(target, self.pose)
        self.packet[0, 0] += 1
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        self.assertAlmostEqual(target[0, 0], 1.0, places=6)
        np.testing.assert_allclose(target[1], self.pose[1])

    def test_tracking_loss_requires_release(self):
        self.engage()
        self.packet[0, 9] = 0
        self.mapper.update(self.packet, self.pose, 0.04)
        self.packet[0, 9] = 1
        self.packet[0, 0] = 2
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_allclose(target, self.pose)
        self.assertIsNone(self.mapper.anchor[0])
        self.packet[0, 8] = 0
        self.mapper.update(self.packet, self.pose, 0.04)
        self.packet[0, 8] = 1
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_allclose(target, self.pose)

    def test_pause_and_invalid_values_hold_gripper(self):
        self.engage()
        self.packet[:, 7] = 1
        _, grip = self.mapper.update(self.packet, self.pose, 0.04)
        self.packet[:, 7] = 0
        _, held = self.mapper.update(self.packet, self.pose, 0.04, active=False)
        np.testing.assert_array_equal(grip, held)
        self.packet[:, 0] = np.nan
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        self.assertTrue(np.isfinite(target).all())

    def test_singular_ik_is_finite_and_bounded(self):
        target = self.pose[0].copy()
        target[0] = 100
        limits = torch.tensor([[-0.01, 0.01]] * 7)
        for jac in (torch.zeros(6, 7), torch.eye(6, 7)):
            q = solve_ik(jac, self.pose[0], target, torch.zeros(7), limits, 0.04)
            self.assertTrue(torch.isfinite(q).all())
            self.assertTrue((q.abs() <= 0.01).all())

    def test_thumbsticks_integrate_body_target(self):
        body = BodyTargetMapper(planar_speed=0.1, vertical_speed=0.2, yaw_speed=1.0)
        measured = self.pose[0]
        self.packet[0, 13:15] = [0.5, 1.0]
        self.packet[1, 13:15] = [1.0, 0.5]
        target, active = body.update(self.packet, measured, 0.1)
        self.assertTrue(active)
        np.testing.assert_allclose(target[:3], [0.005, -0.01, 0.01], atol=1e-7)
        np.testing.assert_allclose(Rotation.from_quat(target[3:]).as_rotvec(), [0, 0, -0.1], atol=1e-7)
        self.packet[:, 13:15] = 0
        _, catching_up = body.update(self.packet, measured, 0.1)
        self.assertTrue(catching_up)
        _, settled = body.update(self.packet, target, 0.1)
        self.assertFalse(settled)

    def test_absolute_world_rotation_and_recalibration_gripper(self):
        self.pose[:, 3:7] = Rotation.from_euler("x", 45, degrees=True).as_quat()
        target, _ = self.engage()
        # The wrist's orientation at clutch time does not offset the controller.
        np.testing.assert_allclose(target[0, 3:7], [0, 0, 0, 1], atol=1e-6)
        self.packet[0, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_allclose(
            Rotation.from_quat(target[0, 3:7]).as_rotvec(), [0, 0, np.pi / 2], atol=1e-6,
        )
        # Re-clutching elsewhere must not redefine rotational zero.
        self.packet[0, 8] = 0
        self.mapper.update(self.packet, self.pose, 0.04)
        self.packet[0, 3:7] = Rotation.from_euler("y", 30, degrees=True).as_quat()
        self.packet[0, 8] = 1
        target, _ = self.mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_allclose(
            Rotation.from_quat(target[0, 3:7]).as_rotvec(), [0, np.pi / 6, 0], atol=1e-6,
        )
        self.mapper.reset(grippers=np.array([0.01, 0.02]))
        _, grip = self.mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_array_equal(grip, [0.01, 0.02])

    def test_fixed_local_tool_rotation_offset(self):
        mapper = ClutchMapper(rotation_offset_deg=(0, 0, 90))
        mapper.update(self.packet, self.pose, 0.04)
        self.packet[:, 8] = 1
        target, _ = mapper.update(self.packet, self.pose, 0.04)
        np.testing.assert_allclose(
            Rotation.from_quat(target[:, 3:7]).as_rotvec(),
            [[0, 0, np.pi / 2], [0, 0, np.pi / 2]], atol=1e-6,
        )

    def test_body_relative_target_moves_with_body(self):
        body0 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        wrist_world0 = np.array([0.2, -0.4, 0.1, 0, 0, 0, 1], dtype=np.float32)
        wrist_body = relative_pose(wrist_world0, body0)
        body1 = np.array([0, -0.3, 0, 0, 0, 0, 1], dtype=np.float32)
        wrist_world1 = compose_pose(body1, wrist_body)
        np.testing.assert_allclose(wrist_world1[:3], [0.2, -0.7, 0.1], atol=1e-7)

    def test_recording_alignment_labels_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demos.hdf5"
            recorder = EpisodeRecorder(path, {"test": True})
            image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            recorder.begin({"joint_pos": np.zeros(24), "head_rgb": image})
            recorder.append({"joint_pos": np.zeros(24), "head_rgb": image}, np.ones(22),
                            {"joint_pos": np.ones(24), "head_rgb": image},
                            self.packet, self.pose, 10.0, 0.04)
            recorder.finish("success")
            recorder.begin({"joint_pos": np.ones(24), "head_rgb": image})
            recorder.close()
            with h5py.File(path) as file:
                self.assertEqual(file.attrs["schema_version"], 4)
                demo = file["data/demo_000000"]
                self.assertEqual(demo["actions"].shape, (1, 22))
                self.assertEqual(demo["obs/joint_pos"][0, 0], 0)
                self.assertEqual(demo["next_obs/joint_pos"][0, 0], 1)
                np.testing.assert_array_equal(demo["obs/head_rgb"][0], image)
                self.assertEqual(demo["obs/head_rgb"].compression, "lzf")
                self.assertEqual(demo["initial_state/head_rgb"].compression, "lzf")
                self.assertTrue(demo.attrs["success"])
                self.assertEqual(file["data/demo_000001"].attrs["status"], "interrupted")
            with self.assertRaises(FileExistsError):
                EpisodeRecorder(path, {})

    def test_controller_graph_missing_tracking(self):
        try:
            from ultra_scene.teleop.input import PackControllers, build_pipeline
            from isaacteleop.retargeting_engine.interface import OptionalTensorGroup, OptionalType
            from isaacteleop.retargeting_engine.tensor_types import ControllerInput
        except ImportError:
            self.skipTest("Install the teleop extra to test the Isaac Teleop graph")
        node = PackControllers("test")
        inputs = {side: OptionalTensorGroup(OptionalType(ControllerInput())) for side in ("left", "right")}
        for group in inputs.values():
            group.set_none()
        outputs = node._allocate_outputs()
        node.compute(inputs, outputs)
        np.testing.assert_array_equal(np.asarray(outputs["action"][0]).reshape(2, 15)[:, 9], 0)
        self.assertIsNotNone(build_pipeline())


if __name__ == "__main__":
    unittest.main()
