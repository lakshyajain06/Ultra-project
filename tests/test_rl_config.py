"""Fast regressions for the manager-based Ultra task; no simulator launch needed."""

import unittest
import math
from types import SimpleNamespace

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from data.schema import TASK_STATE_NAMES, ULTRA_ACTION_JOINT_NAMES
from sim import ULTRA_CONTROLLED_JOINT_NAMES
from sim.envs import (
    DualUltraEnvCfg,
    FacingDualUltraEnvCfg,
    UltraCubePlateEnvCfg,
    configure_cameras,
    configure_dual_cameras,
)
from sim.envs.dual_ultra import (
    CONTROLLED_ARMS,
    CUBE_START_POSITION,
    HANDOVER_POSITION,
    PLATE_POSITION,
    ROBOT_BASE_POSITIONS,
)
from sim.envs.dual_ultra import mdp as dual_mdp
from sim.envs.facing_dual_ultra import (
    FACING_CAMERA_PATHS,
    FACING_CUBE_START_POSITION,
    FACING_HANDOVER_POSITION,
    FACING_PLATE_POSITION,
    FACING_ROBOT_BASE_POSITIONS,
)
from sim.envs.cube_plate_pick_place import mdp


def _asset(**values):
    return SimpleNamespace(data=SimpleNamespace(**values))


class ManagerBasedRLConfigTests(unittest.TestCase):
    def test_action_and_timing_match_demonstrations(self):
        cfg = UltraCubePlateEnvCfg()
        self.assertEqual(cfg.decimation, 2)
        self.assertEqual(cfg.sim.dt, 0.02)
        self.assertEqual(cfg.episode_length_s, 120.0)
        self.assertEqual(len(cfg.actions.joint_position.joint_names), 22)
        self.assertFalse(cfg.actions.joint_position.use_default_offset)
        self.assertTrue(cfg.actions.joint_position.preserve_order)
        proprio = cfg.observations.policy.proprio.params
        self.assertEqual(proprio["torso_cfg"].joint_names, [f"torso_j{i}" for i in range(1, 7)])
        self.assertEqual(proprio["eef_cfg"].body_names, ["la_gripper", "ra_gripper"])
        self.assertEqual(proprio["gripper_cfg"].joint_names, ["la_gripper_joint", "ra_gripper_joint"])

    def test_registered_manager_environment(self):
        self.assertEqual(
            gym.spec("Isaac-Ultra-Cube-Plate-v0").kwargs["env_cfg_entry_point"],
            "sim.envs.cube_plate_pick_place.ultra_cube_plate_env_cfg:UltraCubePlateEnvCfg",
        )
        cfg = UltraCubePlateEnvCfg(enabled_cameras=("head_rgb", "left_wrist_rgb", "right_wrist_rgb"))
        self.assertEqual(cfg.scene.num_envs, 64)
        self.assertIsNotNone(cfg.scene.head_camera)
        self.assertEqual(cfg.scene.head_camera.width, 320)
        self.assertEqual(cfg.scene.head_camera.height, 240)
        self.assertIsNotNone(cfg.observations.policy.head_rgb)
        with self.assertRaises(gym.error.Error):
            gym.spec("Isaac-Ultra-Cube-Plate-Vision-v0")

    def test_registered_dual_ultra_environment(self):
        spec = gym.spec("Isaac-Dual-Ultra-Shared-Workspace-v0")
        self.assertEqual(
            spec.kwargs["env_cfg_entry_point"],
            "sim.envs.dual_ultra.dual_ultra_env_cfg:DualUltraEnvCfg",
        )
        cfg = DualUltraEnvCfg()
        self.assertEqual(len(cfg.actions.robot_left.joint_names), 8)
        self.assertEqual(len(cfg.actions.robot_right.joint_names), 8)
        self.assertEqual(cfg.actions.robot_left.joint_names[-1], "ra_gripper_joint")
        self.assertEqual(cfg.actions.robot_right.joint_names[-1], "la_gripper_joint")
        self.assertNotEqual(cfg.scene.robot_left.prim_path, cfg.scene.robot_right.prim_path)
        self.assertEqual(CONTROLLED_ARMS, (("robot_left", "ra"), ("robot_right", "la")))
        configure_dual_cameras(cfg, ("task_rgb",), width=160, height=120)
        self.assertEqual(cfg.scene.task_camera.width, 160)
        self.assertIsNotNone(cfg.observations.policy.task_rgb)
        self.assertIsNone(cfg.scene.left_controlled_wrist_camera)


    def test_handover_layout_has_exclusive_end_zones_and_shared_middle(self):
        left = ROBOT_BASE_POSITIONS["robot_left"]
        right = ROBOT_BASE_POSITIONS["robot_right"]
        self.assertLess(math.dist(left, CUBE_START_POSITION), 1.15)
        self.assertGreater(math.dist(right, CUBE_START_POSITION), 1.4)
        self.assertLess(math.dist(right, PLATE_POSITION), 1.15)
        self.assertGreater(math.dist(left, PLATE_POSITION), 1.4)
        self.assertLess(math.dist(left, HANDOVER_POSITION), 1.25)
        self.assertLess(math.dist(right, HANDOVER_POSITION), 1.25)

    def test_facing_handover_uses_opposed_robots_and_both_head_cameras(self):
        spec = gym.spec("Isaac-Facing-Dual-Ultra-Handover-v0")
        self.assertEqual(
            spec.kwargs["env_cfg_entry_point"],
            "sim.envs.facing_dual_ultra.facing_dual_ultra_env_cfg:FacingDualUltraEnvCfg",
        )
        cfg = FacingDualUltraEnvCfg(
            enabled_cameras=("robot_left_head_rgb", "robot_right_head_rgb"),
            camera_width=160,
            camera_height=120,
        )
        left_rotation = cfg.scene.robot_left.init_state.rot
        right_rotation = cfg.scene.robot_right.init_state.rot
        self.assertAlmostEqual(left_rotation[2], -right_rotation[2])
        self.assertEqual(set(FACING_CAMERA_PATHS), {
            "robot_left_head_rgb",
            "robot_left_left_wrist_rgb",
            "robot_left_right_wrist_rgb",
            "robot_right_head_rgb",
            "robot_right_left_wrist_rgb",
            "robot_right_right_wrist_rgb",
        })
        self.assertEqual(cfg.scene.robot_left_head_camera.width, 160)
        self.assertEqual(cfg.scene.robot_right_head_camera.height, 120)
        left = FACING_ROBOT_BASE_POSITIONS["robot_left"]
        right = FACING_ROBOT_BASE_POSITIONS["robot_right"]
        self.assertLess(math.dist(left, FACING_CUBE_START_POSITION), 1.15)
        self.assertGreater(math.dist(right, FACING_CUBE_START_POSITION), 1.4)
        self.assertLess(math.dist(right, FACING_PLATE_POSITION), 1.15)
        self.assertGreater(math.dist(left, FACING_PLATE_POSITION), 1.4)
        self.assertLess(math.dist(left, FACING_HANDOVER_POSITION), 1.25)
        self.assertLess(math.dist(right, FACING_HANDOVER_POSITION), 1.25)

    def test_success_requires_observed_pick_handover_and_stable_placement(self):
        cube_position = torch.tensor([[0.0, 0.0, 0.90]])
        zero = torch.zeros((1, 3))
        left_pose = torch.tensor([[[0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 1.0]]])
        right_pose = torch.tensor([[[1.0, 0.0, 0.90, 0.0, 0.0, 0.0, 1.0]]])
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene={
                "cube": _asset(
                    root_pos_w=cube_position,
                    root_lin_vel_w=zero.clone(),
                ),
                "plate": _asset(
                    root_pos_w=torch.tensor([[0.0, 0.0, 0.80]]),
                    root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                    root_lin_vel_w=zero.clone(),
                ),
                "robot_left": _asset(joint_pos=torch.tensor([[0.0]]), body_link_pose_w=left_pose),
                "robot_right": _asset(joint_pos=torch.tensor([[0.045]]), body_link_pose_w=right_pose),
            },
        )
        params = {
            "giver_eef_cfg": SceneEntityCfg("robot_left", body_ids=[0]),
            "giver_gripper_cfg": SceneEntityCfg("robot_left", joint_ids=[0]),
            "receiver_eef_cfg": SceneEntityCfg("robot_right", body_ids=[0]),
            "receiver_gripper_cfg": SceneEntityCfg("robot_right", joint_ids=[0]),
        }
        term = dual_mdp.SustainedHandoverPlacement(SimpleNamespace(), env)
        self.assertFalse(term(env, hold_steps=3, **params).item())
        env.scene["robot_left"].data.joint_pos[0, 0] = 0.045
        env.scene["robot_right"].data.joint_pos[0, 0] = 0.0
        right_pose[0, 0, :3] = cube_position[0]
        self.assertFalse(term(env, hold_steps=3, **params).item())
        cube_position[0] = torch.tensor([0.0, 0.0, 0.8375])
        left_pose[0, 0, :3] = 1.0
        right_pose[0, 0, :3] = 1.0
        env.scene["robot_right"].data.joint_pos[0, 0] = 0.045
        self.assertFalse(term(env, hold_steps=3, **params).item())
        self.assertFalse(term(env, hold_steps=3, **params).item())
        self.assertTrue(term(env, hold_steps=3, **params).item())

    def test_cameras_can_be_independently_configured(self):
        cfg = UltraCubePlateEnvCfg(enabled_cameras=("head_rgb",), camera_width=160, camera_height=120)
        self.assertEqual(cfg.scene.head_camera.width, 160)
        self.assertIsNone(cfg.scene.left_wrist_camera)
        self.assertIsNone(cfg.observations.policy.right_wrist_rgb)
        configure_cameras(cfg, ("right_wrist_rgb",), width=80, height=60)
        self.assertIsNone(cfg.scene.head_camera)
        self.assertEqual(cfg.scene.right_wrist_camera.width, 80)
        self.assertIsNotNone(cfg.observations.policy.right_wrist_rgb)
        with self.assertRaises(ValueError):
            configure_cameras(cfg, ("overhead_rgb",))

    def test_random_cube_reset_matches_teleop_range(self):
        cfg = UltraCubePlateEnvCfg(randomize_cube_position=True, cube_position_range_xy=(0.08, 0.04))
        self.assertEqual(
            cfg.events.reset_all.params,
            {"randomize_cube_position": True, "cube_position_range_xy": (0.08, 0.04)},
        )
        with self.assertRaises(ValueError):
            UltraCubePlateEnvCfg(cube_position_range_xy=(-0.01, 0.08))

    def test_success_requires_position_height_and_release(self):
        cube = torch.tensor([[0.0, 0.0, 0.85]])
        plate = torch.tensor([[0.0, 0.0, 0.80]])
        joints = torch.tensor([[0.045, 0.045]])
        zero_velocity = torch.zeros((1, 3))
        env = SimpleNamespace(
            scene={
                "cube": _asset(
                    root_pos_w=cube,
                    root_lin_vel_w=zero_velocity.clone(),
                    root_ang_vel_w=zero_velocity.clone(),
                ),
                "plate": _asset(
                    root_pos_w=plate,
                    root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                    root_lin_vel_w=zero_velocity.clone(),
                    root_ang_vel_w=zero_velocity.clone(),
                ),
                "robot": _asset(
                    joint_pos=joints,
                    body_link_pose_w=torch.tensor(
                        [[[0.0, 0.0, 0.87, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0, 0, 0, 1]]]
                    ),
                ),
            },
        )
        robot_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])
        gripper_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
        self.assertTrue(mdp.cube_on_plate(env, robot_cfg=robot_cfg, gripper_cfg=gripper_cfg).item())
        joints[0, 0] = 0.0
        self.assertFalse(mdp.cube_on_plate(env, robot_cfg=robot_cfg, gripper_cfg=gripper_cfg).item())
        env.scene["robot"].data.body_link_pose_w[0, 0, :3] = 1.0
        self.assertTrue(mdp.cube_on_plate(env, robot_cfg=robot_cfg, gripper_cfg=gripper_cfg).item())
        joints[0, 0] = 0.045
        cube[0, 0] = 0.2
        self.assertFalse(mdp.cube_on_plate(env, robot_cfg=robot_cfg, gripper_cfg=gripper_cfg).item())

    def test_success_rejects_relative_motion_and_requires_dwell(self):
        zero = torch.zeros((1, 3))
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene={
                "cube": _asset(
                    root_pos_w=torch.tensor([[0.0, 0.0, 0.85]]),
                    root_lin_vel_w=zero.clone(),
                    root_ang_vel_w=zero.clone(),
                ),
                "plate": _asset(
                    root_pos_w=torch.tensor([[0.0, 0.0, 0.80]]),
                    root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                    root_lin_vel_w=zero.clone(),
                    root_ang_vel_w=zero.clone(),
                ),
                "robot": _asset(
                    joint_pos=torch.tensor([[0.045, 0.045]]),
                    body_link_pose_w=torch.tensor([[[1.0, 1.0, 1.0, 0, 0, 0, 1], [1.0, 1.0, 1.0, 0, 0, 0, 1]]]),
                ),
            },
        )
        kwargs = {
            "robot_cfg": SceneEntityCfg("robot", joint_ids=[0, 1]),
            "gripper_cfg": SceneEntityCfg("robot", body_ids=[0, 1]),
        }
        env.scene["cube"].data.root_lin_vel_w[0, 0] = 0.2
        self.assertFalse(mdp.cube_on_plate(env, **kwargs).item())
        env.scene["cube"].data.root_lin_vel_w.zero_()
        term = mdp.SustainedCubeOnPlate(SimpleNamespace(), env)
        self.assertFalse(term(env, hold_steps=3, **kwargs).item())
        self.assertFalse(term(env, hold_steps=3, **kwargs).item())
        self.assertTrue(term(env, hold_steps=3, **kwargs).item())
        env.scene["cube"].data.root_pos_w[0, 0] = 0.2
        self.assertFalse(term(env, hold_steps=3, **kwargs).item())

    def test_proprioception_has_canonical_teleop_task_space_order(self):
        env = SimpleNamespace(
            scene={
                "robot": _asset(
                    joint_pos=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01, 0.02]]),
                    body_link_pose_w=torch.tensor(
                        [
                            [
                                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                                [2.0, 2.0, 3.0, 0.0, 0.0, 0.0, -1.0],
                                [1.0, 4.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                            ]
                        ]
                    ),
                )
            }
        )
        proprio = mdp.proprioception(
            env,
            torso_cfg=SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5]),
            eef_cfg=SceneEntityCfg("robot", body_ids=[1, 2]),
            body_cfg=SceneEntityCfg("robot", body_ids=[0]),
            gripper_cfg=SceneEntityCfg("robot", joint_ids=[6, 7]),
        )
        expected = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.02,
                ]
            ]
        )
        self.assertEqual(proprio.shape, (1, 22))
        torch.testing.assert_close(proprio, expected)
        self.assertGreaterEqual(proprio[0, 12].item(), 0.0)

    def test_learning_and_manager_contracts_match(self):
        cfg = UltraCubePlateEnvCfg()
        self.assertEqual(tuple(cfg.actions.joint_position.joint_names), ULTRA_CONTROLLED_JOINT_NAMES)
        self.assertEqual(ULTRA_ACTION_JOINT_NAMES, ULTRA_CONTROLLED_JOINT_NAMES)
        self.assertEqual(len(TASK_STATE_NAMES), 22)
        self.assertEqual(len(TASK_STATE_NAMES), len(ULTRA_CONTROLLED_JOINT_NAMES))


if __name__ == "__main__":
    unittest.main()
