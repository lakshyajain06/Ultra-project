"""Manager-based configuration for the two-Ultra shared workspace."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..cube_plate_pick_place.mdp import BoundedJointPositionAction
from . import mdp
from .dual_ultra_scene_cfg import DUAL_CAMERA_PATHS, DualUltraSceneCfg, dual_robot_camera_cfg


GIVER_EEF = SceneEntityCfg("robot_left", body_names=["ra_gripper"])
GIVER_BODY = SceneEntityCfg("robot_left", body_names=["fr30_6"])
GIVER_GRIPPER = SceneEntityCfg("robot_left", joint_names=["ra_gripper_joint"])
RECEIVER_EEF = SceneEntityCfg("robot_right", body_names=["la_gripper"])
RECEIVER_BODY = SceneEntityCfg("robot_right", body_names=["fr30_6"])
RECEIVER_GRIPPER = SceneEntityCfg("robot_right", joint_names=["la_gripper_joint"])
HANDOVER_PARAMS = {
    "giver_eef_cfg": GIVER_EEF,
    "giver_gripper_cfg": GIVER_GRIPPER,
    "receiver_eef_cfg": RECEIVER_EEF,
    "receiver_gripper_cfg": RECEIVER_GRIPPER,
}


@configclass
class ActionsCfg:
    robot_left = JointPositionActionCfg(
        class_type=BoundedJointPositionAction,
        asset_name="robot_left",
        joint_names=[*(f"ra_j{i}" for i in range(1, 8)), "ra_gripper_joint"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )
    robot_right = JointPositionActionCfg(
        class_type=BoundedJointPositionAction,
        asset_name="robot_right",
        joint_names=[*(f"la_j{i}" for i in range(1, 8)), "la_gripper_joint"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.selected_arm_proprioception,
            params={
                "left_eef_cfg": GIVER_EEF,
                "left_body_cfg": GIVER_BODY,
                "left_gripper_cfg": GIVER_GRIPPER,
                "right_eef_cfg": RECEIVER_EEF,
                "right_body_cfg": RECEIVER_BODY,
                "right_gripper_cfg": RECEIVER_GRIPPER,
            },
        )
        task_rgb = None
        left_controlled_wrist_rgb = None
        right_controlled_wrist_rgb = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_dual_to_defaults, mode="reset")


@configclass
class RewardsCfg:
    giver_reach_cube = RewTerm(
        func=mdp.gripper_cube_proximity,
        weight=1.0,
        params={"eef_cfg": GIVER_EEF, "std": 0.20},
    )
    lift_cube = RewTerm(func=mdp.cube_lifted, weight=2.0, params={"minimum_height": 0.86})
    receiver_reach_lifted_cube = RewTerm(
        func=mdp.receiver_cube_proximity,
        weight=2.0,
        params={"eef_cfg": RECEIVER_EEF, "std": 0.20, "minimum_height": 0.86},
    )
    handover = RewTerm(func=mdp.handover_complete, weight=5.0, params=HANDOVER_PARAMS)
    move_received_cube_to_plate = RewTerm(
        func=mdp.cube_plate_progress,
        weight=3.0,
        params={**HANDOVER_PARAMS, "std": 0.25},
    )
    place_success = RewTerm(
        func=mdp.SustainedHandoverPlacement,
        weight=15.0,
        params={**HANDOVER_PARAMS, "hold_steps": 5, "minimum_height": 0.86},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    left_joint_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot_left", joint_names=[*(f"ra_j{i}" for i in range(1, 8)), "ra_gripper_joint"]
            )
        },
    )
    right_joint_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot_right", joint_names=[*(f"la_j{i}" for i in range(1, 8)), "la_gripper_joint"]
            )
        },
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.SustainedHandoverPlacement,
        params={**HANDOVER_PARAMS, "hold_steps": 5, "minimum_height": 0.86},
    )
    cube_dropped = DoneTerm(func=mdp.cube_dropped, params={"minimum_height": 0.70})
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class DualUltraEnvCfg(ManagerBasedRLEnvCfg):
    scene: DualUltraSceneCfg = DualUltraSceneCfg(num_envs=32, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    enabled_cameras: tuple[str, ...] = ()
    camera_width: int = 320
    camera_height: int = 240

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 120.0
        self.is_finite_horizon = False
        self.sim = sim_utils.SimulationCfg(dt=0.02, render_interval=self.decimation)
        self.viewer.eye = (0.0, 3.8, 2.5)
        self.viewer.lookat = (0.0, 0.2, 0.85)
        configure_dual_cameras(self, self.enabled_cameras, self.camera_width, self.camera_height)
        self.num_rerenders_on_reset = int(bool(self.enabled_cameras))


def configure_dual_cameras(cfg, enabled, width=320, height=240):
    unknown = set(enabled).difference(DUAL_CAMERA_PATHS)
    if unknown:
        raise ValueError(f"Unknown dual camera streams: {sorted(unknown)}")
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    enabled = set(enabled)
    for stream in DUAL_CAMERA_PATHS:
        sensor_name = stream.removesuffix("_rgb") + "_camera"
        setattr(cfg.scene, sensor_name, dual_robot_camera_cfg(stream, width, height) if stream in enabled else None)
        setattr(
            cfg.observations.policy,
            stream,
            ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg(sensor_name)})
            if stream in enabled else None,
        )
