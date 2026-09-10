"""Manager-based RL configurations for Ultra cube-on-plate evaluation."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from ultra_scene import ULTRA_CONTROLLED_JOINT_NAMES, UltraTabletopSceneCfg

from . import mdp

_ROBOT_CAMERA_CHAIN = "Geometry/world/fr30_1/fr30_2/fr30_3/fr30_4/fr30_5/fr30_6"
CAMERA_STREAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
_CAMERA_PATHS = {
    "head_rgb": f"{{ENV_REGEX_NS}}/Ultra/{_ROBOT_CAMERA_CHAIN}/zed_left",
    "left_wrist_rgb": (
        f"{{ENV_REGEX_NS}}/Ultra/{_ROBOT_CAMERA_CHAIN}/la_1/la_2/la_3/la_4/la_5/la_6/la_gripper/la_wrist_fisheye"
    ),
    "right_wrist_rgb": (
        f"{{ENV_REGEX_NS}}/Ultra/{_ROBOT_CAMERA_CHAIN}/ra_1/ra_2/ra_3/ra_4/ra_5/ra_6/ra_gripper/ra_wrist_fisheye"
    ),
}


def _robot_camera(prim_path: str, width: int = 320, height: int = 240) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=None,
        update_period=0.04,
        height=height,
        width=width,
        data_types=["rgb"],
    )


@configclass
class UltraRLSceneCfg(UltraTabletopSceneCfg):
    """Training scene with a local ground asset (no Nucleus dependency)."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.20), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )


@configclass
class UltraVisionSceneCfg(UltraRLSceneCfg):
    """Ultra scene with the three camera streams present in collected datasets."""

    head_camera = _robot_camera(_CAMERA_PATHS["head_rgb"])
    left_wrist_camera = _robot_camera(_CAMERA_PATHS["left_wrist_rgb"])
    right_wrist_camera = _robot_camera(_CAMERA_PATHS["right_wrist_rgb"])


@configclass
class ActionsCfg:
    """Absolute joint targets matching the HDF5 action contract exactly."""

    joint_position = JointPositionActionCfg(
        class_type=mdp.BoundedJointPositionAction,
        asset_name="robot",
        joint_names=list(ULTRA_CONTROLLED_JOINT_NAMES),
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Oracle state policy observations for non-visual RL experiments."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.proprioception,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(ULTRA_CONTROLLED_JOINT_NAMES), preserve_order=True
                )
            },
        )
        eef_pose_body = ObsTerm(
            func=mdp.eef_pose_body,
            params={
                "eef_cfg": SceneEntityCfg("robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True),
                "body_cfg": SceneEntityCfg("robot", body_names=["fr30_6"]),
            },
        )
        cube_pose = ObsTerm(func=mdp.object_pose, params={"asset_cfg": SceneEntityCfg("cube")})
        plate_pose = ObsTerm(func=mdp.object_pose, params={"asset_cfg": SceneEntityCfg("plate")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisionObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.proprioception,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(ULTRA_CONTROLLED_JOINT_NAMES), preserve_order=True
                )
            },
        )
        head_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("head_camera")})
        left_wrist_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("left_wrist_camera")})
        right_wrist_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("right_wrist_camera")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_to_recording_defaults, mode="reset")


@configclass
class RewardsCfg:
    cube_to_plate = RewTerm(func=mdp.cube_plate_distance, weight=1.0, params={"std": 0.15})
    lift_cube = RewTerm(func=mdp.cube_lifted, weight=0.5, params={"minimum_height": 0.84})
    place_success = RewTerm(
        func=mdp.cube_on_plate,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=["la_gripper_joint", "ra_gripper_joint"], preserve_order=True
            ),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True),
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ULTRA_CONTROLLED_JOINT_NAMES))},
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.SustainedCubeOnPlate,
        params={
            "hold_steps": 5,
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=["la_gripper_joint", "ra_gripper_joint"], preserve_order=True
            ),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["la_gripper", "ra_gripper"], preserve_order=True),
        },
    )
    cube_dropped = DoneTerm(func=mdp.cube_dropped, params={"minimum_height": 0.70})
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class UltraCubePlateEnvCfg(ManagerBasedRLEnvCfg):
    """State-based, vectorized Ultra cube-on-plate environment."""

    scene: UltraRLSceneCfg = UltraRLSceneCfg(num_envs=64, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 120.0
        self.is_finite_horizon = False
        self.sim = sim_utils.SimulationCfg(dt=0.02, render_interval=self.decimation)
        self.viewer.eye = (3.5, 5.0, 2.4)
        self.viewer.lookat = (0.0, 0.65, 0.80)
        # The generic capture camera is unrelated to the demonstration streams.
        self.scene.camera = None


@configclass
class UltraCubePlateVisionEnvCfg(UltraCubePlateEnvCfg):
    """Camera-enabled evaluation config matching the recorded RGB streams."""

    scene: UltraVisionSceneCfg = UltraVisionSceneCfg(num_envs=1, env_spacing=3.0)
    observations: VisionObservationsCfg = VisionObservationsCfg()
    enabled_cameras: tuple[str, ...] = CAMERA_STREAMS
    camera_width: int = 320
    camera_height: int = 240

    def __post_init__(self):
        super().__post_init__()
        configure_cameras(self, self.enabled_cameras, self.camera_width, self.camera_height)
        self.num_rerenders_on_reset = 1


def configure_cameras(
    cfg: UltraCubePlateVisionEnvCfg,
    enabled: tuple[str, ...] | list[str] = CAMERA_STREAMS,
    width: int = 320,
    height: int = 240,
) -> None:
    """Enable any subset of policy cameras on an existing vision config."""
    unknown = set(enabled).difference(CAMERA_STREAMS)
    if unknown:
        raise ValueError(f"Unknown camera streams: {sorted(unknown)}; choose from {CAMERA_STREAMS}")
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    enabled = set(enabled)
    for stream in CAMERA_STREAMS:
        sensor_name = stream.removesuffix("_rgb") + "_camera"
        sensor_cfg = _robot_camera(_CAMERA_PATHS[stream], width, height) if stream in enabled else None
        observation_cfg = (
            ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg(sensor_name)})
            if stream in enabled
            else None
        )
        setattr(cfg.scene, sensor_name, sensor_cfg)
        setattr(cfg.observations.policy, stream, observation_cfg)
