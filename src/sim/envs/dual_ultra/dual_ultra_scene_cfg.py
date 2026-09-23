"""Two Ultra robots sharing one wide tabletop workspace."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from ...robots.ultra import ULTRA_CFG
from ..cube_plate_pick_place.ultra_tabletop_scene_cfg import _dynamic_material, _static_box


# Spatially natural operator mapping: each Quest hand drives the robot on the
# same side of the operator, using the two arms nearest the shared workspace.
CONTROLLED_ARMS = (("robot_left", "ra"), ("robot_right", "la"))
ROBOT_BASE_POSITIONS = {"robot_left": (-0.62, 1.0, 0.67), "robot_right": (0.62, 1.0, 0.67)}
CUBE_START_POSITION = (-0.78, 0.10, 0.806)
HANDOVER_POSITION = (0.0, 0.10, 1.02)
PLATE_POSITION = (0.78, 0.12, 0.7925)


def _camera_path(robot_prim: str, camera_prim: str) -> str:
    side_chain = "ra_1/ra_2/ra_3/ra_4/ra_5/ra_6/ra_gripper" if camera_prim.startswith("ra_") else (
        "la_1/la_2/la_3/la_4/la_5/la_6/la_gripper"
    )
    return (
        f"{{ENV_REGEX_NS}}/{robot_prim}/Geometry/world/fr30_1/fr30_2/fr30_3/fr30_4/fr30_5/fr30_6/"
        f"{side_chain}/{camera_prim}"
    )


DUAL_CAMERA_PATHS = {
    "task_rgb": "{ENV_REGEX_NS}/TaskCamera",
    "left_controlled_wrist_rgb": _camera_path("UltraLeft", "ra_wrist_fisheye"),
    "right_controlled_wrist_rgb": _camera_path("UltraRight", "la_wrist_fisheye"),
}


def dual_robot_camera_cfg(stream: str, width: int = 320, height: int = 240) -> CameraCfg:
    if stream == "task_rgb":
        return task_camera_cfg(width, height)
    try:
        prim_path = DUAL_CAMERA_PATHS[stream]
    except KeyError as error:
        raise ValueError(f"Unknown dual-Ultra camera stream: {stream}") from error
    return CameraCfg(
        prim_path=prim_path,
        spawn=None,
        update_period=0.04,
        height=height,
        width=width,
        data_types=["rgb"],
    )


def task_camera_cfg(width: int = 320, height: int = 240) -> CameraCfg:
    return CameraCfg(
        prim_path=DUAL_CAMERA_PATHS["task_rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 2.4, 1.75),
            # World-convention +X looks toward the shared table, with +Z up.
            rot=(0.13238676, 0.13238676, -0.6946033, 0.6946033),
            convention="world",
        ),
        update_period=0.04,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=4.0,
            horizontal_aperture=24.0,
            clipping_range=(0.1, 100.0),
        ),
    )


@configclass
class DualUltraSceneCfg(InteractiveSceneCfg):
    """Side-by-side Ultras, a shared table, cube, and plate."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8, restitution=0.0
            )
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.88, 1.0)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(intensity=16000.0, color=(1.0, 0.92, 0.82), radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.5, 3.0, 4.0)),
    )

    robot_left: ArticulationCfg = ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UltraLeft",
        init_state=ULTRA_CFG.init_state.replace(pos=ROBOT_BASE_POSITIONS["robot_left"]),
    )
    robot_right: ArticulationCfg = ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UltraRight",
        init_state=ULTRA_CFG.init_state.replace(pos=ROBOT_BASE_POSITIONS["robot_right"]),
    )

    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=_static_box((2.45, 0.90, 0.06), (0.30, 0.20, 0.12)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.20, 0.75)),
    )
    table_leg_fl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFL",
        spawn=_static_box((0.10, 0.10, 0.72), (0.18, 0.12, 0.08)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.10, -0.08, 0.36)),
    )
    table_leg_fr = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegFR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.10, -0.08, 0.36)),
    )
    table_leg_bl = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBL",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.10, 0.48, 0.36)),
    )
    table_leg_br = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.10, 0.48, 0.36)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), **_dynamic_material((0.10, 0.35, 0.95))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_START_POSITION),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.14,
            height=0.025,
            axis="Z",
            **_dynamic_material((0.92, 0.92, 0.88)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_POSITION),
    )

    task_camera: CameraCfg | None = None
    left_controlled_wrist_camera: CameraCfg | None = None
    right_controlled_wrist_camera: CameraCfg | None = None
