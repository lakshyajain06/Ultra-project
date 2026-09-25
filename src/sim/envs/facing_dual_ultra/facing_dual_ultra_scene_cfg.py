"""Two Ultra robots facing each other across a shared table."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from ...robots.ultra import ULTRA_CFG
from ..cube_plate_pick_place.ultra_tabletop_scene_cfg import _dynamic_material, _static_box


# Keep the schema/controller ordering used by the side-by-side dual task. The
# two selected arms occupy the same side of the table and meet head-on.
FACING_CONTROLLED_ARMS = (("robot_left", "ra"), ("robot_right", "la"))
FACING_ROBOT_BASE_POSITIONS = {
    "robot_left": (0.0, 1.25, 0.67),
    "robot_right": (0.0, -0.85, 0.67),
}
FACING_CUBE_START_POSITION = (0.0, 0.65, 0.806)
FACING_HANDOVER_POSITION = (0.0, 0.20, 1.02)
FACING_PLATE_POSITION = (0.0, -0.25, 0.7925)


def _robot_camera_path(robot_prim: str, camera_prim: str) -> str:
    if camera_prim == "zed_left":
        suffix = camera_prim
    else:
        arm = camera_prim[:2]
        suffix = f"{arm}_1/{arm}_2/{arm}_3/{arm}_4/{arm}_5/{arm}_6/{arm}_gripper/{camera_prim}"
    return (
        f"{{ENV_REGEX_NS}}/{robot_prim}/Geometry/world/fr30_1/fr30_2/fr30_3/fr30_4/fr30_5/fr30_6/"
        f"{suffix}"
    )


FACING_TASK_CAMERA_PATH = "{ENV_REGEX_NS}/TaskCamera"
FACING_CAMERA_PATHS = {
    "robot_left_head_rgb": _robot_camera_path("UltraLeft", "zed_left"),
    "robot_left_left_wrist_rgb": _robot_camera_path("UltraLeft", "la_wrist_fisheye"),
    "robot_left_right_wrist_rgb": _robot_camera_path("UltraLeft", "ra_wrist_fisheye"),
    "robot_right_head_rgb": _robot_camera_path("UltraRight", "zed_left"),
    "robot_right_left_wrist_rgb": _robot_camera_path("UltraRight", "la_wrist_fisheye"),
    "robot_right_right_wrist_rgb": _robot_camera_path("UltraRight", "ra_wrist_fisheye"),
}
FACING_EVALUATION_CAMERA_STREAMS = ("task_rgb", *FACING_CAMERA_PATHS)


def facing_robot_camera_cfg(stream: str, width: int = 320, height: int = 240) -> CameraCfg:
    if stream in FACING_CAMERA_PATHS:
        return CameraCfg(
            prim_path=FACING_CAMERA_PATHS[stream],
            spawn=None,
            update_period=0.04,
            height=height,
            width=width,
            data_types=["rgb"],
        )
    if stream != "task_rgb":
        raise ValueError(f"Unknown facing dual-Ultra camera stream: {stream}")
    return CameraCfg(
        prim_path=FACING_TASK_CAMERA_PATH,
        offset=CameraCfg.OffsetCfg(
            pos=(1.75, 0.20, 1.75),
            # Camera +X looks at the table center while +Z remains upright.
            rot=(-0.23297306, -0.00332751, 0.97237832, -0.01388829),
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
class FacingDualUltraSceneCfg(InteractiveSceneCfg):
    """Face-to-face Ultras with exclusive pick/place zones and a shared middle."""

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
        spawn=sim_utils.SphereLightCfg(
            intensity=16000.0, color=(1.0, 0.92, 0.82), radius=1.0
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.5, 2.5, 4.0)),
    )

    robot_left: ArticulationCfg = ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UltraLeft",
        init_state=ULTRA_CFG.init_state.replace(
            pos=FACING_ROBOT_BASE_POSITIONS["robot_left"],
            rot=(0.0, 0.0, -0.70710678, 0.70710678),
        ),
    )
    robot_right: ArticulationCfg = ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UltraRight",
        init_state=ULTRA_CFG.init_state.replace(
            pos=FACING_ROBOT_BASE_POSITIONS["robot_right"],
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=_static_box((1.25, 1.50, 0.06), (0.30, 0.20, 0.12)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.20, 0.75)),
    )
    table_leg_fl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFL",
        spawn=_static_box((0.10, 0.10, 0.72), (0.18, 0.12, 0.08)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.52, -0.42, 0.36)),
    )
    table_leg_fr = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegFR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, -0.42, 0.36)),
    )
    table_leg_bl = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBL",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.52, 0.82, 0.36)),
    )
    table_leg_br = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, 0.82, 0.36)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), **_dynamic_material((0.10, 0.35, 0.95))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FACING_CUBE_START_POSITION),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.14,
            height=0.025,
            axis="Z",
            **_dynamic_material((0.92, 0.92, 0.88)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FACING_PLATE_POSITION),
    )

    task_camera: CameraCfg | None = None
    robot_left_head_camera: CameraCfg | None = None
    robot_left_left_wrist_camera: CameraCfg | None = None
    robot_left_right_wrist_camera: CameraCfg | None = None
    robot_right_head_camera: CameraCfg | None = None
    robot_right_left_wrist_camera: CameraCfg | None = None
    robot_right_right_wrist_camera: CameraCfg | None = None
