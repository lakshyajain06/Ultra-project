"""A small vectorizable tabletop scene for manipulation experiments."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .robots.ultra import ULTRA_CFG


def _static_box(size, color):
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
    )


def _dynamic_material(color):
    return {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=3.0,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.12),
        "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.65,
            restitution=0.02,
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
    }


@configclass
class UltraTabletopSceneCfg(InteractiveSceneCfg):
    """Ultra, a table, one cube, and one plate in each cloned environment."""

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
            intensity=14000.0,
            color=(1.0, 0.92, 0.82),
            radius=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.5, 3.0, 4.0)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=9000.0,
            color=(0.72, 0.82, 1.0),
            radius=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.5, 0.0, 3.0)),
    )

    camera: CameraCfg | None = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CaptureCamera",
        update_period=0.0,
        height=544,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )
    # Existing robot camera prims are attached by the teleop environment only;
    # keeping them optional avoids rendering overhead in the generic scene.
    head_camera: CameraCfg | None = None
    left_wrist_camera: CameraCfg | None = None
    right_wrist_camera: CameraCfg | None = None

    robot: ArticulationCfg = ULTRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Ultra",
        # Apply the source table2_gui placement rule to this raised table:
        # base 0.11 m below the z=0.78 top and 0.98 m behind its y=0.66 edge.
        init_state=ULTRA_CFG.init_state.replace(pos=(0.0, 1.0, 0.67)),
    )

    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=_static_box((1.40, 0.82, 0.06), (0.30, 0.20, 0.12)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.25, 0.75)),
    )
    table_leg_fl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFL",
        spawn=_static_box((0.08, 0.08, 0.72), (0.18, 0.12, 0.08)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.60, -0.05, 0.36)),
    )
    table_leg_fr = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegFR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, -0.05, 0.36)),
    )
    table_leg_bl = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBL",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.60, 0.55, 0.36)),
    )
    table_leg_br = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBR",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, 0.55, 0.36)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.075, 0.075, 0.075), **_dynamic_material((0.10, 0.35, 0.95))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.18, 0.12, 0.818)),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.14,
            height=0.025,
            axis="Z",
            **_dynamic_material((0.92, 0.92, 0.88)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.15, 0.793)),
    )
