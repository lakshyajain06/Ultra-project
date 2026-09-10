"""Source-faithful Isaac Lab configuration for the Ultra robot."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[4]
ULTRA_USD = Path(
    os.environ.get("ULTRA_USD_PATH", REPO_ROOT / "assets/robots/ultra/ultra.usd")
).expanduser()


ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ULTRA_USD),
        activate_contact_sensors=True,
        # Match real2sim2real.scene.setup_physics(). These iterations are
        # required by Ultra's intentionally stiff source drives at 50 Hz.
        articulation_props=sim_utils.PhysxArticulationRootPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Source default placement for its floor-level simple_room table.
        pos=(0.0, 1.50, -0.10),
        # Isaac Lab 3 uses XYZW quaternions: -90 degrees about world Z.
        rot=(0.0, 0.0, -0.70710678, 0.70710678),
        joint_pos={
            "torso_j[1-6]": 0.0,
            "la_j1": 0.4697,
            "la_j2": -0.5580,
            "la_j3": 2.5408,
            "la_j4": -2.2712,
            "la_j5": 2.3473,
            "la_j6": 0.3108,
            "la_j7": -1.9000,
            "ra_j1": -0.4071,
            "ra_j2": 0.0615,
            "ra_j3": 0.2357,
            "ra_j4": -0.0232,
            "ra_j5": -0.7456,
            "ra_j6": -0.2560,
            "ra_j7": 0.3500,
            ".*_gripper_joint": 0.045,
        },
    ),
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j[1-6]"],
            effort_limit_sim=8000.0,
            stiffness=3.0e5,
            damping=1.2e4,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["[lr]a_j[1-7]"],
            effort_limit_sim=8000.0,
            stiffness=3.0e5,
            damping=1.2e4,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["[lr]a_gripper_joint"],
            effort_limit_sim=400.0,
            stiffness=8.0e3,
            damping=2.0e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Ultra configuration with the source pose, placement, and actuator gains."""
