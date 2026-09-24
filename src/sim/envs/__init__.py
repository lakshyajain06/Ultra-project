"""Registered manager-based simulation environments for Ultra."""

import gymnasium as gym

from .cube_plate_pick_place import CAMERA_STREAMS, UltraCubePlateEnvCfg, configure_cameras
from .dual_ultra import DualUltraEnvCfg, DualUltraSceneCfg, configure_dual_cameras
from .facing_dual_ultra import FacingDualUltraEnvCfg, FacingDualUltraSceneCfg, configure_facing_cameras

gym.register(
    id="Isaac-Ultra-Cube-Plate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.cube_plate_pick_place.ultra_cube_plate_env_cfg:UltraCubePlateEnvCfg"
        )
    },
)

gym.register(
    id="Isaac-Facing-Dual-Ultra-Handover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.facing_dual_ultra.facing_dual_ultra_env_cfg:FacingDualUltraEnvCfg"
        )
    },
)

gym.register(
    id="Isaac-Dual-Ultra-Shared-Workspace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.dual_ultra.dual_ultra_env_cfg:DualUltraEnvCfg"},
)

__all__ = [
    "CAMERA_STREAMS",
    "UltraCubePlateEnvCfg",
    "DualUltraEnvCfg",
    "DualUltraSceneCfg",
    "FacingDualUltraEnvCfg",
    "FacingDualUltraSceneCfg",
    "configure_cameras",
    "configure_dual_cameras",
    "configure_facing_cameras",
]
