"""Registered manager-based simulation environments for Ultra."""

import gymnasium as gym

from .cube_plate_pick_place import CAMERA_STREAMS, UltraCubePlateEnvCfg, configure_cameras

gym.register(
    id="Isaac-Ultra-Cube-Plate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.cube_plate_pick_place.env_cfg:UltraCubePlateEnvCfg"},
)

__all__ = [
    "CAMERA_STREAMS",
    "UltraCubePlateEnvCfg",
    "configure_cameras",
]
