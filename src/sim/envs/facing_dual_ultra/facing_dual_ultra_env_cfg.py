"""Manager environment for face-to-face dual-Ultra handover."""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..dual_ultra import mdp
from ..dual_ultra.dual_ultra_env_cfg import DualUltraEnvCfg
from .facing_dual_ultra_scene_cfg import (
    FACING_CAMERA_PATHS,
    FACING_EVALUATION_CAMERA_STREAMS,
    FacingDualUltraSceneCfg,
    facing_robot_camera_cfg,
)


@configclass
class FacingDualUltraEnvCfg(DualUltraEnvCfg):
    scene: FacingDualUltraSceneCfg = FacingDualUltraSceneCfg(num_envs=32, env_spacing=4.0)

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 120.0
        self.is_finite_horizon = False
        self.sim = sim_utils.SimulationCfg(dt=0.02, render_interval=self.decimation)
        self.viewer.eye = (1.9, 0.2, 2.2)
        self.viewer.lookat = (0.0, 0.2, 0.85)
        configure_facing_cameras(
            self, self.enabled_cameras, self.camera_width, self.camera_height
        )
        self.num_rerenders_on_reset = int(bool(self.enabled_cameras))


def configure_facing_cameras(cfg, enabled, width=320, height=240):
    unknown = set(enabled).difference(FACING_EVALUATION_CAMERA_STREAMS)
    if unknown:
        raise ValueError(f"Unknown facing dual camera streams: {sorted(unknown)}")
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    enabled = set(enabled)
    for stream in FACING_EVALUATION_CAMERA_STREAMS:
        sensor_name = stream.removesuffix("_rgb") + "_camera"
        setattr(
            cfg.scene,
            sensor_name,
            facing_robot_camera_cfg(stream, width, height) if stream in enabled else None,
        )
        setattr(
            cfg.observations.policy,
            stream,
            ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg(sensor_name)})
            if stream in enabled else None,
        )
