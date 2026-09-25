"""Hydra composition and validation for dual-Ultra policy evaluation."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf"


def compose_bimanual_evaluation_config(overrides=(), config_name="bimanual_eval"):
    """Compose a bimanual evaluation config through the project Hydra tree."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(config_name=config_name, overrides=list(overrides))
    config = OmegaConf.to_container(config, resolve=True)

    policy = config["policy"]
    policy_type = policy.get("type")
    if policy_type == "checkpoint":
        checkpoint = policy.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("policy.checkpoint must name a checkpoint file")
        if not Path(checkpoint).is_file():
            raise ValueError(f"Policy checkpoint does not exist: {checkpoint}")
    elif policy_type == "replay":
        dataset = policy.get("dataset")
        episode = policy.get("episode")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("policy.dataset must name an HDF5 dataset in replay mode")
        if not Path(dataset).is_file():
            raise ValueError(f"Replay dataset does not exist: {dataset}")
        if not isinstance(episode, str) or not episode.strip():
            raise ValueError("policy.episode must name an episode in replay mode")
    else:
        raise ValueError("policy.type must be checkpoint or replay")
    if config["environment"] not in ("facing", "side_by_side"):
        raise ValueError("environment must be 'facing' or 'side_by_side'")
    method = config["inference"]["method"]
    if method not in ("receding", "chunk", "temporal_ensemble"):
        raise ValueError("inference.method must be receding, chunk, or temporal_ensemble")
    positive_integers = (
        ("simulation.num_envs", config["simulation"]["num_envs"]),
        ("simulation.camera_width", config["simulation"]["camera_width"]),
        ("simulation.camera_height", config["simulation"]["camera_height"]),
        ("rollout.episodes", config["rollout"]["episodes"]),
        ("rollout.max_steps", config["rollout"]["max_steps"]),
        ("recording.fps", config["recording"]["fps"]),
    )
    for name, value in positive_integers:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if method == "chunk":
        steps = config["inference"]["steps"]
        if steps is not None and (
            not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0
        ):
            raise ValueError("inference.steps must be null or a positive integer")
    if not isinstance(config["simulation"]["env_spacing"], (int, float)) or (
        config["simulation"]["env_spacing"] <= 0
    ):
        raise ValueError("simulation.env_spacing must be positive")
    if method == "temporal_ensemble":
        decay = config["inference"]["decay"]
        if not isinstance(decay, (int, float)) or isinstance(decay, bool) or decay < 0:
            raise ValueError("inference.decay must be nonnegative")
    if config["policy_device"] is None:
        config["policy_device"] = config["simulation"]["device"]
    return config


__all__ = ["CONFIG_DIR", "compose_bimanual_evaluation_config"]
