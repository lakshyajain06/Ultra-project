"""Tests for Hydra-composed dual-Ultra evaluation."""

from pathlib import Path

import pytest

from learning.evaluation_config import compose_bimanual_evaluation_config


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    return checkpoint


def test_bimanual_evaluation_config_composes_hydra_overrides(tmp_path):
    config = compose_bimanual_evaluation_config(
        (
            f"policy.checkpoint={_checkpoint(tmp_path)}",
            "environment=side_by_side",
            "simulation.num_envs=4",
            "rollout.episodes=12",
            "rollout.max_steps=600",
            "inference=temporal_ensemble",
            "inference.decay=0.02",
        )
    )
    assert config["environment"] == "side_by_side"
    assert config["simulation"]["num_envs"] == 4
    assert config["rollout"] == {"episodes": 12, "max_steps": 600, "seed": 0}
    assert config["inference"] == {"method": "temporal_ensemble", "decay": 0.02}
    assert config["policy"] == {
        "type": "checkpoint", "checkpoint": str(tmp_path / "policy.pt")
    }
    assert config["policy_device"] == "cuda:0"


def test_bimanual_evaluation_config_selects_replay_dataset_and_episode(tmp_path):
    dataset = tmp_path / "demonstrations.hdf5"
    dataset.touch()
    config = compose_bimanual_evaluation_config(
        (
            "policy=replay",
            f"policy.dataset={dataset}",
            "policy.episode=data/demo_000042",
            "recording.enabled=false",
        )
    )
    assert config["policy"] == {
        "type": "replay",
        "dataset": str(dataset),
        "episode": "data/demo_000042",
    }


@pytest.mark.parametrize(
    "override, message",
    (
        ("environment=overhead", "environment"),
        ("rollout.episodes=0", "rollout.episodes"),
    ),
)
def test_bimanual_evaluation_config_rejects_invalid_values(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        compose_bimanual_evaluation_config((f"policy.checkpoint={_checkpoint(tmp_path)}", override))


def test_bimanual_evaluation_config_requires_replay_source(tmp_path):
    with pytest.raises(ValueError, match="policy.dataset"):
        compose_bimanual_evaluation_config(("policy=replay",))
