"""Structured Hydra configuration for policy experiments."""

from dataclasses import dataclass, field
from typing import Any

from data.datasets import DatasetConfig


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 0


@dataclass
class WandbConfig:
    project: str = "${oc.env:WANDB_PROJECT,ultra-policy}"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "${oc.env:WANDB_MODE,online}"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(paths=()))
    model: Any = field(default_factory=dict)
    trainer: Any = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    device: str = "auto"
    output: str | None = None
    resume: str | None = None
