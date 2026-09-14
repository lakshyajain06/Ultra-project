"""Structured Hydra configuration for ACT experiments."""

from dataclasses import dataclass, field

from data.datasets import DatasetConfig


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    kl_weight: float = 10.0
    num_workers: int = 4
    seed: int = 0


@dataclass
class ACTArchitectureConfig:
    hidden_dim: int = 256
    latent_dim: int = 32
    feedforward_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


@dataclass
class WandbConfig:
    project: str = "${oc.env:WANDB_PROJECT,ultra-act}"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "${oc.env:WANDB_MODE,online}"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(paths=()))
    model: ACTArchitectureConfig = field(default_factory=ACTArchitectureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    device: str = "auto"
    output: str = "outputs/act"
    resume: str | None = None
