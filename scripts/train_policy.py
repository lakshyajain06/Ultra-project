#!/usr/bin/env python3
"""Train a configured policy model on Ultra HDF5 demonstrations."""

import hydra
from hydra.utils import get_class
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from learning.training.config import ExperimentConfig


@hydra.main(config_path="../src/learning/training/conf", config_name="config")
def train(config: DictConfig):
    validated = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    experiment = OmegaConf.to_object(validated)
    target = str(validated.trainer["_target_"])
    if not target.startswith("learning.training."):
        raise ValueError(f"Trainer target must be in learning.training, got {target!r}")
    trainer_type = get_class(target)
    trainer_type(experiment).fit()


def main():
    load_dotenv()
    train()


if __name__ == "__main__":
    main()
