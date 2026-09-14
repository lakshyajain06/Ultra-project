#!/usr/bin/env python3
"""Train a configured policy model on Ultra HDF5 demonstrations."""

from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_class
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from learning.training.config import ExperimentConfig


@hydra.main(config_path="../src/learning/training/conf", config_name="config")
def train(config: DictConfig):
    validated = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    if validated.output is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
        output = Path("outputs/training") / timestamp
        suffix = 1
        while output.exists():
            output = Path("outputs/training") / f"{timestamp}_{suffix:02d}"
            suffix += 1
        validated.output = str(output)
    output = Path(validated.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(validated, resolve=True))

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
