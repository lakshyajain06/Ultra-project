"""Hydra entry point for ACT training."""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from .config import ExperimentConfig
from .engine import Trainer


@hydra.main(config_path="conf", config_name="config")
def hydra_main(config: DictConfig):
    validated = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    Trainer(OmegaConf.to_object(validated)).fit()


def main():
    load_dotenv()
    hydra_main()
