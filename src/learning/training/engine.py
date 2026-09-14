"""Stateful ACT training engine and self-contained checkpoints."""

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import Normalizer, build_datasets, dataset_manifest
from learning.models import ACTConfig, ACTPolicy

from .config import ExperimentConfig
from .tracking import WandbTracker, experiment_config


class Trainer:
    """Own the complete lifecycle of one ACT training run."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        if self.device == "auto":
            self.device = "cpu"
        self.output = Path(config.output)
        self._seed_everything(config.train.seed)
        self.train_data, self.validation_data, self.normalizer = build_datasets(config.dataset)
        self.manifest = dataset_manifest(config.dataset, self.train_data, self.validation_data)
        self.model_config = ACTConfig(
            state_dim=self.normalizer.state_mean.size,
            action_dim=self.normalizer.action_mean.size,
            chunk_size=config.dataset.chunk_size,
            camera_names=config.dataset.camera_names,
            **asdict(config.model),
        )
        self.model = ACTPolicy(self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay,
        )
        self.start_epoch, self.step, self.best = 0, 0, float("inf")
        if config.resume:
            self.restore(config.resume)
        self.validation_loader = DataLoader(
            self.validation_data, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
        )
        self.action_std = torch.as_tensor(self.normalizer.action_std, device=self.device)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def fit(self):
        tracker = WandbTracker.start(
            **asdict(self.config.wandb), output=self.output,
            config=experiment_config(self.config, self.model_config),
            resume=self.config.resume is not None,
        )
        try:
            for epoch in range(self.start_epoch, self.config.train.epochs):
                train_loader = DataLoader(
                    self.train_data, batch_size=self.config.train.batch_size, shuffle=True,
                    num_workers=self.config.train.num_workers,
                    generator=torch.Generator().manual_seed(self.config.train.seed + epoch),
                )
                train_metrics = self.run_epoch(train_loader, training=True)
                validation_metrics = self.run_epoch(self.validation_loader, training=False)
                self.step += len(train_loader)
                selection = validation_metrics.get("mae_action_units", train_metrics["mae_action_units"])
                if selection < self.best:
                    self.best = selection
                    self.save_checkpoint(self.output / "best.pt", epoch)
                self.save_checkpoint(self.output / "latest.pt", epoch)
                tracker.log_epoch(epoch, self.step, train_metrics, validation_metrics, self.best)
                print(json.dumps({
                    "epoch": epoch, "train": train_metrics,
                    "validation": validation_metrics, "best": self.best,
                }))
        finally:
            tracker.finish(self.best)

    def run_epoch(self, loader, *, training):
        self.model.train(training)
        totals, samples = {}, 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                state = batch["state"].to(self.device)
                actions = batch["actions"].to(self.device)
                is_pad = batch["is_pad"].to(self.device)
                images = {key: value.to(self.device) for key, value in batch["images"].items()}
                posterior_actions = actions.masked_fill(is_pad.unsqueeze(-1), 0) if training else None
                prediction, mu, logvar = self.model(
                    state, images, posterior_actions, is_pad if training else None,
                )
                reconstruction, metrics = self.loss_and_metrics(prediction, actions, is_pad, mu, logvar)
                loss = reconstruction + self.config.train.kl_weight * metrics["kl"]
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                metrics["loss"] = loss
                size = state.shape[0]
                samples += size
                valid_steps = int((~is_pad).sum())
                for key, value in metrics.items():
                    weight = size if key in ("kl", "loss") else valid_steps
                    total, denominator = totals.get(key, (0.0, 0))
                    totals[key] = (total + float(value.detach()) * weight, denominator + weight)
        return {key: total / denominator for key, (total, denominator) in totals.items()} if samples else {}

    def loss_and_metrics(self, prediction, target, is_pad, mu, logvar):
        valid = (~is_pad).unsqueeze(-1)
        difference = (prediction - target).abs()
        normalized_mae = difference.masked_select(valid.expand_as(difference)).mean()
        physical_difference = difference * self.action_std.view(1, 1, -1)
        physical_mae = physical_difference.masked_select(valid.expand_as(difference)).mean()
        jaw_indices = torch.tensor([13, 21], device=prediction.device)
        joint_indices = torch.tensor(
            [index for index in range(prediction.shape[-1]) if index not in (13, 21)], device=prediction.device,
        )
        joint_mae = physical_difference.index_select(-1, joint_indices).masked_select(
            valid.expand(*valid.shape[:-1], joint_indices.numel())
        ).mean()
        jaw_mae = physical_difference.index_select(-1, jaw_indices).masked_select(
            valid.expand(*valid.shape[:-1], jaw_indices.numel())
        ).mean()
        kl = prediction.new_zeros(()) if mu is None else (
            -0.5 * (1 + logvar - mu.square() - logvar.exp())
        ).sum(-1).mean()
        return normalized_mae, {
            "mae_normalized": normalized_mae, "mae_action_units": physical_mae,
            "mae_joint_radians": joint_mae, "mae_jaw_metres": jaw_mae, "kl": kl,
        }

    def save_checkpoint(self, path, epoch):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "model_config": self.model_config.to_dict(), "train_config": asdict(self.config.train),
            "experiment_config": asdict(self.config), "normalizer": self.normalizer.to_dict(),
            "manifest": self.manifest, "epoch": epoch, "step": self.step, "best_metric": self.best,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        }
        if torch.cuda.is_available():
            payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, path, *, restore_rng=True):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["model_config"] != self.model_config.to_dict():
            raise ValueError("Resume model configuration differs from the checkpoint")
        if payload["manifest"] != self.manifest:
            raise ValueError("Resume dataset/split or dataset file fingerprint differs from the checkpoint")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.normalizer = Normalizer.from_dict(payload["normalizer"])
        self.train_data.normalizer = self.validation_data.normalizer = self.normalizer
        self.start_epoch, self.step, self.best = payload["epoch"] + 1, payload["step"], payload["best_metric"]
        if restore_rng and "rng" in payload:
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in payload["rng"]:
                torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        return payload

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
