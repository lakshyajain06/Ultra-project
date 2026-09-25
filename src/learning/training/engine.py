"""Stateful policy training engine and self-contained checkpoints."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import Normalizer, build_datasets, dataset_manifest
from .tracking import WandbTracker, experiment_config


class PolicyTrainer(ABC):
    """Shared lifecycle for a trainer selected and instantiated by Hydra."""

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"]
        if self.device == "auto":
            self.device = "cpu"
        self.output = Path(config["output"])
        self._seed_everything(config["train"]["seed"])
        model = dict(config["model"])
        self.model_name = model.pop("name")
        chunk_size = model.pop("chunk_size")
        inputs = dict(model.pop("inputs"))
        model_parameters = dict(model.pop("architecture"))
        self.model_loss = dict(model.pop("loss", {}))
        model_parameters["image_encoder"] = dict(model.pop("encoder"))
        if model:
            raise ValueError(f"Unexpected model configuration keys: {sorted(model)}")
        data_config = {**config["dataset"], **inputs, "chunk_size": chunk_size}
        self.train_data, self.validation_data, self.normalizer = build_datasets(data_config)
        self.manifest = dataset_manifest(data_config, self.train_data, self.validation_data)
        self.jaw_indices = tuple(
            index for index, name in enumerate(self.manifest["controlled_joint_names"])
            if name.endswith("gripper_joint")
        )
        self.model, self.model_config = self.create_model(
            self.model_name,
            model_parameters,
            state_dim=self.normalizer.state_mean.size,
            action_dim=self.normalizer.action_mean.size,
            chunk_size=chunk_size,
            camera_names=inputs["camera_names"],
        )
        self.model.to(self.device)
        optimizer = config["train"]["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=optimizer["learning_rate"], weight_decay=optimizer["weight_decay"],
        )
        self.start_epoch, self.step, self.best = 0, 0, float("inf")
        if config["resume"]:
            self.restore(config["resume"])
        self.validation_loader = DataLoader(
            self.validation_data, batch_size=config["train"]["batch_size"],
            num_workers=config["train"]["num_workers"],
        )
        self.action_std = torch.as_tensor(self.normalizer.action_std, device=self.device)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def fit(self):
        tracker = WandbTracker.start(
            **self.config["wandb"], output=self.output,
            config=experiment_config(self.config, self.model_config),
            resume=self.config["resume"] is not None,
        )
        tracker.set_summary({
            "data/train_samples": len(self.train_data),
            "data/validation_samples": len(self.validation_data),
            "data/train_episodes": len(self.train_data.episodes),
            "data/validation_episodes": len(self.validation_data.episodes),
            "model/parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "model/trainable_parameters": sum(
                parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
            ),
        })
        try:
            for epoch in range(self.start_epoch, self.config["train"]["epochs"]):
                train_loader = DataLoader(
                    self.train_data, batch_size=self.config["train"]["batch_size"], shuffle=True,
                    num_workers=self.config["train"]["num_workers"],
                    generator=torch.Generator().manual_seed(self.config["train"]["seed"] + epoch),
                )
                train_metrics = self.timed_epoch(train_loader, training=True)
                validation_metrics = self.timed_epoch(self.validation_loader, training=False)
                train_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
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

    def timed_epoch(self, loader, *, training):
        started = perf_counter()
        metrics = self.run_epoch(loader, training=training)
        elapsed = perf_counter() - started
        metrics["epoch_seconds"] = elapsed
        metrics["samples_per_second"] = len(loader.dataset) / elapsed if elapsed else float("inf")
        return metrics

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
                prediction, auxiliary = self.training_forward(
                    self.model, state, images, actions, is_pad, training,
                )
                reconstruction, metrics = self.loss_and_metrics(prediction, actions, is_pad)
                loss, objective_metrics = self.objective(
                    reconstruction, auxiliary, self.config["train"], self.model_loss,
                )
                metrics.update(objective_metrics)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["train"]["gradient_clip_norm"],
                    )
                    self.optimizer.step()
                    metrics["gradient_norm"] = gradient_norm
                metrics.update({
                    "reconstruction_loss": reconstruction,
                    "loss": loss,
                })
                size = state.shape[0]
                samples += size
                valid_steps = int((~is_pad).sum())
                for key, value in metrics.items():
                    weight = size if key in (
                        "reconstruction_loss", "kl", "weighted_kl", "loss", "gradient_norm"
                    ) else valid_steps
                    total, denominator = totals.get(key, (0.0, 0))
                    totals[key] = (total + float(value.detach()) * weight, denominator + weight)
        return {key: total / denominator for key, (total, denominator) in totals.items()} if samples else {}

    def loss_and_metrics(self, prediction, target, is_pad):
        valid = (~is_pad).unsqueeze(-1)
        difference = (prediction - target).abs()
        normalized_mae = difference.masked_select(valid.expand_as(difference)).mean()
        physical_difference = difference * self.action_std.view(1, 1, -1)
        physical_mae = physical_difference.masked_select(valid.expand_as(difference)).mean()
        jaw_indices = torch.tensor(self.jaw_indices, device=prediction.device)
        joint_indices = torch.tensor(
            [index for index in range(prediction.shape[-1]) if index not in self.jaw_indices],
            device=prediction.device,
        )
        joint_mae = physical_difference.index_select(-1, joint_indices).masked_select(
            valid.expand(*valid.shape[:-1], joint_indices.numel())
        ).mean()
        jaw_mae = physical_difference.index_select(-1, jaw_indices).masked_select(
            valid.expand(*valid.shape[:-1], jaw_indices.numel())
        ).mean()
        return normalized_mae, {
            "mae_normalized": normalized_mae, "mae_action_units": physical_mae,
            "mae_joint_radians": joint_mae, "mae_jaw_metres": jaw_mae,
        }

    @abstractmethod
    def create_model(self, model_name, parameters, **contract):
        """Construct the configured policy model and its serializable config."""

    @abstractmethod
    def training_forward(self, model, state, images, actions, is_pad, training):
        """Run the model's training or validation forward pass."""

    @abstractmethod
    def objective(self, reconstruction, auxiliary, train_config, model_training):
        """Combine reconstruction and model-specific objective terms."""

    def save_checkpoint(self, path, epoch):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "model_name": self.model_name,
            "model_config": self.model_config, "train_config": self.config["train"],
            "experiment_config": self.config, "normalizer": self.normalizer.to_dict(),
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
        if payload["model_name"] != self.model_name:
            raise ValueError("Resume model family differs from the checkpoint")
        if payload["model_config"] != self.model_config:
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
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
        return payload

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
