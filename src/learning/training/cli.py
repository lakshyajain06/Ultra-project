"""Command-line ACT trainer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import DatasetConfig, Normalizer, build_datasets, dataset_manifest
from learning.models import ACTConfig, ACTPolicy
from .engine import TrainConfig, restore_checkpoint, run_epoch, save_checkpoint, seed_everything


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("datasets", nargs="+", help="Ultra HDF5 files")
    value.add_argument("--output", type=Path, default=Path("outputs/act"))
    value.add_argument("--resume", type=Path)
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--learning-rate", type=float, default=1e-4)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--kl-weight", type=float, default=10.0)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--chunk-size", type=int, default=25)
    value.add_argument("--validation-fraction", type=float, default=0.1)
    value.add_argument("--statuses", nargs="+", default=["success"])
    value.add_argument("--state-keys", nargs="+", default=["proprio"])
    value.add_argument("--cameras", nargs="*", default=["head_rgb", "left_wrist_rgb", "right_wrist_rgb"])
    value.add_argument("--hidden-dim", type=int, default=256)
    value.add_argument("--latent-dim", type=int, default=32)
    value.add_argument("--feedforward-dim", type=int, default=1024)
    value.add_argument("--num-heads", type=int, default=8)
    value.add_argument("--num-layers", type=int, default=4)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    train_config = TrainConfig(
        args.epochs, args.batch_size, args.learning_rate, args.weight_decay,
        args.kl_weight, args.num_workers, args.seed,
    )
    seed_everything(args.seed)
    dataset_config = DatasetConfig(
        tuple(args.datasets), args.chunk_size, tuple(args.state_keys), tuple(args.cameras),
        tuple(args.statuses), args.validation_fraction, args.seed,
    )
    train_data, validation_data, normalizer = build_datasets(dataset_config)
    action_dim = normalizer.action_mean.size
    model_config = ACTConfig(
        normalizer.state_mean.size, action_dim, args.chunk_size, tuple(args.cameras),
        args.hidden_dim, args.latent_dim, args.feedforward_dim, args.num_heads, args.num_layers,
    )
    model = ACTPolicy(model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch, step, best = 0, 0, float("inf")
    manifest = dataset_manifest(dataset_config, train_data, validation_data)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("Resume model configuration differs from the checkpoint")
        if checkpoint["manifest"] != manifest:
            raise ValueError("Resume dataset/split or dataset file fingerprint differs from the checkpoint")
        # The checkpoint is the source of truth for preprocessing on resume.
        normalizer = Normalizer.from_dict(checkpoint["normalizer"])
        train_data.normalizer = validation_data.normalizer = normalizer
        payload = restore_checkpoint(args.resume, model, optimizer, args.device)
        start_epoch, step, best = payload["epoch"] + 1, payload["step"], payload["best_metric"]
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.num_workers)
    action_std = torch.as_tensor(normalizer.action_std, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs):
        # Epoch-derived ordering makes a resumed run match an uninterrupted run.
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        train_metrics = run_epoch(model, train_loader, args.device, action_std, optimizer, args.kl_weight)
        validation_metrics = run_epoch(model, validation_loader, args.device, action_std)
        step += len(train_loader)
        selection = validation_metrics.get("mae_action_units", train_metrics["mae_action_units"])
        if selection < best:
            best = selection
            save_checkpoint(
                args.output / "best.pt", model, optimizer, epoch, step, normalizer,
                train_config, manifest, best,
            )
        save_checkpoint(
            args.output / "latest.pt", model, optimizer, epoch, step, normalizer,
            train_config, manifest, best,
        )
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "best": best}))
