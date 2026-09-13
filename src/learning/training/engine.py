"""Training loop, metrics, and self-contained checkpoints."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    kl_weight: float = 10.0
    num_workers: int = 4
    seed: int = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_and_metrics(prediction, target, is_pad, mu, logvar, action_std):
    valid = (~is_pad).unsqueeze(-1)
    difference = (prediction - target).abs()
    normalized_mae = difference.masked_select(valid.expand_as(difference)).mean()
    physical_difference = difference * action_std.view(1, 1, -1)
    physical_mae = physical_difference.masked_select(valid.expand_as(difference)).mean()
    jaw_indices = torch.tensor([13, 21], device=prediction.device)
    joint_indices = torch.tensor([index for index in range(prediction.shape[-1]) if index not in (13, 21)],
                                 device=prediction.device)
    joint_mae = physical_difference.index_select(-1, joint_indices).masked_select(
        valid.expand(*valid.shape[:-1], joint_indices.numel())
    ).mean()
    jaw_mae = physical_difference.index_select(-1, jaw_indices).masked_select(
        valid.expand(*valid.shape[:-1], jaw_indices.numel())
    ).mean()
    kl = prediction.new_zeros(()) if mu is None else (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    return normalized_mae, {
        "mae_normalized": normalized_mae, "mae_action_units": physical_mae,
        "mae_joint_radians": joint_mae, "mae_jaw_metres": jaw_mae, "kl": kl,
    }


def run_epoch(model, loader, device, action_std, optimizer=None, kl_weight=10.0):
    training = optimizer is not None
    model.train(training)
    totals, samples = {}, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            is_pad = batch["is_pad"].to(device)
            images = {key: value.to(device) for key, value in batch["images"].items()}
            posterior_actions = actions.masked_fill(is_pad.unsqueeze(-1), 0) if training else None
            prediction, mu, logvar = model(state, images, posterior_actions, is_pad if training else None)
            reconstruction, metrics = loss_and_metrics(prediction, actions, is_pad, mu, logvar, action_std)
            loss = reconstruction + kl_weight * metrics["kl"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics["loss"] = loss
            size = state.shape[0]
            samples += size
            valid_steps = int((~is_pad).sum())
            for key, value in metrics.items():
                weight = size if key in ("kl", "loss") else valid_steps
                total, denominator = totals.get(key, (0.0, 0))
                totals[key] = (total + float(value.detach()) * weight, denominator + weight)
    return {key: total / denominator for key, (total, denominator) in totals.items()} if samples else {}


def save_checkpoint(path, model, optimizer, epoch, step, normalizer, train_config, manifest, best_metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "model_config": model.config.to_dict(), "train_config": asdict(train_config),
        "normalizer": normalizer.to_dict(), "manifest": manifest,
        "epoch": epoch, "step": step, "best_metric": best_metric,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer=None, device="cpu", restore_rng=True):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and "rng" in payload:
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if torch.cuda.is_available() and "cuda" in payload["rng"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload
