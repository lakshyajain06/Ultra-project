"""Optional Weights & Biases integration for training runs."""

import json
from pathlib import Path

import wandb


class WandbTracker:
    """Small adapter that keeps wandb out of the core training loop."""

    RUN_ID_FILE = "wandb_run_id.txt"

    @classmethod
    def start(cls, *, project, entity, name, group, tags, mode, output, config, resume):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        run_id_path = output / cls.RUN_ID_FILE
        run_id = run_id_path.read_text().strip() if resume and run_id_path.exists() else None
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=list(tags) or None,
            mode=mode,
            dir=str(output),
            config=config,
            id=run_id,
            resume="allow" if resume else None,
        )
        if run is None:
            raise RuntimeError("wandb.init() did not create a run")
        run_id_path.write_text(f"{run.id}\n")
        return cls(run)

    def __init__(self, run):
        self.run = run

    def log_epoch(self, epoch, step, train_metrics, validation_metrics, best):
        metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        metrics.update({f"validation/{key}": value for key, value in validation_metrics.items()})
        metrics.update({"epoch": epoch, "best_mae_action_units": best})
        self.run.log(metrics, step=step)

    def set_summary(self, values):
        for key, value in values.items():
            self.run.summary[key] = value

    def finish(self, best):
        self.run.summary["best_mae_action_units"] = best
        self.run.finish()


def experiment_config(config, model_config, model_name=None, model_training=None):
    """Return a JSON-compatible snapshot for the wandb run configuration."""
    from dataclasses import asdict

    model = model_config.to_dict()
    if model_name is not None:
        model = {"name": model_name, "parameters": model, "training": model_training or {}}
    return json.loads(json.dumps({**asdict(config), "model": model}))
