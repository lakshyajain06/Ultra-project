"""ACT-specific trainer implementation selected by Hydra."""

from learning.models import ACTPolicy

from .engine import PolicyTrainer


class ACTTrainer(PolicyTrainer):
    """Own ACT construction, posterior conditioning, and KL optimization."""

    def create_model(self, model_name, parameters, **contract):
        if model_name != "act":
            raise ValueError(f"ACTTrainer requires model.name=act, got {model_name!r}")
        config = {**contract, **parameters}
        return ACTPolicy(config), config

    def training_forward(self, model, state, images, actions, is_pad, training):
        posterior_actions = actions.masked_fill(is_pad.unsqueeze(-1), 0) if training else None
        prediction, mu, logvar = model(
            state, images, posterior_actions, is_pad if training else None,
        )
        return prediction, {"mu": mu, "logvar": logvar}

    def objective(self, reconstruction, auxiliary, _train_config, model_training):
        mu, logvar = auxiliary["mu"], auxiliary["logvar"]
        kl = reconstruction.new_zeros(()) if mu is None else (
            -0.5 * (1 + logvar - mu.square() - logvar.exp())
        ).sum(-1).mean()
        weighted_kl = model_training.get("kl_weight", 10.0) * kl
        return reconstruction + weighted_kl, {"kl": kl, "weighted_kl": weighted_kl}
