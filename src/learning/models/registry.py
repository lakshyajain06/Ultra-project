"""Registry used to reconstruct policy models from checkpoints."""

from dataclasses import dataclass

from .act import ACTConfig, ACTPolicy


@dataclass(frozen=True)
class ModelFamily:
    config_type: type
    model_type: type


MODEL_FAMILIES = {
    "act": ModelFamily(ACTConfig, ACTPolicy),
}


def model_family(name):
    try:
        return MODEL_FAMILIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown model family {name!r}; available: {sorted(MODEL_FAMILIES)}") from error


def load_model(name, config, state_dict):
    family = model_family(name)
    model = family.model_type(family.config_type.from_dict(config))
    model.load_state_dict(state_dict)
    return model
