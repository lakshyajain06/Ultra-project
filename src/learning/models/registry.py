"""Registry used to reconstruct policy models from checkpoints."""

from .act import ACTPolicy


MODEL_FAMILIES = {
    "act": ACTPolicy,
}


def model_family(name):
    try:
        return MODEL_FAMILIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown model family {name!r}; available: {sorted(MODEL_FAMILIES)}") from error


def load_model(name, config, state_dict):
    model = model_family(name)(config)
    model.load_state_dict(state_dict)
    return model
