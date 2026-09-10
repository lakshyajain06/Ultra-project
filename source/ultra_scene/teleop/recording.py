"""Incremental HDF5 transitions. Incomplete attempts remain explicitly labelled."""

import json
from pathlib import Path
import h5py
import numpy as np

PROPRIO_LAYOUT = (
    *(f"torso_j{i}" for i in range(1, 7)),
    *(f"left_eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "left_gripper_opening",
    *(f"right_eef_body_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    "right_gripper_opening",
)


def _storage_kwargs(key):
    """Use fast lossless compression for image tensors, not numeric state."""
    if key.rsplit("/", 1)[-1].endswith("_rgb"):
        return {"compression": "lzf", "shuffle": True}
    return {}


class EpisodeRecorder:
    def __init__(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")  # Never replace an existing dataset.
        # Version 6 adds the canonical 22-D ``obs/proprio`` stream while
        # retaining the raw state fields needed by replay and older tools.
        self.file.attrs["schema_version"] = 6
        metadata = dict(metadata)
        metadata.setdefault("proprio_layout", list(PROPRIO_LAYOUT))
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.data = self.file.create_group("data")
        self.episode = None
        self.count = 0

    def begin(self, initial_state):
        self._validate_proprio(initial_state, "initial_state")
        self.episode = self.data.create_group(f"demo_{self.count:06d}")
        self.count += 1
        self.episode.attrs["status"] = "incomplete"
        self.episode.attrs["success"] = False
        for key, value in initial_state.items():
            self.episode.create_dataset(f"initial_state/{key}", data=value, **_storage_kwargs(key))
        self.file.flush()

    def append(self, obs, action, next_obs, packet, targets, wall_time, sim_time, controller_rotation_offsets=None):
        self._validate_proprio(obs, "obs")
        self._validate_proprio(next_obs, "next_obs")
        values = {"actions": action, "quest": packet, "eef_targets": targets,
                  "wall_time": wall_time, "sim_time": sim_time}
        if controller_rotation_offsets is not None:
            values["controller_rotation_offsets"] = controller_rotation_offsets
        values.update({f"obs/{k}": v for k, v in obs.items()})
        values.update({f"next_obs/{k}": v for k, v in next_obs.items()})
        for key, value in values.items():
            value = np.asarray(value)
            if key not in self.episode:
                self.episode.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                            dtype=value.dtype, chunks=(1, *value.shape), **_storage_kwargs(key))
            dataset = self.episode[key]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = value
        # Flush each transition: a process crash leaves an explicitly incomplete attempt.
        self.file.flush()

    @staticmethod
    def _validate_proprio(values, label):
        if "proprio" not in values:
            raise KeyError(f"{label} must contain canonical 22-D proprio")
        proprio = np.asarray(values["proprio"])
        if proprio.shape != (22,) or not np.isfinite(proprio).all():
            raise ValueError(f"{label}/proprio must be 22 finite values; got {proprio.shape}")

    def finish(self, status):
        if self.episode is not None:
            self.episode.attrs["status"] = status
            self.episode.attrs["success"] = status == "success"
            self.episode.attrs["num_samples"] = len(self.episode.get("actions", []))
            self.file.flush()
            self.episode = None

    def close(self):
        self.finish("interrupted")
        self.file.close()
