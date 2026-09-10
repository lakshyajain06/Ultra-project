"""Incremental HDF5 transitions. Incomplete attempts remain explicitly labelled."""

import json
from pathlib import Path
import h5py
import numpy as np


class EpisodeRecorder:
    def __init__(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")  # Never replace an existing dataset.
        self.file.attrs["schema_version"] = 3
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.data = self.file.create_group("data")
        self.episode = None
        self.count = 0

    def begin(self, initial_state):
        self.episode = self.data.create_group(f"demo_{self.count:06d}")
        self.count += 1
        self.episode.attrs["status"] = "incomplete"
        self.episode.attrs["success"] = False
        for key, value in initial_state.items():
            self.episode.create_dataset(f"initial_state/{key}", data=value)
        self.file.flush()

    def append(self, obs, action, next_obs, packet, targets, wall_time, sim_time):
        values = {"actions": action, "quest": packet, "eef_targets": targets,
                  "wall_time": wall_time, "sim_time": sim_time}
        values.update({f"obs/{k}": v for k, v in obs.items()})
        values.update({f"next_obs/{k}": v for k, v in next_obs.items()})
        for key, value in values.items():
            value = np.asarray(value)
            if key not in self.episode:
                self.episode.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                            dtype=value.dtype, chunks=True)
            dataset = self.episode[key]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = value
        # Flush each transition: a process crash leaves an explicitly incomplete attempt.
        self.file.flush()

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
