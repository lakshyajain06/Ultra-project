"""Direct HDF5 label-correction tests."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/annotate_data.py"
SPEC = importlib.util.spec_from_file_location("annotate_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_annotation_updates_hdf5_and_preserves_original_status():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "episodes.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["metadata"] = json.dumps({"task": "test", "control_hz": 25})
            demo = handle.create_group("data/demo_000000")
            demo.attrs["status"] = "aborted"
            demo.attrs["success"] = False
            demo.create_dataset("actions", data=np.zeros((2, 22), dtype=np.float32))
            demo.create_dataset("obs/head_rgb", data=np.zeros((2, 3, 4, 3), dtype=np.uint8))

        annotator = MODULE.Annotator(path)
        if shutil.which("ffmpeg"):
            loaded = annotator.load_episode("demo_000000")
            assert loaded["video_bytes"] > 0
            video = annotator.video("demo_000000")
            assert b"ftyp" in video[:32]
        _, episode = annotator.annotate("demo_000000", "success", "Actually completed")
        assert episode["recorded_status"] == "aborted"
        assert episode["effective_status"] == "success"
        assert episode["annotated"] is True

        annotator.annotate("demo_000000", "rejected", "Bad camera view")
        with h5py.File(path, "r") as handle:
            attrs = handle["data/demo_000000"].attrs
            assert attrs["original_status"] == "aborted"
            assert attrs["status"] == "rejected"
            assert not bool(attrs["success"])
            assert attrs["annotation_notes"] == "Bad camera view"
            assert attrs["annotated_utc"]
