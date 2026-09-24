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


def test_dual_ultra_camera_streams_are_discovered_and_encoded():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dual_episodes.hdf5"
        camera_names = (
            "task_rgb", "left_controlled_wrist_rgb", "right_controlled_wrist_rgb",
        )
        with h5py.File(path, "w") as handle:
            handle.attrs["metadata"] = json.dumps({
                "task": "dual_ultra_handover",
                "control_hz": 25,
                "camera_streams": list(camera_names),
            })
            demo = handle.create_group("data/demo_000000")
            demo.attrs["status"] = "success"
            demo.create_dataset("actions", data=np.zeros((2, 16), dtype=np.float32))
            for camera in camera_names:
                demo.create_dataset(f"obs/{camera}", data=np.zeros((2, 4, 6, 3), dtype=np.uint8))

        annotator = MODULE.Annotator(path)
        episode = annotator.summary()["episodes"][0]
        assert episode["cameras"] == list(camera_names)
        if shutil.which("ffmpeg"):
            loaded = annotator.load_episode("demo_000000")
            assert loaded["video_bytes"] > 0


def test_facing_dual_six_camera_triangle_layout():
    names = MODULE.FACING_DUAL_CAMERAS
    frames = [np.full((1, 4, 6, 3), index + 1, np.uint8) for index in range(6)]
    mosaic = MODULE.compose_mosaic_frame(frames, 0, names)
    assert mosaic.shape == (8, 24, 3)
    # Heads are centered above each pair of wrist cameras.
    assert np.all(mosaic[0:4, 3:9] == 1)
    assert np.all(mosaic[0:4, 15:21] == 4)
    assert np.all(mosaic[4:8, 0:6] == 2)
    assert np.all(mosaic[4:8, 6:12] == 3)
    assert np.all(mosaic[4:8, 12:18] == 5)
    assert np.all(mosaic[4:8, 18:24] == 6)
    if shutil.which("ffmpeg"):
        video = MODULE.encode_mosaic_video(frames, 25, names)
        assert b"ftyp" in video[:32]


def test_metadata_camera_contract_excludes_archived_streams():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "upgraded.hdf5"
        selected = list(MODULE.FACING_DUAL_CAMERAS)
        with h5py.File(path, "w") as handle:
            handle.attrs["metadata"] = json.dumps({"camera_streams": selected})
            demo = handle.create_group("data/demo_000000")
            demo.create_dataset("actions", data=np.zeros((1, 16), dtype=np.float32))
            for camera in (*selected, "task_rgb"):
                demo.create_dataset(f"obs/{camera}", data=np.zeros((1, 4, 6, 3), dtype=np.uint8))
        with h5py.File(path) as handle:
            assert MODULE.episode_cameras(
                handle["data/demo_000000"], json.loads(handle.attrs["metadata"])
            ) == selected
