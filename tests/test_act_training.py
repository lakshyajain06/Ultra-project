"""CPU-only ACT pipeline tests with tiny synthetic HDF5 demonstrations."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ultra_scene.learning.inference import ACTInference
from ultra_scene.learning.models import ACTConfig, ACTPolicy
from ultra_scene.learning.training.data import DatasetConfig, build_datasets, dataset_manifest
from ultra_scene.learning.training.engine import TrainConfig, restore_checkpoint, run_epoch, save_checkpoint


CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")


def make_dataset(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 5
        action_names = [*(f"torso_j{i}" for i in range(1, 7)), *(f"la_j{i}" for i in range(1, 8)),
                        "la_gripper_joint", *(f"ra_j{i}" for i in range(1, 8)), "ra_gripper_joint"]
        observation_names = ["la_jaw_l_joint", *reversed(action_names), "ra_jaw_l_joint"]
        handle.attrs["metadata"] = json.dumps({
            "action_joint_names": action_names, "observation_joint_names": observation_names,
        })
        data = handle.create_group("data")
        for episode_index, (length, status) in enumerate(((4, "success"), (3, "success"), (2, "aborted"))):
            demo = data.create_group(f"demo_{episode_index:06d}")
            demo.attrs["status"] = status
            demo.attrs["success"] = status == "success"
            time = np.arange(length, dtype=np.float32)[:, None]
            demo.create_dataset("actions", data=np.tile(time, (1, 22)) + episode_index)
            obs = demo.create_group("obs")
            offsets = np.arange(24, dtype=np.float32)[None]
            obs.create_dataset("joint_pos", data=np.tile(time, (1, 24)) + offsets + episode_index)
            obs.create_dataset("joint_vel", data=np.tile(time * 0.1, (1, 24)) + offsets * 0.01)
            for camera_index, camera in enumerate(CAMERAS):
                obs.create_dataset(camera, data=np.full((length, 12, 16, 3), 20 * camera_index, np.uint8))


class ACTTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "synthetic.hdf5"
        make_dataset(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **overrides):
        values = dict(paths=(str(self.path),), chunk_size=3, validation_fraction=0.5, seed=7)
        values.update(overrides)
        return DatasetConfig(**values)

    def test_episode_split_normalization_and_padding(self):
        train, validation, normalizer = build_datasets(self.config())
        self.assertEqual(ACTConfig().state_dim, 44)
        self.assertEqual((len(train.episodes), len(validation.episodes)), (1, 1))
        self.assertTrue(set(e.identity for e in train.episodes).isdisjoint(e.identity for e in validation.episodes))
        self.assertEqual(normalizer.state_mean.shape, (44,))
        sample = train[len(train) - 1]
        self.assertEqual(sample["actions"].shape, (3, 22))
        self.assertEqual(sample["is_pad"].tolist(), [False, True, True])
        self.assertEqual(sample["images"]["head_rgb"].shape, (3, 12, 16))

    def test_status_error_explains_available_labels(self):
        with self.assertRaisesRegex(ValueError, "found statuses=.*aborted.*success"):
            build_datasets(self.config(statuses=("timeout",)))

    def test_duplicate_path_is_rejected(self):
        duplicate = self.config(paths=(str(self.path), str(self.path.resolve())))
        with self.assertRaisesRegex(ValueError, "Duplicate dataset path"):
            build_datasets(duplicate)

    def test_missing_controlled_observation_joint_is_rejected(self):
        with h5py.File(self.path, "r+") as handle:
            metadata = json.loads(handle.attrs["metadata"])
            metadata["observation_joint_names"][1] = "unknown_joint"
            handle.attrs.modify("metadata", json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "observation_joint_names is missing"):
            build_datasets(self.config())

    def test_cameras_can_be_disabled(self):
        train, _, normalizer = build_datasets(self.config(camera_names=(), validation_fraction=0))
        sample = train[0]
        self.assertEqual(sample["images"], {})
        model = ACTPolicy(ACTConfig(
            state_dim=44, action_dim=22, chunk_size=3, camera_names=(),
            hidden_dim=16, latent_dim=2, feedforward_dim=32, num_heads=2, num_layers=1, dropout=0,
        ))
        prediction = model.predict(sample["state"][None], {})
        self.assertEqual(prediction.shape, (1, 3, 22))
        self.assertEqual(normalizer.action_mean.shape, (22,))

    def test_forward_train_checkpoint_and_inference(self):
        config = self.config(validation_fraction=0)
        train, validation, normalizer = build_datasets(config)
        model_config = ACTConfig(
            state_dim=44, action_dim=22, chunk_size=3, camera_names=CAMERAS,
            hidden_dim=32, latent_dim=4, feedforward_dim=64, num_heads=4, num_layers=1, dropout=0,
        )
        model = ACTPolicy(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(train, batch_size=2)
        metrics = run_epoch(model, loader, "cpu", torch.from_numpy(normalizer.action_std), optimizer, kl_weight=0.1)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        checkpoint = Path(self.temporary.name) / "checkpoint.pt"
        manifest = dataset_manifest(config, train, validation)
        save_checkpoint(checkpoint, model, optimizer, 0, len(loader), normalizer, TrainConfig(), manifest, 1.0)
        clone = ACTPolicy(model_config)
        payload = restore_checkpoint(checkpoint, clone, device="cpu", restore_rng=False)
        self.assertEqual(payload["model_config"]["action_dim"], 22)
        runner = ACTInference.from_checkpoint(checkpoint)
        with h5py.File(self.path) as handle:
            demo = handle["data/demo_000000/obs"]
            observation = {"joint_pos": demo["joint_pos"][0], "joint_vel": demo["joint_vel"][0]}
            observation.update({name: demo[name][0] for name in CAMERAS})
            observation_names = json.loads(handle.attrs["metadata"])["observation_joint_names"]
        prediction = runner.predict(observation, observation_names)
        self.assertEqual(prediction.shape, (3, 22))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
