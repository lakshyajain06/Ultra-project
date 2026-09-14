"""CPU-only ACT pipeline tests with tiny synthetic HDF5 demonstrations."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.datasets import DatasetConfig, build_datasets, compose_task_state
from learning.inference import PolicyInference
from learning.models import ACTConfig, ACTPolicy
from learning.training.config import ExperimentConfig, TrainConfig
from learning.training.act_trainer import ACTTrainer
from learning.training.tracking import WandbTracker, experiment_config


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
            eef = np.zeros((length, 2, 7), dtype=np.float32)
            eef[..., :3] = np.arange(length, dtype=np.float32)[:, None, None] + np.array([[[1], [2]]])
            eef[..., 6] = 1
            obs.create_dataset("eef_pose_body", data=eef)
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

    def test_hydra_structured_config_round_trip(self):
        structured = OmegaConf.structured(ExperimentConfig)
        merged = OmegaConf.merge(structured, {
            "dataset": {"paths": [str(self.path)], "camera_names": []},
            "model": {"name": "act", "parameters": {"hidden_dim": 32}, "training": {"kl_weight": 0.1}},
            "train": {"batch_size": 8},
            "wandb": {"mode": "disabled"},
        })
        config = OmegaConf.to_object(merged)
        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.dataset.paths, (str(self.path),))
        self.assertEqual(config.dataset.camera_names, ())
        self.assertEqual(config.train.batch_size, 8)
        self.assertEqual(config.model["name"], "act")

    def test_episode_split_normalization_and_padding(self):
        train, validation, normalizer = build_datasets(self.config())
        self.assertEqual(ACTConfig().state_dim, 22)
        self.assertEqual((len(train.episodes), len(validation.episodes)), (1, 1))
        self.assertTrue(set(e.identity for e in train.episodes).isdisjoint(e.identity for e in validation.episodes))
        self.assertEqual(normalizer.state_mean.shape, (22,))
        sample = train[len(train) - 1]
        self.assertEqual(sample["actions"].shape, (3, 22))
        self.assertEqual(sample["is_pad"].tolist(), [False, True, True])
        self.assertEqual(sample["images"]["head_rgb"].shape, (3, 12, 16))
        episode, step = train.samples[0]
        with h5py.File(episode.path) as handle:
            demo = handle[episode.key]
            expected = compose_task_state(
                demo["obs/joint_pos"][step], demo["obs/eef_pose_body"][step],
                episode.controlled_joint_indices,
            )
        actual = train[0]["state"].numpy() * normalizer.state_std + normalizer.state_mean
        np.testing.assert_allclose(actual, expected, atol=1e-5)

        legacy, _, legacy_normalizer = build_datasets(self.config(
            state_keys=("joint_pos", "joint_vel"), validation_fraction=0,
        ))
        self.assertEqual(legacy[0]["state"].shape, (44,))
        self.assertEqual(legacy_normalizer.state_mean.shape, (44,))

    def test_status_error_explains_available_labels(self):
        with self.assertRaisesRegex(ValueError, "found statuses=.*aborted.*success"):
            build_datasets(self.config(statuses=("timeout",)))

    def test_episode_key_selects_exactly_one_episode(self):
        train, validation, _ = build_datasets(self.config(
            episode_keys=("data/demo_000001",), validation_fraction=0,
        ))
        self.assertEqual([episode.key for episode in train.episodes], ["data/demo_000001"])
        self.assertEqual(validation.episodes, [])

        with self.assertRaisesRegex(ValueError, "were not found.*demo_999999"):
            build_datasets(self.config(episode_keys=("demo_999999",)))

    def test_episode_key_must_be_unique_across_files(self):
        second = Path(self.temporary.name) / "synthetic-copy.hdf5"
        make_dataset(second)
        with self.assertRaisesRegex(ValueError, "ambiguous.*demo_000000"):
            build_datasets(self.config(
                paths=(str(self.path), str(second)), episode_keys=("demo_000000",),
            ))

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

    def test_malformed_eef_pose_is_rejected(self):
        with h5py.File(self.path, "r+") as handle:
            demo = handle["data/demo_000000/obs"]
            del demo["eef_pose_body"]
            demo.create_dataset("eef_pose_body", data=np.zeros((4, 2, 6), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"eef_pose_body must be \[T,2,7\]"):
            build_datasets(self.config())

    def test_cameras_can_be_disabled(self):
        train, _, normalizer = build_datasets(self.config(camera_names=(), validation_fraction=0))
        sample = train[0]
        self.assertEqual(sample["images"], {})
        model = ACTPolicy(ACTConfig(
            state_dim=22, action_dim=22, chunk_size=3, camera_names=(),
            hidden_dim=16, latent_dim=2, feedforward_dim=32, num_heads=2, num_layers=1, dropout=0,
        ))
        prediction = model.predict(sample["state"][None], {})
        self.assertEqual(prediction.shape, (1, 3, 22))
        self.assertEqual(normalizer.action_mean.shape, (22,))

    def test_forward_train_checkpoint_and_inference(self):
        config = self.config(validation_fraction=0)
        experiment = ExperimentConfig(
            dataset=config,
            model={
                "name": "act",
                "parameters": {
                    "hidden_dim": 32, "latent_dim": 4, "feedforward_dim": 64,
                    "num_heads": 4, "num_layers": 1, "dropout": 0,
                },
                "training": {"kl_weight": 0.1},
            },
            train=TrainConfig(batch_size=2, learning_rate=1e-3, num_workers=0),
            device="cpu",
            output=str(Path(self.temporary.name) / "run"),
        )
        trainer = ACTTrainer(experiment)
        loader = DataLoader(trainer.train_data, batch_size=2)
        metrics = trainer.run_epoch(loader, training=True)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("gradient_norm", metrics)
        self.assertAlmostEqual(
            metrics["loss"], metrics["reconstruction_loss"] + metrics["weighted_kl"], places=5,
        )
        checkpoint = Path(self.temporary.name) / "checkpoint.pt"
        trainer.step = len(loader)
        trainer.best = 1.0
        trainer.save_checkpoint(checkpoint, epoch=0)
        clone = ACTTrainer(experiment)
        payload = clone.restore(checkpoint, restore_rng=False)
        self.assertEqual(payload["model_name"], "act")
        self.assertEqual(payload["model_config"]["action_dim"], 22)
        runner = PolicyInference.from_checkpoint(checkpoint)
        with h5py.File(self.path) as handle:
            demo = handle["data/demo_000000/obs"]
            observation = {"joint_pos": demo["joint_pos"][0], "eef_pose_body": demo["eef_pose_body"][0]}
            observation.update({name: demo[name][0] for name in CAMERAS})
            observation_names = json.loads(handle.attrs["metadata"])["observation_joint_names"]
        prediction = runner.predict(observation, observation_names)
        self.assertEqual(prediction.shape, (3, 22))
        self.assertTrue(np.isfinite(prediction).all())
        controlled_indices = [observation_names.index(name) for name in runner.controlled_joint_names]
        task_state = compose_task_state(
            observation["joint_pos"], observation["eef_pose_body"], controlled_indices,
        )
        manager_observation = {
            "proprio": task_state[None],
            **{name: observation[name][None] for name in CAMERAS},
        }
        manager_prediction = runner.predict(manager_observation)
        np.testing.assert_allclose(manager_prediction, prediction)
        torch_observation = {key: torch.from_numpy(value) for key, value in manager_observation.items()}
        torch_prediction = runner.predict(torch_observation)
        np.testing.assert_allclose(torch_prediction, manager_prediction)
        with self.assertRaisesRegex(ValueError, "supports one environment"):
            runner.predict({**manager_observation, "proprio": np.repeat(manager_observation["proprio"], 2, axis=0)})

    def test_wandb_metric_names_and_config_are_stable(self):
        class FakeRun:
            def __init__(self):
                self.logged = []
                self.summary = {}
                self.finished = False

            def log(self, metrics, step):
                self.logged.append((metrics, step))

            def finish(self):
                self.finished = True

        run = FakeRun()
        tracker = WandbTracker(run)
        tracker.set_summary({"model/parameters": 123})
        tracker.log_epoch(2, 17, {"loss": 1.2}, {"mae_action_units": 0.3}, 0.3)
        tracker.finish(0.3)
        self.assertEqual(run.logged, [({
            "train/loss": 1.2, "validation/mae_action_units": 0.3,
            "epoch": 2, "best_mae_action_units": 0.3,
        }, 17)])
        self.assertEqual(run.summary["best_mae_action_units"], 0.3)
        self.assertEqual(run.summary["model/parameters"], 123)
        self.assertTrue(run.finished)

        config = experiment_config(
            ExperimentConfig(dataset=self.config(), device="cpu", output="outputs/test"), ACTConfig(),
        )
        self.assertEqual(config["dataset"]["paths"], [str(self.path)])
        self.assertEqual(config["model"]["state_dim"], 22)


if __name__ == "__main__":
    unittest.main()
