import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from utils.training_checkpoint import (
    checkpoint_metadata_path,
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)


CONFIG = {
    "architecture": "synthetic",
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
}


def make_model_and_optimizer(device="cpu"):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.GELU(),
        torch.nn.Linear(4, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    return model, optimizer


def take_step(model, optimizer):
    optimizer.zero_grad()
    loss = model(torch.arange(12, dtype=torch.float32).reshape(4, 3).to(next(model.parameters()).device)).sum()
    loss.backward()
    optimizer.step()


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "resume.pth"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_restores_model_optimizer_and_metadata(self):
        model, optimizer = make_model_and_optimizer()
        take_step(model, optimizer)
        expected_model = {key: value.clone() for key, value in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()

        save_training_checkpoint(self.path, model, optimizer, 7, 0.8125, 123.25, CONFIG)
        restored_model, restored_optimizer = make_model_and_optimizer()
        metadata = load_training_checkpoint(
            self.path, restored_model, restored_optimizer, CONFIG
        )

        self.assertEqual(metadata["epoch"], 7)
        self.assertEqual(metadata["best_dice"], 0.8125)
        self.assertEqual(metadata["total_train_time"], 123.25)
        sidecar = read_training_checkpoint_metadata(self.path)
        self.assertEqual(sidecar["epoch"], 7)
        self.assertEqual(sidecar["best_dice"], 0.8125)
        for key, value in restored_model.state_dict().items():
            torch.testing.assert_close(value, expected_model[key])

        restored_state = restored_optimizer.state_dict()
        self.assertEqual(restored_state["param_groups"], expected_optimizer["param_groups"])
        self.assertTrue(restored_state["state"])
        for parameter_state in restored_optimizer.state.values():
            self.assertIn("exp_avg", parameter_state)
            self.assertIn("exp_avg_sq", parameter_state)
            for value in parameter_state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device, next(restored_model.parameters()).device)

    def test_rng_round_trip(self):
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        model, optimizer = make_model_and_optimizer()
        save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)
        expected = (random.random(), np.random.rand(), torch.rand(4))

        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        load_training_checkpoint(self.path, model, optimizer, CONFIG)
        actual = (random.random(), np.random.rand(), torch.rand(4))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_rng_and_optimizer_device_round_trip(self):
        device = "cuda"
        torch.cuda.manual_seed_all(94)
        model, optimizer = make_model_and_optimizer(device)
        take_step(model, optimizer)
        save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)
        expected = torch.rand(4, device=device)

        torch.cuda.manual_seed_all(4)
        restored_model, restored_optimizer = make_model_and_optimizer(device)
        load_training_checkpoint(self.path, restored_model, restored_optimizer, CONFIG)
        torch.testing.assert_close(torch.rand(4, device=device), expected)
        for parameter, state in restored_optimizer.state.items():
            for value in state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device, parameter.device)

    def test_configuration_mismatch_does_not_mutate_model(self):
        model, optimizer = make_model_and_optimizer()
        save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)
        target_model, target_optimizer = make_model_and_optimizer()
        before = {key: value.clone() for key, value in target_model.state_dict().items()}

        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            load_training_checkpoint(
                self.path,
                target_model,
                target_optimizer,
                {**CONFIG, "learning_rate": 2e-4},
            )
        for key, value in target_model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_missing_and_corrupt_checkpoint_errors(self):
        model, optimizer = make_model_and_optimizer()
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(self.path, model, optimizer, CONFIG)

        self.path.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(ValueError, "Could not read training checkpoint"):
            load_training_checkpoint(self.path, model, optimizer, CONFIG)

    def test_atomic_replace_and_temporary_file_cleanup(self):
        self.path.write_bytes(b"old checkpoint")
        model, optimizer = make_model_and_optimizer()
        real_replace = __import__("os").replace
        observations = []

        def inspect_then_replace(source, destination):
            if Path(destination) == self.path:
                observations.append((Path(source).is_file(), Path(destination).read_bytes()))
            real_replace(source, destination)

        with mock.patch("utils.training_checkpoint.os.replace", side_effect=inspect_then_replace) as replace:
            save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)

        self.assertEqual(replace.call_count, 2)
        self.assertEqual(observations, [(True, b"old checkpoint")])
        self.assertGreater(self.path.stat().st_size, len(b"old checkpoint"))
        self.assertEqual(list(self.path.parent.glob("resume.pth.*.tmp")), [])
        self.assertTrue(checkpoint_metadata_path(self.path).is_file())

    def test_metadata_detects_checkpoint_replacement(self):
        model, optimizer = make_model_and_optimizer()
        save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)
        self.path.write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            read_training_checkpoint_metadata(self.path)

    def test_temporary_file_is_removed_when_save_fails(self):
        model, optimizer = make_model_and_optimizer()
        with mock.patch(
            "utils.training_checkpoint.torch.save", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_training_checkpoint(self.path, model, optimizer, 1, 0.5, 2.0, CONFIG)

        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob("resume.pth.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
