import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import train_polyp
from utils.training_checkpoint import load_training_checkpoint, save_training_checkpoint


class TinyFourOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, images):
        prediction = self.output(images)
        return prediction, prediction, prediction, prediction


class TrainingLoopResumeSmokeTests(unittest.TestCase):
    def test_protocol_constants_are_unchanged(self):
        self.assertEqual(train_polyp.SIZE_RATES, [0.75, 1.0, 1.25])
        self.assertEqual(len([subset for subset in train_polyp.ss if subset]), 15)

    def test_fresh_output_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "experiment-last.pth"
            existing.write_bytes(b"protected")
            with self.assertRaisesRegex(FileExistsError, "would overwrite"):
                train_polyp.ensure_fresh_outputs_do_not_exist(
                    directory, "experiment", None
                )
            self.assertEqual(existing.read_bytes(), b"protected")

    def test_learning_rate_behavior_continues_from_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyFourOutputModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            train_polyp.adjust_lr(optimizer, 1e-4, 300, 0.1, 300)
            checkpoint = Path(directory) / "resume.pth"
            config = {"schedule": "upstream"}
            save_training_checkpoint(
                checkpoint, model, optimizer, 300, 0.8, 10.0, config
            )

            train_polyp.adjust_lr(optimizer, 1e-4, 301, 0.1, 300)
            uninterrupted_lr = optimizer.param_groups[0]["lr"]
            restored_model = TinyFourOutputModel()
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=1e-4
            )
            load_training_checkpoint(
                checkpoint, restored_model, restored_optimizer, config
            )
            train_polyp.adjust_lr(restored_optimizer, 1e-4, 301, 0.1, 300)
            self.assertEqual(
                restored_optimizer.param_groups[0]["lr"], uninterrupted_lr
            )

    @unittest.skipUnless(torch.cuda.is_available(), "training loop requires CUDA")
    def test_completed_epoch_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            train_polyp.opt = SimpleNamespace(
                batchsize=1,
                img_size=32,
                clip=0.5,
                epoch=3,
                train_save=directory + "\\",
            )
            train_polyp.total_step = 1
            train_polyp.best = 0.0
            train_polyp.total_train_time = 0.0
            train_polyp.dict_plot = {
                "CVC-ClinicDB": [],
                "Kvasir": [],
                "CVC-ColonDB": [],
                "CVC-300": [],
                "test": [],
            }
            model = TinyFourOutputModel().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loader = [
                (
                    torch.rand(1, 3, 32, 32),
                    torch.rand(1, 1, 32, 32),
                )
            ]
            resume_path = Path(directory) / "resume.pth"
            config = {"architecture": "tiny-four-output"}

            with mock.patch("train_polyp.test", return_value=(0.75, 2)):
                train_polyp.train(
                    loader,
                    model,
                    optimizer,
                    epoch=1,
                    test_path="unused",
                    model_name="experiment",
                    resume_checkpoint_path=resume_path,
                    resume_config=config,
                )

            self.assertTrue((Path(directory) / "experiment-last.pth").is_file())
            self.assertTrue((Path(directory) / "experiment-best.pth").is_file())
            self.assertTrue(resume_path.is_file())
            restored_model = TinyFourOutputModel().cuda()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
            metadata = load_training_checkpoint(
                resume_path, restored_model, restored_optimizer, config
            )
            self.assertEqual(metadata["epoch"], 1)
            self.assertEqual(metadata["best_dice"], 0.75)
            self.assertGreater(metadata["total_train_time"], 0.0)
            self.assertTrue(restored_optimizer.state)


if __name__ == "__main__":
    unittest.main()
