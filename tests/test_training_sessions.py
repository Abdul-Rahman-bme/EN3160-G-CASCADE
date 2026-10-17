import tempfile
import unittest
from pathlib import Path

import torch

from scripts.training_progress import progress
from utils.training_checkpoint import save_training_checkpoint
from utils.training_sessions import all_session_bounds, session_plan


class TrainingSessionTests(unittest.TestCase):
    def test_all_sessions_cover_epochs_exactly_once(self):
        bounds = all_session_bounds()
        self.assertEqual(len(bounds), 40)
        covered = []
        for session in bounds:
            covered.extend(range(session["start_epoch"], session["end_epoch"] + 1))
            self.assertEqual(
                session["exclusive_epoch_limit"], session["end_epoch"] + 1
            )
        self.assertEqual(covered, list(range(1, 200)))
        self.assertEqual(bounds[0], {
            "session_number": 1,
            "start_epoch": 1,
            "end_epoch": 5,
            "exclusive_epoch_limit": 6,
        })
        self.assertEqual(bounds[39], {
            "session_number": 40,
            "start_epoch": 196,
            "end_epoch": 199,
            "exclusive_epoch_limit": 200,
        })

    def test_plans_resume_and_partial_session(self):
        self.assertEqual(session_plan(5)["start_epoch"], 6)
        self.assertEqual(session_plan(5)["exclusive_epoch_limit"], 11)
        partial = session_plan(7)
        self.assertEqual(partial["session_number"], 2)
        self.assertEqual(partial["start_epoch"], 8)
        self.assertEqual(partial["end_epoch"], 10)
        self.assertEqual(partial["exclusive_epoch_limit"], 11)

    def test_completed_experiment_has_no_next_session(self):
        plan = session_plan(199)
        self.assertTrue(plan["completed"])
        self.assertIsNone(plan["exclusive_epoch_limit"])
        with self.assertRaises(ValueError):
            session_plan(200)

    def test_progress_reads_checkpoint_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pth"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            save_training_checkpoint(
                checkpoint, model, optimizer, 10, 0.8, 100.0, {"run": "test"}
            )
            result = progress(checkpoint)
            self.assertEqual(result["completed_epoch"], 10)
            self.assertEqual(result["session_number"], 3)
            self.assertEqual(result["start_epoch"], 11)
            self.assertEqual(result["end_epoch"], 15)
            self.assertEqual(result["exclusive_epoch_limit"], 16)
            self.assertEqual(result["best_dice"], 0.8)
            self.assertFalse(result["simulated"])

    def test_fresh_and_simulated_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing.pth"
            fresh = progress(checkpoint)
            self.assertFalse(fresh["checkpoint_exists"])
            self.assertEqual(fresh["session_number"], 1)
            simulated = progress(checkpoint, completed_epoch=195)
            self.assertTrue(simulated["simulated"])
            self.assertEqual(simulated["session_number"], 40)
            self.assertEqual(simulated["exclusive_epoch_limit"], 200)


if __name__ == "__main__":
    unittest.main()
