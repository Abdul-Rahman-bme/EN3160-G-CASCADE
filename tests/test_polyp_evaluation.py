import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.polyp_evaluation import (
    conventional_binary_metrics,
    ensure_empty_output,
    evaluate_experiment,
    official_binary_metrics,
    pair_dataset,
)


class TinyEvaluationModel(torch.nn.Module):
    def forward(self, images):
        logits = images[:, :1]
        return logits, logits, logits, logits


def write_pair(dataset, stem, image_value=128, mask_value=255):
    image = np.full((12, 16, 3), image_value, dtype=np.uint8)
    mask = np.full((12, 16), mask_value, dtype=np.uint8)
    Image.fromarray(image).save(dataset / "images" / f"{stem}.jpg")
    Image.fromarray(mask).save(dataset / "masks" / f"{stem}.png")


class PolypEvaluationTests(unittest.TestCase):
    def test_official_metrics_match_literal_reference(self):
        prediction = np.array([[0.6, 0.4], [0.9, 0.2]], dtype=np.float32)
        target = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        predicted_binary = prediction >= 0.5
        target_binary = target >= 0.5
        intersection = (predicted_binary.reshape(-1) * target_binary.reshape(-1)).sum()
        union = np.logical_or(predicted_binary, target_binary).sum()
        expected_dice = float(
            "{:.4f}".format(
                (2 * intersection + 1)
                / (predicted_binary.sum() + target_binary.sum() + 1)
            )
        )
        expected_iou = float("{:.4f}".format((intersection + 1) / (union + 1)))
        self.assertEqual(
            official_binary_metrics(prediction, target),
            (expected_dice, expected_iou),
        )
        conventional = conventional_binary_metrics(prediction, target)
        self.assertNotEqual(conventional[0], expected_dice)

    def test_dataset_pairing_rejects_missing_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "images").mkdir()
            (dataset / "masks").mkdir()
            write_pair(dataset, "paired")
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                dataset / "images" / "missing.jpg"
            )
            with self.assertRaisesRegex(ValueError, "pairing mismatch"):
                pair_dataset(dataset)

    def test_evaluation_writes_csv_json_and_preserves_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "Tiny"
            (dataset / "images").mkdir(parents=True)
            (dataset / "masks").mkdir()
            write_pair(dataset, "one", image_value=80, mask_value=0)
            write_pair(dataset, "two", image_value=180, mask_value=255)
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"checkpoint identity only")
            output = root / "output"
            seen_mask_hashes = {
                path.name: path.read_bytes()
                for path in (dataset / "masks").iterdir()
            }

            report = evaluate_experiment(
                TinyEvaluationModel(),
                root / "datasets",
                ["Tiny"],
                output,
                checkpoint,
                "test",
                device="cpu",
                test_size=16,
                export_masks=True,
            )

            self.assertEqual(report["total_images"], 2)
            with (output / "Tiny" / "per_image.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            with (output / "evaluation.json").open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["total_images"], 2)
            self.assertEqual(
                seen_mask_hashes,
                {path.name: path.read_bytes() for path in (dataset / "masks").iterdir()},
            )
            self.assertEqual(
                len(list((output / "Tiny" / "predicted_masks").glob("*.png"))), 2
            )
            with self.assertRaises(FileExistsError):
                ensure_empty_output(output)


if __name__ == "__main__":
    unittest.main()
