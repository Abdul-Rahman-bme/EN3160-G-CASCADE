import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_robustness import deterministic_seed, write_svg_plot
from utils.image_corruptions import CORRUPTIONS, apply_corruption, apply_recovery


class ImageCorruptionTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[0:32, 0:48]
        array = np.stack(
            ((x * 5) % 256, (y * 7) % 256, ((x + y) * 3) % 256), axis=-1
        ).astype(np.uint8)
        self.image = Image.fromarray(array, mode="RGB")

    def test_all_corruptions_are_reproducible_and_preserve_shape(self):
        original = np.asarray(self.image).copy()
        for corruption in CORRUPTIONS:
            first = np.asarray(apply_corruption(self.image, corruption, 3, seed=123))
            second = np.asarray(apply_corruption(self.image, corruption, 3, seed=123))
            self.assertEqual(first.shape, original.shape, corruption)
            np.testing.assert_array_equal(first, second, err_msg=corruption)
        np.testing.assert_array_equal(np.asarray(self.image), original)

    def test_seed_changes_stochastic_corruption(self):
        first = np.asarray(apply_corruption(self.image, "gaussian_noise", 2, seed=1))
        second = np.asarray(apply_corruption(self.image, "gaussian_noise", 2, seed=2))
        self.assertFalse(np.array_equal(first, second))

    def test_recovery_and_validation(self):
        for recovery in ("none", "clahe", "gamma"):
            result = apply_recovery(self.image, recovery)
            self.assertEqual(result.size, self.image.size)
        with self.assertRaises(ValueError):
            apply_corruption(self.image, "gaussian_noise", 0)
        with self.assertRaises(ValueError):
            apply_corruption(self.image, "unknown", 1)

    def test_seed_derivation_and_svg_output(self):
        seed_a = deterministic_seed(3160, "image.jpg", "motion_blur", 2)
        seed_b = deterministic_seed(3160, "image.jpg", "motion_blur", 2)
        self.assertEqual(seed_a, seed_b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.svg"
            rows = [
                {
                    "corruption": "gaussian_noise",
                    "severity": severity,
                    "weighted_official_dice": 0.9 - severity * 0.05,
                }
                for severity in range(1, 6)
            ]
            write_svg_plot(path, rows, clean_score=0.9)
            content = path.read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            self.assertIn("gaussian_noise", content)


if __name__ == "__main__":
    unittest.main()
