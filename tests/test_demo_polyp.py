import unittest

import numpy as np
from PIL import Image

from scripts.demo_polyp import labeled_panel, overlay_prediction


class DemoPolypTests(unittest.TestCase):
    def test_overlay_and_panel(self):
        image = Image.fromarray(np.full((20, 30, 3), 100, dtype=np.uint8))
        prediction = np.zeros((20, 30), dtype=np.float32)
        prediction[:, 15:] = 1.0
        overlay = overlay_prediction(image, prediction)
        self.assertEqual(overlay.size, image.size)
        overlay_array = np.asarray(overlay)
        self.assertGreater(overlay_array[:, 20:, 0].mean(), overlay_array[:, :10, 0].mean())
        panel = labeled_panel(
            [("Input", image), ("Overlay", overlay)], "Dice=0.5", height=100
        )
        self.assertGreater(panel.width, image.width)
        self.assertEqual(panel.height, 172)


if __name__ == "__main__":
    unittest.main()
