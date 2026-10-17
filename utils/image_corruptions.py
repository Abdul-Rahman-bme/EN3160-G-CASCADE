"""Deterministic, input-only image corruptions for robustness evaluation."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


CORRUPTIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "low_illumination",
    "contrast_reduction",
    "jpeg_compression",
)


def _array(image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _severity_index(severity):
    if isinstance(severity, bool) or severity not in (1, 2, 3, 4, 5):
        raise ValueError("severity must be an integer from 1 through 5")
    return severity - 1


def apply_corruption(image, corruption, severity, seed=0):
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption}")
    index = _severity_index(severity)
    source = _array(image)
    rng = np.random.default_rng(seed)

    if corruption == "gaussian_noise":
        sigma = (5, 10, 20, 30, 40)[index]
        result = np.clip(
            source.astype(np.float32) + rng.normal(0, sigma, source.shape), 0, 255
        ).astype(np.uint8)
    elif corruption == "gaussian_blur":
        sigma = (0.5, 1.0, 1.5, 2.0, 3.0)[index]
        result = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif corruption == "motion_blur":
        size = (3, 5, 7, 9, 13)[index]
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        angle = float(rng.uniform(-45.0, 45.0))
        rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (size, size))
        kernel /= kernel.sum()
        result = cv2.filter2D(source, -1, kernel)
    elif corruption == "low_illumination":
        factor = (0.8, 0.65, 0.5, 0.35, 0.2)[index]
        result = np.clip(source.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif corruption == "contrast_reduction":
        factor = (0.8, 0.65, 0.5, 0.35, 0.2)[index]
        result = np.clip(
            (source.astype(np.float32) - 127.5) * factor + 127.5, 0, 255
        ).astype(np.uint8)
    else:
        quality = (85, 70, 50, 30, 10)[index]
        buffer = BytesIO()
        Image.fromarray(source).save(
            buffer, format="JPEG", quality=quality, subsampling=2, optimize=False
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            result = np.asarray(compressed.convert("RGB"), dtype=np.uint8)

    if result.shape != source.shape:
        raise RuntimeError(f"Corruption changed image shape: {source.shape} -> {result.shape}")
    return Image.fromarray(result, mode="RGB")


def apply_recovery(image, recovery):
    if recovery in (None, "none"):
        return image.copy()
    source = _array(image)
    if recovery == "clahe":
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        result = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB
        )
    elif recovery == "gamma":
        gamma = 0.8
        lookup = np.uint8(
            np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255)
        )
        result = cv2.LUT(source, lookup)
    else:
        raise ValueError(f"Unknown recovery method: {recovery}")
    return Image.fromarray(result, mode="RGB")
