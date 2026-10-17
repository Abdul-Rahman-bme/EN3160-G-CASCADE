"""Reproducible PVT-GCASCADE polyp evaluation utilities."""

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


OFFICIAL_DATASETS = ("CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "CVC-300")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _files_by_stem(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"Duplicate case-insensitive stem '{path.stem}' in {directory}")
            result[key] = path
    return result


def pair_dataset(dataset_directory):
    dataset_directory = Path(dataset_directory)
    images = _files_by_stem(dataset_directory / "images")
    masks = _files_by_stem(dataset_directory / "masks")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Dataset pairing mismatch in {dataset_directory}: "
            f"missing_masks={missing_masks}, missing_images={missing_images}"
        )
    if not images:
        raise ValueError(f"No supported image/mask pairs found in {dataset_directory}")
    return [(images[key], masks[key]) for key in sorted(images)]


def official_binary_metrics(prediction, ground_truth):
    """Match test_polyp.py: threshold, smooth=1, round each image to 4 dp."""
    predicted = np.asarray(prediction) >= 0.5
    target = np.asarray(ground_truth) >= 0.5
    intersection = np.logical_and(predicted, target).sum(dtype=np.float64)
    union = np.logical_or(predicted, target).sum(dtype=np.float64)
    smooth = 1.0
    dice = (2.0 * intersection + smooth) / (
        predicted.sum(dtype=np.float64) + target.sum(dtype=np.float64) + smooth
    )
    iou = (intersection + smooth) / (union + smooth)
    return float(f"{dice:.4f}"), float(f"{iou:.4f}")


def conventional_binary_metrics(prediction, ground_truth):
    predicted = np.asarray(prediction) >= 0.5
    target = np.asarray(ground_truth) >= 0.5
    intersection = np.logical_and(predicted, target).sum(dtype=np.float64)
    predicted_count = predicted.sum(dtype=np.float64)
    target_count = target.sum(dtype=np.float64)
    union = np.logical_or(predicted, target).sum(dtype=np.float64)
    dice_denominator = predicted_count + target_count
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def normalize_official_prediction(logits, output_shape):
    resized = F.interpolate(
        logits,
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    prediction = resized.sigmoid().detach().cpu().numpy().squeeze()
    return (prediction - prediction.min()) / (
        prediction.max() - prediction.min() + 1e-8
    )


def image_transform(test_size):
    return transforms.Compose(
        [
            transforms.Resize((test_size, test_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def load_model(checkpoint_path, device="cuda", test_size=352):
    from lib.networks import PVT_GCASCADE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model = PVT_GCASCADE(
        n_class=1,
        img_size=test_size,
        k=11,
        padding=5,
        conv="mr",
        gcb_act="gelu",
        skip_aggregation="additive",
    )
    # Checkpoints must be local/trusted because full-state files include Python RNG data.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        checkpoint_kind = "full_training_state"
    else:
        state_dict = payload
        checkpoint_kind = "model_only"
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device)).eval()
    return model, checkpoint_kind


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repository_root):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def ensure_empty_output(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(
    model,
    dataset_directory,
    output_directory,
    device="cuda",
    test_size=352,
    export_masks=False,
    input_operator=None,
):
    pairs = pair_dataset(dataset_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    masks_directory = output_directory / "predicted_masks"
    if export_masks:
        masks_directory.mkdir(parents=True, exist_ok=True)
    preprocess = image_transform(test_size)
    rows = []

    for image_path, mask_path in pairs:
        with Image.open(image_path) as handle:
            original = handle.convert("RGB")
        evaluated_image = input_operator(original.copy(), image_path) if input_operator else original
        if evaluated_image.size != original.size:
            raise ValueError(f"Input operator changed image dimensions for {image_path}")
        image_tensor = preprocess(evaluated_image).unsqueeze(0).to(device)
        with Image.open(mask_path) as handle:
            ground_truth = np.asarray(handle.convert("L"), np.float32)
        ground_truth /= ground_truth.max() + 1e-8

        with torch.no_grad():
            outputs = model(image_tensor)
            combined_logits = outputs[0] + outputs[1] + outputs[2] + outputs[3]
        prediction = normalize_official_prediction(combined_logits, ground_truth.shape)
        official_dice, official_iou = official_binary_metrics(prediction, ground_truth)
        conventional_dice, conventional_iou = conventional_binary_metrics(
            prediction, ground_truth
        )
        rows.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                "official_dice": official_dice,
                "official_iou": official_iou,
                "conventional_dice": conventional_dice,
                "conventional_iou": conventional_iou,
            }
        )
        if export_masks:
            mask = Image.fromarray(np.uint8(np.clip(prediction * 255.0, 0, 255)))
            mask.save(masks_directory / f"{image_path.stem}.png")

    fields = list(rows[0])
    _write_csv(output_directory / "per_image.csv", rows, fields)
    summary = {
        "dataset": Path(dataset_directory).name,
        "image_count": len(rows),
        "official_dice": float(np.mean([row["official_dice"] for row in rows])),
        "official_iou": float(np.mean([row["official_iou"] for row in rows])),
        "conventional_dice": float(
            np.mean([row["conventional_dice"] for row in rows])
        ),
        "conventional_iou": float(
            np.mean([row["conventional_iou"] for row in rows])
        ),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary, rows


def evaluate_experiment(
    model,
    dataset_root,
    datasets,
    output_directory,
    checkpoint_path,
    checkpoint_kind,
    device="cuda",
    test_size=352,
    export_masks=False,
    input_operator=None,
    extra_metadata=None,
    checkpoint_sha256=None,
):
    output_directory = Path(output_directory)
    ensure_empty_output(output_directory)
    summaries = []
    for dataset in datasets:
        summary, _ = evaluate_dataset(
            model,
            Path(dataset_root) / dataset,
            output_directory / dataset,
            device=device,
            test_size=test_size,
            export_masks=export_masks,
            input_operator=input_operator,
        )
        summaries.append(summary)

    fields = list(summaries[0])
    _write_csv(output_directory / "dataset_summary.csv", summaries, fields)
    total_images = sum(item["image_count"] for item in summaries)
    weighted = {
        metric: sum(item[metric] * item["image_count"] for item in summaries)
        / total_images
        for metric in (
            "official_dice",
            "official_iou",
            "conventional_dice",
            "conventional_iou",
        )
    }
    repository_root = Path(__file__).resolve().parents[1]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256 or sha256_file(checkpoint_path),
        "checkpoint_kind": checkpoint_kind,
        "git": git_metadata(repository_root),
        "dataset_root": str(Path(dataset_root).resolve()),
        "datasets": list(datasets),
        "test_size": test_size,
        "threshold": 0.5,
        "prediction_aggregation": "sum_four_decoder_logits",
        "prediction_normalization": "sigmoid_then_per_image_minmax",
        "official_metric": "smooth_1_round_each_image_4dp_then_mean",
        "total_images": total_images,
        "weighted_summary": weighted,
        "dataset_summaries": summaries,
        "extra_metadata": extra_metadata or {},
    }
    with (output_directory / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report
