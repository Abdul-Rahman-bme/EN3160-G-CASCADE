"""Resumable training checkpoints for PVT-GCASCADE."""

import math
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_VERSION = 1
METADATA_VERSION = 1
REQUIRED_CHECKPOINT_FIELDS = (
    "epoch",
    "best_dice",
    "total_train_time",
    "model_state_dict",
    "optimizer_state_dict",
    "rng_state",
    "config",
)
REQUIRED_RNG_FIELDS = ("python", "numpy", "torch", "cuda")


def checkpoint_metadata_path(path):
    path = Path(path)
    return path.with_name(path.name + ".metadata.json")


def _write_json_atomically(path, payload):
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _metadata_payload(path, epoch, best_dice, total_train_time, config):
    stat = Path(path).stat()
    return {
        "metadata_version": METADATA_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_name": Path(path).name,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "total_train_time": float(total_train_time),
        "config": dict(config),
    }


def read_training_checkpoint_metadata(path):
    """Read and validate the small scheduling sidecar without loading tensors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")
    metadata_path = checkpoint_metadata_path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata sidecar not found: {metadata_path}"
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read checkpoint metadata {metadata_path}: {error}") from error

    stat = path.stat()
    expected = {
        "metadata_version": METADATA_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_name": path.name,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }
    differences = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if differences:
        raise ValueError(
            f"Checkpoint metadata does not match checkpoint file: {differences}"
        )
    for key in ("epoch", "best_dice", "total_train_time", "config"):
        if key not in metadata:
            raise ValueError(f"Checkpoint metadata is missing field: {key}")
    return metadata


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(states):
    missing = [key for key in REQUIRED_RNG_FIELDS if key not in states]
    if missing:
        raise ValueError(f"Missing RNG state fields: {missing}")

    _validate_rng_compatibility(states)

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None:
        torch.cuda.set_rng_state_all(states["cuda"])


def _validate_rng_compatibility(states):
    """Validate device-dependent RNG state without changing live streams."""
    if states["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG without CUDA.")
        if len(states["cuda"]) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG state count does not match the available CUDA devices."
            )


def _validate_model_state_dict(model, saved_state):
    """Reject incompatible model state before mutating the live model."""
    current_state = model.state_dict()
    missing = sorted(set(current_state) - set(saved_state))
    unexpected = sorted(set(saved_state) - set(current_state))
    incompatible_shapes = {
        key: (tuple(saved_state[key].shape), tuple(current_state[key].shape))
        for key in set(current_state).intersection(saved_state)
        if saved_state[key].shape != current_state[key].shape
    }
    if missing or unexpected or incompatible_shapes:
        raise ValueError(
            "Model state is incompatible: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={incompatible_shapes}"
        )


def _validate_optimizer_state_dict(optimizer, saved_state):
    """Check optimizer parameter-group structure before loading any state."""
    if (
        not isinstance(saved_state, dict)
        or "state" not in saved_state
        or "param_groups" not in saved_state
    ):
        raise ValueError("Optimizer state dictionary is malformed.")

    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = saved_state["param_groups"]
    if len(current_groups) != len(saved_groups):
        raise ValueError("Optimizer parameter-group count does not match.")
    for index, (current, saved) in enumerate(zip(current_groups, saved_groups)):
        if len(current["params"]) != len(saved.get("params", ())):
            raise ValueError(
                f"Optimizer parameter count differs in group {index}."
            )


def _validate_checkpoint(checkpoint, model, optimizer, expected_config):
    if not isinstance(checkpoint, dict):
        raise ValueError("Training checkpoint payload must be a dictionary.")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint version.")

    missing = [key for key in REQUIRED_CHECKPOINT_FIELDS if key not in checkpoint]
    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")

    if not isinstance(checkpoint["config"], dict):
        raise ValueError("Checkpoint configuration must be a dictionary.")
    if expected_config is not None:
        saved_config = checkpoint["config"]
        differences = {
            key: (saved_config.get(key), value)
            for key, value in expected_config.items()
            if key not in saved_config or saved_config[key] != value
        }
        if differences:
            raise ValueError(f"Training configuration mismatch: {differences}")

    epoch = checkpoint["epoch"]
    best_dice = checkpoint["best_dice"]
    total_train_time = checkpoint["total_train_time"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a non-negative integer.")
    if not isinstance(best_dice, (int, float)) or not math.isfinite(best_dice):
        raise ValueError("Checkpoint best Dice must be finite.")
    if (
        not isinstance(total_train_time, (int, float))
        or not math.isfinite(total_train_time)
        or total_train_time < 0
    ):
        raise ValueError(
            "Checkpoint total training time must be finite and non-negative."
        )
    if not isinstance(checkpoint["rng_state"], dict):
        raise ValueError("Checkpoint RNG state must be a dictionary.")
    missing_rng = [
        key for key in REQUIRED_RNG_FIELDS if key not in checkpoint["rng_state"]
    ]
    if missing_rng:
        raise ValueError(f"Missing RNG state fields: {missing_rng}")
    _validate_rng_compatibility(checkpoint["rng_state"])

    _validate_model_state_dict(model, checkpoint["model_state_dict"])
    _validate_optimizer_state_dict(optimizer, checkpoint["optimizer_state_dict"])


def save_training_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_dice,
    total_train_time,
    config,
):
    """Atomically write a checkpoint after an epoch has completed validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "total_train_time": float(total_train_time),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "config": dict(config),
    }

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    _write_json_atomically(
        checkpoint_metadata_path(path),
        _metadata_payload(path, epoch, best_dice, total_train_time, config),
    )


def load_training_checkpoint(path, model, optimizer, expected_config=None):
    """Restore model, optimizer and random states from a trusted checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")

    # Only load checkpoints generated locally or from a trusted source.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Could not read training checkpoint {path}: {error}") from error

    _validate_checkpoint(checkpoint, model, optimizer, expected_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move optimizer tensors to the corresponding parameter device.
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)

    restore_rng_state(checkpoint["rng_state"])

    return {
        "epoch": int(checkpoint["epoch"]),
        "best_dice": float(checkpoint["best_dice"]),
        "total_train_time": float(checkpoint["total_train_time"]),
        "config": checkpoint["config"],
    }
