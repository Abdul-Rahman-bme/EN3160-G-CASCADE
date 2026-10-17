"""Short, manual real-model compatibility check; it does not train an epoch."""

import gc
import tempfile
from pathlib import Path

import torch

from lib.networks import PVT_GCASCADE
from utils.training_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


MODEL_DIRECTORY = Path("model_pth/smoke_official_01")
MODEL_NAMES = (
    "Polyp_PVT_GCASCADE_img_size352bs4_Run1-best.pth",
    "Polyp_PVT_GCASCADE_img_size352bs4_Run1-last.pth",
)
CONFIG = {"architecture": "PVT_GCASCADE_real_smoke"}


def build_model():
    return PVT_GCASCADE(
        n_class=1,
        img_size=352,
        k=11,
        padding=5,
        conv="mr",
        gcb_act="gelu",
        skip_aggregation="additive",
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real PVT inference check.")

    model = build_model().cuda().eval()
    sample = torch.zeros(1, 3, 352, 352, device="cuda")
    for name in MODEL_NAMES:
        path = MODEL_DIRECTORY / name
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            outputs = model(sample)
        shapes = [tuple(output.shape) for output in outputs]
        assert shapes == [(1, 1, 352, 352)] * 4, shapes
        assert all(torch.isfinite(output).all() for output in outputs)
        print("PASS model-only inference:", path, shapes)

    del outputs, sample
    model = model.cpu().train()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.zeros_like(first_parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state, "AdamW state was not created"

    with tempfile.TemporaryDirectory() as directory:
        resume_path = Path(directory) / "real-pvt-resume.pth"
        save_training_checkpoint(
            resume_path, model, optimizer, 1, 0.6961616, 2794.76, CONFIG
        )
        del optimizer, model
        gc.collect()

        restored_model = build_model()
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        metadata = load_training_checkpoint(
            resume_path, restored_model, restored_optimizer, CONFIG
        )
        assert metadata["epoch"] == 1
        assert metadata["best_dice"] == 0.6961616
        assert restored_optimizer.state
        print("PASS real-model full-state resume with AdamW state")


if __name__ == "__main__":
    main()
