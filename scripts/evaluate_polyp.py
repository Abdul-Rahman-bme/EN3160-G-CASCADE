import argparse
from pathlib import Path

import torch

from utils.polyp_evaluation import (
    OFFICIAL_DATASETS,
    evaluate_experiment,
    load_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="data/polyp/TestDataset")
    parser.add_argument("--datasets", nargs="+", default=list(OFFICIAL_DATASETS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-size", type=int, default=352)
    parser.add_argument("--export-masks", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable.")
    unknown = sorted(set(args.datasets) - set(OFFICIAL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown official dataset(s): {unknown}")

    model, checkpoint_kind = load_model(
        args.checkpoint, device=args.device, test_size=args.test_size
    )
    report = evaluate_experiment(
        model,
        args.dataset_root,
        args.datasets,
        Path(args.output),
        args.checkpoint,
        checkpoint_kind,
        device=args.device,
        test_size=args.test_size,
        export_masks=args.export_masks,
    )
    print("Evaluation complete:", Path(args.output).resolve())
    for summary in report["dataset_summaries"]:
        print(
            "{dataset}: n={image_count}, official Dice={official_dice:.6f}, "
            "official IoU={official_iou:.6f}".format(**summary)
        )
    print("Image-count-weighted official Dice:", report["weighted_summary"]["official_dice"])


if __name__ == "__main__":
    main()
