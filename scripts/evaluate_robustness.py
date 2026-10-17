import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from utils.image_corruptions import CORRUPTIONS, apply_corruption, apply_recovery
from utils.polyp_evaluation import (
    OFFICIAL_DATASETS,
    ensure_empty_output,
    evaluate_experiment,
    load_model,
    sha256_file,
)


def deterministic_seed(base_seed, image_path, corruption, severity):
    value = f"{base_seed}|{Path(image_path).resolve()}|{corruption}|{severity}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def write_summary_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg_plot(path, scenario_rows, clean_score):
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 35, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"]
    by_corruption = {}
    for row in scenario_rows:
        if row["corruption"] != "clean":
            by_corruption.setdefault(row["corruption"], []).append(row)

    def x(severity):
        return left + (severity - 1) * plot_width / 4

    def y(score):
        return top + (1.0 - max(0.0, min(1.0, score))) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Weighted official Dice by corruption severity</text>',
    ]
    for tick in range(0, 11, 2):
        score = tick / 10
        yy = y(score)
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>')
        elements.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{score:.1f}</text>')
    for severity in range(1, 6):
        xx = x(severity)
        elements.append(f'<text x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="12">{severity}</text>')
    elements.append(f'<line x1="{left}" y1="{y(clean_score):.1f}" x2="{width-right}" y2="{y(clean_score):.1f}" stroke="#333333" stroke-dasharray="6,4"/>')
    elements.append(f'<text x="{width-right}" y="{y(clean_score)-5:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">clean {clean_score:.4f}</text>')
    legend_y = height - 18
    for index, (name, rows) in enumerate(sorted(by_corruption.items())):
        rows = sorted(rows, key=lambda item: item["severity"])
        points = " ".join(f"{x(row['severity']):.1f},{y(row['weighted_official_dice']):.1f}" for row in rows)
        color = colors[index % len(colors)]
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in rows:
            elements.append(f'<circle cx="{x(row["severity"]):.1f}" cy="{y(row["weighted_official_dice"]):.1f}" r="3" fill="{color}"/>')
        legend_x = left + index * 130
        elements.append(f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="11" fill="{color}">{name}</text>')
    elements.append(f'<text x="{left + plot_width/2:.1f}" y="{height-38}" text-anchor="middle" font-family="sans-serif" font-size="13">Severity</text>')
    elements.append(f'<text transform="translate(18 {top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Official Dice</text>')
    elements.append("</svg>")
    Path(path).write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="data/polyp/TestDataset")
    parser.add_argument("--datasets", nargs="+", default=list(OFFICIAL_DATASETS))
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=3160)
    parser.add_argument("--recovery", choices=["none", "clahe", "gamma"], default="none")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-size", type=int, default=352)
    args = parser.parse_args()

    unknown = sorted(set(args.corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruption(s): {unknown}")
    if any(severity not in range(1, 6) for severity in args.severities):
        raise ValueError("Severities must be integers from 1 through 5.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable.")

    output = Path(args.output)
    ensure_empty_output(output)
    model, checkpoint_kind = load_model(
        args.checkpoint, device=args.device, test_size=args.test_size
    )
    checkpoint_hash = sha256_file(args.checkpoint)

    def recovered_clean(image, _path):
        return apply_recovery(image, args.recovery)

    clean_report = evaluate_experiment(
        model,
        args.dataset_root,
        args.datasets,
        output / "clean",
        args.checkpoint,
        checkpoint_kind,
        device=args.device,
        test_size=args.test_size,
        input_operator=recovered_clean,
        extra_metadata={"corruption": "clean", "recovery": args.recovery},
        checkpoint_sha256=checkpoint_hash,
    )
    clean_by_dataset = {
        item["dataset"]: item for item in clean_report["dataset_summaries"]
    }
    rows = [
        {
            "corruption": "clean",
            "severity": 0,
            "dataset": item["dataset"],
            "image_count": item["image_count"],
            "official_dice": item["official_dice"],
            "official_iou": item["official_iou"],
            "dice_degradation": 0.0,
            "iou_degradation": 0.0,
            "recovery": args.recovery,
        }
        for item in clean_report["dataset_summaries"]
    ]
    scenario_rows = [
        {
            "corruption": "clean",
            "severity": 0,
            "weighted_official_dice": clean_report["weighted_summary"]["official_dice"],
            "weighted_official_iou": clean_report["weighted_summary"]["official_iou"],
            "dice_degradation": 0.0,
            "iou_degradation": 0.0,
        }
    ]

    for corruption in args.corruptions:
        for severity in args.severities:
            def operator(image, image_path, corruption=corruption, severity=severity):
                seed = deterministic_seed(args.seed, image_path, corruption, severity)
                corrupted = apply_corruption(image, corruption, severity, seed)
                return apply_recovery(corrupted, args.recovery)

            scenario = f"{corruption}_severity_{severity}"
            report = evaluate_experiment(
                model,
                args.dataset_root,
                args.datasets,
                output / scenario,
                args.checkpoint,
                checkpoint_kind,
                device=args.device,
                test_size=args.test_size,
                input_operator=operator,
                extra_metadata={
                    "corruption": corruption,
                    "severity": severity,
                    "seed": args.seed,
                    "recovery": args.recovery,
                },
                checkpoint_sha256=checkpoint_hash,
            )
            for item in report["dataset_summaries"]:
                clean = clean_by_dataset[item["dataset"]]
                rows.append(
                    {
                        "corruption": corruption,
                        "severity": severity,
                        "dataset": item["dataset"],
                        "image_count": item["image_count"],
                        "official_dice": item["official_dice"],
                        "official_iou": item["official_iou"],
                        "dice_degradation": clean["official_dice"] - item["official_dice"],
                        "iou_degradation": clean["official_iou"] - item["official_iou"],
                        "recovery": args.recovery,
                    }
                )
            weighted = report["weighted_summary"]
            scenario_rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "weighted_official_dice": weighted["official_dice"],
                    "weighted_official_iou": weighted["official_iou"],
                    "dice_degradation": clean_report["weighted_summary"]["official_dice"] - weighted["official_dice"],
                    "iou_degradation": clean_report["weighted_summary"]["official_iou"] - weighted["official_iou"],
                }
            )

    write_summary_csv(output / "robustness_by_dataset.csv", rows)
    write_summary_csv(output / "robustness_scenarios.csv", scenario_rows)
    with (output / "robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "seed": args.seed,
                "recovery": args.recovery,
                "scenarios": scenario_rows,
                "by_dataset": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    write_svg_plot(
        output / "robustness_official_dice.svg",
        scenario_rows,
        clean_report["weighted_summary"]["official_dice"],
    )
    print("Robustness evaluation complete:", output.resolve())


if __name__ == "__main__":
    main()
