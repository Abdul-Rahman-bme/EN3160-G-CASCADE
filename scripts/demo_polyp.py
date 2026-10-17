"""Small local PVT-GCASCADE image/mask/overlay viewer using Tkinter and Pillow."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk

from utils.image_corruptions import CORRUPTIONS, apply_corruption, apply_recovery
from utils.polyp_evaluation import (
    conventional_binary_metrics,
    image_transform,
    load_model,
    normalize_official_prediction,
    official_binary_metrics,
)


def predict_image(model, image, device="cuda", test_size=352, output_shape=None):
    tensor = image_transform(test_size)(image).unsqueeze(0).to(device)
    if output_shape is None:
        output_shape = (image.height, image.width)
    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs[0] + outputs[1] + outputs[2] + outputs[3]
    return normalize_official_prediction(logits, output_shape)


def overlay_prediction(image, prediction):
    binary = prediction >= 0.5
    base = np.asarray(
        image.resize((prediction.shape[1], prediction.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).copy()
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[binary] = 0.55 * base[binary] + 0.45 * red[binary]
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)))


def labeled_panel(items, metrics_text, height=360):
    panels = []
    font = ImageFont.load_default()
    for label, image in items:
        ratio = height / image.height
        resized = image.resize(
            (max(1, int(image.width * ratio)), height), Image.Resampling.BILINEAR
        )
        panel = Image.new("RGB", (resized.width, height + 42), "white")
        panel.paste(resized, (0, 22))
        ImageDraw.Draw(panel).text((8, 5), label, fill="black", font=font)
        panels.append(panel)
    width = sum(panel.width for panel in panels)
    result = Image.new("RGB", (width, height + 72), "white")
    offset = 0
    for panel in panels:
        result.paste(panel, (offset, 0))
        offset += panel.width
    ImageDraw.Draw(result).text((8, height + 50), metrics_text, fill="black", font=font)
    return result


def choose_image():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select an endoscopy image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")],
    )
    root.destroy()
    if not selected:
        raise RuntimeError("No image was selected.")
    return Path(selected)


def show_panel(panel, title):
    import tkinter as tk

    root = tk.Tk()
    root.title(title)
    display = panel.copy()
    max_width, max_height = 1500, 850
    scale = min(1.0, max_width / display.width, max_height / display.height)
    if scale < 1.0:
        display = display.resize(
            (int(display.width * scale), int(display.height * scale)),
            Image.Resampling.LANCZOS,
        )
    photo = ImageTk.PhotoImage(display)
    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image")
    parser.add_argument("--ground-truth")
    parser.add_argument("--corruption", choices=["clean", *CORRUPTIONS], default="clean")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3160)
    parser.add_argument("--recovery", choices=["none", "clahe", "gamma"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-size", type=int, default=352)
    parser.add_argument("--save-panel")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    with Image.open(image_path) as handle:
        clean = handle.convert("RGB")
    evaluated = clean.copy()
    if args.corruption != "clean":
        evaluated = apply_corruption(
            evaluated, args.corruption, args.severity, args.seed
        )
    evaluated = apply_recovery(evaluated, args.recovery)

    ground_truth = None
    if args.ground_truth:
        mask_path = Path(args.ground_truth)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
        with Image.open(mask_path) as handle:
            ground_truth = np.asarray(handle.convert("L"), np.float32)
        ground_truth /= ground_truth.max() + 1e-8

    model, checkpoint_kind = load_model(
        args.checkpoint, device=args.device, test_size=args.test_size
    )
    output_shape = ground_truth.shape if ground_truth is not None else None
    prediction = predict_image(
        model,
        evaluated,
        device=args.device,
        test_size=args.test_size,
        output_shape=output_shape,
    )
    predicted_mask = Image.fromarray(np.uint8((prediction >= 0.5) * 255), mode="L").convert("RGB")
    overlay = overlay_prediction(evaluated, prediction)
    items = [("Clean input", clean), ("Evaluated input", evaluated), ("Prediction", predicted_mask), ("Overlay", overlay)]
    metrics_text = f"checkpoint={checkpoint_kind}; corruption={args.corruption}; recovery={args.recovery}"
    if ground_truth is not None:
        official = official_binary_metrics(prediction, ground_truth)
        conventional = conventional_binary_metrics(prediction, ground_truth)
        ground_truth_image = Image.fromarray(np.uint8((ground_truth >= 0.5) * 255), mode="L").convert("RGB")
        items.append(("Ground truth", ground_truth_image))
        metrics_text += (
            f"; official Dice/IoU={official[0]:.4f}/{official[1]:.4f}; "
            f"conventional Dice/IoU={conventional[0]:.4f}/{conventional[1]:.4f}"
        )
    panel = labeled_panel(items, metrics_text)
    if args.save_panel:
        output_path = Path(args.save_panel)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"Demo panel already exists: {output_path}")
        panel.save(output_path)
        print("Saved demo panel:", output_path.resolve())
    else:
        show_panel(panel, "G-CASCADE polyp segmentation demo")


if __name__ == "__main__":
    main()
