"""Local Streamlit UI for the verified PVT-GCASCADE polyp segmentation pipeline.

Run from the repository root:
    python -m streamlit run scripts/web_demo_polyp.py

Uses the repository's existing model loader, preprocessing, prediction and metrics.
Nothing is uploaded to an external service by this app.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from scripts.demo_polyp import overlay_prediction, predict_image
from utils.polyp_evaluation import (
    OFFICIAL_DATASETS,
    conventional_binary_metrics,
    load_model,
    official_binary_metrics,
    pair_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments/official_run03/models/"
    / "Polyp_PVT_GCASCADE_img_size352bs4_Run3-best.pth"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data/polyp/TestDataset"
TEST_SIZE = 352

st.set_page_config(
    page_title="G-CASCADE | Polyp segmentation", page_icon="🔬", layout="wide"
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; max-width: 1500px;}
      .eyebrow {font-size: .83rem; letter-spacing: .09em; text-transform: uppercase; color: #70b7de;}
      .subline {color: #9ba9bb; font-size: .94rem;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def as_mask(image: Image.Image) -> np.ndarray:
    mask = np.asarray(image.convert("L"), dtype=np.float32)
    return mask / (float(mask.max()) + 1e-8)


@st.cache_resource(show_spinner="Loading PVT-GCASCADE best checkpoint…")
def cached_model(checkpoint: str, device: str, test_size: int, modification_ns: int):
    # modification_ns is a cache key to invalidate when the checkpoint changes.
    del modification_ns
    return load_model(checkpoint, device=device, test_size=test_size)


def open_image_file(upload) -> Image.Image:
    with Image.open(upload) as img:
        return img.convert("RGB")


def show_image(label: str, image: Image.Image, caption: str = "") -> None:
    st.markdown(f"**{label}**")
    st.image(image, use_container_width=True, caption=caption or None)


def main():
    st.markdown('<div class="eyebrow">EN3160 · WACV 2024 reproduction</div>', unsafe_allow_html=True)
    st.title("G-CASCADE · Polyp segmentation")
    st.markdown(
        '<div class="subline">PVTv2-B2 encoder · additive G-CASCADE decoder · '
        'best checkpoint: epoch 19 · 352 × 352 input</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This demo uses the saved **epoch-19 best checkpoint**. "
        "Dice/IoU are shown only when a ground-truth mask is supplied. "
        "The four reference datasets were used for checkpoint selection; "
        "their results are validation metrics, not independent held-out test scores."
    )

    with st.sidebar:
        st.header("Model settings")
        checkpoint = st.text_input("Best checkpoint", str(DEFAULT_CHECKPOINT))
        dataset_root = st.text_input("Validation dataset root", str(DEFAULT_DATA_ROOT))
        available_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Device", available_devices)
        st.caption("The app runs locally. It does not train or update model weights.")
        st.divider()
        st.markdown("**Reproduction summary**")
        st.metric("Weighted validation Dice", "0.849587")
        st.caption("602 validation images · best epoch 19 · stopped after epoch 90 / 199")

    mode = st.radio(
        "Image source",
        ["Choose from validation dataset", "Upload an image"],
        horizontal=True,
    )

    image = None
    mask_image = None
    selected_name = "uploaded_image"

    if mode == "Choose from validation dataset":
        a, b = st.columns([1, 2])
        with a:
            dataset = st.selectbox("Dataset", list(OFFICIAL_DATASETS))
        folder = Path(dataset_root).expanduser() / dataset
        try:
            pairs = pair_dataset(folder)
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Cannot open dataset: {exc}")
            st.stop()
        with b:
            options = {image_path.name: (image_path, mask_path) for image_path, mask_path in pairs}
            name = st.selectbox("Test image", list(options))
        image_path, mask_path = options[name]
        selected_name = f"{dataset}_{image_path.stem}"
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        with Image.open(mask_path) as img:
            mask_image = img.convert("L")
        st.caption(f"Ground-truth annotation automatically matched to `{mask_path.name}`.")
    else:
        a, b = st.columns(2)
        with a:
            uploaded_image = st.file_uploader(
                "Upload a colonoscopy image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key="uploaded_image",
            )
        with b:
            uploaded_mask = st.file_uploader(
                "Optional ground-truth mask", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key="uploaded_mask",
            )
        if uploaded_image is not None:
            try:
                image = open_image_file(uploaded_image)
                selected_name = Path(uploaded_image.name).stem
            except (OSError, ValueError) as exc:
                st.error(f"Could not read image: {exc}")
                st.stop()
        if uploaded_mask is not None:
            try:
                with Image.open(uploaded_mask) as img:
                    mask_image = img.convert("L")
            except (OSError, ValueError) as exc:
                st.error(f"Could not read ground truth: {exc}")
                st.stop()
        if image is None:
            st.info("Choose an image to enable inference.")

    st.write("")
    run_clicked = st.button("▶ Run segmentation", type="primary", disabled=image is None)
    if not run_clicked:
        st.caption("Select a dataset image or upload one, then click **Run segmentation**.")
        return

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        st.stop()
    if mask_image is not None and image is not None and mask_image.size != image.size:
        st.error(
            f"Ground-truth dimensions {mask_image.size} do not match image dimensions {image.size}. "
            "Please choose a matching mask."
        )
        st.stop()

    try:
        with st.spinner("Running inference with the saved model…"):
            model, checkpoint_kind = cached_model(
                str(checkpoint_path.resolve()), device, TEST_SIZE, checkpoint_path.stat().st_mtime_ns
            )
            output_shape = (
                (mask_image.height, mask_image.width)
                if mask_image is not None else (image.height, image.width)
            )
            prediction = predict_image(
                model, image, device=device, test_size=TEST_SIZE, output_shape=output_shape
            )
    except Exception as exc:
        st.error(f"Inference failed: {type(exc).__name__}: {exc}")
        st.stop()

    predicted_binary = Image.fromarray(
        np.asarray((prediction >= 0.5) * 255, dtype=np.uint8), mode="L"
    )
    overlay = overlay_prediction(image, prediction)
    st.success(f"Inference complete · {selected_name} · {checkpoint_kind}")

    if mask_image is not None:
        ground_truth = as_mask(mask_image)
        official_dice, official_iou = official_binary_metrics(prediction, ground_truth)
        conventional_dice, conventional_iou = conventional_binary_metrics(prediction, ground_truth)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Official Dice", f"{official_dice:.4f}")
        m2.metric("Official IoU", f"{official_iou:.4f}")
        m3.metric("Conventional Dice", f"{conventional_dice:.4f}")
        m4.metric("Conventional IoU", f"{conventional_iou:.4f}")
    else:
        st.warning("No ground-truth mask was supplied. Prediction is available, but Dice/IoU cannot be calculated.")

    panels = [
        ("Input image", image),
        ("Predicted mask", predicted_binary),
        ("Red overlay", overlay),
    ]
    if mask_image is not None:
        panels.append(("Ground truth", mask_image))
    cols = st.columns(len(panels))
    for col, (label, panel) in zip(cols, panels):
        with col:
            show_image(label, panel)

    a, b = st.columns(2)
    with a:
        st.download_button(
            "Download predicted mask", png_bytes(predicted_binary),
            file_name=f"{selected_name}_predicted_mask.png", mime="image/png",
        )
    with b:
        st.download_button(
            "Download red overlay", png_bytes(overlay),
            file_name=f"{selected_name}_overlay.png", mime="image/png",
        )

    with st.expander("How this prediction is produced"):
        st.markdown(
            "Image → ImageNet normalization at 352×352 → PVTv2-B2 encoder "
            "→ G-CASCADE decoder → four prediction logits → sum → bilinear resize "
            "→ sigmoid → per-image min–max normalization → threshold at 0.5. "
            "The official metric rounds each image's Dice/IoU to four decimals."
        )


if __name__ == "__main__":
    main()
