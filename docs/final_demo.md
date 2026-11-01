# PVT-GCASCADE Final Demonstration

## Reproduction status

- Baseline training stopped early after 90 of 199 planned epochs.
- The best validation checkpoint was selected at epoch 19.
- Model: PVTv2-B2 encoder with additive G-CASCADE decoder.
- Evaluation input size: 352 x 352.
- Evaluated 602 images across four validation datasets.
- This is a partial reproduction, not a completed 199-epoch experiment.

## Checkpoint and protocol

- Checkpoint: `experiments/official_run03/models/Polyp_PVT_GCASCADE_img_size352bs4_Run3-best.pth` (local only).
- The four decoder logits are summed for prediction.
- Predictions are resized, sigmoid-transformed, and per-image min-max normalized.
- Official metrics use a threshold of 0.5, smoothing of 1, per-image rounding to four decimal places, and dataset averaging.
- These datasets were also used in checkpoint selection; results are validation results, not independent held-out test estimates.

## Verified dataset results

| Dataset | Images | Official Dice | Official IoU |
|---|---:|---:|---:|
| CVC-ClinicDB | 62 | 0.928477 | 0.880758 |
| Kvasir | 100 | 0.916251 | 0.864641 |
| CVC-ColonDB | 380 | 0.811656 | 0.729943 |
| CVC-300 | 60 | 0.897185 | 0.831153 |
| **Image-count-weighted** | **602** | **0.849587** | **0.777938** |

## Run evaluation

From the repository root, activate the `gcascade` environment.

```powershell
python -m scripts.evaluate_polyp `
    --checkpoint .\experiments\official_run03\models\Polyp_PVT_GCASCADE_img_size352bs4_Run3-best.pth `
    --dataset-root .\data\polyp\TestDataset `
    --output .\experiments\official_run03\demo_recheck `
    --device cuda --test-size 352 --export-masks
```

The output directory must not already contain results.

## Run the interactive demo

```powershell
python -m scripts.demo_polyp `
    --checkpoint .\experiments\official_run03\models\Polyp_PVT_GCASCADE_img_size352bs4_Run3-best.pth `
    --device cuda --test-size 352
```

Choose a colonoscopy image when prompted. Add `--ground-truth`
with a matching mask to display per-image Dice and IoU.

## Demo workflow

1. Show the input colonoscopy image and annotated ground truth.
2. Load the best checkpoint and perform inference.
3. Show the predicted binary mask and red overlay.
4. Explain four decoder outputs and the final prediction.
5. Explain Dice/IoU and disclose the validation-data limitation.

The saved sample panel and exported masks remain local.
Model weights, datasets, and generated masks are not committed.
