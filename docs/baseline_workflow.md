# EN3160 PVT-GCASCADE baseline workflow

This document covers segmented training, official-compatible evaluation, controlled corruption experiments, and the local demo. It supplements `docs/resumable_training.md`. No command below implies that the 199-epoch reproduction or complete robustness study has already been run.

## Ownership and credit

This EN3160 project is shared by **Abdul** and **Mokshan**. Preserve both contributors' work and use separate experiment names/directories. The upstream method is G-CASCADE by Md Mostafijur Rahman and Radu Marculescu, WACV 2024.

## Required layout

```text
data/polyp/
  TrainDataset/{images,masks}/
  TestDataset/<dataset>/{images,masks}/
pretrained_pth/pvt/pvt_v2_b2.pth
experiments/<experiment>/
  models/
  checkpoints/training-resume.pth
  checkpoints/training-resume.pth.metadata.json
  evaluation/
  robustness/
```

Datasets, pretrained weights, experiment outputs, logs, predicted masks, and checkpoints are ignored by Git. Never use `git add -A` in this repository.

## Five-epoch training sessions

Activate `gcascade`, then inspect Session 1 without training:

```powershell
.\scripts\run_polyp_session.ps1 -DryRun
```

Launch Session 1 only after explicit approval:

```powershell
.\scripts\run_polyp_session.ps1
```

Run the same command later for Session 2 and every subsequent session. The script reads the checkpoint metadata sidecar, validates that it still matches the full checkpoint's filename, size, and nanosecond modification time, and chooses the fixed session boundary. It does not infer progress from filenames.

The 40 fixed session boundaries cover epochs 1–199 exactly once:

- Session 1: epochs 1–5, exclusive `--epoch 6`
- Session 2: epochs 6–10, exclusive `--epoch 11`
- Session 3: epochs 11–15, exclusive `--epoch 16`
- Session 40: epochs 196–199, exclusive `--epoch 200`

If execution stops after an intermediate completed epoch, the next invocation finishes only the remainder of that fixed five-epoch block. Each completed epoch has a full-state checkpoint; incomplete epochs are repeated.

Use a different stable experiment root/name when needed:

```powershell
.\scripts\run_polyp_session.ps1 `
  -ExperimentName Polyp_PVT_GCASCADE_img_size352bs4_Run3 `
  -ExperimentRoot .\experiments\official_run03
```

The launcher verifies the branch, required data/pretrained files, `gcascade` environment, PyTorch, CUDA, and GPU. A non-empty experiment directory without a valid checkpoint is rejected. Fresh training cannot overwrite existing model, checkpoint, or log files. After training returns successfully, the launcher verifies that the checkpoint reports the planned ending epoch.

### Progress only

This reads the small validated sidecar and does not deserialize model/optimizer tensors:

```powershell
python -m scripts.training_progress `
  --checkpoint .\experiments\Polyp_PVT_GCASCADE_img_size352bs4_Run2\checkpoints\training-resume.pth
```

It reports the completed epoch, best Dice, current/next session, remaining epochs, checkpoint timestamp, next bounds, and completion status.

## Safe stop and restart

Wait for `Saved resumable checkpoint:` before stopping normally with `Ctrl+C`. If interrupted mid-epoch, the most recent completed epoch remains the restart point. Do not power off while a checkpoint is being written. The `.metadata.json` sidecar must remain beside its `.pth` file; a mismatch is treated as corruption or an incomplete external copy.

For archival integrity:

```powershell
Get-FileHash .\experiments\<experiment>\checkpoints\training-resume.pth -Algorithm SHA256
```

## Baseline evaluation

Evaluate a completed model-only `-best.pth` file into a new output directory:

```powershell
python -m scripts.evaluate_polyp `
  --checkpoint .\experiments\<experiment>\models\<experiment>-best.pth `
  --dataset-root .\data\polyp\TestDataset `
  --output .\experiments\<experiment>\evaluation\best
```

Add `--export-masks` to write normalized grayscale predictions. The evaluator refuses a non-empty output directory. It validates image/mask pairs by case-insensitive stem and records checkpoint path/SHA-256, checkpoint kind, Git commit/dirty state, dataset root, image counts, test size, threshold, aggregation, and normalization.

Outputs include:

- `<dataset>/per_image.csv`
- `<dataset>/summary.json`
- `dataset_summary.csv`
- `evaluation.json`
- optional `<dataset>/predicted_masks/*.png`

### Metric compatibility

The named `official_*` columns preserve `test_polyp.py` behavior:

1. Add the four decoder logits.
2. Bilinearly resize with `align_corners=False`.
3. Apply sigmoid.
4. Apply per-image min-max normalization.
5. Threshold prediction and ground truth at 0.5.
6. Use smoothing value 1.
7. Round every image's Dice/IoU to four decimals before dataset averaging.

The `conventional_*` columns use unsmoothed, unrounded binary Dice/IoU (with an empty/empty score of 1). Keep these labels separate in reports.

The upstream `test_polyp.py` currently defines `--testsize` but references `opt.img_size` during PVT construction, and it scatters CSV output in the repository root. The wrapper leaves that upstream file untouched while correcting the CLI plumbing and output organization; its metric path was verified against the upstream calculation and the recorded epoch-1 scores.

The training script chooses its best checkpoint using image-count-weighted Dice across these same four evaluation folders. This is model selection on evaluation/test data, so the protocol has test-set leakage and is not a completely independent held-out assessment. Report that limitation with all final results.

Paper values are references, not reproduced values: CVC-ClinicDB 94.68/90.18, Kvasir 92.74/87.90, ColonDB 82.61/74.60, and EndoScene/CVC-300 90.56/83.87 Dice/mIoU (%). Source: [WACV 2024 supplementary material](https://openaccess.thecvf.com/content/WACV2024/supplemental/Rahman_G-CASCADE_Efficient_Cascaded_WACV_2024_supplemental.pdf).

## Robustness evaluation

Run clean plus all six corruptions and severities 1–5 only when the compute cost is acceptable:

```powershell
python -m scripts.evaluate_robustness `
  --checkpoint .\experiments\<experiment>\models\<experiment>-best.pth `
  --dataset-root .\data\polyp\TestDataset `
  --output .\experiments\<experiment>\robustness\baseline `
  --seed 3160
```

For a bounded check:

```powershell
python -m scripts.evaluate_robustness `
  --checkpoint .\model_pth\smoke_official_01\Polyp_PVT_GCASCADE_img_size352bs4_Run1-last.pth `
  --datasets CVC-300 `
  --corruptions gaussian_noise `
  --severities 2 `
  --output .\experiments\robustness_smoke
```

Implemented input-only corruptions are Gaussian noise, Gaussian blur, motion blur, low illumination, contrast reduction, and JPEG compression. Operators preserve dimensions and uint8 input range. Per-image seeds are derived from the global seed, image path, corruption, and severity. Ground-truth masks are never passed to corruption code and original files are never modified.

Outputs contain each scenario's ordinary evaluation report, `robustness_by_dataset.csv`, `robustness_scenarios.csv`, `robustness_summary.json`, and the measured-data SVG plot `robustness_official_dice.svg`. Degradation is clean score minus corrupted score.

Optional recovery uses the same checkpoint and metric path:

```powershell
# Run in a separate output directory and compare against the baseline run.
python -m scripts.evaluate_robustness <same arguments> --recovery clahe
```

`gamma` is also available. Recovery is applied to clean and corrupted inputs so clean-performance regressions remain visible. These are preprocessing comparisons, not trained robustness improvements.

## Local demo

Open a file-selection dialog and show input, evaluated input, prediction, and overlay:

```powershell
python -m scripts.demo_polyp --checkpoint .\experiments\<experiment>\models\<experiment>-best.pth
```

With a known image/mask and corruption comparison:

```powershell
python -m scripts.demo_polyp `
  --checkpoint .\model_pth\smoke_official_01\Polyp_PVT_GCASCADE_img_size352bs4_Run1-last.pth `
  --image .\data\polyp\TestDataset\CVC-300\images\<image>.png `
  --ground-truth .\data\polyp\TestDataset\CVC-300\masks\<image>.png `
  --corruption gaussian_noise --severity 2 --seed 3160
```

Use `--save-panel .\demo_outputs\panel.png` for a non-interactive PNG. When ground truth is supplied, both official and conventional Dice/IoU are shown. Epoch-1 checkpoints must be described as preliminary.

## Tests

```powershell
Get-ChildItem utils,scripts,tests -Filter *.py | ForEach-Object {
  python -m py_compile $_.FullName
}
python -m unittest -v `
  tests.test_training_checkpoint `
  tests.test_train_polyp_resume `
  tests.test_training_sessions `
  tests.test_polyp_evaluation `
  tests.test_image_corruptions `
  tests.test_demo_polyp
python -m tests.verify_real_pvt_checkpoint
```

Exact bitwise resumed equivalence is not claimed. CUDA nondeterminism and worker scheduling remain possible, and no uninterrupted-versus-resumed multi-epoch equality study has been performed.
