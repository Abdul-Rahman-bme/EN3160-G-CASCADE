# PVT-GCASCADE baseline: resumable training runbook

## Status and verified environment

This runbook prepares the official PVT-GCASCADE polyp baseline for an epoch-boundary, full-state resume. It does **not** claim that the existing epoch-1 weight-only files can resume training, and it does not authorize launching the full run.

Verified on October 17, 2026:

- Windows 11, branch `abu-resumable-training`, base commit `903f095`
- Python 3.10.21
- PyTorch 2.2.2+cu121 and torchvision 0.17.2+cu121
- NumPy 1.26.4 and SciPy 1.11.4
- NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM; CUDA available

The missing `typing_extensions` dependency was repaired on October 17, 2026 (`typing_extensions 4.16.0`). The code also no longer eagerly imports plotting packages that are irrelevant to polyp training, so the training CLI imports cleanly in the actual `gcascade` environment without a shim.

## Fixed scientific protocol

- PVTv2-B2 encoder and G-CASCADE decoder
- Additive skip aggregation
- Image size 352 × 352; batch size 4
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- Gradient clipping at 0.5
- Multi-scale rates `[0.75, 1.0, 1.25]`
- Weighted BCE + weighted IoU structure loss
- All 15 non-empty combinations of the four decoder outputs
- Validation on CVC-ClinicDB, Kvasir, CVC-ColonDB, and CVC-300
- Image-count-weighted aggregate Dice for best-model selection

The upstream loop remains `range(1, opt.epoch)`. Therefore `--epoch 200` executes epochs 1 through 199, not 200 epochs.

## Results: do not mix these two sources

Our preliminary first completed epoch on October 14, 2026 produced:

| Dataset | Our epoch-1 Dice |
|---|---:|
| CVC-ClinicDB | 0.814395 |
| Kvasir | 0.850391 |
| CVC-ColonDB | 0.606508 |
| CVC-300 | 0.884743 |
| Image-count-weighted aggregate | 0.6961616 |

These are **not** final reproduction results.

The paper's supplementary Table 3 reports the following PVT-GCASCADE results, averaged over five runs and expressed below as fractions rather than percentages:

| Dataset in paper | Published Dice | Published mIoU |
|---|---:|---:|
| CVC-ClinicDB | 0.9468 | 0.9018 |
| Kvasir | 0.9274 | 0.8790 |
| ColonDB | 0.8261 | 0.7460 |
| EndoScene | 0.9056 | 0.8387 |

Source: [G-CASCADE WACV 2024 supplementary material](https://openaccess.thecvf.com/content/WACV2024/supplemental/Rahman_G-CASCADE_Efficient_Cascaded_WACV_2024_supplemental.pdf). The repository labels its corresponding fourth validation folder `CVC-300`; retain the repository's naming in experiment logs and do not directly compare protocols without checking the dataset split.

## Checkpoint types and locations

Model-only files remain compatible with the official evaluation scripts:

- `<train_save>/<experiment_name>-last.pth`: model weights after the latest training epoch
- `<train_save>/<experiment_name>-best.pth`: model weights at the best weighted validation Dice

The new full-state file specified by `--save_resume_checkpoint` contains model weights, optimizer and AdamW momentum state, last completed epoch, best Dice, cumulative training time, RNG states, and strict continuation configuration. It is written to a temporary file, flushed, and atomically replaced only after validation and best-score updates complete.

The protected epoch-1 files in `model_pth/smoke_official_01/` are model-only. Never pass them to `--resume`.

## Fresh full-run command (prepare, then launch manually)

Activate the repaired `gcascade` environment, choose a new run name, confirm that none of the output paths exists, then run from the repository root:

```powershell
python train_polyp.py `
  --encoder PVT `
  --skip_aggregation additive `
  --epoch 200 `
  --lr 0.0001 `
  --optimizer AdamW `
  --batchsize 4 `
  --img_size 352 `
  --clip 0.5 `
  --decay_rate 0.1 `
  --decay_epoch 300 `
  --train_path .\data\polyp\TrainDataset\ `
  --test_path .\data\polyp\TestDataset\ `
  --experiment_name Polyp_PVT_GCASCADE_img_size352bs4_Run2 `
  --train_save .\model_pth\official_pvt_gcascade_run02\ `
  --save_resume_checkpoint .\checkpoints\official_pvt_gcascade_run02-resume.pth
```

Fresh training refuses to start if its best, last, or full-state destination already exists. Use a new experiment name/directory instead of deleting or overwriting an earlier run.

## Resume command

Use the same scientific settings and dataset roots. `--resume` loads the full state; when `--save_resume_checkpoint` is omitted, the resume file itself is atomically updated after each subsequent completed epoch.

```powershell
python train_polyp.py `
  --encoder PVT `
  --skip_aggregation additive `
  --epoch 200 `
  --lr 0.0001 `
  --optimizer AdamW `
  --batchsize 4 `
  --img_size 352 `
  --clip 0.5 `
  --decay_rate 0.1 `
  --decay_epoch 300 `
  --train_path .\data\polyp\TrainDataset\ `
  --test_path .\data\polyp\TestDataset\ `
  --experiment_name Polyp_PVT_GCASCADE_img_size352bs4_Run2 `
  --train_save .\model_pth\official_pvt_gcascade_run02\ `
  --resume .\checkpoints\official_pvt_gcascade_run02-resume.pth
```

The loader rejects changes to architecture, optimizer, learning rate, weight decay, image size, batch size, clipping, LR schedule, augmentation, dataset roots, loss identifier, decoder-combination count, or multi-scale rates. `--epoch` is intentionally not fixed in the checkpoint, so the exclusive stopping limit can be extended.

## Interruption, restart, and integrity checks

1. Monitor console output, `<train_save>/train.log`, GPU utilization, free disk space, and the modification times of the three output files.
2. A full-state checkpoint is safe only after `Saved resumable checkpoint:` appears. If interrupted earlier, resume from the previous completed epoch; partial temporary files are removed on handled save errors and are never used as the destination.
3. Stop with `Ctrl+C`. Do not power off while a checkpoint is being written. On restart, use the resume command and verify the reported completed and next epoch numbers.
4. Verify a checkpoint without modifying it by loading it into the same model/optimizer with `load_training_checkpoint`; missing, corrupt, version-incompatible, structurally incompatible, and configuration-mismatched files fail explicitly. For archival integrity, record a checksum, for example `Get-FileHash <path> -Algorithm SHA256`.
5. Keep model/checkpoint directories and logs out of Git. Stage only explicit source and documentation paths; never use `git add -A` for this project.

At approximately 46.58 minutes of training work per epoch from the first completed epoch, 199 epochs imply roughly 154.5 hours (about 6.4 days) of training-loop time, plus startup and validation overhead. Laptop thermals and validation make wall-clock time variable.

## Reproducibility limitations

The full-state checkpoint restores Python, NumPy, PyTorch CPU, and all visible CUDA RNG states. Restoration occurs after model, optimizer, profiler, and DataLoader construction and before the next DataLoader iterator, so shuffle and default worker seeds continue from the captured epoch boundary. The loader uses four non-persistent workers and PyTorch's default worker seeding.

This is verified as **functional continuation**, not bitwise deterministic continuation. CUDA kernels can be nondeterministic, deterministic-algorithm flags are not enabled, worker scheduling/prefetch can vary, and no uninterrupted-versus-resumed multi-epoch equality experiment has been run. Mid-epoch continuation is unsupported; at most the current incomplete epoch is repeated.

## Verification commands

These checks are short and do not launch a complete dataset epoch:

```powershell
python -m py_compile utils\training_checkpoint.py train_polyp.py
python -m unittest -v tests.test_training_checkpoint tests.test_train_polyp_resume
python -m tests.verify_real_pvt_checkpoint
```

The last command strictly loads both protected epoch-1 model-only files, performs batch-1 inference at 352 × 352, and round-trips a temporary real-PVT full-state checkpoint. Its temporary checkpoint is automatically removed.
