# PVT-GCASCADE Baseline Training Progress

## Current status

- Experiment: `Polyp_PVT_GCASCADE_img_size352bs4_Run3`
- Completed epochs: **90 / 199**
- Completed sessions: **18 / 40**
- Best validation epoch: **19**
- Best weighted validation Dice: **0.849587**
- Training status: **Stopped early after epoch 90**
- Last saved checkpoint: `2026-11-01T11:27:33.581160+05:30`
- Source commit: `a9b3b240172b53943f50f46f806954f9d75ee870`

## Session history

| Session | Completed epochs | Best epoch so far | Best weighted Dice |
|---|---|---:|---:|
| 1 | 1-5 | 4 | 0.809619 |
| 2 | 6-10 | 10 | 0.836800 |
| 3 | 11-15 | 10 | 0.836800 |
| 4 | 16-20 | 19 | 0.849587 |
| 5 | 21-25 | 19 | 0.849587 |
| 6 | 26-30 | 19 | 0.849587 |
| 7 | 31-35 | 19 | 0.849587 |
| 8 | 36-40 | 19 | 0.849587 |
| 9 | 41-45 | 19 | 0.849587 |
| 10 | 46-50 | 19 | 0.849587 |
| 11 | 51-55 | 19 | 0.849587 |
| 12 | 56-60 | 19 | 0.849587 |
| 13 | 61-65 | 19 | 0.849587 |
| 14 | 66-70 | 19 | 0.849587 |
| 15 | 71-75 | 19 | 0.849587 |
| 16 | 76-80 | 19 | 0.849587 |
| 17 | 81-85 | 19 | 0.849587 |
| 18 | 86-90 | 19 | 0.849587 |

## Best epoch validation metrics

| Dataset | Number of images | Dice |
|---|---:|---:|
| CVC-ClinicDB | 62 | 0.928477 |
| Kvasir | 100 | 0.916251 |
| CVC-ColonDB | 380 | 0.811656 |
| CVC-300 | 60 | 0.897185 |
| **Weighted overall** | **602** | **0.849587** |

## Reproduction details

- Encoder: PVTv2-B2.
- Decoder: G-CASCADE with additive skip aggregation.
- Training image size: 352; batch size: 4.
- AdamW learning rate: 0.0001; weight decay: 0.0001.
- Official three-scale training and combined structure loss.
- Model selection uses image-count-weighted Dice across four datasets.
- **These are validation scores, not independent final test results.**
- Full epoch-level metrics are in `results/baseline/session_metrics.csv`.
- Large weights and resumable checkpoints are excluded from ordinary Git.
- Baseline training was stopped early at epoch 90; the epoch-90 checkpoint remains resumable.

## Baseline training freeze

- Training stopped voluntarily at epoch **90 / 199** after **18 sessions**, before the original target.
- Best weighted validation Dice: **0.849586710963455**, from epoch **19**.
- The locally saved best-model weights remain the selected model for evaluation and demonstration.
- The epoch-90 resumable checkpoint and its metadata are preserved, allowing training to resume later.
- **109 planned epochs were not completed.**
- Training was stopped to prioritize final demonstration and code explanation; this is a partial reproduction, not a completed 199-epoch experiment.
- The recorded Dice scores are checkpoint-selection validation metrics, not independent final test results.
