# PVT-GCASCADE Baseline Training Progress

## Current status

- Experiment: `Polyp_PVT_GCASCADE_img_size352bs4_Run3`
- Completed epochs: **30 / 199**
- Completed sessions: **6 / 40**
- Best validation epoch: **19**
- Best weighted validation Dice: **0.849587**
- Next session: **7, epochs 31-35**
- Last saved checkpoint: `2026-10-23T09:25:21.774010+05:30`
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
- The baseline remains in progress until epoch 199.
