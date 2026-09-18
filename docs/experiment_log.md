# EN3160 G-CASCADE Experiment Log

## Project

Reproduction of PVT-GCASCADE for binary colonoscopy polyp segmentation.

## Team structure

- Abu: Official PVT-GCASCADE baseline reproduction
- Mokshu: Robust PVT-GCASCADE augmentation extension

---

## Upstream

Official repository:
https://github.com/SLDGroup/G-CASCADE

Starting commit:
08d76880027bfe14ed79def1eea3d8a4957fbab6

Tag:
upstream-baseline

---

## Baseline Run 01

### Status

Environment setup not started.

### Intended architecture

PVT-GCASCADE

### Intended baseline configuration

- Encoder: PVTv2-b2
- Decoder: G-CASCADE
- Input resolution: 352 x 352
- Epochs: 200
- Batch size: 4 if GPU memory permits
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Gradient clip: 0.5
- Graph K: 11
- Graph convolution: Max-Relative
- Skip aggregation: additive
- Multi-scale rates: 0.75, 1.0, 1.25

### Deviations

None yet.

### Notes

Environment setup pending.