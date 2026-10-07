# Dataset Description

## Overview

This dataset supports **Blind Spot**, a challenge about predicting where a
black-box image classifier is weak.

The imagery is **EuroSAT RGB** — 64×64 Sentinel-2 satellite patches over Europe,
labelled with ten land-cover classes (Helber et al., *EuroSAT: A Novel Dataset
and Deep Learning Benchmark for Land Use and Land Cover Classification*).
Source: <https://zenodo.org/records/7711810>. Licence: **MIT** — commercial use
and redistribution permitted. The imagery derives from Copernicus Sentinel-2
data, which is free and open under the Copernicus licence.

On top of that imagery we add the part that makes the challenge: the
**measured response of a hidden classifier**. A convolutional network `H` was
trained on a disjoint 45% slice of EuroSAT and is never released. For every
image in this dataset, and for every one of the 400 ordered pairs drawn from a
20-operation catalog, we recorded the full probability vector `H` outputs after
the image is passed through that pair of operations — **2.4 million forward
passes** (6,000 images × 400 pairs), yielding **24 million recorded
probabilities** (× 10 classes).

The dataset is therefore not a collection of labels but a collection of
**behaviour**: a dense map of how one specific model responds to one specific,
strength-calibrated family of image degradations.

### How the operation catalog was built

A naive catalog makes the challenge trivial: if one operation is far more
destructive than the others, the best answer is the same everywhere and can be
read off a lookup table. Every operation here is therefore **strength
calibrated** — its parameter was bisected so that, applied alone, it flips `H`
on ~8% of a held-out calibration split. Measured solo break rates span
0.071–0.087 across all 19 non-identity operations, so no operation dominates by
brute strength and the best pair depends on the individual image.

Calibration used images from `H`'s own training pool only, never from the
released splits, so no test information reached the catalog definition.

## File Structure

```
raw/
├── images/<image_id>.png   6,000 RGB PNGs, 64×64, one per image
├── meta.csv                per-image label and clean-prediction record
├── ops.csv                 the 20-operation catalog and its calibrated strengths
├── responses.npy           (6000, 20, 20, 10) float16 — the hidden model's behaviour
├── meta.json               provenance and generation parameters
└── LICENSE-EuroSAT.txt     upstream MIT licence for the redistributed imagery
```

The images are redistributed under EuroSAT's MIT licence, whose terms require
the copyright notice to accompany any substantial portion of the data;
`LICENSE-EuroSAT.txt` is that notice, reproduced verbatim from the upstream
repository (Copyright © 2023 Patrick Helber). The derived artifacts in this
dataset — `responses.npy`, `ops.csv`, `meta.csv` — are original work created for
this challenge and are not covered by that notice.

`responses.npy[i, a, b, c]` is `p_H(class c | op_b(op_a(image i)))`. Axis order
is (image, first operation, second operation, class). Operations are applied
**first `op_a`, then `op_b`**; the pair is ordered and the two orders generally
differ (`jpeg → noise` is not `noise → jpeg`).

## Features

### `meta.csv`

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Unique image identifier, e.g. `Forest_864`; matches `images/<image_id>.png` |
| `true_class` | int | Land-cover class, 0–9 (see Notes) |
| `clean_correct` | int | 1 if `H` classifies the untouched image correctly, else 0 |
| `clean_p_true` | float | `p_H(true_class)` on the untouched image |

### `ops.csv`

| Column | Type | Description |
|---|---|---|
| `op_index` | int | Operation index 0–19, used to address `responses.npy` |
| `op_name` | str | Human-readable name, e.g. `jpeg`, `motion_h`, `desat` |
| `strength_param` | float | Calibrated strength in [0,1] fed to that operation |

### `responses.npy`

| Axis | Size | Description |
|---|---|---|
| 0 | 6000 | image, aligned row-for-row with `meta.csv` |
| 1 | 20 | index of the **first** operation applied |
| 2 | 20 | index of the **second** operation applied |
| 3 | 10 | class probability from the hidden model, sums to 1 |

Stored as float16 to keep the file small; cast to float32 before arithmetic.

## Notes

**Class codes.** 0 `AnnualCrop`, 1 `Forest`, 2 `HerbaceousVegetation`,
3 `Highway`, 4 `Industrial`, 5 `Pasture`, 6 `PermanentCrop`, 7 `Residential`,
8 `River`, 9 `SeaLake`.

**The hidden model.** A four-block CNN (32→64→128→256 channels, BatchNorm,
global average pooling) trained for 30 epochs with strong augmentation —
flips, 90° rotations, brightness/contrast/saturation jitter, per-channel gain,
mild blur and noise. Clean validation accuracy **95.85%**. The augmentation
matters: a model trained without it is *blanket-brittle*, so any photometric
shift destroys it and the best attack becomes a property of the class rather
than of the image. Training `H` to be robust is what makes its residual weak
points idiosyncratic and image-specific.

**Operations are mild and legal.** Every operation is an ordinary image
processing step — compression, resampling, blur, photometric adjustment — at a
strength that leaves the image plainly recognisable. Nothing here is a
gradient-based adversarial perturbation, and no pixel-level optimisation is
involved. The question is which *ordinary* degradations happen to land in a
given image's blind spot.

**Stochastic operations are seeded.** `noise` and `speckle` draw from a
generator seeded by a SHA-256 hash of the image id, so the recorded responses
are exactly reproducible and re-running generation reproduces the table.

**Not every image is usable.** `H` misclassifies ~4% of clean images. Those
have no intact prediction to steer away from and are dropped by `prepare.py`.

**Class balance.** EuroSAT is mildly imbalanced (2,000–3,000 images per class);
that imbalance carries through to the sampled 6,000.
