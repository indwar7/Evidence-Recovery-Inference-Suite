# Dataset Description

## Overview

This dataset records **how one convolutional galaxy-morphology classifier
actually behaves when the same galaxy is delivered under different observing
conditions**.

The source imagery is real: **Galaxy10 SDSS (astroNN)** — 21,785 galaxy cutouts,
69x69 RGB, taken from the Sloan Digital Sky Survey, with morphology labels
contributed by Galaxy Zoo volunteers.
Source: https://zenodo.org/records/10844811 · Licence **CC-BY-4.0**
(redistribution and commercial use permitted with attribution) ·
DOI 10.5281/zenodo.10844811. SDSS imagery is public.

Layered on top of that imagery is the measured part. A hidden morphology
classifier `H` was trained on a **disjoint** 45% of the galaxies. For every
galaxy in the released 55%, and for all **400 ordered pairs** drawn from a
**20-condition catalog of real survey systematics**, the degraded cutout was
rendered and passed through `H`, and `H`'s full 8-class probability vector was
recorded. That is 400 measurements per galaxy.

The ground truth is therefore exact by construction: it is not a human
annotation and not another algorithm's output, it is what the model did. No
synthetic imagery and no LLM output appears anywhere in the pipeline.

The catalog covers the genuine error budget of an imaging survey — atmosphere
(seeing, airmass reddening, cirrus), telescope (guiding trail, coadd
misregistration), sky (background pedestal, scattered-light gradient), detector
(read noise, shot noise, saturation, nonlinearity, cosmic rays, bleed column),
contaminants (blended foreground star, deblending residual) and pipeline
(pixel binning, cutout scale error, delivery compression).

## File Structure

```
images/IMAGE_ID.png   clean 69x69 RGB SDSS cutout, one per released galaxy
meta.csv                per-galaxy label and clean-model outcome
ops.csv                 the 20-condition catalog with calibrated strengths
classes.csv             class index -> Galaxy Zoo morphology name
catalog.py              condition implementations, verbatim and runnable
responses.npy           [N, 20, 20, 8] float16 measured response atlas
meta.json               provenance, seed, strengths, clean accuracy
```

## Features

### `meta.csv` — one row per released galaxy

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Galaxy identifier, `g%06d`; matches `images/IMAGE_ID.png` and seeds the random conditions |
| `true_class` | int | Galaxy Zoo morphology, 0–7, indexes `classes.csv` |
| `clean_correct` | bool | Whether `H` classifies the undegraded cutout correctly |
| `clean_p_true` | float | `H`'s probability for the true morphology on the clean cutout |

### `ops.csv` — one row per observing condition

| Column | Type | Description |
|---|---|---|
| `op_index` | int | 0–19; the value submitted as `op1` / `op2`. Index 0 is `identity` |
| `op_name` | str | Condition name, e.g. `seeing`, `trail_h`, `sky_bkg` |
| `family` | str | One of `none`, `atmosphere`, `telescope`, `sky`, `detector`, `contaminant`, `pipeline` |
| `strength_param` | float | Calibrated strength in [0,1] passed to the condition |
| `description` | str | One-line physical description of the effect |

### `classes.csv` — one row per morphology

| Column | Type | Description |
|---|---|---|
| `class_index` | int | 0–7, as used in `true_class` and `target_class` |
| `morphology` | str | Galaxy Zoo morphology name, e.g. `Disk, Edge-on, Rounded Bulge` |
| `galaxy10_index` | int | Original Galaxy10 class index before the rare classes were dropped |

### `responses.npy` — the measured atlas

float16 array of shape `[N_galaxies, 20, 20, 8]`, aligned row-for-row with
`meta.csv`. Element `[i, a, b, c]` is the probability the hidden model assigns to
morphology `c` for galaxy `i` after applying condition `a` to the clean cutout
and then condition `b` to the result. Values are probabilities in [0,1] summing
to 1 over the last axis.

## Notes

**Order is physical.** `[i, a, b]` and `[i, b, a]` are different measurements,
not duplicates. Clipping a saturated core and then convolving with a seeing
kernel spreads a flat-topped core; blurring first and clipping after removes the
peak entirely.

**Strength calibration.** Each condition's single strength parameter was set by
bisection so that, applied alone, it flips `H` on the same fraction of galaxies
(~8%). Without this the answer degenerates to "name the strongest condition",
which is a fact about the catalog and identical for every galaxy. Calibration
targets *break rate* rather than mean pixel damage deliberately: mean damage has
a floor at the model's own baseline uncertainty, so equalising it leaves the
conditions at very unequal real strengths, whereas break rate has a floor at
zero.

**The stimulus is public, the response is not.** `catalog.py` is the exact code
used to build the atlas and reproduces the degraded image for any pair. The
conditions with a random realisation — cosmic-ray positions, the blended star's
offset, the scattered-light direction — are seeded from `image_id`, so they are
fixed per galaxy and reproducible by anyone holding the data. `H`'s weights are
never shipped.

**Orientation matters.** `trail_h` and `trail_v` are separate conditions because
which one damages a galaxy depends on its position angle on the sky, which is
essentially random and is not a property of its morphology class. Note that a
90-degree rotation of an image exchanges these two conditions, so any
augmentation that rotates must permute the response axes to match; horizontal
and vertical flips map every condition to itself and are safe.

**Class handling.** Galaxy10 is severely imbalanced. Two classes are dropped as
too rare for any classifier to learn — "Smooth, Cigar shaped" (349 examples) and
"Disk, Edge-on, Boxy Bulge" (17) — leaving eight morphologies. The rest are
capped at 2,500 each so `H` is not dominated by the three most common. Residual
imbalance of about 4.8:1 is kept and documented rather than equalised away,
because that is what a real morphology sample looks like.

**Label noise is real.** Galaxy Zoo labels are volunteer consensus, so the eight
classes carry genuine ambiguity. This is a property of the source, and it is why
`H` tops out well below the accuracy a classifier reaches on a clean-cut
dataset.

**Approximation caveat.** Conditions are applied to already-reduced,
display-space cutouts rather than to raw detector counts, so effects such as
shot noise are physically motivated approximations rather than exact radiometric
simulations.

**Reproducibility.** `dataset/generator/gen_raw.py` rebuilds the atlas from the
Zenodo download with seed 20260731. The pool split, the calibration, the
per-galaxy random realisations and the model weights are all fixed, so the same
responses come back byte for byte.
