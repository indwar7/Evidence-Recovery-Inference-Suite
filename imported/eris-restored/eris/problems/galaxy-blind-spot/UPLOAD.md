# Upload checklist — Observing Conditions

Submission is a **manual web-portal flow** (Draft -> Pending -> Accepted) and
**credits are consumed on submit**, so nothing here is automated. This file is
the ordered list of what to upload and the checks it has already passed.

---

## Step 1 — Dataset

Upload the contents of `dataset/raw/` and paste `dataset/dataset-description.md`
as the description.

| item | file |
|---|---|
| raw data | `dataset/raw/images/`, `meta.csv`, `ops.csv`, `classes.csv`, `catalog.py`, `responses.npy`, `meta.json` |
| description | `dataset/dataset-description.md` |
| reproducible generator | `dataset/generator/gen_raw.py` (+ `model.py`, `catalog.py`, `hidden.pt`, `calib.json`, `norm_stats.npz`) |

**Do NOT upload** `dataset/generator/hidden.pt` to any solver-visible location.
It is the model under study; publishing it makes the task trivial. It belongs
with the generator only, for reproducibility.

### Guide's dataset checklist

- [x] Dataset description documents all columns and types — Features tables for
      `meta.csv`, `ops.csv`, `classes.csv`, and the `responses.npy` array layout.
- [x] Dataset has sufficiently many samples — see ROWS_PLACEHOLDER.
- [x] Prepare script is reproducible and deterministic — `prepare.py`, seed
      20260731, no RNG outside the fixed-seed split.
- [x] License allows commercial use — Galaxy10 SDSS is **CC-BY-4.0**
      (redistribution and commercial use permitted with attribution).
- [x] Source URL documented — <https://zenodo.org/records/10844811>,
      DOI 10.5281/zenodo.10844811.

---

## Step 2 — Challenge

| item | file |
|---|---|
| problem description | `problem-description.md` |
| grading script | `grade.py` |
| config | `config.yaml` |
| initial rubrics | `rubrics.md` |
| reference solution | `solution.ipynb` (and `reference_solution.py`) |

### Platform constraints

- [x] GPU-leveraging — the intended solution renders 400 degraded views per
      galaxy and trains a CNN surface model; A10G, 10 cores, 62 GB.
- [x] Runtime under 1.5 h — RUNTIME_PLACEHOLDER
- [x] Deterministic evaluation — `grade.py` is a pure lookup into a precomputed
      table; it never runs a model, so two grading runs cannot disagree.
- [x] Deterministic dataset generation — seed 20260731 throughout.
- [x] Public training set, hidden test labels — responses for test rows live
      only in `prepared/private/answers.csv`.
- [x] CSV submission only — `submission.csv` with `row_id, op1, op2`.
- [x] Self-contained, no internet needed at solve time.
- [x] Kaggle-image libraries only — numpy, pandas, scipy, Pillow, torch.

### Verified before submission

- [x] Grader returns **1.0** for the oracle and **0.0** for the anti-oracle.
- [x] Grader rejects duplicate `row_id`, out-of-range ops, missing rows, float
      ops, and op-name strings, each with a specific message.
- [x] Split is **by galaxy**, not by row — one galaxy contributes up to 7 rows;
      verified no `image_id` appears in both `train.csv` and `test.csv`.
- [x] Every graded row is winnable — galaxies `H` misclassifies when clean are
      dropped, and rows whose target never reaches p = 0.30 are dropped.
- [x] Difficulty gate measured before the spec was frozen — see `DESIGN.md`.

---

## Anchors to quote in the listing

ANCHORS_PLACEHOLDER

---

## Known review risks to be ready for

RISKS_PLACEHOLDER
