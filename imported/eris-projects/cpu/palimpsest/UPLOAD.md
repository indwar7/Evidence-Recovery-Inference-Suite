# Palimpsest — Shipd Upload Cheat-Sheet

Every form field, in platform order, paste-ready. Long fields point to the file
whose FULL content you paste.

---

## 1 · DATASET form

**Name (slug)**
```
palimpsest-dataset
```

**Title**
```
Wikivoyage Edit-Lineage Windows: Real Revision Histories With the Revision Graph as Ground Truth
```

**Short title**
```
Wikivoyage Edit-Lineage Windows
```

**Description** → paste the FULL content of `dataset/dataset-description.md`

> ⚠️ **DO NOT paste `problem-description.md` here.** That file describes the
> *prepared* split solvers see (`public/`, `train.csv`, `test.csv`,
> `snapshots/`) and belongs in the CHALLENGE form (section 2). Pasting it in
> the dataset form triggers "Validation Failed — fundamental mismatch between
> the described file structure and the actual files present", because the raw
> upload contains only `windows.jsonl`, `attribution.csv`, `meta.json`.

**Source URL**
```
https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-meta-history.xml.7z
```

**License**
```
CC BY-SA 4.0 — commercial use and redistribution permitted with attribution and
share-alike (verified at en.wikivoyage.org/wiki/Wikivoyage:Copyleft, 2026-08-06).
Attribution is satisfied by raw/attribution.csv, which lists every article title
and revision id used with permanent links (78,400 revisions). Derived data is
released under CC BY-SA 4.0 as required by share-alike.
```

**Citation / provenance (if asked)**
```
English Wikivoyage contributors. Wikimedia Foundation full-history XML dump,
2026-08-05 (enwikivoyage-latest-pages-meta-history.xml.7z). Text © Wikivoyage
contributors, CC BY-SA 4.0.
```

**Raw data upload** → ZIP the three files in `dataset/raw/` first:
```
cd cpu/palimpsest/dataset/raw && zip -9 ../palimpsest-raw.zip windows.jsonl attribution.csv meta.json
```
Upload `palimpsest-raw.zip` (NOT the bare folder — Babel Gauge lesson).
Contents: `windows.jsonl` (5,600 article windows × 14 revisions, with text),
`attribution.csv`, `meta.json`.

**Generation / reproducibility note (if asked)**
```
Deterministic, no randomness: dataset/generator/build_raw.py streams the dump
once (7z e -so | xml.etree.iterparse) and keeps, for each eligible article
(namespace 0, no redirect, ≥20 revisions, median window size 2,500–40,000
bytes), the consecutive 14-revision window maximising
churn / (1 + |net byte drift|), min 200 bytes churn — computed only from byte
sizes, never from text or labels. Early-stops after 6,000 windows = "first
eligible articles in dump page-id order", reproducible for the pinned dump.
Run: python build_raw.py <dump.7z> <out_dir> 6000
```

**Prepare script** → paste/upload `prepare.py` (repo root of the challenge).
Deterministic, seed 20260805; verified byte-identical across two runs.

---

## 2 · CHALLENGE form

**Name (slug)**
```
palimpsest
```

**Title**
```
Untangling Mixed Edit Histories
```

**Domain / category**
```
natural-language-processing
```

**Tags** — select exactly these three: `text`, `clustering`, `feature-engineering`.
Do NOT tag `classification`/`regression`/`ranking`/`sequence-labeling` (the
reviewer rubric penalises variations of those, and the output is a genealogy
forest, not any of them). Skip `large-scale`/`small-data` — 900 items / 212 MB
is neither.

**Problem description** → paste the FULL content of `problem-description.md`

**Grading script** → upload `grade.py`
(usage: `python grade.py submission.csv answers.csv`; returns parent accuracy
in [0,1]; raises InvalidSubmission only for structural violations — missing
rows, duplicates, out-of-item parents, self-parents, cycles. Rows beyond the
answer key are tolerated, so public/private portion scoring works.)

**Metric name (if asked)**
```
Immediate-parent accuracy
```

**Metric description (if asked)**
```
Fraction of test snapshots whose predicted parent_id equals the true parent_id
(ROOT counted as a normal value), micro-averaged over all 6,241 test snapshots.
Submissions must be structurally valid forests; the identical checks ship in
public/validate_submission.py so solvers can verify offline before submitting.
```

**Expected score range / difficulty (if asked)**
```
Baselines: random 0.053, all-ROOT 0.252, cluster+byte-size 0.310, best
similarity heuristic 0.390, linear model 0.493. Reference solution ~0.55
(graded on the private test). Similarity-only ceiling ~0.85, oracle 1.0.
A frontier coding agent given only public/ and 75 min CPU scored 0.685.
```

---

## 3 · RUBRICS form (9 entries — text in `rubrics.md`)

| # | Title | Type | Importance |
| - | ----- | ---- | ---------- |
| 1 | Infers the number of lineages rather than assuming it | DATA_HANDLING | REQUIRED |
| 2 | Validates on an item-level split | TRAINING | REQUIRED |
| 3 | Models edit direction, not just similarity | MODELING | RECOMMENDED |
| 4 | Decodes a globally consistent forest | MODELING | RECOMMENDED |
| 5 | Does not lean on byte length as the ordering signal | FEATURE_ENGINEERING | RECOMMENDED |
| 6 | Emits a structurally valid submission | CODE_QUALITY | REQUIRED |
| 7 | Beats the similarity-nearest-neighbour baseline (0.401) | TRAINING | RECOMMENDED |
| 8 | Probes for shortcut signals instead of assuming clean text | AGENT_BEHAVIOR | RECOMMENDED |
| 9 | Completes within the compute budget | CODE_QUALITY | REQUIRED |

For each entry, paste the corresponding section body from `rubrics.md`
(criterion paragraph + its *Fails if* line).

---

## 4 · SOLUTION form

**Notebook** → upload `solution.ipynb` (already executed; outputs saved).

**Summary (if asked)**
```
Three-stage progression. v1: tf-idf clustering with silhouette-selected k +
byte-size chaining (no learning) — 0.32 validation. v2: neural ordered-pair
scorer — hashed bags of the tokens each candidate edit adds/removes, embedded
via EmbeddingBag, trained as an (n+1)-way softmax over candidate parents within
each item — greedy decode, 0.55 validation. v3: same scorer decoded as an
assignment problem (Hungarian matching, each snapshot parents at most one
child) plus cycle-breaking; trained on all 630 items. CPU-only, well inside
the 1.5 h budget.
```

**Runtime note (if asked)**
```
~35 min wall, measured end-to-end CPU-only (torch pinned to CPU
threads): pair featurisation + 12 epochs training + matching decode. Reads
./dataset/public/, writes ./working/submission.csv. Only standard Kaggle-image
libraries (numpy, pandas, scikit-learn, scipy, torch).
```

---

## Pre-submit checklist

- [ ] `palimpsest-raw.zip` uploaded (zip made from raw/, not the folder)
- [ ] prepare.py attached to the dataset, not the challenge
- [ ] problem-description pasted complete (submission table has 6,241 rows note)
- [ ] all 9 rubrics entered with correct Type + Importance
- [ ] solution.ipynb uploaded with outputs visible
