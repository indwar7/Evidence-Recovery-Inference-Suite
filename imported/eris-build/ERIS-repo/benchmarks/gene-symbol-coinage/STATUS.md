# Build status — Coining Gene Symbols from Gene Names

**~60% complete.** Not submission-ready. This file says exactly what is done,
what is measured, and what is left.

## The task in one line

Given the HGNC's approved gene **name** (`solute carrier family 2 member 1`),
produce the approved gene **symbol** (`SLC2A1`) — the short form a nomenclature
committee coined for it.

- **Source:** HGNC complete gene set, quarterly snapshot 2026-07-07
- **Source URL:** `https://www.genenames.org/download/archive/` (HTTP 200)
- **Licence:** CC0 1.0, quoted verbatim from
  `https://www.genenames.org/about/license/`
- **Shape:** sequence-to-sequence generation; submitted artifact is a coined
  string
- **Metric:** prefix agreement — shared leading characters over the longer of
  the two lengths
- **Split:** by the archive's own recorded effective date of the current
  symbol; cutoff 2010-01-01
- **Size:** 13,863 train / 2,270 test (16.4% ratio)

## Done and verified

| Item | Status |
|---|---|
| Gate 1 — licence is CC0, quoted from the HGNC's own licence page | PASS |
| Gate 2 — source is new to the register; title and shape share nothing | PASS |
| Gate 3 — both name and symbol are the archive's own recorded values | PASS |
| Gate 4 — seq2seq generation; not tabular, not anomaly detection | PASS |
| Gate 5 — source URL answers `curl -sI` with 200 | PASS |
| `dataset/generator/build_raw.py` — pinned snapshot + sha256 check | done |
| `prepare.py` — date-based split, all disjointness asserted | done |
| `grade.py` — prefix agreement, degrade-not-reject, `InvalidSubmission(ValueError)` | done |
| `dataset/validate_submission.py` | done |
| `dataset-description.md` — all 8 sections, 7 measured limitations | done |
| Baseline ladder — 13 probes measured on the shipped split | done |

Every number in `dataset-description.md` was re-derived against the shipped
snapshot and three were corrected (82.9% not 84.4%, 268 not 240, 3,076 not
3,074).

## The measured ladder (prefix agreement, on the shipped split)

| Rung | Score |
|---|---|
| blank submission | 0.000 |
| constant most-common train symbol | 0.003 |
| name-length bucket to modal symbol | 0.028 |
| capitalised token + trailing number | 0.053 |
| copy the name through | 0.061 |
| first letter of the name only | 0.175 |
| first word uppercased | 0.217 |
| first 3 letters + trailing number | 0.266 |
| **harvest capitalised tokens (adversarial)** | **0.303** |
| initials of each word (the sample submission) | 0.311 |
| **nearest train name, designator swapped (lookup adversary)** | **0.366** |
| character seq2seq, 25 epochs, 3.2M params (CPU/MPS floor) | 0.447 |
| oracle | 1.000 |

The reference also splits cleanly by difficulty: **0.680** on the 43.5% of test
rows whose symbol stem already existed in train, **0.253** on the 56.5% that
needed a new stem. That is the gap the task is measuring.

## What a measurement changed mid-build

**A family-disjoint split was built first and abandoned.** Holding out whole
gene families (connected components over symbol stem, name skeleton, and two
prefix keys) looked like the cleaner test of coinage. Measured on that split, a
trained seq2seq scored **0.258 against 0.337 for a one-line initials
heuristic** — the reference landed *below* a trivial baseline, meaning the split
had removed the signal along with the shortcut.

Replaced with the archive's own approval dates. The date split is also the
honest framing: train is what the committee had already decided, test is what it
decided next.

## What is left

1. **Close the adversary margin.** The lookup adversary sits at 0.366 against
   the reference's 0.447 — a **+0.081 margin**, below the ~0.18 target. The fix
   is written but unmeasured: `work/ref_ra.py`, a retrieval-augmented variant
   that feeds the model its nearest training siblings as context, so the lookup
   adversary becomes a strict ablation of the reference rather than a rival to
   it. Expected 0.50–0.55. **~25 min to train on MPS.** Two background runs of
   this were killed by session restarts and never produced a number.
2. **Build the 5+ ablation ladder.** The guide requires five faithful partial
   implementations of the real solver, distinct and strictly ordered — not just
   baselines. Currently only the plain seq2seq exists. Natural ablations of the
   retrieval-augmented reference: no retrieval, retrieval without the
   sibling-dropout rehearsal, 1 neighbour instead of 2, no initials hint.
3. **Write the remaining files:** `config.yaml`, `problem-description.md`,
   `rubrics.md`, `README.md`, `DESIGN.md`, `solution.ipynb`.
4. **Write and run `run_checklist.py`** — the guide's Part 11 checklist as an
   actual script. Nothing has been verified by script yet; the ladder above was
   measured, but the schema, leakage, grader and doc checks have not been run.
5. **Build the raw zip** from `dataset/raw` + `dataset/generator` only, then
   extract it clean, run `prepare.py`, and prove every generated file comes back
   byte-identical.

## How to resume

```bash
cd gene-symbol-coinage
python3 dataset/generator/build_raw.py     # re-fetch + verify sha256 if raw/ is absent
python3 prepare.py                          # rebuild the split
python3 work/baselines.py                   # re-measure the 13 probes
PYTHONUNBUFFERED=1 python3 work/ref_ra.py 28 192 3 work/ra_sub.csv | tee work/ra.log
```

`work/` holds the measurement harness and is not part of the package:
`metrics.py` (the four candidate metrics), `baselines.py` (the 13 probes),
`ref.py` (plain seq2seq), `ref_ra.py` (retrieval-augmented, unmeasured),
`wordtable.py` (an alignment-based CPU rung, 0.312), `temporal*.py` (the split
prototypes).

**Note on local training:** PyTorch MPS needs the encoder built explicitly with
`enable_nested_tensor=False` and passed as `custom_encoder`, or
`nn.Transformer` with `norm_first=True` hits an unimplemented op. That fix is
already in `work/ref.py` and `work/ref_ra.py`.
