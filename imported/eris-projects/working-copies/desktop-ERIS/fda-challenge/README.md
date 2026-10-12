# What Does It Do In There?

A drug label's Clinical Pharmacology section spends several thousand words
describing what a drug does in the body. The FDA separately assigns that drug
a mechanism-of-action class from its own vocabulary — one short phrase naming
the thing those thousands of words describe.

This challenge masks every mechanism word out of the pharmacology text, hands
the solver twelve candidate classes, and asks which ones the FDA assigned.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge metadata, split notes, measured anchors |
| `problem-description.md` | The prompt shown to the solving agent |
| `dataset-description.md` | Source, fetch method, features, limitations, licence |
| `rubrics.md` | Eight task-specific rubrics |
| `prepare.py` | `prepare(raw, public, private)` — deterministic, seed 20260925 |
| `grade.py` | `grade(submission, answers) -> float` — chance-corrected set F1 |
| `reference_solution.py` | CPU reference: TF-IDF pair scorer + tuned threshold |
| `dataset/generator/build_raw.py` | The openFDA fetch and masking script |
| `dataset/validate_submission.py` | Offline structural checker, no answer key |
| `solution.ipynb` | Reference solution notebook |
| `DESIGN.md` | The measured design record |

## Reproduce

```bash
# 1. fetch and mask (one worker, rate-limited)
cd dataset/generator
python build_raw.py --out ../raw

# 2. build the split and the candidate pools
cd ../..
python prepare.py

# 3. run the reference and score it
python reference_solution.py
python grade.py submission.csv --answers dataset/private/answers.csv
```

## Task at a glance

- **Input** — `route`, `pharmacology` (masked), `candidates` (12, shuffled)
- **Output** — `moa_classes`, a pipe-separated subset of that row's candidates
- **Metric** — chance-corrected set F1: raw F1 minus the expected F1 of a
  same-size random draw from the row's own pool
- **Split** — by shared active ingredient (union-find over salt-normalised
  ingredients), so no test drug shares an ingredient with any training drug

## Two things that were measured, not assumed

**The text gives the answer away, so it is masked.** 57% of labels state their
own mechanism verbatim in the pharmacology prose. Masking uses a vocabulary
built from *every* class in the corpus — never just the row's own, since that
would make the masking pattern itself the answer. Measured on 1,401 labels:
1,860 of 2,082 label-class pairs stopped being keyword-recoverable, with a
median ~7,300 characters of real pharmacology surviving.

**Raw F1 pays for distributional luck, so the metric corrects for chance.**
Always naming the single most common class scored 0.182 raw F1 while reading
nothing. Subtracting the expected random score makes padding worthless —
naming all twelve candidates scores exactly **0.000** — though it does not
zero out luck: per-row scores are clipped at 0, so a random single guess still
scores 0.085. Every content-free probe lands in 0.070–0.193; the reference
scores **0.462**.

The full baseline ladder is in `config.yaml`; the reasoning and the numbers
that were wrong along the way are in `DESIGN.md`.

## Data provenance

openFDA Drug Label API, U.S. Food and Drug Administration.
**CC0 1.0 Universal** — public domain dedication.

Only the drug label endpoint is used. Device endpoints are deliberately
avoided: they carry GMDN Agency nomenclature that openFDA notes is separately
licensed and not covered by the CC0 dedication.

openFDA's disclaimer applies — the data should not be relied on for decisions
about medical care, and results should be assumed unvalidated. This is a
research benchmark on deliberately masked text with the drug's identity
withheld. It is not medical information. Full notice in
`dataset/raw/ATTRIBUTION.txt`.
