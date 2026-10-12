# The Vanished Clause

A regulation is amended. The old wording is not struck through or footnoted —
it is replaced in place, and anyone reading afterwards sees only the new text.
This challenge gives a solver a CFR section as it reads *after* an amendment,
the wording the amendment removed as loose fragments, and a set of marked
places — one in four of them real — and asks it to rebuild how the section
read *before*.

The amendment boundary is the eCFR's own recorded `amendment_date`, and both
versions are the agency's published wording on those dates. Nothing is
annotated and no model produced any value in the dataset.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge metadata, split notes, measured anchors |
| `problem-description.md` | The prompt shown to the solving agent |
| `dataset-description.md` | Source, fetch method, features, limitations, licence |
| `rubrics.md` | Ten task-specific rubrics |
| `prepare.py` | `prepare(raw, public, private)` — deterministic; the split uses no randomness (sorted, bounded search over CFR parts), so there is no seed |
| `grade.py` | `grade(submission, answers) -> float` — gap-anchored changed-token F1 |
| `dataset/generator/build_raw.py` | The eCFR fetch script |
| `dataset/validate_submission.py` | Offline structural checker, no answer key |
| `solution.ipynb` | Reference solution, three progressive submissions |
| `reference_solution.py` | The same three stages as a script |

## Reproduce

```bash
# 1. fetch the raw pairs (one worker, rate-limited — allow several hours)
cd dataset/generator
python build_raw.py --out ../raw --since 2022-01-01

# 2. build the split
cd ../..
python prepare.py

# 3. score a submission
python grade.py submission.csv --answers dataset/private/answers.csv
```

## Task at a glance

- **Input** — `text_after` with `[GAP]` marks, and `removed_fragments`, the
  wording the amendment removed as a JSON list in an order that carries
  nothing. The section's own number is masked as `[SECTION]`, and no title,
  part or section name is published
- **Marks** — every gap a fragment came out of, plus three decoys per real
  gap (never fewer than five per row)
- **Output** — `text_before`, the prior wording, as free text with the marks
  removed
- **Metric** — gap-anchored changed-token F1: only the tokens the amendment
  removed are scored, each in the gap where it sat, balanced against what the
  prediction adds
- **Split** — by CFR part, so drafting conventions must generalise rather than
  be memorised; test holds one pair per section, so no test row's shown text
  is another test row's answer
- **Train labels** — `train_slots.csv` gives the mark each training fragment
  came from
- **Row independence** — every test row is predicted from the training files
  and its own fields; nothing is fitted on `test.csv` as a corpus

## Why the task is shaped this way

The first build showed `text_after` alone and asked for the prior wording
outright. Nothing scored: the seq2seq reference reached 0.002 and three agent
runs 0.020 / 0.000 / 0.000. A rule that now says *ten days* does not record
that it used to say *thirty*; the prior wording is not inferable. Where
wording belongs is, so the removed wording is handed over and the solver has
to place it.

## Why the metric is what it is

The median training amendment changes **2.2%** of its section's tokens. Under
whole-string similarity, the input with its marks removed would be ~98%
correct and score near the maximum. Scoring only the changed tokens sends it
to exactly 0.

Those tokens are matched by **gap** — the right token in the right place — and
balanced against precision. Position-free, appending every fragment to the
end of the section would be a perfect answer; keyed by gap it scores 0.045.
Under recall alone, every fragment at every mark would be perfect; with the
precision term it scores 0.112.

| Baseline | Score |
|---|---|
| Blank submission | 0.000 |
| Marks removed, nothing put back (sample submission) | 0.000 |
| Every fragment appended to the end | 0.045 |
| Fragments dealt to marks at random | 0.091 |
| Every fragment at every mark | 0.112 |
| Tightest marks, fragments at random (best content-free probe) | 0.119 |
| Fragment-blind mark classifier | 0.188 |
| Bigram-count heuristic, no labels | 0.295 |
| Learned pair scorer + assignment (`solution.ipynb`) | 0.497 |
| Ceiling: every fragment at its true mark | 0.982 |
| Oracle | 1.000 |

## Data provenance

Electronic Code of Federal Regulations, versioner API v1. Office of the
Federal Register (NARA) with the U.S. Government Publishing Office.
**U.S. Government Work** — public domain under 17 U.S.C. § 105.

The eCFR is an editorial compilation, **not** the official legal edition of
the CFR. This is a research benchmark and must not be used for any legal or
compliance purpose. Full notice in `dataset/raw/ATTRIBUTION.txt`.
