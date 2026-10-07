# Usage

Every benchmark is self-contained, offline, and identically shaped. Once you can run
one, you can run them all.

---

## Install

```bash
git clone https://github.com/indwar7/eris-ml.git
cd eris-ml
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn
```

No API keys. No downloads. No GPU. The built corpora are committed; the optional
regeneration dependencies are listed, commented out, in `requirements.txt`.

---

## The four-minute tour

```bash
python tools/verify.py --reference
```

This re-runs every shipped grader against the shipped data, retrains every CPU reference
solution, and checks all eight published anchors. It exits non-zero if any anchor has
moved. Expect `8/8 anchors reproduced` in about 25 seconds.

---

## Anatomy of a benchmark

```
benchmarks/<name>/
├── config.yaml              the spec — read this first
├── problem-description.md   what the solver is given and asked for
├── dataset-description.md   what the data is and where it came from
├── rubrics.md               the rules a submission must satisfy
├── prepare.py               raw corpus  ->  public/private split (seeded)
├── grade.py                 the official scorer
├── solution.py / .ipynb     the runnable reference
└── dataset/
    ├── generator/build_raw.py
    ├── raw/                 the corpus as collected, plus ATTRIBUTION.txt
    ├── public/              train.csv · test.csv · sample_submission.csv
    │                        dataset_stats.json · validate_submission.py
    └── private/             answers.csv — the sealed key
```

`config.yaml` is the single source of truth: metric definition and formula, every
measured anchor, split policy, provenance, licence, and the runtime the task was
calibrated against. When this documentation and a `config.yaml` disagree, the config
wins.

---

## Running a solver

Every benchmark follows the same three steps. Using `experimental-order` as the example:

```bash
cd benchmarks/experimental-order

# 1. train and predict
python solution.py --out submission.csv

# 2. check the submission is structurally valid, without touching the key
python dataset/public/validate_submission.py submission.csv

# 3. score it
python grade.py submission.csv dataset/private/answers.csv
# 0.3585
```

`validate_submission.py` needs no answers and is safe to run inside a solver loop. It
catches the structural errors that the grader rejects outright — a missing column, a
duplicated identifier — before you spend a scoring run on them.

---

## Submission formats

| Benchmark | Columns | One row per | Grader invocation |
|---|---|---|---|
| accent-transfer | `row_id, phonemes` | test row | `python grade.py sub.csv dataset/private/answers.csv` |
| prompt-edit-attribution | `item_id, rank_1..rank_4` | test item | `python grade.py sub.csv --answers dataset/private/answers.csv` |
| pipeline-attribution | `bag_id, batch_id` | **bag**, not snippet | `python grade.py sub.csv dataset/private/answers.csv` |
| experimental-order | `paper_id, predicted_order` | test paper | `python grade.py sub.csv dataset/private/answers.csv` |
| citation-structure | `query_id, cited_positions` | test query | `python grade.py sub.csv dataset/private/answers.csv` |
| reference-order | `unit_id, rank_A, rank_B, rank_C` | test paragraph | `python grade.py sub.csv dataset/private/answers.csv` |
| mechanism-recovery | `row_id, moa_classes` | test label | `python grade.py sub.csv --answers dataset/private/answers.csv` |
| amendment-reversal | `row_id, text_before` | test section | `python grade.py sub.csv --answers dataset/private/answers.csv` |
| rag-sufficiency | `item_id, diagnosis, action, reason_code` | test item | `python grade.py sub.csv --answers dataset/private/answers.csv` |
| gene-symbol-coinage | `row_id, symbol` | test gene | `python grade.py sub.csv dataset/private/answers.csv` |
| clause-responsibility | `item_id, ranking, flags` | test item | `python grade.py sub.csv --answers dataset/private/answers.csv` |

Four formats deserve a closer look.

**`pipeline-attribution` submits one row per bag.** The `batch_id` cell holds one label
per snippet, whitespace-separated, **in `test.csv` snippet order**. The labels are
arbitrary tokens: they never need to match across bags and never need to name a real
batch, because only co-membership *within* a bag is scored. One row per bag is
deliberate — a per-snippet key would let a row-wise leaderboard slice cut a bag in half
and score each fragment as its own bag.

**`citation-structure` and `mechanism-recovery` submit a set.** `cited_positions` is a
whitespace-separated list of 0-indexed candidate positions; `moa_classes` is a
pipe-separated list of class names from that row's own pool. How many to select is not
given and is part of the problem. Selecting everything scores exactly zero on both.

**`citation-structure` needs one build step.** Its `train.csv` is 142 MB, over GitHub's
file limit, so it is not committed. Run `python prepare.py` in that directory once; the
file it writes is byte-identical to the one the anchors were measured on.

**`rag-sufficiency` submits a joint trace.** The three fields are scored together per
item, and they must be mutually consistent: `reason_code=gold_absent` implies
`action=abstain`, never `answer_as_is`. Getting the action right and the reason wrong is
not full credit.

---

## Calling the graders from Python

Every grader exposes the same function, so a harness can treat them all uniformly:

```python
import pandas as pd
from importlib import import_module
import sys

sys.path.insert(0, "benchmarks/experimental-order")
grade = import_module("grade").grade

submission = pd.read_csv("submission.csv", keep_default_na=False)
answers    = pd.read_csv("benchmarks/experimental-order/dataset/private/answers.csv",
                         keep_default_na=False)
print(grade(submission, answers))   # 0.3585
```

Pass `keep_default_na=False`. Several corpora contain literal strings that pandas
otherwise coerces to `NaN`.

---

## What the graders do with a bad submission

The contract is uniform across the suite:

| Situation | Behaviour |
|---|---|
| Missing column | `InvalidSubmissionError` (a `ValueError` subclass) — rejected outright |
| Duplicated identifier where uniqueness is load-bearing | `InvalidSubmissionError` |
| Missing row for a test item | That item scores `0.0`; the rest is scored normally |
| Malformed cell — non-permutation, wrong label count | That item scores `0.0`; no exception |
| Extra rows not in the test set | Ignored |

The design intent is that one malformed item never zeroes an otherwise valid
submission, while a structurally meaningless file is refused rather than silently scored.

---

## Rebuilding a corpus from source

Not needed to use the suite — every corpus is committed — but fully supported.

```bash
cd benchmarks/<name>
python dataset/generator/build_raw.py --out dataset/raw    # refetch from source
python prepare.py                                          # rebuild public/private
```

Generators are network-bound and rate-limited by their sources; the PLOS generator, for
example, sustains about one request every 1.2 seconds and supports `--articles` and
`--start` for resumable scaling. Preparation scripts are seeded, so a rebuild from the
same raw corpus is byte-identical. Uncomment the regeneration dependencies in
`requirements.txt` first.

---

## Using this suite for model evaluation

Some practical notes for an evaluation harness:

1. **Hold out the key.** `dataset/private/answers.csv` is committed so that scores are
   verifiable. If you are running an agent that can read the filesystem, remove or move
   `dataset/private/` before the run — several of these tasks are trivially solved by a
   solver that finds the key.
2. **Enforce the no-external-data rule.** Two corpora are verbatim published text and
   their sources are publicly searchable. The rule is `[REQUIRED]` in every `rubrics.md`
   and the solver environment is expected offline. A score at or near `1.000` on
   `experimental-order` or `pipeline-attribution` should be read as source lookup, not
   ability — see [METHODOLOGY.md § 5](METHODOLOGY.md).
3. **Read the banned-shortcut list.** Each `rubrics.md` names the attacks that are
   prohibited by rule rather than blocked by construction, with their measured scores.
   A harness that does not enforce them is measuring something other than the task.
4. **Compare against the ladder, not the reference alone.** A model that beats the
   reference but not the strongest shortcut attack has found the shortcut. The ladders
   are in [BENCHMARKS.md](BENCHMARKS.md).
5. **Budget.** Each task was calibrated against a single A10G with 10 cores and 62 GB,
   under 1.5 hours. Every reference in this repository runs on CPU in seconds.

---

## Getting help

The spec for any benchmark is its `config.yaml`, and the reasoning behind any design
decision is in that file's `note_band`, `band_evidence` and `split_note` fields — or, for
`pipeline-attribution`, in its `DESIGN.md`. Both are written for a reviewer, not for a
marketing page, and they document the defects found during construction as carefully as
the results.
