# Reconstructing Local Citation Structure in Case Law

**Domain: NLP — relational inference over document text.** You are given
one court opinion and a pool of twelve candidate opinions. Some of the
candidates are cited by the query opinion; most are not. Every formal
citation string and every party name has been stripped from all thirteen
texts, so the answer cannot be read off the page. It has to be inferred
from what the opinions are arguing about.

## How this differs from adjacent tasks

- **Not citation extraction.** Nothing here is scored by pulling a
  citation string out of text. Every reporter citation and case-name
  mention is removed before you see the text, and the shipped split is
  verified to contain zero of either, so extraction is not a viable
  strategy at all.
- **Not retrieval against an external corpus.** You never search a
  database or fetch a document from outside the row you are given. The
  complete candidate set is handed to you upfront — twelve texts, in the
  same CSV row as the query.
- **Not document classification.** There is no fixed label set. The
  target is a *relation* between a specific pair of documents, and the
  same candidate text could be cited by one query and not by another.
- **Not a ranking task.** You must commit to a subset. Deciding *how
  many* candidates to select is part of the problem, and both degenerate
  answers — select none, select all — score exactly zero.

## Task

For each query opinion, decide which of its twelve candidates it cites.

- The number of true citations varies from query to query (1 to 4). It
  is not given to you.
- Candidate order is randomised per query and carries no signal.
- The citation edges are real, drawn from an independently maintained
  citation-extraction database, not constructed for this task.
- The opinions come from United States courts at every level. The
  collection is not restricted to one jurisdiction, so a query and the
  candidate it cites need not come from the same court system.

## Dataset

Discover the data directory rather than assuming a fixed path:

```python
from pathlib import Path

DATA = next(
    p for p in (Path("/data"), Path("dataset/public"),
                Path("/kaggle/input/citation-structure-dataset/public"))
    if (p / "test.csv").exists()
)
```

Files in the data directory:

- `train.csv` — 1,954 labelled queries. Columns: `query_id`, `query_text`,
  `candidate_00_text` … `candidate_11_text`, and `cited_positions` (the
  answer, as a whitespace-separated list of 0-indexed candidate
  positions). Use this however you like: supervised training, threshold
  calibration, feature engineering, or fine-tuning.
- `test.csv` — 400 unlabelled queries. Same columns except
  `cited_positions` is absent.
- `sample_submission.csv` — the exact output format, one row per test
  query. Its `cited_positions` values are format examples, not
  predictions.
- `dataset_stats.json` — split sizes and the redaction leak scan.

The train/test split is by **connected component** of the real citation
graph: no citation edge connects a test query to any training query or
candidate. A solution cannot succeed by memorising a citing pair it has
already seen.

## Rules

**Use only the data directory.** The opinion text is real and technically
findable in public legal archives. Recovering the answer by matching
redacted text back to a public source, rather than reasoning about the
text itself, is not solving the task; a submission built that way is
invalid regardless of score. The solver environment is expected to have
no network access.

**Score each query using only its own candidate pool.** Do not pool
candidates across test queries. Every query is graded independently, and
a solution that depends on seeing other test rows would not work on a
single query presented in isolation — which is the deployment shape of
this task.

## Evaluation

Submissions are scored using **mean per-query Matthews Correlation
Coefficient (MCC)** over the twelve-candidate pool, clipped to `[0, 1]`.

```python
def score_query(true_pos: set, pred_pos: set, n_candidates: int) -> float:
    tp = len(true_pos & pred_pos)
    fp = len(pred_pos - true_pos)
    fn = len(true_pos - pred_pos)
    tn = n_candidates - tp - fp - fn
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return max(0.0, min(1.0, num / den)) if den else 0.0
```

**Why MCC rather than F1.** Each pool has a minority of true positives,
so F1 over the selected set rewards over-selection: selecting all twelve
candidates earns a substantial raw F1 purely from recall, without reading
anything. MCC is computed over the full confusion matrix including true
negatives, which charges a submission for the distractors it wrongly
accepts. Measured on the shipped test set:

| Strategy                          | MCC    |
|-----------------------------------|--------|
| select none                       | 0.0000 |
| select all twelve                 | 0.0000 |
| always positions 0 and 1          | 0.1253 |
| random two per query (5-run mean) | 0.1257 |
| random three per query            | 0.1306 |
| two longest candidates            | 0.1262 |
| two shortest candidates           | 0.0963 |
| oracle                            | 1.0000 |

The two content-free position baselines confirm the candidate ordering
carries no exploitable signal: always picking positions 0 and 1 (0.1253)
is indistinguishable from picking two at random (0.1257).

**Reading the whole opinion matters.** These opinions run to roughly
1,500 tokens, and the passage that signals a citation is often deep in
the discussion. Measured on the same test set, using a pre-trained
sentence encoder:

| Approach                                        | MCC    |
|-------------------------------------------------|--------|
| encoder truncated to its 256-token window       | 0.6481 |
| same encoder over sliding windows, max-pooled   | 0.6494 |

Both use identical per-query normalisation and thresholding. On this split
the two are close: reading the whole document helps, but only slightly, and
the honest reading is that most of the signal survives truncation.

What matters far more is the decision rule. A fixed top-2 cut-off on the same
windowed scores drops to 0.5517, and a fixed top-1 to 0.5810 — per-query
normalisation and an adaptive threshold are worth more than the extra
context.

## Submission

Submit a CSV named `submission.csv` with one row per test query:

- `query_id` — copied from `test.csv`.
- `cited_positions` — the candidate positions you believe are cited, as
  whitespace-separated 0-indexed integers. An empty value means "cites
  none of these twelve".

```csv
query_id,cited_positions
q_0001,0 4
q_0002,2 5 9
q_0003,
```

**Requirements**

- Exactly one row per query in `test.csv`, each `query_id` exactly once.
- Every position must be an integer in `0..11`, with no repeats within a
  row.
- Include a header row.

**Rejected outright** (the run does not score): a submission missing a
required column.

**Everything else costs points, not the run:**

- A missing query scores 0 for that query only.
- A malformed, duplicated or out-of-range position scores 0 for that
  query only.
- A `query_id` listed more than once: every conflicting row for it is
  discarded and that query scores 0 — the grader does not pick one of the
  duplicates. The rest is still graded.
- Rows for queries outside the current grading slice are ignored.

## Compute

This task runs on **CPU only**. No GPU is provided or required at any
stage, including for the reference solution. You have 10 CPU cores and
62 GB RAM, with a 1.5 hour runtime budget.
