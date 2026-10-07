# Digitization Pipeline Attribution

Real historical American newspapers were digitized in batches by the Library
of Congress's National Digital Newspaper Program (NDNP). Different batches
were processed by different OCR engines and digitization vendors over many
years. Your task is to recover which digitization batch produced each piece
of OCR'd text — using only the systematic character-error fingerprint that
batch's OCR pipeline left behind, never the actual words.

## Task

You are given `bags` of text snippets. Every snippet in a bag is drawn from
a DIFFERENT real newspaper article, so the article's subject matter tells you
nothing about which snippets belong together — only the OCR noise pattern
does. Some unknown number of snippets in a bag came from each of several real
digitization batches, and your job is to recover that grouping.

- You do not need to name which real batch a group came from — only which
  snippets travelled together.
- The number of true groups in a bag is not given and varies bag to bag.
- Each snippet's text is genuinely noisy, uncorrected OCR output: character
  substitutions, junk characters, and broken word boundaries are real
  artifacts of period OCR engines running on period newspaper print.

## Why this is not a text-generation or classification task

This is a program-writing and modeling task, not a text-generation task. You
are not asked to clean the text, name a batch, or classify a snippet into a
known category — every test batch is entirely unseen in training. You are
asked to recover a **partition**: which items share a hidden real-world
grouping variable. Write a fallback `submission.csv` (every snippet in its
own group is a safe, valid, scoreable starting point) before you start
building your real approach.

## Dataset

Discover the data directory rather than assuming a fixed path — it may be
mounted differently across environments:

```python
from pathlib import Path

DATA = next(
    p for p in (Path("/data"), Path("dataset/public"),
                Path("/kaggle/input/pipeline-attribution-dataset/public"))
    if (p / "test.csv").exists()
)
```

Files in the data directory:

- `train.csv` — columns `snippet_id, text, batch_id`. Every row is labeled
  with its real digitization batch. Use this however you like: train a
  classifier, an embedding model, or hand-engineer features — the point is to
  learn what makes one pipeline's OCR noise look different from another's, in
  a way that generalizes to pipelines you have never seen labeled examples
  from.
- `test.csv` — columns `row_id, bag_id, snippet_id, text`. No labels.
  `row_id` uniquely identifies a `(bag_id, snippet_id)` pair (the same
  `snippet_id` can legitimately appear in more than one bag). Rows sharing
  the same `bag_id` are the snippets you must partition. A bag's snippets are
  listed in shuffled order — position carries no information.
- `sample_submission.csv` — a valid, minimum-viable submission (every snippet
  in its own bag placed in group `0`). It scores 0.000, not an error.
- `dataset_stats.json` — counts and split statistics for the shipped data.

## Evaluation

Submissions are scored using **mean per-bag Adjusted Rand Index (ARI)**,
clipped to `[0, 1]` and averaged across all test bags.

```python
from sklearn.metrics import adjusted_rand_score

def score_bag(true_batch_ids: list, submitted_group_labels: list) -> float:
    return max(0.0, adjusted_rand_score(true_batch_ids, submitted_group_labels))
```

- `group_label` values are entirely up to you — any string or integer. They
  only need to be consistent WITHIN one bag (two snippets sharing a
  `group_label` means "I believe these came from the same batch"). They do
  not need to match any real batch name, and reusing the same label across
  different bags has no special meaning — every bag is scored independently.
- ARI is chance-corrected: putting every snippet in one group, or every
  snippet in its own group, both score close to 0.000. This is intentional —
  those submissions carry no real information.
- No group-count guess is required to be exactly right; ARI degrades
  gracefully with imperfect partitions rather than requiring an exact match.

## Submission

Submit a CSV file named `submission.csv` with the following columns:

- `row_id` — copy this directly from `test.csv`.
- `bag_id` — the bag this row belongs to, from `test.csv`.
- `snippet_id` — the snippet this row belongs to, from `test.csv`.
- `group_label` — your predicted group for this snippet, any string or int.

**Requirements**

- One row per `row_id` in `test.csv`.
- Include a header row.

**Structural slips cost points, not the run.** The grader degrades rather
than rejects imperfect submissions:

- A missing bag scores 0 for that bag only; the rest of your submission is
  still graded.
- A missing snippet row within a bag is treated as an unrecoverable
  prediction for that snippet only.
- Extra rows for bags not in the current grading slice are ignored.
- Only a submission missing required columns entirely is rejected outright.

## Compute

This task benefits genuinely from GPU acceleration: the labeled training set
has thousands of real, noisy text snippets across 19 real digitization
batches, and a from-scratch neural encoder trained to recognize pipeline
fingerprints (rather than hand-engineered noise statistics) has real, though
not yet fully realized, headroom above the strongest baseline measured
during this task's construction. You have a 1x NVIDIA A10G GPU, 10 CPU cores,
and 62 GB RAM, with a 1.5 hour runtime budget.
