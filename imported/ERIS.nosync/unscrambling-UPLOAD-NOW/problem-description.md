# The Reference Shuffle

You are given one paragraph of real, published reference-article prose
with every inline citation marker replaced by a neutral `[CITE]` token, and a
shuffled pool of the paragraph's own reference cards — the exact
bibliographic sources that paragraph really cites, each shown as
structured metadata (title, author, venue, year). Every card in the pool
is a genuine citation of this paragraph; nothing has been withheld or
added. The only thing you are not told is **which `[CITE]` marker each
card belongs to** — the order in which the paragraph's authors actually
cited them has been scrambled. Your job is to put it back.

## How this differs from adjacent tasks

- **Not support-selection.** Every reference card in the pool is a true
  citation of the paragraph — there is no true/false decision to make
  and no distractor to reject. The withheld information is purely
  *order*, not *membership*. (A separate, earlier challenge on this
  account, "Reconstructing Local Citation Structure in Case Law", is a
  membership-selection task scored by set-overlap; this task is
  deliberately a different shape — the submitted artifact here is a
  permutation, not a subset, and nothing is scored by counting true/false
  positives.)
- **Not citation-string extraction.** The formal citation markup is
  removed before you see the paragraph. Nothing here is solved by parsing
  markup syntax.
- **Not retrieval against an external corpus.** The complete candidate
  pool — every card you need — is handed to you in the same row. You
  never search outside it.
- **Not document/paragraph classification.** There is no fixed label set.
  The target is an ordering relation specific to one paragraph's own
  reference pool.

## Task

Every paragraph in this dataset has **exactly 3** references, shown as
cards `ref_A_card`, `ref_B_card`, `ref_C_card`. For each paragraph,
recover the true reading-order position (0, 1, or 2) of each lettered
card — i.e. figure out which card's citation was the first one actually
made in the paragraph, which was second, and which was third.

- Reference cards are shuffled per paragraph under a fixed seed and
  labelled with opaque letters (A, B, C) that carry no ordering
  information themselves.
- Every citation is real: taken from the structured citation records the
  paragraph's authors wrote, with the field values (title, author, venue,
  year) kept as recorded.
- The true order is the source's own recorded structure — the physical
  left-to-right position of each citation marker in the original text —
  not an annotation created for this task.

## Dataset

Discover the data directory rather than assuming a fixed path:

```python
from pathlib import Path

DATA = next(
    p for p in (Path("/data"), Path("dataset/public"),
                Path("/kaggle/input/citation-order-dataset/public"))
    if (p / "test.csv").exists()
)
```

Files in the data directory:

- `train.csv` — labelled paragraphs. Columns: `unit_id`, `text` (the
  paragraph with `[CITE]` markers), `ref_A_card`, `ref_B_card`,
  `ref_C_card`, and the answer as three columns `rank_A`, `rank_B`,
  `rank_C` — each an integer in `{0, 1, 2}` giving that card's true
  reading-order position.
- `test.csv` — unlabelled paragraphs. Same columns except `rank_A` /
  `rank_B` / `rank_C` are absent.
- `sample_submission.csv` — the exact output format, one row per test
  unit. Its `rank_A` / `rank_B` / `rank_C` values are format examples,
  not predictions.
- `dataset_stats.json` — split statistics.

The train/test split is by **source article**: no two paragraphs from the
same source article appear on both sides, so a solution cannot succeed by
having memorised a specific article's citation habits from train and
reapplying them to a held-out paragraph of the *same* article in test.

## Rules

**Use only the data directory.** The source paragraphs are real
published text and, in principle, findable online by searching for a
distinctive phrase. Recovering the answer by matching the paragraph back
to its original source and reading its real, un-redacted citation order — rather than reasoning about which reference
card's title, author, venue and year best fits which sentence — is not
solving the task; a submission built that way is invalid regardless of
score. The solver environment is expected to have no network access, and
your solution must run to completion from the data directory alone.

**Score each unit using only its own reference pool.** Every paragraph is
graded independently against its own three cards. A solution that
depends on seeing other test rows to resolve the current one would not
work on a single paragraph presented in isolation, which is the
deployment shape of this task.

- Learn from the provided data only. No pretrained language models
  fine-tuned specifically to recover citation structure, no LLM calls of
  any kind, and no external data — including no live or cached copy of
  the source text, any knowledge base, or any citation index.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated output may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

Submissions are scored using **mean rescaled Kendall's tau rank
correlation** between the predicted and true `(rank_A, rank_B, rank_C)`
triple, averaged over all test units.

```python
def score_unit(true_ranks: list, pred_ranks: list) -> float:
    # true_ranks, pred_ranks: [rank_A, rank_B, rank_C], each a
    # permutation of {0, 1, 2}
    n = len(true_ranks)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_sign = true_ranks[i] - true_ranks[j]
            pred_sign = pred_ranks[i] - pred_ranks[j]
            if true_sign * pred_sign > 0:
                concordant += 1
            else:
                discordant += 1
    tau = (concordant - discordant) / (concordant + discordant)
    return (tau + 1.0) / 2.0
```

**Why rescaled Kendall's tau, and not exact-match or per-column
accuracy.** Exact-match on the whole triple is all-or-nothing: a
prediction that swaps only two of the three ranks scores identically to
one that reverses all three, which throws away almost every signal a
near-correct answer carries. Treating `rank_A`/`rank_B`/`rank_C` as three
independent classification targets breaks the constraint that they must
jointly form a permutation of `{0, 1, 2}` and does not penalise a
submission that ignores that constraint. Kendall's tau counts concordant
vs. discordant **pairs** across all three columns jointly, so a near-miss
keeps most of its credit and a reversal is scored as badly as it should
be.

**Why this has an exact 0.5 chance floor.** A uniformly random permutation
of 3 labels has an *exact* expected rescaled tau of 0.5 — verified here by
exhaustive enumeration over all 6 permutations, not by sampling. This
means a submission that reads no text at all and just guesses a random
order is centred at a *known*, fixed number — the task-design pattern
this account uses for every ordering/ranking challenge. A non-attempt
(missing row, malformed value, duplicated id) is NOT scored at this
floor; it is scored 0.0, matching the standard degrade rule used
throughout this account's challenges (see Submission below) — the gap
between 0.0 and ~0.5 is the incentive to attempt every row rather than
skip it.

Every measured baseline on the shipped test set (109 test units, 70
distinct articles):

| Strategy | Rescaled Kendall's tau |
|---|---|
| blank / non-attempt (see below) | 0.0000 |
| pool order (A=0, B=1, C=2, unchanged) | 0.4771 |
| adversarial: global venue→position model, fit on train, ignores per-unit content | 0.4924 |
| random permutation (mean over 5 seeds) | 0.4954 |
| reversed pool order | 0.5229 |
| year-sort ascending | 0.5352 |
| title-length sort (strongest content-free probe) | 0.5413 |
| **reference** (word overlap + author-surname bonus + year-consistency bonus, Hungarian assignment) | **0.7248** |
| oracle | 1.0000 |

Every content-free and structural probe clusters near the 0.5 chance
floor, confirming the shuffle carries no residual signal and no
position/length/pool-order shortcut exists. The strongest probe,
title-length sort, reaches 0.5413. The reference solution clears every
probe by at least 0.183 (0.7248 − 0.5413), using word overlap between
each `[CITE]` marker's surrounding text and each candidate card, plus a
small author-surname and year-consistency bonus, never treating any
content-free signal as forbidden.

## Submission

Submit a CSV named `submission.csv` with one row per test unit:

- `unit_id` — copied from `test.csv`.
- `rank_A`, `rank_B`, `rank_C` — the reading-order position (0, 1, or 2)
  you believe each lettered card's citation truly occupies in the
  paragraph. Together the three values for a row must form a permutation
  of `{0, 1, 2}`.

```csv
unit_id,rank_A,rank_B,rank_C
test_00001,1,0,2
test_00002,0,2,1
test_00003,2,0,1
```

**Requirements**

- Exactly one row per unit in `test.csv`, each `unit_id` exactly once.
- `rank_A`, `rank_B`, `rank_C` must jointly be a permutation of
  `{0, 1, 2}` — no repeats, no omissions, no value outside that range.
- Include a header row.

**Rejected outright** (the run does not score): a submission missing a
required column.

**Everything else costs points, not the run — scored 0.0 for that unit
only, the rest is still graded:**

- A missing unit.
- A malformed, non-integer, out-of-range, or non-permutation
  `(rank_A, rank_B, rank_C)` triple.
- A `unit_id` listed more than once: every conflicting row for it is
  discarded and that unit scores 0.0 — the grader does not pick one of
  the duplicates.
- Rows for units outside the current grading slice are ignored.

## Compute

This task runs on **CPU only**. No GPU is provided or required at any
stage, including for the reference solution. You have 10 CPU cores and
62 GB RAM, with a 1.5 hour runtime budget.
