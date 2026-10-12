# The Reference Shuffle

You are given one paragraph of real, published reference-article prose
with every inline citation marker replaced by a neutral `[CITE]` token, and a
shuffled pool of the paragraph's own reference cards — the exact
bibliographic sources that paragraph really cites, each shown as
structured metadata (title, author, venue, year). Every card in the pool
is a genuine citation of this paragraph; nothing has been withheld from
the pool and nothing has been added to it. Two things are withheld from
you. First, **which `[CITE]` marker each card belongs to** — the order in
which the paragraph's authors actually cited them has been scrambled.
Second, **one recorded field on every card** — each card shows three of
its four fields, and the fourth is shown as `[withheld]`. Your job is to
put the cards back in order from what is left.

## Task

Every paragraph in this dataset has **exactly 4** references, shown as
cards `ref_A_card`, `ref_B_card`, `ref_C_card`, `ref_D_card`. Where the
original paragraph cites more than four sources, it is shown with four
consecutive citations kept; the markers of the others are deleted from
the text, and the four kept citations keep their true relative order.
For each paragraph, recover the true reading-order position (0, 1, 2, or 3) of
each lettered card — i.e. figure out which card's citation was the first
one actually made in the paragraph, which was second, third and fourth.

- Reference cards are shuffled per paragraph under a fixed seed and
  labelled with opaque letters (A, B, C, D) that carry no ordering
  information themselves.
- Every citation is real: taken from the structured citation records the
  paragraph's authors wrote, with the field values (title, author, venue,
  year) kept as recorded.
- Every card has the same four slots in the same order, `title | author
  | venue | year`. Exactly one slot that the editors recorded is shown as
  `[withheld]`; which slot is drawn per card under the fixed seed,
  independently of the card's true position. A slot the editors never
  recorded is shown as its placeholder (`(uncredited)`, `(unspecified)`,
  `(undated)`) and is never the withheld one. The withheld value is not
  written anywhere in the data directory.
- The true order is the source's own recorded structure — the physical
  left-to-right position of each citation marker in the original text —
  not an annotation created for this task.

A card looks like this:

```
title: [withheld] | author: Smith, J. | venue: The Guardian | year: 2004
```

## Dataset

The data directory is mounted by the solver environment. Locate it from
the environment you are running in rather than hard-coding a path. It
contains these files:

- `train.csv` — labelled paragraphs. Columns: `unit_id`, `text` (the
  paragraph with `[CITE]` markers), `ref_A_card`, `ref_B_card`,
  `ref_C_card`, `ref_D_card`, and the answer as four columns `rank_A`,
  `rank_B`, `rank_C`, `rank_D` — each an integer in `{0, 1, 2, 3}` giving
  that card's true reading-order position. Train cards are withheld the
  same way as test cards.
- `test.csv` — unlabelled paragraphs. Same columns except `rank_A` /
  `rank_B` / `rank_C` / `rank_D` are absent.
- `sample_submission.csv` — the exact output format, one row per test
  unit. Its `rank_*` values are format examples, not predictions.

The train/test split is by **source article**: no two paragraphs from the
same source article appear on both sides, so a solution cannot succeed by
having memorised a specific article's citation habits from train and
reapplying them to a held-out paragraph of the *same* article in test.
385 train units from 205 articles; 319 test units from 163 articles.

## Rules

**Use only the data directory.** The source paragraphs are real
published text and, in principle, findable online by searching for a
distinctive phrase. Recovering the answer by matching the paragraph back
to its original source and reading its real, un-redacted citation order —
rather than reasoning about which reference card's visible title, author,
venue and year best fits which sentence — is not solving the task; a
submission built that way is invalid regardless of score. The same
applies to recovering a withheld field from any source outside the data
directory. The solver environment is expected to have no network access,
and your solution must run to completion from the data directory alone.

**Score each unit using only its own reference pool.** Every paragraph is
graded independently against its own four cards. A solution that
depends on seeing other test rows to resolve the current one would not
work on a single paragraph presented in isolation, which is the
deployment shape of this task.

- Learn from the provided data only.
- Seed everything; your run should be reproducible.

## What not to use

Each of these makes a submission invalid regardless of its score.

- **Any language model, hosted or local**, for generating, ranking,
  labelling, or rewriting anything — no API calls, no locally loaded
  instruction-tuned models, and no LLM-generated output anywhere in the
  pipeline.
- **Any model trained on citation or bibliographic data** — citation
  recommenders, reference matchers, retrievers trained on encyclopedia
  text or on scholarly citation graphs. The task measures whether the
  paragraph and the cards, as shipped, carry the signal; a model that has already seen
  the citation structure of the web does not test that.
- **Any external data at solve time** — no encyclopedia dump or page, no
  citation index, no DOI, ISBN or URL resolver, no knowledge base, and no
  cached copy of the source text. A withheld field stays withheld. The
  data directory is the whole world.
- **Statistics fitted on the test set.** Term frequencies, document
  frequencies, n-gram counts or any other corpus-level statistic must
  come from `train.csv` only. Fitting them on `test.csv`, or on train and
  test pooled, is transductive adaptation and is treated the same as
  reading another test row.
- **Cross-row reasoning on test.** Nothing from one test unit may inform
  the prediction for another — no shared venue tables built from test, no
  intersecting candidate pools across rows, no row-order or `unit_id`
  features.
- **Hand-labelled test units**, in whole or in part.

Using `train.csv` in any way you like — supervised training, feature
engineering, calibration, held-out validation — is the intended path.

## Evaluation

Submissions are scored by **mean per-unit position accuracy**: for each
test unit, the fraction of its four cards placed at their true
reading-order position, averaged over all test units. A row whose four
values are not a permutation of `{0, 1, 2, 3}` scores 0.0, and a
uniformly random order scores 0.25 in expectation.

## Submission

Submit a CSV named `submission.csv` with one row per test unit:

- `unit_id` — copied from `test.csv`.
- `rank_A`, `rank_B`, `rank_C`, `rank_D` — the reading-order position (0,
  1, 2, or 3) you believe each lettered card's citation truly occupies in
  the paragraph. Together the four values for a row must form a
  permutation of `{0, 1, 2, 3}`.

```csv
unit_id,rank_A,rank_B,rank_C,rank_D
test_00000,1,0,3,2
test_00001,0,2,1,3
test_00002,3,0,1,2
```

**Requirements**

- Exactly one row per unit in `test.csv`, each `unit_id` exactly once.
- `rank_A`, `rank_B`, `rank_C`, `rank_D` must jointly be a permutation of
  `{0, 1, 2, 3}` — no repeats, no omissions, no value outside that range.
- Include a header row.

**Rejected outright** (the run does not score): a submission missing a
required column.

**Everything else costs points, not the run — scored 0.0 for that unit
only, the rest is still graded:**

- A missing unit.
- A malformed, non-integer, out-of-range, or non-permutation
  `(rank_A, rank_B, rank_C, rank_D)` tuple.
- A `unit_id` listed more than once: every conflicting row for it is
  discarded and that unit scores 0.0 — the grader does not pick one of
  the duplicates.

## Compute

This task runs on **CPU only**. No GPU is provided or required at any
stage, including for the reference solution. You have 10 CPU cores and
62 GB RAM, with a 1.5 hour runtime budget.
