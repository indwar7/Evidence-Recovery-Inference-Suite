# The Reference Shuffle

Given one real published paragraph with its inline citation markers
replaced by a neutral token, and a shuffled pool of that paragraph's own
real reference cards (title, author, venue, year), recover the order in
which the references were actually cited.

Every card in the pool is a genuine citation of the paragraph — nothing
is a distractor. The only withheld information is order.

## Package contents

```
config.yaml                        challenge manifest
problem-description.md             the task, as the solver sees it
dataset-description.md             provenance, schema, intended use, limitations
rubrics.md                         scoring bands
DESIGN.md                          measured design record
prepare.py                         raw/ -> public/ + private/ (deterministic)
grade.py                           grade(submission, answers) -> float
solution.ipynb                     worked reference solution
dataset/
  raw/                             source material + provenance
    wiki_citation_units.jsonl      one JSON object per extracted paragraph unit
    _fetched_titles.json           fetch bookkeeping
    build_meta.json                build-time counts
    meta.json                      schema, checksums, licence
    ATTRIBUTION.txt                source, licence, list of modifications
  public/
    train.csv                      labelled paragraphs WITH rank_A/rank_B/rank_C
    test.csv                       unlabelled paragraphs
    sample_submission.csv          format example
    dataset_stats.json             split statistics
  private/
    answers.csv                    the answer key
```

## Submission format

One row per test unit:

```csv
unit_id,rank_A,rank_B,rank_C
test_00001,1,0,2
test_00002,0,2,1
```

`rank_A`, `rank_B`, `rank_C` jointly form a permutation of `{0, 1, 2}` —
each is the 0-indexed reading-order position you believe that lettered
card's citation truly occupies in the paragraph.

## Metric

Mean rescaled Kendall's tau rank correlation over each unit's reference
order, averaged across all test units.

A uniformly random permutation of `{0, 1, 2}` has an *exact* expected
score of 0.5 (verified by exhaustive enumeration over all 6 permutations
— see DESIGN.md). A non-attempt (missing row, malformed value, duplicate
id) scores 0.0, not 0.5 — the standard degrade rule this account's
challenges use, so there is a real incentive to attempt every row rather
than skip it.

| Strategy | Score |
|---|---|
| blank / non-attempt | 0.0000 |
| pool order (unchanged) | 0.4771 |
| adversarial (global venue-position, train-fit) | 0.4924 |
| random permutation (5-seed mean) | 0.4954 |
| reversed pool order | 0.5229 |
| year-sort ascending | 0.5352 |
| title-length sort | 0.5413 |
| **reference** (word overlap + author/year signal, Hungarian assignment) | **0.7248** |
| oracle | 1.0000 |

Measured on the final shipped test split: 109 units, 70 articles.

## Reproducing the data

```bash
python prepare.py
```

Reads `dataset/raw/`, writes `dataset/public/` and `dataset/private/`.
Deterministic: two runs produce byte-identical outputs.

## Environment

CPU only. No GPU is required at any stage.
