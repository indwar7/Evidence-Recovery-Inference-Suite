# The Reference Shuffle

Given one real published paragraph with its inline citation markers
replaced by a neutral token, and a shuffled pool of that paragraph's own
real reference cards (title, author, venue, year), recover the order in
which the references were actually cited.

Every card in the pool is a genuine citation of the paragraph — nothing
is a distractor. What is withheld is the order, and one recorded field
on every card (shown as `[withheld]`), so a card may have to be placed
from its remaining fields or by elimination.

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
    train.csv                      labelled paragraphs WITH rank_A..rank_D
    test.csv                       unlabelled paragraphs
    sample_submission.csv          format example
  private/
    answers.csv                    the answer key
```

## Submission format

One row per test unit:

```csv
unit_id,rank_A,rank_B,rank_C,rank_D
test_00000,1,0,3,2
test_00001,0,2,1,3
```

`rank_A`, `rank_B`, `rank_C`, `rank_D` jointly form a permutation of
`{0, 1, 2, 3}` — each is the 0-indexed reading-order position you believe
that lettered card's citation truly occupies in the paragraph.

## Metric

Mean per-unit position accuracy: the fraction of each unit's four cards
placed at their true reading-order position, averaged across all test
units. A tuple that is not a permutation of `{0, 1, 2, 3}` scores 0 for
that unit.

A uniformly random permutation of four cards has an exact expected
score of 1/4. A non-attempt (missing row, malformed value, duplicate id)
scores 0.0.

Paragraphs that cite more than four sources keep one seeded window of
four consecutive citations; the other markers are deleted from the text.

| Strategy | Score |
|---|---|
| blank / non-attempt | 0.0000 |
| random permutation (5-seed mean) | 0.2382 |
| title-withheld cards placed first | 0.2398 |
| title-withheld cards placed last | 0.2445 |
| title-length sort | 0.2492 |
| pool order (unchanged) | 0.2516 |
| author-surname bonus alone | 0.2555 |
| reversed pool order | 0.2578 |
| adversarial (global venue-position, train-fit) | 0.2633 |
| year-sort ascending | 0.3033 |
| word overlap only (visible fields), Hungarian assignment | 0.4843 |
| **reference** (word overlap + year-consistency term, Hungarian assignment) | **0.4914** |
| oracle | 1.0000 |

Measured on the shipped test split: 319 units, 163 articles (train: 385
units, 205 articles).

## Reproducing the data

```bash
python prepare.py
```

Reads `dataset/raw/`, writes `dataset/public/` and `dataset/private/`.
Deterministic: two runs produce byte-identical outputs.

## Environment

CPU only. No GPU is required at any stage.
