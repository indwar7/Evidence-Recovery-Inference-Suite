# The Vanished Clause

**Domain:** natural-language-processing · **Tags:** text, seq2seq, legal, generation
**Metric:** gap-anchored changed-token F1 (copying the input scores 0.000) · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** U.S. Government Work

> **No shipped solution clears the floor on this benchmark.** The reference
> scores 0.002. The data, the split, the metric and the ladder are measured
> and reproducible; a working baseline is the open problem. The two
> diagnostic ceilings below show the score is reachable: perfectly restoring
> one sentence per section is worth 0.749.

## The task

A regulation is amended. The old wording is not struck through or footnoted;
it is replaced in place, and anyone reading afterwards sees only the new
text. You are given a section of the Code of Federal Regulations as it reads
*after* an amendment. Reconstruct how it read *before*.

The Federal Register instruction that made each change ("remove the words X
and add in their place Y") is withheld, so the change has to be inferred from
how the surviving text reads. The section's own number is masked as
`[SECTION]`, and no title, part or section name is published, because
together they address the section in the public version history.

**In the test split the median amendment changes 3.9% of its section's tokens.** Only those
tokens are scored, each in the gap where it sat.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and 11 measured anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 8 rubrics — 4 REQUIRED, 3 RECOMMENDED, 1 UNIVERSAL |
| `DESIGN.md` | The measured design record |
| `prepare.py` | Deterministic raw → public/private split; the split uses no randomness |
| `grade.py` | `grade(submission, answers) -> float` |
| `reference_solution.py` | CPU reference: sentence locator and mined substitution table |
| `solution.ipynb` | Reference notebook, including the seq2seq submission |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                --answers dataset/private/answers.csv    # -> 0.000 (input copied through)
python reference_solution.py --out submission.csv        # about one minute, CPU
python grade.py submission.csv --answers dataset/private/answers.csv
```

## Data

Consecutive published versions of federal regulations from the Electronic
Code of Federal Regulations. The amendment boundary is the archive's own
recorded `amendment_date`, and both versions are the agency's published
wording on those dates — no annotation created for this dataset, no
model-generated content.

1,404 raw pairs across nine CFR titles reduce to 410 train and 353 test. The
split is by **CFR part**, and test holds exactly one pair per section, so no
test row's shown text is another test row's answer.

**U.S. Government Work** — public domain under 17 U.S.C. § 105. The eCFR is
an editorial compilation, not the official legal edition of the CFR; this
benchmark must not be used for any legal or compliance purpose. See
`dataset/raw/ATTRIBUTION.txt`.
