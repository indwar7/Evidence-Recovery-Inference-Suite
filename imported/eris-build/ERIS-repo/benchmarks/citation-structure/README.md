# Reconstructing Local Citation Structure in Case Law

**Domain:** natural-language-processing · **Tags:** text
**Metric:** mean per-query Matthews correlation coefficient (both degenerate selections score 0.000) · **Runtime:** CPU only
**Licence:** Public Domain Mark

## The task

You are given one U.S. court opinion and a fixed pool of twelve candidate
opinions. Decide which of the twelve the query opinion actually cites.

Every formal citation string and every party name has been stripped from the
text, so the answer cannot be read off the page. It has to be inferred from
what the opinions are arguing about.

**Between one and four candidates are correct, and the number is not given.**
Deciding how many to select is part of the problem: selecting nothing and
selecting all twelve both score exactly 0.000.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and 15 measured anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 7 rubrics — 4 REQUIRED, 3 RECOMMENDED |
| `DESIGN.md` | The measured design record, including the shape that was abandoned |
| `prepare.py` | Deterministic raw → public/private split (seed 20260921) |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution: frozen sentence encoder over sliding windows, blended with lexical overlap |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                dataset/private/answers.csv              # -> 0.141 (format example)
```

> **`dataset/public/train.csv` is not committed.** It is 142 MB, over GitHub's
> 100 MB file limit. `python prepare.py` rebuilds it from `dataset/raw/`, and
> the rebuild is byte-identical to the file the anchors were measured on.
> `test.csv`, `sample_submission.csv` and `answers.csv` are committed, so
> every published anchor can be re-scored without running it.

The reference needs `sentence-transformers` and downloads a pre-trained
encoder on first use, so it is not part of the offline `tools/verify.py` run.

## Data

Real U.S. court opinions from the Free Law Project's CourtListener bulk
export, snapshot 2022-09-30. The label is the archive's own recorded citation
edge between two opinions — no annotation created for this dataset, no
model-generated content.

The split is by **connected component of the citation graph**: 521
components go to train and 3 to test, so no real citation edge crosses the
split. 1,954 train queries, 400 test queries, 4,151 opinions.

**Public Domain Mark** — the Free Law Project states that its bulk data files
are "free of known copyright restrictions", and U.S. judicial opinions are not
copyrightable. Commercial use permitted. See `dataset/raw/ATTRIBUTION.txt`.
