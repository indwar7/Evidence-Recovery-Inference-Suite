# Reconstructing Local Citation Structure in Case Law

Given one U.S. court opinion and a fixed pool of twelve candidate
opinions, decide which candidates the query opinion actually cites.

Every formal citation string and every party name has been stripped from
the text, so the answer cannot be read off the page. It has to be
inferred from what the opinions are arguing about.

## Package contents

```
config.yaml                        challenge manifest
problem-description.md             the task, as the solver sees it
dataset-description.md             provenance, schema, intended use, limitations
rubrics.md                         scoring bands
prepare.py                         raw/ -> public/ + private/ (deterministic)
grade.py                           grade(submission, answers) -> float
solution.ipynb                     worked reference solution
dataset/
  raw/                             source material + provenance
    opinions.csv                   id, cluster_id, plain_text (redacted)
    opinion_clusters.csv           id, date_filed, case_name, docket_id, ...
    citation_edges.csv             citing_opinion_id, cited_opinion_id
    build_meta.json                build-time counts
    meta.json                      schema, checksums, licence
    ATTRIBUTION.txt                source, licence, list of modifications
  public/
    train.csv                      1,954 queries WITH labels
    test.csv                       400 queries WITHOUT labels
    sample_submission.csv          format example
    dataset_stats.json             split statistics and leak scan
  private/
    answers.csv                    the answer key
```

## Submission format

One row per test query:

```csv
query_id,cited_positions
q_0001,0 4
q_0002,2 5 9
q_0003,
```

`cited_positions` is a whitespace-separated list of 0-indexed candidate
positions — position `k` means the `candidate_{k:02d}_text` column of
that query's row in `test.csv`. An empty value is legal and means "this
query cites none of these twelve candidates".

## Metric

Mean per-query Matthews Correlation Coefficient over the candidate pool,
clipped to `[0, 1]`.

MCC is used rather than F1 because the candidate pool has a minority of
true positives, and MCC accounts for true negatives. Both degenerate
strategies score exactly zero:

| Submission                        | MCC    |
|-----------------------------------|--------|
| select nothing                    | 0.000  |
| select all twelve                 | 0.000  |
| random pair per query             | ~0.126 |
| always positions 0 and 1          | 0.125  |
| reference (windowed encoder + lexical) | 0.672 |
| perfect                           | 1.000  |

Always picking positions 0 and 1 scores the same as picking two at
random, which is the check that candidate ordering carries no signal.

## Reproducing the data

```bash
python prepare.py
```

Reads `dataset/raw/`, writes `dataset/public/` and `dataset/private/`.
Deterministic: two runs produce byte-identical outputs.

## Environment

CPU only. The reference solution uses a pre-trained sentence embedding
model on CPU; no GPU is required at any stage.
