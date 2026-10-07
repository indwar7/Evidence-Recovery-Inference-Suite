# Coining Gene Symbols from Gene Names

**Domain:** natural-language-processing · **Task type:** sequence generation
**Metric:** prefix agreement · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC0 1.0

> **Status: in build (~60%).** The corpus, the split, the grader and a
> 13-rung ladder are built and measured. **The reference does not yet clear
> the strongest adversary by the suite's margin** (+0.081 against a target of
> about +0.18), and `config.yaml`, the problem description, the rubrics and
> the notebook are not written. The ladder below is provisional until that
> margin is closed. Full status and remaining work: [`STATUS.md`](STATUS.md)
> and [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md).

## The task

Every human gene has an approved name and an approved symbol, both assigned
by a nomenclature committee. You are given the name,
`solute carrier family 2 member 1`, and asked to produce the symbol the
committee coined for it, `SLC2A1`.

A symbol is read from the left: the stem says which family the gene belongs
to, and the designators after it narrow that down to one member. The metric
follows the same order. It counts the leading characters a prediction shares
with the approved symbol and divides by the longer of the two lengths, so
padding a prediction cannot raise its score.

The split is by **date**. Train is every symbol the committee had approved
before 2010; test is what it approved afterwards.

## Files

| File | Purpose |
|---|---|
| `STATUS.md` | What is done, what is measured, what is left |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `prepare.py` | Deterministic raw → public/private split by approval date |
| `grade.py` | `grade(submission, answers) -> float` |
| `dataset/generator/build_raw.py` | Fetches the pinned snapshot and checks its sha256 |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                dataset/private/answers.csv              # -> 0.311 (initials of each word)
```

## Provisional ladder

Measured on the shipped split. Published as provisional; see the status note.

| Rung | Score |
|---|---|
| Blank submission | `0.000` |
| Copy the name through | `0.061` |
| First word, upper-cased | `0.217` |
| Adversarial: harvest capitalised tokens | `0.303` |
| **Initials of each word** — the shipped sample submission | `0.311` |
| Adversarial: nearest train name, designator swapped | `0.366` |
| Character seq2seq, 3.2M parameters | `0.447` |
| Oracle | `1.000` |

## Data

The HGNC complete gene set, quarterly snapshot 2026-07-07. Both the name and
the symbol are the committee's own recorded values — no annotation created
for this dataset, no model-generated content. 13,863 train and 2,270 test.

**CC0 1.0** — stated on the HGNC's own licence page. Commercial use
permitted, no attribution required. See `dataset/raw/ATTRIBUTION.txt`.
