# The Accent Translator

**Domain:** natural-language-processing · **Task type:** sequence-to-sequence
**Metric:** change-segment accuracy · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC BY-SA 4.0

## The task

Given a word and a source/target accent pair (RP, GA, AU, NZ), predict the
word's phoneme string in the target accent. The source transcription is
hidden; the model works from spelling plus accent labels.

The metric scores **only the segments that differ from the source accent**.
Copying the spelling scores 0.09; the CPU letter-context reference scores 0.46.
Test words are unseen in training, one row per word.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and measured baseline anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 9 rubrics — 4 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL |
| `prepare.py` | Deterministic raw → public/private split (seed 20260918), word-disjoint |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution, 3 progressive submissions (GPU final) |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py
python grade.py dataset/public/sample_submission.csv dataset/private/answers.csv   # ≈0.09 floor
```

## Data

English Wiktionary, official dump. Accent tags and IPA are the archive's own
`{{IPA|en|…|a=…}}` template values — no annotation, no generated content.
**CC BY-SA 4.0**, confirmed via the MediaWiki siteinfo API. See
`dataset/raw/ATTRIBUTION.txt`.
