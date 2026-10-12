# Restoring Tone Marks on Vietnamese Place Names Built from Unseen Syllables

Recover the native Vietnamese spelling of a place name from its ASCII-folded
form. Vietnamese writes tone and vowel quality with diacritics (for example
`A Nong` → `A Nông`, `Bao Loc` → `Bảo Lộc`); when text passes through an
ASCII-only system those marks are lost. The task is to put them back.

A row gives one folded place name plus named context (its province, district
and five nearest training settlements). The evaluation set is defined by
held-out syllables: every test name contains at least one syllable that occurs
in no training name, so a per-syllable or per-name lookup table cannot reach
it, and every hand-coded table measured scores below a character-level
Transformer trained from scratch. Numbers are in `config.yaml` and `DESIGN.md`.

## Files

- `problem-description.md` — the prompt shown to the solving agent.
- `dataset-description.md` — what the dataset ships, its source, licence and limits.
- `rubrics.md` — task-specific evaluation rubrics.
- `config.yaml` — metadata, dataset block, prepare/grade pointers, the measured ladder.
- `prepare.py` — `prepare(raw_root, public, private)`; deterministic, seedless.
- `grade.py` — `grade(submission, answers) -> float`; changed-position F1.
- `solution.ipynb` — three progressive submissions; the final one trains on GPU.
- `run_checklist.py` — the pre-upload verification checklist (run it; all PASS).
- `DESIGN.md` — the measured design record (why this shape, split, metric; the ladder).
- `dataset/`
  - `raw/` — `VN.txt` (GeoNames Vietnam export), `meta.json`, `ATTRIBUTION.txt`.
  - `generator/build_raw.py` — fetches and pins the raw dump.
  - `public/` — `train.csv`, `test.csv`, `sample_submission.csv`, `dataset_stats.json`, `validate_submission.py` *(produced by `prepare.py`)*.
  - `private/` — `answers.csv`, `split_manifest.json` *(produced by `prepare.py`)*.
- `restoring-vietnamese-diacritics-raw.zip` — the dataset upload: `dataset/raw` + `dataset/generator` only.

## Reproduce

```bash
# 1. fetch the raw dump (or use the pinned dataset/raw/VN.txt already present)
python dataset/generator/build_raw.py

# 2. build the public/private split (deterministic)
python prepare.py

# 3. verify everything
python run_checklist.py
```

## Provenance

Data: GeoNames geographical database, Vietnam export
(`https://www.geonames.org/search.html?q=&country=VN`, file `https://download.geonames.org/export/dump/VN.zip`), licensed **CC BY 4.0**
(`https://creativecommons.org/licenses/by/4.0/`, stated in the GeoNames
`readme.txt`). The `name`, `asciiname`, `alternatenames`, coordinates and admin
codes of populated-place and ADM1/ADM2 records are used. Snapshot pinned 2026-10-10 with a SHA-256 recorded in
`dataset/raw/meta.json`.
