# Compiling Amended Regulations — Shipd Upload Cheat-Sheet

Every form field, in platform order, paste-ready. Long fields point to the file
whose FULL content you paste.

> **Paste warning.** Copy from the RAW file, never from a rendered preview.
> Rendered copies drop pipe characters and closing backticks. Nothing in these
> files uses a markdown table, precisely because of that.

---

## 1 · DATASET form

**Name (slug)**
```
recompile-dataset
```

**Title**
```
eCFR Point-in-Time Section Pairs With the Amendments That Produced Them
```

**Short title**
```
eCFR Point-in-Time Amendment Pairs
```

**Description** → paste the FULL content of `dataset/dataset-description.md`

> ⚠️ **Do NOT paste `problem-description.md` here.** That file describes the
> *prepared* split solvers see (`public/`, `items/`, `train.csv`, `test.csv`) and
> belongs in the CHALLENGE form. Pasting it here triggers "Validation Failed —
> mismatch between the described file structure and the actual files present",
> because the raw upload contains only `items.jsonl`, `attribution.csv`,
> `meta.json`.

**Source URL** — use the eCFR API as the primary. That is genuinely where every
label comes from, and the Federal Register bulk feed is shared with other
datasets on the platform.
```
https://www.ecfr.gov/api/versioner/v1
```

**Secondary source (if there is a field for it)**
```
https://www.govinfo.gov/bulkdata/FR
```

**If the platform flags a duplicate source URL**, paste this:
```
The Federal Register feed supplies only the <AMDPAR> instruction strings. Every
label in this dataset -- both the before and the after text -- comes from the eCFR
point-in-time API. The unit of data is a versioned pair: one CFR section captured
on two dates either side of one rule, reconciled against eCFR's own recorded
change dates. A corpus of Federal Register documents contains no such pairs. This
dataset is not citation extraction, not citation linking, and not classification
or summarization over regulatory text.
```

**License**
```
Public domain. The Federal Register and the Code of Federal Regulations are works
of the United States Government and are not subject to copyright in the United
States (17 U.S.C. 105). Commercial use and redistribution are unrestricted.
attribution.csv records the Federal Register document number and citation behind
every record, so any row can be traced to the rule that produced it.
```

**Citation / provenance (if asked)**
```
United States Government. Federal Register, govinfo bulk XML
(https://www.govinfo.gov/bulkdata/FR), and Code of Federal Regulations, eCFR
point-in-time API (https://www.ecfr.gov/api/versioner/v1). Final rules published
2021-01-05 to 2021-03-08.
```

**Raw data upload** → ZIP the three files in `dataset/raw/` first:
```
cd cpu/recompile/dataset/raw && zip -9 ../../recompile-raw.zip items.jsonl attribution.csv meta.json
```
Upload `recompile-raw.zip` (the ZIP, never the bare folder).
Do **not** include `_cache/` — it is the download cache, not data.

**Generation / reproducibility note (if asked)**
```
Deterministic scan, oldest-first, no randomness: dataset/generator/build_raw.py
lists final rules from the Federal Register API for 2021-2025, keeps those with a
stated effective date, a single CFR title and at most three parts, reads the
amendatory instructions and printed replacement text from the govinfo daily FR
XML, and pairs each printed section with eCFR point-in-time text either side of
the change eCFR records for it. Every download is disk-cached with atomic renames,
so a killed run resumes exactly where it stopped.
Run: python build_raw.py <out_dir> <n_rules>
```

**Prepare script** → attach `prepare.py` **to the dataset, not the challenge**,
then press **Rebuild** and confirm the Prepared Dataset shows `items/`,
`train.csv`, `test.csv`, `sample_submission.csv`, `validate_submission.py`,
`dataset_stats.json`.

> ⚠️ **Check this.** If the Prepared Dataset still lists `items.jsonl`, the prepare
> step did not run and you would be shipping the answer key — `items.jsonl`
> contains the `after` field.

Deterministic, seed 20260810; verified byte-identical across two runs.

---

## 2 · CHALLENGE form

**Name (slug)**
```
recompile
```

**Title**
```
Compiling Amended Regulations
```

**Domain / category**
```
natural-language-processing
```

**Tags** — select exactly these three: `text`, `feature-engineering`,
`information-extraction`. Do NOT tag `classification`/`regression`/`ranking` — the
output is a document, not any of those, and the reviewer rubric penalises
mislabelled task types. Skip `large-scale`/`small-data`.

**Problem description** → paste the FULL content of `problem-description.md`

**Grading script** → upload `grade.py`
(usage: `python grade.py submission.csv answers.csv`; returns weighted
edit-operation F1 in [0,1]. Structural problems **degrade rather than abort**:
missing, blank and duplicated rows are scored as "no amendment applied" (0 for
that item) and rows beyond the answer key are ignored, so public/private portion
scoring works. InvalidSubmission is raised only when nothing is scorable at all —
the file is not readable as a CSV carrying `item_id` and `after_text`, or it shares
no id with the answer key.)

**Metric name (if asked)**
```
Weighted edit-operation F1
```

**Metric description (if asked)**
```
Token-level edit scripts are computed with difflib from the shared before text to
the official after text and to the submitted text. Score per item is F1 over those
operations, each weighted by the number of tokens it moves. Operations are aligned
by position and compared by content rather than intersected exactly, so an
amendment applied to the right paragraph with the wrong extent earns partial
credit: half of an operation's credit is the span IoU, half the token F1 of the
replacement text, and operations aimed at different places never match. Averaged
over test items. Returning the section unchanged scores 0.000; returning the
official text scores 1.000. A plain text-similarity metric would not work here: an
amendment moves a few dozen words in a section of several hundred, so doing nothing
scores about 0.97.
```

**Expected score range / difficulty (if asked)**
```
Baselines measured on the shipped split: section unchanged 0.000, printed block
appended 0.155, printed block only 0.189, regex splice
0.192, hierarchy splice 0.400. Reference solution
0.522 on the private test set.
Oracle 1.000.

Five faithful partial implementations -- the shapes a solver actually lands on
when it gets part of the problem right -- were measured on the private test set
and span the intended band without colliding:
  hierarchy splice, no "* * *" handling        0.400
  flat designators + no word removals          0.419
  flat designators (no sentence-boundary test) 0.445
  full hierarchy, no word removals             0.494
  reference                                    0.522
A solver that never builds a hierarchy scores 0.192 and falls out of the band from
below, which is the intended penalty for skipping the central difficulty.
PENDING - agent gate not yet run
```

---

## 3 · RUBRICS form (9 entries — text in `rubrics.md`)

Enter each with its Type and Importance:

1. Builds the paragraph hierarchy instead of matching designators flatly — DATA_HANDLING — REQUIRED
2. Separates paragraph openings from cross-references — FEATURE_ENGINEERING — REQUIRED
3. Treats `* * *` as "the rest is unchanged" — MODELING — RECOMMENDED
4. Applies a run of instructions in order, including redesignations — MODELING — RECOMMENDED
5. Validates on a part-level split — TRAINING — REQUIRED
6. Leaves untouched text untouched — CODE_QUALITY — REQUIRED
7. Beats the paragraph-splice baseline — TRAINING — RECOMMENDED
8. Checks the data for shortcuts instead of assuming it is clean — AGENT_BEHAVIOR — RECOMMENDED
9. Produces a valid submission inside the compute budget — CODE_QUALITY — REQUIRED

For each entry, paste the corresponding section body from `rubrics.md` (criterion
paragraph plus its *Fails if* line).

---

## 4 · SOLUTION form

**Notebook** → upload `solution.ipynb` (already executed; outputs saved).

**Summary (if asked)**
```
Three stages, each measured on held-out items. Stage 1: find the addressed
paragraph by regular expression and splice the agency's printed block over it,
stopping at the next designator of any kind. Stage 2: build a real paragraph
hierarchy -- levels inferred from the sequence a designator continues, since "(i)"
after "(h)" is a first-level letter while "(i)" after "(1)" opens a third level --
and treat a designator as a paragraph opening only when the preceding token ends a
sentence, which is what separates an opening from the cross-references
regulations make to their own paragraphs constantly. Stage 3: handle partial
replacements marked "* * *" (the remainder survives, so the printed fragment
displaces only as many sentences as it contains), quoted removals that print no
replacement block, and runs of instructions applied in order so redesignations
move later targets. No training; pure Python and pandas.
```

**Runtime note (if asked)**
```
Under 3 minutes wall, CPU-only, well inside the 1.5 h budget. Reads
./dataset/public/, writes ./working/submission.csv. Standard Kaggle-image
libraries only (pandas plus the standard library).
```

---

## Pre-submit checklist

- [ ] `recompile-raw.zip` built from `dataset/raw/` — three files, no `_cache/`
- [ ] `prepare.py` attached to the DATASET, Rebuild pressed, prepared listing checked
- [ ] Prepared Dataset does NOT show `items.jsonl` (that file holds the answers)
- [ ] `problem-description.md` pasted complete, from the raw file
- [ ] all 9 rubrics entered with correct Type and Importance
- [ ] `solution.ipynb` uploaded with outputs visible
