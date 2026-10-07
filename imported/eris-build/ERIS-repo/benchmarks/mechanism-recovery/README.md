# What Does It Do In There?

**Domain:** natural-language-processing · **Tags:** text, set-selection, biomedical
**Metric:** chance-corrected set F1 (naming all twelve candidates scores 0.000) · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC0 1.0

## The task

A drug label's Clinical Pharmacology section spends several thousand words
on what the drug does in the body. The FDA separately assigns that drug a
mechanism-of-action class from its own vocabulary: one short phrase naming
the thing those thousands of words describe.

Every mechanism word is masked out of the pharmacology text. You are given
the masked text and twelve candidate classes, and asked which ones the FDA
assigned.

**57% of labels state their own mechanism verbatim**, which is why the text
is masked. The mask vocabulary is built from every class name in the corpus,
never from the row's own class, so which words are masked gives nothing away.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and 16 measured anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 8 rubrics — 4 REQUIRED, 3 RECOMMENDED, 1 UNIVERSAL |
| `DESIGN.md` | The measured design record, including two references that were rejected |
| `prepare.py` | Deterministic raw → public/private split (seed 20260925) |
| `grade.py` | `grade(submission, answers) -> float` |
| `reference_solution.py` | CPU reference: one classifier per class, scored against the row's own pool |
| `solution.ipynb` | Reference notebook, 3 progressive submissions |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                --answers dataset/private/answers.csv    # -> 0.070 (first candidate)
python reference_solution.py --out submission.csv        # a few seconds, CPU
python grade.py submission.csv --answers dataset/private/answers.csv   # -> 0.462
```

## Data

Prescription drug labels from the openFDA Drug Label API. The label is the
FDA's own pharmacologic class assignment, recorded in the same label — no
annotation created for this dataset, no model-generated content. The
unmasked text is never written to disk.

3,091 fetched labels reduce to 505 train and 125 test across 22 classes;
946 were exact duplicates of another label and 468 near-duplicates, because
generic drugs are labelled separately by every manufacturer. The split is by
**shared active ingredient**, so no test drug shares an ingredient with any
training drug.

**CC0 1.0** — openFDA dedicates its data to the public domain. Only the drug
label endpoint is used; device records carry separately licensed
nomenclature and are avoided. openFDA's disclaimer applies: this is not
medical information. See `dataset/raw/ATTRIBUTION.txt`.
