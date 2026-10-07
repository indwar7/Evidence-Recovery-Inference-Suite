# Recovering OCR Batch Origin from Character Noise

**Domain:** natural-language-processing · **Tags:** text, clustering, unsupervised learning
**Metric:** mean per-bag Adjusted Rand Index · **Runtime:** CPU only, <1.5h
**Licence:** Public Domain Mark

## The task

Historical newspapers were digitized in batches, each by a real scanning pipeline with
its own OCR engine, settings and systematic error signature. You are handed a bag of
unlabelled OCR snippets drawn from 2–4 different batches and asked to group them by the
pipeline that produced them.

Every snippet in a bag is a **different article**, so topic and authorship are
deliberately confounded away. The only evidence is each pipeline's character-error
fingerprint.

This is not OCR accuracy benchmarking — there is no corrected transcript to score
against. It is not stylometry — the thing being separated is a machine, not a writer.

Both degenerate submissions score exactly `0.000`. The reference scores `0.342`. The
strongest adversarial attack — pooled global clustering across the whole test corpus,
banned by rule — scores `0.094`.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and all fourteen measured baseline anchors |
| `DESIGN.md` | The full construction writeup, including three review rounds and nine fixed defects |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, disclosed confounds |
| `rubrics.md` | The rules a submission must satisfy, including the banned pooled-clustering attack |
| `prepare.py` | Deterministic raw → public/private split (seed 20260812), batch-disjoint |
| `grade.py` | `grade(submission, answers) -> float`; 21 unit tests |
| `reference_solution.py` | Reference — RandomForest-probability embedding + per-bag agglomerative |
| `solution.ipynb` | The same approach as a notebook |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Submission format

**One row per bag**, not per snippet. The `batch_id` cell holds one label per snippet,
whitespace-separated, in `test.csv` snippet order. Labels are arbitrary tokens — they
never need to match across bags or name a real batch, because only co-membership within
a bag is scored.

## Reproduce

```bash
python grade.py dataset/public/sample_submission.csv \
                dataset/private/answers.csv              # -> 0.0000 (floor)

python reference_solution.py submission.csv              # ~15 s, CPU
python grade.py submission.csv dataset/private/answers.csv   # -> ~0.33
```

## Data

7,881 distinct snippets of raw, uncorrected OCR from 35 real Library of Congress NDNP
scanning batches — ALTO XML `<String CONTENT="...">` values, nothing cleaned or
paraphrased. Real per-batch vendor metadata (`softwareCreator`, `softwareName`,
`processingAgency`) is read directly from each batch's own ALTO XML.

Split is **batch-disjoint and open-set**: the 16 test batches never appear among the 19
train batches. Batch identities are published nowhere — labels are opaque aliases — and
every shipped snippet is a deterministic 80–97% character window of its source region, so
no test row is byte-identical to any archive region.

**Public Domain Mark.** The Library of Congress states that newspapers in Chronicling
America are in the public domain or have no known copyright restrictions. Issue dates
span roughly 1831–1960; per the Library's own advisory, snippets dated within the last 95
years may contain copyrighted third-party material. See `dataset/raw/ATTRIBUTION.txt`.

## A note on honesty

`DESIGN.md` records what went wrong during construction as carefully as what went right:
a metric that let "everyone in one group" beat a trained model, a harness scoring bug
that understated every result, a reviewer who scored `1.000` by looking the answers up in
the source archive, and a length confound that could not be removed and was therefore
disclosed to solvers instead. Read it before trusting any number on this page.
