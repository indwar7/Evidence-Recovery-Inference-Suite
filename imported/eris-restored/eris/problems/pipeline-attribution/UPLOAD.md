# Digitization Pipeline Attribution — Shipd Upload Cheat-Sheet

Every form field, in platform order, paste-ready. Long fields point to the
file whose FULL content you paste. Copy from the RAW file, never a rendered
preview — rendered copies drop pipe characters and closing backticks. Nothing
here uses a markdown table for exactly that reason.

---

## 1 · DATASET form

**Name (slug)**
```
pipeline-attribution-dataset
```

**Title**
```
Real NDNP Newspaper OCR Text With Its Own Digitization Batch Identity
```

**Short title**
```
NDNP OCR Batch-Fingerprint Corpus
```

**Description** → paste the FULL content of `dataset/dataset-description.md`

> Do NOT paste `problem-description.md` here — that describes the prepared
> split solvers see, and the raw upload only contains `raw/` + `generator/`.
> Pasting the wrong one triggers "Validation Failed — mismatch between the
> described file structure and the actual files present."

**Source URL**
```
https://chroniclingamerica.loc.gov/data/batches/
```

**License** → paste the `license:` block from `config.yaml`'s `dataset:`
section (the honest NDNP rights-clearance caveat — do not shorten it to a
bare "public domain" claim).

**Raw file upload** — a FLAT zip (the platform's validator cannot see inside
a nested top-level folder) containing exactly:
```
raw/snippets.jsonl
raw/batches_meta.csv
raw/meta.json
raw/skip_log.csv
raw/ATTRIBUTION.txt
generator/build_raw.py
generator/README.md
```
Build it from the `dataset/` directory:
```
cd dataset && zip -r ../pipeline-attribution-raw.zip raw generator -x "*/__pycache__/*"
```

**IMPORTANT — also include the prepared split in the SAME zip.** This
project's own history (repeat-harvest) hit "Grade Sample Validation" failures
twice because the grading job reads `dataset/public/sample_submission.csv`
**directly from the extracted dataset zip** — it does not reliably run
`prepare.py` first. Run `python3 prepare.py` locally, then zip `public/` and
`private/` in alongside `raw/` and `generator/`:
```
cd dataset && zip -r ../pipeline-attribution-raw.zip raw generator public private -x "*/__pycache__/*"
```
This was verified end-to-end from an extracted copy of this exact zip before
upload (see DESIGN.md pre-flight checklist).

---

## 2 · CHALLENGE form

**Name (slug)**
```
pipeline-attribution
```

**Title**
```
Digitization Pipeline Attribution
```

**Domain**
```
natural-language-processing
```

**Problem description** → paste the FULL content of `problem-description.md`

**Config** → paste the FULL content of `config.yaml`

**Grading script** → upload/paste `grade.py`

**Rubrics** → paste the FULL content of `rubrics.md` (5 rubrics: 3 REQUIRED,
3 RECOMMENDED, 1 UNIVERSAL)

**Reference solution** → upload `solution.ipynb` (already executed, 3
submissions of increasing sophistication, final cell scores 0.316)

**Environment**
```
kaggle-python-docker-image, 1x NVIDIA A10G, 10 CPU cores, 62 GB RAM
```

---

## 3 · Pre-upload local verification (already done, keep this record)

- [x] `prepare.py` byte-identical across two runs (md5-verified)
- [x] `work/test_grade.py` — 15/15 structural edge cases pass
- [x] Full baseline + adversarial ladder measured, reference (0.316) on top
      of every non-oracle baseline including both adversarial-clustering
      variants (0.072 realistic / 0.139 oracle)
- [x] No answer leak: `batch_id` absent from `test.csv` and
      `sample_submission.csv`; train/test batches formally disjoint
- [x] Platform entrypoint replayed from an extracted copy of the dataset
      zip — both grading-without-prepare and prepare-then-grade paths work,
      byte-identical to the shipped files
- [x] Reference solution runtime: ~10s locally (CPU), comfortably inside the
      1.5h A10G budget
- [x] Notebook executed end-to-end, every cell `ast.parse`-clean before
      being embedded
- [ ] Leaderboard-slice representativeness measured (10 contiguous slices of
      20 bags: range 0.262–0.359 around a 0.316 full-set score) — moderate
      spread, not tightened further; documented as likely sampling noise
      from small per-slice bag counts rather than a structural ordering
      artifact, since bags are independently randomly generated with no
      sequential clumping mechanism. Revisit if a reviewer flags it.
- [ ] Novelty score — untested on the platform (stochastic; see
      REJECTION-REGISTER.md section 2.1 for the framing levers already
      applied in DESIGN.md's "What is new here" / "Relation to prior work"
      sections)
- [ ] Agent-difficulty gate — can only be run on Shipd once uploaded

See `DESIGN.md` for the full build history including two real bugs found and
fixed (a metric that rewarded a degenerate baseline, and a scoring-harness
key collision that understated every trained baseline), and
`REJECTION-REGISTER.md` (project root) for the general pre-flight checklist
this challenge was built against.
