# Spelling Unseen Akkadian Words in Cuneiform Signs

GPU sequence-to-sequence task built on the Oracc corpus. Given a normalized Akkadian word with its lemmatisation, up to four attested spellings of its lemma, its line and its text metadata, predict the sign-by-sign transliteration the scribe wrote. Evaluation words are forms that occur on no word of any training text, of lemmas seen in training, in held-out texts.

## Files

- `config.yaml`: metadata, dataset block, prepare/grade contracts, measured ladder.
- `problem-description.md`: the task as shown to solvers.
- `dataset-description.md`: the raw dataset, i.e. what the dataset upload ships.
- `rubrics.md`: grading rubrics.
- `prepare.py`: `prepare(raw_root, public_dir, private_dir)`; deterministic and seedless.
- `grade.py`: `grade(submission, answers) -> float`; 0.5 × exact sign sequence + 0.5 × sign-level similarity; degrade, never reject.
- `solution.ipynb`: the reference notebook, executed, with outputs. Its final submission is the published reference.
- `DESIGN.md`: measured design record, including the internal review round and every change it caused.
- `run_checklist.py`: pre-upload verification; every item prints PASS/FAIL.
- `ladder.json`: the measured ladder on the shipped split.
- `work/`: `ladder.py` (baselines), `cpu_rules.py` (strongest rule system), `ceiling.py` (train-only ceiling estimate), `reference.py` (the notebook's model code as a script), `build_notebook.py`, `finalize.py`, `doc_stats.py` (every corpus statistic quoted in the docs).
- `dataset/raw/`: `lines.jsonl`, `texts.jsonl`, `meta.json`, `ATTRIBUTION.txt`.
- `dataset/generator/build_raw.py`: fetches the Oracc JSON zips and writes the raw files.
- `dataset/public/` (`train.csv`, `train_docs.csv`, `test.csv`, `sample_submission.csv`, `dataset_stats.json`) and `dataset/private/` (`answers.csv`, `split_manifest.json`): derived by `prepare.py`.
- `cuneiform-spelling-raw.zip`: the dataset upload. It is flat: the four raw files at the root, plus `generator/build_raw.py`.

## Reproduce

```bash
python3 dataset/generator/build_raw.py --out dataset/raw           # fetch + extract (one worker)
python3 prepare.py dataset/raw dataset/public dataset/private       # deterministic split
jupyter nbconvert --to notebook --execute --inplace solution.ipynb  # reference -> working/submission.csv
python3 work/finalize.py                                            # ladder.json, fills reference numbers, stages upload
python3 run_checklist.py                                            # every item must PASS
cd dataset/raw && zip -q ../../cuneiform-spelling-raw.zip lines.jsonl texts.jsonl meta.json ATTRIBUTION.txt && cd .. && zip -qr ../cuneiform-spelling-raw.zip generator
```

## Provenance

Oracc, The Open Richly Annotated Cuneiform Corpus, JSON exports (`https://oracc.museum.upenn.edu/json/`): 77 project zips fetched, 69 volumes contributing texts, CC BY-SA 3.0 (see `dataset/raw/ATTRIBUTION.txt`). The labels are the editors' own transliterations; nothing was annotated or generated.
