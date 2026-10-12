# Rust Toolchain Local Variable Names

## Files

- `problem-description.md` — the prompt shown to the solving agent.
- `dataset-description.md` — what the dataset ships, its source, licence and limits.
- `rubrics.md` — task-specific evaluation rubrics.
- `config.yaml` — metadata, dataset block, prepare/grade pointers, the measured ladder.
- `prepare.py` — `prepare(raw_root, public, private)`; deterministic, seedless.
- `grade.py` — `grade(submission, answers) -> float`; per-row position accuracy.
- `solution.ipynb` — progressive submissions; the final one trains on GPU.
- `run_checklist.py` — the pre-upload verification checklist (run it; all PASS).
- `DESIGN.md` — the measured design record (why this shape, split, metric; the ladder).
- `dataset/`
  - `raw/` — `library/{core,alloc,std,coretests,alloctests}`, `compiler`, `src/tools/rust-analyzer/crates` (pinned `rust-lang/rust` snapshot), `LICENSE-MIT`, `LICENSE-APACHE`, `ATTRIBUTION.txt`, `meta.json`.
  - `generator/build_raw.py` — fetches and pins the raw snapshot.
  - `generator/validate_submission.py` — standalone submission format checker.
  - `public/` — `train.csv`, `test.csv`, `sample_submission.csv`, `dataset_stats.json` *(produced by `prepare.py`)*.
  - `private/` — `answers.csv` *(produced by `prepare.py`)*.

## Reproduce

```bash
python3 dataset/generator/build_raw.py --out dataset/raw
python3 prepare.py --raw dataset/raw --public dataset/public --private dataset/private
python3 grade.py   # runs the grader's own unit tests
```

## Provenance

Source: the Rust project (`rust-lang/rust`) — the standard library
(`library/{core,alloc,std,coretests,alloctests}`), the rustc compiler
(`compiler/`) and rust-analyzer (`src/tools/rust-analyzer/crates/`), all
pinned at commit `76c90957b7e422c4b9c45192b0197214d7de5a54` (2026-10-09).
Dual-licensed MIT / Apache-2.0 by the Rust Project Developers; see
`dataset/raw/ATTRIBUTION.txt`.
