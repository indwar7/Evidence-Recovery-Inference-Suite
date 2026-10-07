# Legacy Telemetry Log Canonicalization

## Overview

A fleet-monitoring vendor ingests equipment telemetry from six legacy logging systems, each with its own undocumented line format. Your task is to learn the string-to-string transduction from 60,000 paired examples (raw log lines → canonical records) and apply it to 8,000 unlabeled test lines. This is a **sequence-to-sequence learning task** — no pretrained models, no LLMs, and no external data allowed. Any approach is valid (learned rule systems, grammar induction, neural sequence models, hybrids) as long as it is derived from the provided data and runs end-to-end in your notebook.

## Task

Some lines are damaged in transit; all others follow one of the six
undocumented legacy formats. For every raw log line, output one of two things.

**Output 1 — the canonical record:** a single string of exactly six
`key=value` fields joined by the pipe character, in this fixed order
(`ts`, `dev`, `site`, `temp_c`, `pres_kpa`, `status`). Example:

```
ts=2025-04-30T12:06:09Z|dev=VLV-2004|site=UAE3|temp_c=76.7|pres_kpa=311.8|status=OK
```

Each field must follow its exact format:

- **`ts`** — UTC timestamp in `YYYY-MM-DDTHH:MM:SSZ` form (seconds always present).
- **`dev`** — a 3-letter device type (`PMP`, `CMP`, `FAN`, or `VLV`), followed by a hyphen, followed by a 4-digit number.
- **`site`** — the site code, as listed in `sites.csv`.
- **`temp_c`** — temperature in °C, exactly one decimal place (`%.1f`).
- **`pres_kpa`** — pressure in kPa, exactly one decimal place (`%.1f`).
- **`status`** — one of `OK`, `WARN`, or `FAULT`.

**Output 2 — the literal string `REJECT`:** if the line is corrupted
(damaged beyond reliable parsing).

The train split gives you `raw_line → canonical` pairs; the test split
gives you raw lines only. **Every uncorrupted line contains all six facts,
and they are exactly recoverable** — the mapping is deterministic, with no
irreducible ambiguity. The legacy formats differ in far more than layout:
expect differences in clock convention, units, decimal representation,
date-field order, and status vocabulary. None of this is documented —
it must be learned from the pairs.

## Data files

All public files are in `./dataset/public/`.

### `train.csv` — 60,000 rows

| Column | Type | Role |
|---|---|---|
| `line_id` | str | Unique id. |
| `raw_line` | str | The log line as received. |
| `canonical` | str | Target: canonical record, or `REJECT` (~4% of rows). |

### `test.csv` — 8,000 rows

`line_id`, `raw_line` — produce one output per row. Note: the mix of
source systems in the test split **differs from the training split**, and
test device numbers never appear in training (identifiers must be carried
over from the input, not looked up).

### `sites.csv` — 36 rows

`site`, `utc_offset_min` — minutes to **add to UTC** to get each site's
local clock (some sites use half-hour offsets).

### `sample_submission.csv`

The required output format (`line_id, output`), filled with the constant
`REJECT`.

## Metric

**Mean per-line field accuracy** over the 8,000 test lines.

For each test line, compute `line_score(truth, output)` where `truth` is the canonical answer and `output` is your prediction:

```python
FIELDS = ["ts", "dev", "site", "temp_c", "pres_kpa", "status"]

def line_score(truth: str, output: str) -> float:
    output = output.strip()

    # If truth is REJECT, score 1.0 iff output is exactly "REJECT"
    if truth == "REJECT":
        return 1.0 if output == "REJECT" else 0.0

    # If truth is canonical but output is empty or REJECT, score 0.0
    if not output or output == "REJECT":
        return 0.0

    # Parse output as key=value pairs separated by the pipe character
    pieces = output.split("|")
    got = {}
    for piece in pieces:
        if "=" in piece:
            k, v = piece.split("=", 1)
            got.setdefault(k, v)

    # Parse truth (always valid canonical format)
    tru = dict(p.split("=", 1) for p in truth.split("|"))

    # Count matching fields (exact string match required)
    correct = sum(1 for k in FIELDS if got.get(k) == tru[k])

    # len(pieces) is the raw pipe-split count and may be anything from 1 upwards.
    # The max() is what guarantees the denominator is at least 6: an output with
    # more than 6 pieces raises the denominator (extra fields dilute the score),
    # while an output with fewer than 6 pieces still divides by 6 (incomplete
    # output is heavily penalized).
    return correct / max(6, len(pieces))
```

**Final score:** unweighted mean of `line_score` over all 8,000 test lines.

**Range:** [0.0, 1.0], higher is better.

**Important:** Values match by **exact string comparison**. Respect these formats exactly:
- Timestamps: `YYYY-MM-DDTHH:MM:SSZ` format
- Decimals: always exactly one decimal place (`%.1f`)
- Device codes: zero-padded 4-digit numbers (e.g., `PMP-0001`)
- Status values: uppercase (`OK`, `WARN`, `FAULT`)

**Reference baselines:**
- Copy input line as-is: ≈ 0.00
- Output `REJECT` for every line: ≈ 0.05
- Format-aware parser ignoring hidden conventions: ≈ 0.76
- Fully correct transducer (learns all quirks): 1.0

## Submission format

Write a CSV file named **`submission.csv`** to `./working/` with exactly
these two columns:

```
line_id,output
L060001,ts=2025-03-14T18:22:00Z|dev=PMP-8413|site=NE2|temp_c=41.3|pres_kpa=312.9|status=OK
L060002,REJECT
...
```

- Exactly **one row for each `line_id` in `test.csv`** (8,000 rows +
  header), in any order.
- `output` is the canonical record string or `REJECT`. Empty outputs score
  0 for their line.
- The score is always averaged over all 8,000 test `line_id`s. A `line_id`
  you omit scores 0 for that line rather than voiding the submission;
  duplicated ids keep their first row and unknown ids are ignored. Only an
  unreadable file or a missing `line_id`/`output` column is unscorable.
- The canonical string contains pipe characters — write the CSV with
  standard quoting (pandas `to_csv` handles this).

## Rules & environment

- **Learn from the provided data only** — no pretrained or foundation
  models, no LLMs, and no external data.
- Use only libraries available in the **Kaggle Python Docker image**
  (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, …).
- **No LLM-generated outputs** may be used anywhere in your solution.
- Your notebook must run **end-to-end, top to bottom**: load the public
  data → learn the transduction → write `./working/submission.csv`.
- Seed everything; your run should be reproducible.
