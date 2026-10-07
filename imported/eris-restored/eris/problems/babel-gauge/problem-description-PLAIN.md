# Babel Gauge: Recovering What a Lossy Channel Still Determines

## Overview

Six legacy logging systems each report the same six physical facts about a
piece of equipment, but each writes them through a different undocumented
encoding - and several of those encodings destroy information on the way out.
One system rounds temperature to whole degrees Fahrenheit before writing it.
Another drops the seconds from its clock. A third stores pressure as a whole
number of kPa. The canonical record you must produce keeps the true value at
full precision.

The consequence is that this task has a ceiling below 1.0 that no method can
beat, and the ceiling is not uniform: it differs from format to format and
from field to field. Some fields are exactly determined by the line in front
of you. Others are only constrained by it - narrowed to a set of candidates,
with no way to choose among them beyond what the training distribution tells
you. Two of the encodings also collapse their alarm states, writing a single token
where the canonical record distinguishes WARN from FAULT. Which one it was is
not in the line, but it is not independent of the rest of the line either.

So the task is not "parse these logs". It is: from 60,000 paired examples,
work out which parts of each encoding are invertible and which are lossy,
invert what can be inverted exactly, and for the rest emit the most probable
underlying value given the coarse observation. You must find the boundary
between the two yourself - it is nowhere documented, and it does not fall on
tidy format lines.

No pretrained models, no LLMs, no external data. Any approach is valid
(learned rule systems, grammar induction, neural sequence models, hybrids) as
long as it is derived from the provided data and runs end-to-end in your
notebook.

## Task

Some lines are damaged in transit; all others follow one of the six
undocumented legacy formats. For every raw log line, output one of two things.

Output 1, the canonical record: a single string of exactly six key=value
fields joined by the pipe character, in this fixed order (ts, dev, site,
temp_c, pres_kpa, status). Example:

```
ts=2025-04-30T12:06:09Z|dev=VLV-2004|site=UAE3|temp_c=76.7|pres_kpa=311.8|status=OK
```

Each field must follow its exact format:

- ts: UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form, seconds always present.
- dev: a 3-letter device type (PMP, CMP, FAN or VLV), followed by a hyphen,
  followed by a 4-digit number.
- site: the site code, as listed in sites.csv.
- temp_c: temperature in degrees Celsius, exactly one decimal place (%.1f).
- pres_kpa: pressure in kPa, exactly one decimal place (%.1f).
- status: one of OK, WARN or FAULT.

Output 2, the literal string REJECT: if the line is corrupted, meaning damaged
beyond reliable parsing.

The train split gives you raw_line to canonical pairs; the test split gives
you raw lines only. The legacy formats differ in far more than layout. Expect
differences in clock convention, units, decimal representation, date-field
order, and status vocabulary. None of this is documented, so it must be
learned from the pairs.

Important: not every fact survives every format. Several of these systems log
at a coarser precision than the canonical record requires - whole degrees
rather than tenths, whole psi rather than hundredths, and in some formats no
seconds at all. Those digits are simply not present in the input, so they
cannot be recovered by any means. A perfect solver still does not score 1.0.
Your job is to recover everything that IS determined by the input, and for the
rest to emit the most likely value given what you can observe. Both halves are
learnable from the training pairs.

## Data files

All public files are in ./dataset/public/

train.csv, 60,000 rows. Three columns:

- line_id (str): unique id.
- raw_line (str): the log line as received.
- canonical (str): target, the canonical record, or REJECT for about 4% of rows.

test.csv, 8,000 rows. Two columns, line_id and raw_line. Produce one output
per row. Note that the mix of source systems in the test split differs from
the training split, and test device numbers never appear in training, so
identifiers must be carried over from the input rather than looked up.

sites.csv, 36 rows. Two columns, site and utc_offset_min: the minutes to add
to UTC to get each site's local clock. Some sites use half-hour offsets.

sample_submission.csv. The required output format (line_id, output), filled
with the constant REJECT.

## Metric

Mean per-line field accuracy over the 8,000 test lines.

For each test line, compute line_score(truth, output) where truth is the
canonical answer and output is your prediction:

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

    # len(pieces) is the raw pipe-split count and may be anything from 1
    # upwards. The max() is what guarantees the denominator is at least 6:
    # an output with more than 6 pieces raises the denominator (extra fields
    # dilute the score), while an output with fewer than 6 pieces still
    # divides by 6 (incomplete output is heavily penalized).
    return correct / max(6, len(pieces))
```

Final score: unweighted mean of line_score over all 8,000 test lines.

Range: 0.0 to 1.0, higher is better.

Important: values match by exact string comparison. Respect these formats
exactly:

- Timestamps: YYYY-MM-DDTHH:MM:SSZ format.
- Decimals: always exactly one decimal place (%.1f).
- Device codes: zero-padded 4-digit numbers, for example PMP-0001.
- Status values: uppercase OK, WARN, FAULT.

Reference baselines:

- Copy input line as-is: about 0.00
- Output REJECT for every line: about 0.05
- Handles only the two fully-precise formats: about 0.18
- Handles four of the six formats: about 0.40
- Handles five of the six formats: about 0.53
- Reference solution, at the information-theoretic ceiling: about 0.64

The ceiling is below 1.0 by construction, because the coarsely-logged digits
described above are absent from the input. Do not expect to exceed it.

## Submission format

Write a CSV file named submission.csv to ./working/ with exactly these two
columns:

```
line_id,output
L060001,ts=2025-03-14T18:22:00Z|dev=PMP-8413|site=NE2|temp_c=41.3|pres_kpa=312.9|status=OK
L060002,REJECT
...
```

- Exactly one row for each line_id in test.csv, so 8,000 rows plus a header,
  in any order.
- output is the canonical record string or REJECT. Empty outputs score 0 for
  their line.
- The score is always averaged over all 8,000 test line_ids. A line_id you
  omit scores 0 for that line rather than voiding the submission; duplicated
  ids keep their first row and unknown ids are ignored. Only an unreadable
  file or a missing line_id or output column is unscorable.
- The canonical string contains pipe characters, so write the CSV with
  standard quoting. pandas to_csv handles this.

## Rules & environment

- Learn from the provided data only: no pretrained or foundation models, no
  LLMs, and no external data.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end-to-end, top to bottom: load the public data,
  learn the transduction, then write ./working/submission.csv
- Seed everything; your run should be reproducible.
