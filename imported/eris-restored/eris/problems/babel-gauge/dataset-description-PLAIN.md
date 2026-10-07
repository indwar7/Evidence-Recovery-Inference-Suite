# Dataset Description

## Overview

68,000 equipment-telemetry log lines as emitted by six legacy logging systems
with undocumented, mutually incompatible encodings, each paired with the
full-precision record of the six facts it was carrying. Four of the six
encodings are lossy: they round or truncate before writing, so the paired
record cannot be fully reconstructed from the line. The dataset is built to
study transduction under partial identifiability - learning which parts of an
unknown encoding are invertible, and calibrating the rest against the
training distribution.

What the data represents: a fleet-monitoring vendor ingests hourly equipment
telemetry (temperature, pressure, status) from customers running six
generations of logging firmware. Every system reports the same six facts -
timestamp, device, site, temperature, pressure, status - but each renders them
differently: different field orders and separators, site-local vs UTC clocks,
deg F vs deg C, psi vs kPa, implied decimal points in a fixed-width mainframe
format, 12-hour vs 24-hour time, opposite slash-date orders, and per-system
status vocabularies. The ingestion pipeline also occasionally damages lines in
transit; damaged lines must be flagged rather than half-parsed.

The vendor's target is one canonical record string per line:

    ts=2025-04-30T12:06:09Z|dev=VLV-2004|site=UAE3|temp_c=76.7|pres_kpa=311.8|status=OK

or the literal string REJECT for damaged lines.

Data source: fully synthetic. There is no external source dataset. Every line
is produced by mechanistic renderers in generate_dataset.py (numpy RNG draws
plus deterministic string templates). No LLM output is used anywhere in the
data, and no Kaggle competition or external dataset was copied or used.

Reproducibility: the generator is fully deterministic (global seed 20260707;
every line is drawn with a child RNG seeded by its own index, so output does
not depend on iteration order). Running generate_dataset.py twice produces
byte-identical CSVs. It requires only numpy and pandas (Kaggle Python Docker
image). The generator is provided with the task submission for reproducibility
review; it is not part of this dataset upload.

License: CC0 1.0 (public domain dedication). Synthetic data created for this
task; commercial use permitted.

## File Structure

This dataset upload contains exactly five files at the top level:

- train.csv - 60,000 labeled rows: raw log line paired with its canonical record.
- test.csv - 8,000 unlabeled rows: raw log lines only, targets withheld.
- sites.csv - 36 rows: reference table of per-site UTC offsets.
- sample_submission.csv - 8,000 rows: the required output format, filled with REJECT.
- answers.csv - 8,000 rows: the private grading key for the test lines.

The prepare.py script submitted with the task publishes train.csv, test.csv
(line_id and raw_line only), sites.csv and sample_submission.csv to solvers,
and routes answers.csv to the private grading split. Solvers never see
answers.csv.

## Features

### train.csv (60,000 rows)

| Column | Type | Description |
|---|---|---|
| line_id | string | Unique id, from L000001 to L060000. |
| raw_line | string | The log line as received: one of six legacy formats, or a corrupted line. |
| canonical | string | Target. The canonical record described in Overview, or REJECT for corrupted lines (about 4%). |

### test.csv (8,000 rows)

| Column | Type | Description |
|---|---|---|
| line_id | string | Unique id, from L060001 to L068000. |
| raw_line | string | The log line as received. Format mix differs from train, and device numbers are disjoint from train. |

### sites.csv (36 rows)

| Column | Type | Description |
|---|---|---|
| site | string | Site code appearing in the log lines, for example NE2 or IND7. |
| utc_offset_min | int | Minutes to add to UTC to get that site's local clock. Range -600 to +720, including half-hour offsets such as +330 and -210. |

### sample_submission.csv (8,000 rows)

| Column | Type | Description |
|---|---|---|
| line_id | string | Same line_ids as test.csv. |
| output | string | Format template, filled with the constant REJECT. |

### answers.csv (8,000 rows, private grading key)

| Column | Type | Description |
|---|---|---|
| line_id | string | Same line_ids as test.csv. |
| canonical | string | The canonical record for each test line, or REJECT. Routed to the private split by prepare.py and never published to solvers. |

## Generation Process

Full details and exact constants are in generate_dataset.py. Summary:

1. Ground truth first. Each line draws a canonical record: timestamp
   (2025-02-01 through 2025-05-31 UTC), device type PMP, CMP, FAN or VLV plus
   a 4-digit number, site (36 codes), temperature (-5 to 95 deg C, 1 decimal),
   pressure (80 to 900 kPa, 1 decimal), status (OK 76%, WARN 16%, FAULT 8%).
2. Rendering. The record is rendered through one of six format renderers,
   documented format by format in the Format Reference section below.
3. Corruption. About 4% of train and 5% of test lines suffer ingestion damage:
   2 to 4 structural characters plus 1 to 4 arbitrary characters are deleted,
   and one surviving character is garbled. Damage always breaks the line's
   grammar (verified: zero corrupted lines still parse), so REJECT is exactly
   decidable.
4. Split design. Train and test are drawn with disjoint device-number ranges
   (0001-6999 vs 7000-9999), so identifiers must be copied from the input and
   never memorized. The rare format F6 is deliberately scarce in training
   (1.5%) but common in test (20%).

Achievable ceiling (verified by construction and by measurement): because the
coarsely-logged digits are absent from the input, no solver can reconstruct
them. Predicting the most likely underlying value for each coarse reading,
learned from the training pairs, yields the best attainable score. Measured
end to end, that ceiling is 0.6401; the bundled reference solution reaches
0.6401. Corrupted lines remain exactly decidable: zero corrupted lines still
parse under any format grammar, so REJECT is never a guess.

## Format Reference

Each of the six renderers is listed below with the conventions it applies. All
six carry the same six facts; they differ in layout, clock, units, timestamp
encoding and status vocabulary.

| Format | Style | Clock | Temperature logged as | Pressure logged as | Timestamp encoding | Seconds | Status vocabulary |
|---|---|---|---|---|---|---|---|
| F1 | Gateway text, pipe-delimited | site-local | whole degrees F | whole psi | YYYY-MM-DD HH:MM:SS | yes | OK / WARN / FAULT |
| F2 | Bus dump, angle-bracketed | UTC | deg C, 1 decimal | kPa, 1 decimal | YYYYMMDDHHMMSS packed | yes | 0 / 1 / 2 |
| F3 | Shuffled key=value | site-local | whole degrees C | whole kPa | DD/MM/YYYY HH:MM | no | OK / WARN / FAULT |
| F4 | Fixed-width mainframe | site-local | whole degrees C, signed | whole kPa | YYMMDDHHMM packed | no | O / A (WARN and FAULT collapsed) |
| F5 | US regional CSV | site-local | whole degrees F | whole psi | MM/DD/YY hh:mmAM/PM | no | RUN / ALM (WARN and FAULT collapsed) |
| F6 | JSON collector | UTC | deg C, 1 decimal | kPa, 1 decimal | Unix epoch seconds | yes | ok / warn / fault |

Only F2 and F6 carry every fact at full canonical precision. F1, F3, F4 and F5
round their readings, and F3, F4 and F5 transmit no seconds at all, so the
canonical value behind those fields is not derivable from the line.

Notes on the conventions:

- Site-local formats (F1, F3, F4, F5) require subtracting the site's
  utc_offset_min from sites.csv to recover UTC, which can roll the date across
  midnight. F2 and F6 are already UTC.
- F1 and F5 print whole psi and whole degrees F. One whole psi spans about
  6.9 kPa and one whole degree F spans about 0.56 deg C, so many canonical
  values map to the same rendered reading and the original cannot be
  recovered.
- F4 stores temperature and pressure as whole numbers, temperature with an
  explicit sign character, so -007 means about -7 deg C and +077 means about
  77 deg C. The tenths digit is lost.
- F4 is genuinely fixed-width: the site field is space-padded, so a 3-character
  site code such as NE7 is followed by two spaces while a 4-character code such
  as IND2 is followed by one. Splitting on a single space is not sufficient.
- F3 renders day-first (DD/MM/YYYY) while F5 renders month-first (MM/DD/YY),
  so date order cannot be assumed and must be inferred from evidence.
- F5 uses a 12-hour clock where 12AM maps to hour 00 and 12PM maps to hour 12.
- F3, F4 and F5 carry no seconds. The canonical record keeps the true second,
  so the seconds digit of ts is unrecoverable for those formats.

Worked example for each format, taken verbatim from the dataset:

| Format | Raw line | Canonical record |
|---|---|---|
| F1 | 2025-04-30 16:06:09 \| VLV-2004 @ UAE3 \| T=170F P=45psi \| OK | ts=2025-04-30T12:06:09Z\|dev=VLV-2004\|site=UAE3\|temp_c=76.7\|pres_kpa=311.8\|status=OK |
| F2 | <C4664\|ATH2\|20250412035659\|t:59.8\|p:305.8\|s:1> | ts=2025-04-12T03:56:59Z\|dev=CMP-4664\|site=ATH2\|temp_c=59.8\|pres_kpa=305.8\|status=WARN |
| F3 | time=22/03/2025 17:14 site=IND7 flag=WARN temp=44C pressure=156kPa dev=CMP1279 | ts=2025-03-22T11:44:40Z\|dev=CMP-1279\|site=IND7\|temp_c=44.1\|pres_kpa=156.2\|status=WARN |
| F4 | V3965IND2 2502230054+07800180O | ts=2025-02-22T19:24:44Z\|dev=VLV-3965\|site=IND2\|temp_c=77.7\|pres_kpa=180.5\|status=OK |
| F5 | SIN2,VLV-4675,02/08/25 07:07PM,173,94,ALM | ts=2025-02-08T11:07:39Z\|dev=VLV-4675\|site=SIN2\|temp_c=78.1\|pres_kpa=651.4\|status=FAULT |
| F6 | {"st":"IRL2","d":"CMP-5862","e":1747296976,"tc":46.6,"pk":325.0,"x":"ok"} | ts=2025-05-15T08:16:16Z\|dev=CMP-5862\|site=IRL2\|temp_c=46.6\|pres_kpa=325.0\|status=OK |

A corrupted line, which maps to REJECT, looks like this:

    f*agO time=20/04/2025 1115 pressure=739kPa temp=21C devCMP1097 site=DEN3

## Controlled Complexity

| Property | Where | Skill it tests |
|---|---|---|
| Six undocumented grammars | raw_line | inducing structure and rules from paired examples |
| Site-local vs UTC clocks, offsets -600 to +720 min including half-hours | F1/F3/F4/F5 plus sites.csv | learned time normalization including date rollover across midnight |
| deg F to deg C and psi to kPa with exact re-rounding | F1/F5 | discovering numeric transforms from pairs instead of assuming identity |
| Implied 0.1x decimals in fixed-width integers | F4 | fixed-width legacy conventions |
| DD/MM/YYYY (F3) vs MM/DD/YY (F5) plus 12-hour clock where 12AM maps to 00 | F3/F5 | resolving date ambiguity from evidence, not convention |
| Four status vocabularies, two of which collapse WARN and FAULT | F2/F4/F5/F6 | vocabulary alignment, and inferring a state the line does not carry |
| About 4-5% corrupted lines mapping to REJECT | both splits | strict grammar validation vs best-guess parsing |
| F6 is 1.5% of train but 7.9% of test | split design | not neglecting the low-resource format |
| Test device numbers disjoint from train | both splits | copying identifiers instead of memorizing them |

## Data Splits Created by prepare.py

The task submission includes a prepare.py script that, when run on this
dataset, creates the public and private files for solvers:

- Public train.csv (60,000 rows): line_id, raw_line, canonical.
- Public test.csv (8,000 rows): line_id, raw_line, with canonical withheld.
- Public sites.csv: copy of the uploaded file.
- Public sample_submission.csv: required format (line_id, output), filled with
  the constant REJECT.
- Private answers.csv (8,000 rows): line_id, canonical for the test lines,
  routed to the private grading split and never shown to solvers.

## Sample-Size Summary

| Piece | Count |
|---|---|
| Lines | 68,000 total: 60,000 train and 8,000 test |
| Train format mix | F1 28%, F2 22%, F3 20%, F4 18%, F5 10.5%, F6 1.5% |
| Test format mix | F3 24.6%, F4 22.8%, F5 21.3%, F1 18.0%, F6 7.9%, F2 5.4% |
| Corrupted lines mapping to REJECT | 3.9% of train, 5.0% of test |
| Sites | 36 |
| Device types | 4 |
