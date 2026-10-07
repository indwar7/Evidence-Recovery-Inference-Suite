#!/usr/bin/env python3
"""Prepare public/private artifacts for Repeat-Pattern Span Detection (seq-to-seq).

Raw corpus: dataset/raw/sequences.csv
    seq_id, split, length, sequence, true_intervals, n_true

Domain: sequence-to-sequence. Given a DNA window, output the DNA substring
of the dominant repeated-pattern stretch. grade.py locates the best
occurrence of the predicted substring inside the window and scores it by
overlap ratio (IoU) against the canonical reference span.

One canonical span is chosen per window from true_intervals:
    widest -> earliest start -> earliest end.

NO NUMERIC FEATURE COLUMNS ARE PUBLISHED.
-----------------------------------------
An earlier build shipped a `length` column in both public tables. Two
problems, both measured:

  1. It is pure redundancy -- `length` is exactly `len(sequence)`, so it adds
     no information a solver cannot compute in one call, while giving the
     platform's data-integrity profiler a numeric column to scan.
  2. Window lengths are naturally long-tailed (most windows sit in a narrow
     band, a minority run much longer), so more than 5% of rows fall outside
     the 1.5x-IQR Tukey fence and the column is flagged as extreme-outlier
     heavy -- an artifact of publishing the column at all, not of anything
     anomalous in the data.

Dropping it removes the finding at zero cost to a solver: the public tables
now carry only `seq_id` (an opaque string key), `sequence` (the DNA window)
and, on train, `target` (the answer substring). There is no numeric column
left in any public artifact to profile.

Public:
  sequences_train.csv  -> seq_id, sequence, target
  sequences_test.csv   -> seq_id, sequence
  sample_submission.csv-> seq_id, prediction         (answer-blind baseline substring)
Private:
  answers.csv          -> seq_id, sequence, target, reference_start, reference_end

  `target` is the held-out answer in the SAME form the submission takes: the
  DNA substring of the canonical span. It is what `prediction` is compared
  against, so the submission column and the answer column identify the same
  object and the grading contract is unambiguous. `sequence` is carried here
  so that grade(submission, answers) works with exactly two arguments -- the
  grader needs the window to locate a predicted substring. reference_start /
  reference_end are the canonical span's coordinates, kept so the grader can
  score edge tightness (IoU) rather than only exact string equality.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_intervals(value):
    out = []
    if not value:
        return out
    for token in value.split("|"):
        a, b = token.split("-")
        out.append((int(a), int(b)))
    return out


def canonical_interval(intervals):
    """Widest -> earliest start -> earliest end."""
    if not intervals:
        return None
    return sorted(intervals, key=lambda iv: (-(iv[1] - iv[0]), iv[0], iv[1]))[0]


def find_raw(raw_dir):
    if raw_dir and os.path.isfile(raw_dir):
        return raw_dir
    for path in (
        os.path.join(raw_dir or "", "sequences.csv"),
        os.path.join(raw_dir or "", "raw", "sequences.csv"),
        os.path.join(HERE, "dataset", "raw", "sequences.csv"),
    ):
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"sequences.csv not found (raw_dir={raw_dir!r})")


def baseline_prediction(sequence, width=20):
    """Answer-blind baseline: fixed-length centre substring."""
    n = len(sequence)
    w = min(width, n)
    if w <= 0:
        return ""
    start = max(0, (n - w) // 2)
    return sequence[start:start + w]


def prepare(raw_dir=None, public_dir=None, private_dir=None, **kwargs):
    raw_dir = raw_dir or kwargs.get("raw") or kwargs.get("input_dir") \
        or kwargs.get("data_dir") or kwargs.get("dataset_dir")
    public_dir = public_dir or kwargs.get("public") \
        or kwargs.get("output_public") or kwargs.get("public_output")
    private_dir = private_dir or kwargs.get("private") \
        or kwargs.get("output_private") or kwargs.get("private_output")

    raw_path = find_raw(
        raw_dir if raw_dir is not None else os.path.join(HERE, "dataset", "raw")
    )
    public = public_dir or os.path.join(HERE, "public")
    private = private_dir or os.path.join(HERE, "private")
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    def build(row):
        seq = row["sequence"]
        iv = canonical_interval(parse_intervals(row["true_intervals"]))
        if iv is None:
            return None
        s, e = iv
        return row["seq_id"], seq, s, e, seq[s:e]

    # PUBLIC TRAIN -- no numeric column; `length` is len(sequence)
    with open(os.path.join(public, "sequences_train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "sequence", "target"])
        n_train = 0
        for r in train:
            b = build(r)
            if b is None:
                continue
            sid, seq, s, e, target = b
            w.writerow([sid, seq, target])
            n_train += 1

    test_built = [b for b in (build(r) for r in test) if b is not None]

    # PUBLIC TEST (no target, no numeric column)
    with open(os.path.join(public, "sequences_test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "sequence"])
        for sid, seq, s, e, target in test_built:
            w.writerow([sid, seq])

    # SAMPLE SUBMISSION  (seq-to-seq: seq_id, prediction)
    with open(os.path.join(public, "sample_submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "prediction"])
        for sid, seq, s, e, target in test_built:
            w.writerow([sid, baseline_prediction(seq)])

    # PRIVATE ANSWERS
    #   `target` is the answer in the submission's own form (a DNA substring),
    #   so `prediction` and `target` name the same object; the span
    #   coordinates follow for edge-tightness scoring, and `sequence` is
    #   carried so 2-argument grading can locate a prediction.
    with open(os.path.join(private, "answers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "sequence", "target",
                    "reference_start", "reference_end"])
        for sid, seq, s, e, target in test_built:
            w.writerow([sid, seq, target, s, e])

    print(f"train seqs: {n_train}")
    print(f"test seqs:  {len(test_built)}")
    print("public columns: sequences_train.csv = seq_id, sequence, target")
    print("                sequences_test.csv  = seq_id, sequence")
    print("                (no numeric column published; length == len(sequence))")
    return {"n_train": n_train, "n_test": len(test_built)}


if __name__ == "__main__":
    args = sys.argv[1:]
    prepare(
        args[0] if len(args) > 0 else None,
        args[1] if len(args) > 1 else None,
        args[2] if len(args) > 2 else None,
    )
