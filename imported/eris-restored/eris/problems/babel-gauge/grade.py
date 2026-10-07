#!/usr/bin/env python3
"""grade.py - score a submission for the "Babel Gauge: Legacy Telemetry Log
Canonicalization" task.

Metric: mean per-line field accuracy over the 8,000 test lines.

For a line whose ground truth is the literal string REJECT, the line scores
1.0 iff the submitted output (stripped of surrounding whitespace) is exactly
"REJECT", else 0.0.

For every other line the truth is a canonical record with exactly six
`key=value` fields joined by `|`:

    ts=...|dev=...|site=...|temp_c=...|pres_kpa=...|status=...

The submitted output is split on `|`; each piece is parsed as `key=value`
(first `=` only). The line's score is

    (# of the 6 truth fields whose value matches exactly)
    ------------------------------------------------------
            max(6, # of pieces in the submitted output)

so missing, wrong, malformed, and extra fields all cost. Submitting
"REJECT" for a clean line scores 0. The final score is the unweighted mean
over all lines; range [0, 1], higher is better.

Empty/missing outputs score 0 for their line. The score is always averaged
over every line_id in the answer key, so a submission that omits rows is
penalised for them rather than discarded.

A submission is rejected with InvalidSubmissionError only when it genuinely
cannot be scored: an unreadable file or missing required columns. Recoverable
defects are absorbed instead of aborting the run:

  - missing line_ids   -> those lines score 0.0
  - unknown line_ids   -> ignored
  - duplicated line_ids-> first occurrence wins

Usage:
    python grade.py <submission.csv> [--answers prepared/private/answers.csv]

Prints a JSON object: {"mean_field_accuracy": <float>} and exits 0.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import pandas as pd

ID_COL = "line_id"
OUT_COL = "output"
FIELDS = ["ts", "dev", "site", "temp_c", "pres_kpa", "status"]


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _parse_canonical(s):
    """Split a canonical-format string into a key->value dict + piece count."""
    pieces = s.split("|")
    kv = {}
    for p in pieces:
        if "=" in p:
            k, v = p.split("=", 1)
            if k not in kv:            # first occurrence wins
                kv[k] = v
    return kv, len(pieces)


def _line_score(truth, out):
    if out is None or (isinstance(out, float) and out != out):   # None / NaN
        out = ""
    else:
        out = str(out).strip()
        if out.lower() == "nan":       # NaN survived a CSV round-trip as text
            out = ""
    if truth == "REJECT":
        return 1.0 if out == "REJECT" else 0.0
    if not out or out == "REJECT":
        return 0.0
    truth_kv, _ = _parse_canonical(truth)
    got_kv, n_pieces = _parse_canonical(out)
    correct = sum(1 for k in FIELDS if got_kv.get(k) == truth_kv[k])
    return correct / max(6, n_pieces)


def grade(submission, answers):
    """Return mean per-line field accuracy. Raises InvalidSubmissionError
    for truly invalid submissions.

    `submission` and `answers` may each be a pandas DataFrame (as passed by
    the platform's grade entrypoint) or a path to a CSV file (CLI use).
    """
    if isinstance(answers, pd.DataFrame):
        ans = answers.copy()
    else:
        ans = pd.read_csv(answers)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission, keep_default_na=False)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing_cols = [c for c in (ID_COL, OUT_COL) if c not in sub.columns]
    if missing_cols:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing_cols}; "
            f"expected columns: {ID_COL},{OUT_COL} "
            f"(found: {list(sub.columns)})")

    sub = sub[[ID_COL, OUT_COL]].copy()
    sub[ID_COL] = sub[ID_COL].astype(str)

    # Recoverable defects are absorbed, never fatal: a run that produced a
    # submission always receives a number. Duplicates keep their first row;
    # unknown ids are dropped; ids absent from the submission score 0 via the
    # reindex below. The mean is always taken over the full answer key, so
    # omissions cost score instead of voiding the submission.
    sub = sub.drop_duplicates(subset=ID_COL, keep="first")

    answer_ids = ans[ID_COL].astype(str)
    outputs = (sub.set_index(ID_COL)[OUT_COL]
                  .reindex(answer_ids)          # missing ids -> NaN -> 0.0
                  .tolist())

    truth = ans["canonical"].astype(str).tolist()
    scores = [_line_score(t, out) for t, out in zip(truth, outputs)]
    return float(sum(scores) / len(scores))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to submission CSV")
    parser.add_argument("--answers", default=os.path.join(
        here, "prepared", "private", "answers.csv"))
    args = parser.parse_args()

    try:
        score = grade(args.submission, args.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)

    print(json.dumps({"mean_field_accuracy": round(score, 6)}))


if __name__ == "__main__":
    main()
