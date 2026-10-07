#!/usr/bin/env python3
"""grade.py - score a submission for "The Channel That Lies".

Metric: average precision (area under the precision-recall curve) for
detecting the hours in which the low-cost node's ozone channel cannot be
trusted.

Submissions provide a continuous anomaly score per hour, higher meaning more
suspect. Only the ranking matters; the scale is irrelevant. A random or
constant score gives roughly the positive rate (about 0.23 on this test set).

A submission is rejected with InvalidSubmissionError only when it genuinely
cannot be scored: an unreadable file or a missing required column. Recoverable
defects are absorbed so a completed run always returns a number:

  - missing timestamps    -> scored at the bottom of the ranking
  - unknown timestamps    -> ignored
  - duplicated timestamps -> first occurrence wins
  - non-numeric or NaN    -> treated as missing, same as above

Usage:
    python grade.py <submission.csv> [--answers prepared/private/answers.csv]

Prints {"average_precision": <float>} and exits 0.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ID = "timestamp"
SCORE = "score"
LABEL = "is_anomalous"


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _average_precision(y_true, y_score):
    """Area under the precision-recall curve, computed as the step-wise sum
    used by sklearn.metrics.average_precision_score. Ties are broken
    pessimistically by sorting on score alone, which is what a ranking metric
    should do for a submission that cannot separate two hours."""
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = y.sum()
    if total_pos == 0:
        raise InvalidSubmissionError(
            "The answer key contains no positive hours; average precision is "
            "undefined.")
    # sum of precision at each true positive, divided by the positive count
    return float((precision * y).sum() / total_pos)


def grade(submission, answers):
    """Return average precision. `submission` and `answers` may each be a
    DataFrame (as passed by the platform) or a path to a CSV file."""
    ans = answers.copy() if isinstance(answers, pd.DataFrame) \
        else pd.read_csv(answers)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing = [c for c in (ID, SCORE) if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"{ID},{SCORE} (found: {list(sub.columns)})")

    ans[ID] = pd.to_datetime(ans[ID]).astype("int64")
    sub = sub[[ID, SCORE]].copy()
    try:
        sub[ID] = pd.to_datetime(sub[ID]).astype("int64")
    except Exception as exc:
        raise InvalidSubmissionError(
            f"Could not parse the '{ID}' column as timestamps: {exc}") from exc

    sub = sub.drop_duplicates(subset=ID, keep="first")

    y = ans[LABEL].to_numpy(dtype=int)
    s = (sub.set_index(ID)[SCORE]
            .reindex(ans[ID])              # unknown ids dropped, gaps -> NaN
            .to_numpy(dtype=float))

    # an hour with no usable score is ranked last rather than voiding the run
    if np.isfinite(s).any():
        s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1.0)
    else:
        s = np.zeros_like(y, dtype=float)

    return _average_precision(y, s)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission")
    p.add_argument("--answers", default=os.path.join(
        here, "prepared", "private", "answers.csv"))
    a = p.parse_args()
    try:
        score = grade(a.submission, a.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)
    print(json.dumps({"average_precision": round(score, 6)}))


if __name__ == "__main__":
    main()
