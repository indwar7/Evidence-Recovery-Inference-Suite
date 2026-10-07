#!/usr/bin/env python3
"""grade.py - score a submission for "Cross-Lingual Organization Name
Normalization".

Metric: mean token-level F1 between the submitted English name and the
canonical English name, averaged over all test pairs.

Both strings are lowercased and split into alphanumeric tokens. Precision is
the share of submitted tokens that appear in the target, recall is the share
of target tokens that appear in the submission, and the pair scores their
harmonic mean. Token multiplicity is respected, so repeating a word does not
earn extra credit.

Partial credit is deliberate. Organization names are long and a solver that
gets "University of Antwerp" from "Universiteit Antwerpen" but misses one
qualifier should not score the same as one that returns nothing.

A submission is rejected with InvalidSubmissionError only when it genuinely
cannot be scored: an unreadable file or a missing required column. Recoverable
defects are absorbed so a completed run always returns a number:

  - missing pair_ids    -> score 0.0 for those pairs
  - unknown pair_ids    -> ignored
  - duplicated pair_ids -> first occurrence wins
  - empty or NaN output -> scores 0.0 for that pair

Usage:
    python grade.py <submission.csv> [--answers prepared/private/answers.csv]

Score bounds: minimum 0.0, maximum 1.0, higher is better. Token F1 is bounded
below by 0 (no shared tokens) and above by 1 (an exact token match on every
row), so the mean over rows is bounded by the same values.

Prints {"token_f1": <float>} and exits 0.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

ID = "pair_id"
OUT = "output"
TARGET = "target_name"

# ---- declared score bounds (read by the platform validator) -------------- #
MIN_SCORE = 0.0
MAX_SCORE = 1.0
MINIMUM_SCORE = 0.0
MAXIMUM_SCORE = 1.0
SCORE_MIN = 0.0
SCORE_MAX = 1.0
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
DIRECTION = "maximize"
GRADE_DIRECTION = "maximize"
SCORE_BOUNDS = {"min": 0.0, "max": 1.0, "direction": "maximize"}
METRIC_NAME = "token_f1"
TOKEN = re.compile(r"\w+", re.UNICODE)


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _tokens(s):
    return TOKEN.findall(s.lower())


def _pair_f1(pred, target):
    if pred is None or (isinstance(pred, float) and pred != pred):
        return 0.0
    p = Counter(_tokens(str(pred)))
    t = Counter(_tokens(str(target)))
    if not p or not t:
        return 0.0
    overlap = sum((p & t).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(t.values())
    return 2 * precision * recall / (precision + recall)


def grade(submission, answers):
    """Return mean token-level F1. `submission` and `answers` may each be a
    DataFrame (as passed by the platform) or a path to a CSV file."""
    ans = answers.copy() if isinstance(answers, pd.DataFrame) \
        else pd.read_csv(answers, keep_default_na=False)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission, keep_default_na=False)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing = [c for c in (ID, OUT) if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"{ID},{OUT} (found: {list(sub.columns)})")

    sub = sub[[ID, OUT]].copy()
    sub[ID] = sub[ID].astype(str)
    sub = sub.drop_duplicates(subset=ID, keep="first")

    ans[ID] = ans[ID].astype(str)
    preds = (sub.set_index(ID)[OUT]
                .reindex(ans[ID])          # unknown ids dropped, gaps -> NaN
                .tolist())

    targets = ans[TARGET].astype(str).tolist()
    scores = [_pair_f1(p, t) for p, t in zip(preds, targets)]
    mean = float(sum(scores) / len(scores))
    # clamp against floating-point drift so the contract is exact
    return min(MAX_SCORE, max(MIN_SCORE, mean))


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
    print(json.dumps({"token_f1": round(score, 6)}))


if __name__ == "__main__":
    main()
