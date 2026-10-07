#!/usr/bin/env python3
"""grade.py - score a submission for "Predicting Retrieval Sufficiency in
RAG Pipelines".

Metric: mean joint-field accuracy across the three submitted columns
(diagnosis, action, reason_code), averaged per item then averaged over all
test items. Each item's per-item score is:

    (1 if diagnosis correct else 0
     + 1 if action correct else 0
     + 1 if reason_code correct else 0) / 3

This is deliberately NOT "all three fields must match" (too harsh -- would
collapse most partial understanding to 0) and NOT "average each field's
accuracy independently over the whole test set" (would let a solver nail
one easy field and coast) -- per-item joint averaging rewards getting more
of the trace right on each item and produces a continuous, many-valued
per-item score (register section -1: reviewers reward a rich, non-binary
per-item distribution) rather than a bimodal 0/1 signal.

Chance level: diagnosis has 2 valid values, action has 3, reason_code has 4,
but the three fields are NOT independent (some combinations are internally
inconsistent, e.g. reason_code=gold_absent must pair with action=abstain).
The reference "always guess the train-set's globally most common
consistent trace" baseline is the correct chance floor to compare against,
not 1/2 * 1/3 * 1/4 -- see prepare.py / STATE.md for the measured number.

A submission is rejected with InvalidSubmissionError only when it genuinely
cannot be scored: an unreadable file or a missing required column. Every
other defect degrades the score instead of raising:

  - missing item_ids                  -> scored 0 for all 3 fields
  - unknown item_ids                  -> ignored
  - duplicated item_ids               -> first occurrence wins
  - empty / NaN field value           -> scored 0 (wrong) for that field
  - value not in that field's fixed vocabulary -> scored 0 (wrong), never
    raises -- an out-of-domain guess is simply incorrect, not invalid

Usage:
    python grade.py <submission.csv> [--answers dataset/private/answers.csv]

Score bounds: minimum 0.0, maximum 1.0, higher is better.

Prints {"joint_accuracy": <float>} and exits 0.
Only pandas is used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import pandas as pd

ID = "item_id"
FIELDS = ["diagnosis", "action", "reason_code"]

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
METRIC_NAME = "joint_accuracy"


class InvalidSubmissionError(ValueError):
    """Raised when a submission is malformed beyond scoring."""


def grade(submission, answers):
    """Return mean joint-field accuracy. `submission` and `answers` may
    each be a DataFrame (as passed by the platform) or a path to a CSV
    file."""
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

    missing = [c for c in [ID] + FIELDS if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"{ID},{','.join(FIELDS)} (found: {list(sub.columns)})")

    sub = sub[[ID] + FIELDS].copy()
    sub[ID] = sub[ID].astype(str)
    sub = sub.drop_duplicates(subset=ID, keep="first").set_index(ID)

    ans[ID] = ans[ID].astype(str)
    ans_idx = ans.set_index(ID)

    per_item_scores = []
    for item_id in ans_idx.index:
        row_score = 0
        for field in FIELDS:
            target = str(ans_idx.loc[item_id, field]).strip().lower()
            if item_id in sub.index:
                pred_val = sub.loc[item_id, field]
                pred = str(pred_val).strip().lower() if pd.notna(pred_val) else ""
            else:
                pred = ""
            if pred and pred == target:
                row_score += 1
        per_item_scores.append(row_score / len(FIELDS))

    mean = float(sum(per_item_scores) / len(per_item_scores))
    # clamp against floating-point drift so the contract is exact
    return min(MAX_SCORE, max(MIN_SCORE, mean))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission")
    p.add_argument("--answers",
                    default=os.path.join(here, "dataset", "private", "answers.csv"))
    a = p.parse_args()
    try:
        score = grade(a.submission, a.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)
    print(json.dumps({"joint_accuracy": round(score, 6)}))


if __name__ == "__main__":
    main()
