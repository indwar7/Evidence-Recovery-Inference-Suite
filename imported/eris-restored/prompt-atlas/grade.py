#!/usr/bin/env python3
"""grade.py -- score a submission for "Prompt Response Atlas".

Every term is checked against a MEASURED response surface: for each test item
the answers file carries the hidden classifier's probability on the target
category under all 256 ordered component pairs, plus that surface's minimum
and maximum. Nothing here is a heuristic judgement of the submission; it is
compared against numbers that were recorded by running the classifier.

    row_score =
        0.30 * steering_score
      + 0.12 * helpful_component_f1
      + 0.10 * harmful_component_f1
      + 0.10 * default_category_exact
      + 0.08 * best_first_component_exact
      + 0.08 * top_competitor_exact
      + 0.08 * order_sensitive_exact
      + 0.07 * target_reachable_exact
      + 0.07 * peak_probability_bin_exact

steering_score normalises the probability actually obtained by the submitted
pair against that item's own measured floor and ceiling, so an item whose
components barely move the classifier cannot dominate one where they move it a
lot.

The final score is the mean row score over the test items. A row whose
answer_json is missing or unparseable scores zero for that row rather than
voiding the submission.

Usage:
    python3 grade.py <submission.csv> [--answers prepared/private/answers.csv]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

ID_COL = "id"
ANS_COL = "answer_json"

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
METRIC_NAME = "atlas_score"

WEIGHTS = {
    "steering": 0.30,
    "helpful": 0.12,
    "harmful": 0.10,
    "default_category": 0.10,
    "best_first": 0.08,
    "top_competitor": 0.08,
    "order_sensitive": 0.08,
    "target_reachable": 0.07,
    "peak_bin": 0.07,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

MAX_SET = 5


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _f1(pred, true):
    """Set F1. Both empty scores 1.0."""
    pred = set(pred)
    true = set(true)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    hit = len(pred & true)
    if hit == 0:
        return 0.0
    prec = hit / len(pred)
    rec = hit / len(true)
    return 2 * prec * rec / (prec + rec)


def _as_list(x):
    if isinstance(x, list):
        return [str(v) for v in x][:MAX_SET]
    if isinstance(x, str):
        return [x]
    return []


def _norm(x):
    return str(x).strip().lower() if x is not None else ""


def _row_score(sub_ans, row):
    """sub_ans: parsed submission record. row: one answers.csv row."""
    truth = json.loads(row["answer_json"])
    parts = {}

    # --- steering: what the submitted pair actually achieved -------------- #
    pmin = float(row["p_min"])
    pmax = float(row["p_max"])
    pair = sub_ans.get("chosen_pair") or {}
    key = None
    if isinstance(pair, dict):
        f, s = str(pair.get("first", "")), str(pair.get("second", ""))
        key = f"{f}_{s}"
    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
        key = f"{pair[0]}_{pair[1]}"
    if key is not None and key in row.index:
        achieved = float(row[key])
    else:
        achieved = pmin              # unnamed or invalid pair -> the floor
    span = pmax - pmin
    parts["steering"] = 0.0 if span <= 0 else (achieved - pmin) / span

    parts["helpful"] = _f1(_as_list(sub_ans.get("helpful_components")),
                           truth["helpful_components"])
    parts["harmful"] = _f1(_as_list(sub_ans.get("harmful_components")),
                           truth["harmful_components"])

    for field, wkey in (("default_category", "default_category"),
                        ("best_first_component", "best_first"),
                        ("top_competitor", "top_competitor"),
                        ("order_sensitive", "order_sensitive"),
                        ("target_reachable", "target_reachable"),
                        ("peak_probability_bin", "peak_bin")):
        parts[wkey] = 1.0 if _norm(sub_ans.get(field)) == _norm(truth[field]) \
            else 0.0

    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def grade(submission, answers):
    """submission, answers: DataFrame or path. Returns float in [0, 1]."""
    if isinstance(submission, str):
        submission = pd.read_csv(submission)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    if ID_COL not in submission.columns:
        raise InvalidSubmissionError(f"submission missing '{ID_COL}' column")
    if ANS_COL not in submission.columns:
        raise InvalidSubmissionError(f"submission missing '{ANS_COL}' column")

    sub = submission[[ID_COL, ANS_COL]].copy()
    sub[ID_COL] = sub[ID_COL].astype(str)
    sub = sub.drop_duplicates(subset=ID_COL, keep="first")
    lookup = dict(zip(sub[ID_COL], sub[ANS_COL]))

    answers = answers.copy()
    answers[ID_COL] = answers[ID_COL].astype(str)

    total = 0.0
    for _, row in answers.iterrows():
        raw = lookup.get(row[ID_COL])
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue                                  # missing row scores 0
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(parsed, dict):
                continue
            s, _ = _row_score(parsed, row)
        except Exception:
            continue                                  # malformed row scores 0
        total += s

    if len(answers) == 0:
        raise InvalidSubmissionError("answers file is empty")
    return float(total / len(answers))


grade.minimum = 0.0
grade.maximum = 1.0
grade.direction = "maximize"


def score_bounds():
    return {"min": 0.0, "max": 1.0, "direction": "maximize"}


def _cli():
    ap = argparse.ArgumentParser(description="grade a prompt-atlas submission")
    ap.add_argument("submission", nargs="?", default="submission.csv")
    ap.add_argument("answers_pos", nargs="?", default=None)
    ap.add_argument("--answers", default=None)
    ap.add_argument("--breakdown", action="store_true")
    a = ap.parse_args()
    ans_path = a.answers or a.answers_pos or "prepared/private/answers.csv"

    if a.breakdown:
        sub = pd.read_csv(a.submission)
        ans = pd.read_csv(ans_path)
        lookup = dict(zip(sub["id"].astype(str), sub["answer_json"]))
        agg = {k: [] for k in WEIGHTS}
        for _, row in ans.iterrows():
            raw = lookup.get(str(row["id"]))
            if raw is None:
                continue
            _, parts = _row_score(json.loads(raw), row)
            for k, v in parts.items():
                agg[k].append(v)
        for k in WEIGHTS:
            m = np.mean(agg[k]) if agg[k] else 0.0
            print(f"  {k:20s} raw {m:.3f}   weighted {WEIGHTS[k]*m:.4f}")

    print(json.dumps({METRIC_NAME: grade(a.submission, ans_path)}))


if __name__ == "__main__":
    _cli()
