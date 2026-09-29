#!/usr/bin/env python3
"""grade.py - score a submission for "Attributing Assistant Behavior to
System Prompt Clauses".

Metric: for each item, combine a RANKING component (where the true most-
responsible clause landed in the submitted ranking) with a FLAG component
(fraction of clauses whose submitted necessary/correlated flag matches the
true flag), averaged 50/50, then averaged over all test items.

RANKING COMPONENT
-----------------
Positional credit, generalized to a variable number of clauses n (n varies
per item, 3-10 -- see prepare.py): if the true most-responsible clause
lands at 0-indexed position i in the submitted ranking,

    ranking_credit = 1 - i / (n - 1)      (n > 1; 1.0 if n == 1)

so placing it first scores 1.0, last scores 0.0, and every position in
between gets even partial credit -- the same positional-credit shape
proven in "Prompt Edit Effect Attribution", generalized from a fixed
4-candidate case to this task's variable clause count.

FLAG COMPONENT
--------------
Fraction of the item's clauses whose submitted necessary/correlated flag
exactly matches the true flag (each clause's flag is derived from real
ablation deltas at prepare time -- see prepare.py's LABEL RULE).

A submission is rejected with InvalidSubmissionError only when it
genuinely cannot be scored: an unreadable file or missing required
columns. Every other defect degrades the score instead of raising:

  - missing item_ids                    -> scored 0 for that item
  - unknown item_ids                    -> ignored
  - duplicated item_ids                 -> first occurrence wins
  - a ranking that is not a valid permutation of the item's own clause_ids
    (repeated entry, unknown clause id, wrong length) -> ranking_credit
    scored 0 for that item, never rejected
  - a flag value outside {necessary, correlated} for any clause -> that
    clause's flag scored 0 (wrong), never rejected
  - empty / NaN submission fields -> scored 0, never rejected

Usage:
    python grade.py <submission.csv> [--answers dataset/private/answers.csv]
                     [--test dataset/public/test.csv]

Score bounds: minimum 0.0, maximum 1.0, higher is better.

Prints {"combined_score": <float>} and exits 0.
Only pandas is used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import pandas as pd

ID = "item_id"
VALID_FLAGS = {"necessary", "correlated"}

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
METRIC_NAME = "combined_score"
RANKING_WEIGHT = 0.5
FLAG_WEIGHT = 0.5


class InvalidSubmissionError(ValueError):
    """Raised when a submission is malformed beyond scoring."""


def _parse_clause_list(raw):
    """clause_ids and a submitted ranking are both stored as a single
    pipe-delimited string column (e.g. 'c1|c3|c2|c4') to keep the CSV
    schema fixed width despite each item having a different clause
    count. Returns a list of strings, or [] if unparseable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def _ranking_credit(true_clause_ids, top_clause_id, submitted_ranking):
    n = len(true_clause_ids)
    if n <= 1:
        return 1.0
    valid = (sorted(submitted_ranking) == sorted(true_clause_ids))
    if not valid or top_clause_id not in submitted_ranking:
        return 0.0
    pos = submitted_ranking.index(top_clause_id)
    return 1.0 - pos / (n - 1)


def _flag_credit(true_flags, submitted_flags):
    """true_flags / submitted_flags: {clause_id: 'necessary'|'correlated'}.
    Returns fraction of true_flags' clauses correctly matched."""
    if not true_flags:
        return 0.0
    correct = 0
    for cid, true_val in true_flags.items():
        pred_val = submitted_flags.get(cid)
        if pred_val in VALID_FLAGS and pred_val == true_val:
            correct += 1
    return correct / len(true_flags)


def grade(submission, answers):
    """Return combined ranking+flag score. `submission` and `answers` may
    each be a DataFrame (as passed by the platform) or a path to a CSV
    file. `answers` must carry columns: item_id, clause_ids (pipe-
    delimited, true order irrelevant here), top_clause_id, and
    flags (pipe-delimited 'clause_id:flag' pairs, e.g.
    'c1:necessary|c2:correlated|c3:correlated')."""
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

    required = [ID, "ranking", "flags"]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"{','.join(required)} (found: {list(sub.columns)})")

    sub = sub[required].copy()
    sub[ID] = sub[ID].astype(str)
    sub = sub.drop_duplicates(subset=ID, keep="first").set_index(ID)

    ans[ID] = ans[ID].astype(str)
    ans_idx = ans.set_index(ID)

    per_item_scores = []
    for item_id in ans_idx.index:
        true_clause_ids = _parse_clause_list(ans_idx.loc[item_id, "clause_ids"])
        top_clause_id = str(ans_idx.loc[item_id, "top_clause_id"]).strip()
        true_flag_pairs = _parse_clause_list(ans_idx.loc[item_id, "flags"])
        true_flags = {}
        for pair in true_flag_pairs:
            if ":" in pair:
                cid, flag = pair.split(":", 1)
                true_flags[cid.strip()] = flag.strip()

        if item_id in sub.index:
            submitted_ranking = _parse_clause_list(sub.loc[item_id, "ranking"])
            submitted_flag_pairs = _parse_clause_list(sub.loc[item_id, "flags"])
            submitted_flags = {}
            for pair in submitted_flag_pairs:
                if ":" in pair:
                    cid, flag = pair.split(":", 1)
                    submitted_flags[cid.strip()] = flag.strip()
        else:
            submitted_ranking = []
            submitted_flags = {}

        r_credit = _ranking_credit(true_clause_ids, top_clause_id, submitted_ranking)
        f_credit = _flag_credit(true_flags, submitted_flags)
        per_item_scores.append(RANKING_WEIGHT * r_credit + FLAG_WEIGHT * f_credit)

    mean = float(sum(per_item_scores) / len(per_item_scores))
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
    print(json.dumps({"combined_score": round(score, 6)}))


if __name__ == "__main__":
    main()
