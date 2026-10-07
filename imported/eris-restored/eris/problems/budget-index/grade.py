#!/usr/bin/env python3
"""grade.py -- score a submission for "Budget-Constrained Index Construction".

Metric: ANSWERABILITY -- the fraction of evaluation queries answered by the
submitted index. For each document you build a set of INDEX WINDOWS (contiguous
spans of sentences, each at most MAX_CHUNK_TOKENS = 60 tokens). You may build at
most `budget_k` windows per document and they may NOT overlap; the rest of the
document is left un-indexed. A query is answered iff some single submitted
window contains ALL of its required sentences and NONE of its poison sentences.
Range 0 to 1; higher is better. The workload is built so the exact optimum
(verified by a budgeted dynamic program at generation time) is about 0.45 -- no
submission can exceed it.

The answers file holds ONE ROW PER DOCUMENT, keyed by `doc_id` (its first
column, which the platform harness uses to match submission rows). Each row
packs the document's per-sentence token counts, its index budget, and its
evaluation queries (required/poison sentence ids) as JSON, so any row-wise
slice of it -- e.g. the platform's public/private leaderboard split -- remains
a complete, independently gradeable answer set. The score is query-weighted.

A submission gives one index per document: `windows` is a |-separated list of
`start-end` sentence spans (end EXCLUSIVE), e.g. `0-3|7-10`. A document whose
windows exceed the budget, overlap, run out of range, or produce an oversized
window scores 0 for that document's queries only -- the rest of the submission
is still graded. A submission is rejected with InvalidSubmissionError only when
it cannot be scored at all: missing columns or duplicated doc_ids.

Usage:
    python grade.py <submission.csv> [--answers private/answers.csv]

Only pandas is used (Kaggle Python Docker image).
"""

import argparse
import json
import sys

import pandas as pd

MAX_CHUNK_TOKENS = 60
DOC_COL = "doc_id"
WIN_COL = "windows"

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


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def parse_windows(text, n_sents, tok, budget_k):
    """Parse and validate one document's window list. Returns a list of
    (start, end) spans, or None if invalid for this document."""
    if text is None:
        return []
    s = str(text).strip()
    if s == "" or s.lower() == "nan":
        return []
    spans = []
    for part in s.split("|"):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            return None
        a, _, b = part.partition("-")
        try:
            a, b = int(a), int(b)
        except ValueError:
            return None
        if not (0 <= a < b <= n_sents):
            return None
        if sum(tok[a:b]) > MAX_CHUNK_TOKENS:      # token cap
            return None
        spans.append((a, b))
    if len(spans) > budget_k:                     # over budget
        return None
    spans.sort()
    for i in range(1, len(spans)):                # non-overlap
        if spans[i][0] < spans[i - 1][1]:
            return None
    return spans


def _answered(spans, req, poi):
    lo, hi = min(req), max(req)
    for j, i in spans:
        if j <= lo and hi < i and not any(j <= p < i for p in poi):
            return True
    return False


def grade(submission, answers):
    """submission, answers: DataFrame or path to CSV. Returns float in [0, 1]."""
    if isinstance(submission, str):
        submission = pd.read_csv(submission)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    if DOC_COL not in submission.columns:
        raise InvalidSubmissionError(f"submission missing '{DOC_COL}' column")
    if WIN_COL not in submission.columns:
        raise InvalidSubmissionError(f"submission missing '{WIN_COL}' column")
    if submission[DOC_COL].duplicated().any():
        raise InvalidSubmissionError("submission has duplicate doc_id rows")

    sub = dict(zip(submission[DOC_COL].astype(str), submission[WIN_COL]))

    total = 0
    answered = 0
    for r in answers.itertuples(index=False):
        did = str(r.doc_id)
        tok = json.loads(r.tok_json)
        n_sents = len(tok)
        budget_k = int(r.budget_k)
        queries = json.loads(r.queries_json)
        spans = parse_windows(sub.get(did, ""), n_sents, tok, budget_k)
        for q in queries:
            total += 1
            if spans and _answered(spans, q["r"], q["p"]):
                answered += 1

    if total == 0:
        raise InvalidSubmissionError("no gradable queries in answers")
    return answered / total


grade.minimum = 0.0
grade.maximum = 1.0
grade.direction = "maximize"


def score_bounds():
    return {"min": 0.0, "max": 1.0, "direction": "maximize"}


def _cli():
    ap = argparse.ArgumentParser(description="grade budget-index submission")
    ap.add_argument("submission", nargs="?", default="submission.csv")
    ap.add_argument("answers_pos", nargs="?", default=None)
    ap.add_argument("--answers", default=None)
    a = ap.parse_args()
    ans = a.answers or a.answers_pos or "private/answers.csv"
    print(json.dumps({"answerability": grade(a.submission, ans)}))


if __name__ == "__main__":
    _cli()
