"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-query Matthews Correlation Coefficient (MCC) over
candidate-selection, clipped to [0, 1].

SHAPE. One row per query in both submission.csv and answers.csv:
    submission:  query_id, cited_positions
    answers:     query_id, cited_positions, n_candidates
`cited_positions` is a whitespace-separated list of 0-indexed candidate
positions (the `candidate_NN_text` column index in test.csv) that the
query opinion is believed to cite. An empty value means "cites none of
these candidates".

WHY MCC AND NOT F1 OVER THE SELECTED SET. Each query has a fixed
candidate pool with a minority of true positives, so a submission can
score well on F1 without reading anything: measured on the shipped
split, "select every candidate" scores a substantial raw F1 purely from
recall. MCC is computed over the full confusion matrix of the candidate
pool (true positives, false positives, false negatives AND true
negatives), which charges a submission for the distractors it wrongly
accepts. Measured on the shipped test set: "select none" and "select
all" both score exactly 0.000 under MCC, and selecting a random subset
of the correct size scores ~0.13 -- an honest chance floor rather than
an inflated one.

REJECTED OUTRIGHT (raises InvalidSubmission, a ValueError): a required
column missing from the submission. Nothing else raises.

DEGRADES, NOT REJECTS, for everything else -- the query scores 0 and the
rest of the submission is still graded:
  - a query missing from the submission;
  - a query_id appearing MORE THAN ONCE (all conflicting rows are
    dropped; the grader does not pick one);
  - a `cited_positions` value containing a non-integer token, a
    duplicate position, or a position outside [0, n_candidates).

SLICE SAFETY. Only query_ids present in `answers` are scored; extra
query_ids in the submission are ignored, so a row-wise public/private
split of answers.csv grades correctly.
"""
import re

import pandas as pd

REQUIRED_SUBMISSION_COLUMNS = {"query_id", "cited_positions"}
REQUIRED_ANSWER_COLUMNS = {"query_id", "cited_positions", "n_candidates"}

_SPLIT = re.compile(r"[\s,;]+")


class InvalidSubmission(ValueError):
    pass


def _parse_positions(value, n_candidates: int):
    """Parse a cited_positions cell into a set of ints in
    [0, n_candidates). Returns None if the cell is malformed, which the
    caller treats as 'score this query 0'."""
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s:
        return set()
    out = set()
    for tok in _SPLIT.split(s):
        if not tok:
            continue
        if not tok.lstrip("-").isdigit():
            return None
        i = int(tok)
        if i < 0 or i >= n_candidates or i in out:
            return None
        out.add(i)
    return out


def _mcc(true_pos: set, pred_pos: set, n_candidates: int) -> float:
    tp = len(true_pos & pred_pos)
    fp = len(pred_pos - true_pos)
    fn = len(true_pos - pred_pos)
    tn = n_candidates - tp - fp - fn
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if den == 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if answers is None or not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {sorted(REQUIRED_ANSWER_COLUMNS)}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=sorted(REQUIRED_SUBMISSION_COLUMNS))
    missing = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing:
        raise InvalidSubmission(f"submission missing required columns: {sorted(missing)}")

    sub_ids = submission["query_id"].astype(str).str.strip()
    dup_ids = set(sub_ids[sub_ids.duplicated()])
    pred_by_query = {q: v for q, v in zip(sub_ids, submission["cited_positions"])
                     if q not in dup_ids}

    scores = []
    for row in answers.itertuples(index=False):
        qid = str(row.query_id).strip()
        try:
            n_cand = int(row.n_candidates)
        except (TypeError, ValueError):
            continue
        if n_cand < 2:
            continue

        true_pos = _parse_positions(row.cited_positions, n_cand)
        if true_pos is None or not true_pos or len(true_pos) == n_cand:
            # a query with no true positives, or where every candidate is
            # a true positive, cannot be scored discriminatively; skipped
            # by construction this never happens on the shipped split
            continue

        if qid not in pred_by_query:
            scores.append(0.0)
            continue
        pred_pos = _parse_positions(pred_by_query[qid], n_cand)
        if pred_pos is None:
            scores.append(0.0)
            continue
        scores.append(_mcc(true_pos, pred_pos, n_cand))

    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import sys

    sub = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    ans = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    print(grade(sub, ans))
