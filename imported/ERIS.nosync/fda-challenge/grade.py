#!/usr/bin/env python3
"""Grader for "What Does It Do In There?" (chance-corrected set F1).

GRADING CONTRACT
----------------
    submission.csv : row_id, moa_classes
    answers.csv    : row_id, candidate_pool, moa_classes

`moa_classes` is the same object in the same form on both sides: a
pipe-separated set of mechanism-of-action class names, each drawn from that
row's own candidate pool. The submitted set is compared against the held-out
set.

`candidate_pool` is the pool the solver was shown (`candidates` in test.csv,
carried here under a different name so no column name is shared between the
public test table and the answer key). The grader needs it to compute what a
random selection of the same size would have scored -- it is never itself an
answer.

THE METRIC
----------
Per row, set F1 between the predicted and true class sets, then corrected for
what chance alone would earn:

    F1        = 2|P n T| / (|P| + |T|)
    E[F1]     = expected F1 of a uniformly random subset of the same size |P|
                drawn from that row's pool
    row_score = max(0, (F1 - E[F1]) / (1 - E[F1]))

Averaged over rows, in [0, 1].

WHY CHANCE CORRECTION, NOT RAW F1
---------------------------------
Measured before the split was built: the MoA class distribution is heavily
skewed, and simply always naming the most common class scored 0.182 raw F1
while reading nothing at all. Guessing the pool's most frequent handful does
better still. Raw F1 therefore spends a large part of its range on
distributional luck rather than on pharmacology.

Subtracting the expected score of a same-size random draw from the same pool
sends every such strategy to approximately 0 by construction, and a solver is
scored only on the agreement it earns above chance. The size of the submitted
set is part of the decision: guessing more classes raises raw F1's recall but
also raises E[F1], so padding the answer buys nothing.

E[F1] is computed in closed form. For a pool of n candidates, |T| true and
|P| predicted, a uniformly random P has E[|P n T|] = |P|*|T|/n, so

    E[F1] = 2 * (|P| * |T| / n) / (|P| + |T|)

DEGRADATION (degrade, never reject)
-----------------------------------
  missing row_id ................ 0.0 for that row
  unknown row_id ................ ignored
  blank / NaN prediction ........ 0.0 for that row
  class outside the row's pool .. that class is dropped before scoring
  duplicated row_id ............. all its rows discarded, that row scores 0
  missing required column ....... InvalidSubmission (a ValueError)

Harness signature: grade(submission, answers) -- exactly two inputs, no
filesystem access beyond them.
"""

import sys

import pandas as pd

SEP = "|"


class InvalidSubmissionError(ValueError):
    """Structural fault in the submission -- not a wrong answer."""


def parse_set(v):
    """Pipe-separated string -> set of non-empty, stripped class names."""
    if v is None:
        return set()
    if not isinstance(v, str):
        if pd.isna(v):
            return set()
        v = str(v)
    return {p.strip() for p in v.split(SEP) if p.strip()}


def expected_f1(n_pool, n_true, n_pred):
    """E[F1] of a uniformly random size-n_pred subset of an n_pool pool."""
    if n_pool <= 0 or n_pred <= 0 or n_true <= 0:
        return 0.0
    exp_hits = n_pred * n_true / n_pool
    return 2 * exp_hits / (n_pred + n_true)


def score_row(pred, truth, pool):
    """Chance-corrected set F1 for one row, in [0, 1]."""
    pool = set(pool)
    # A class outside the row's pool is not a wrong answer, it is not an
    # answer at all -- drop it rather than let it distort |P|.
    pred = set(pred) & pool
    truth = set(truth) & pool

    if not truth:
        # No gradable target for this row (should not occur; prepare.py
        # guarantees at least one in-pool true class).
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0

    hits = len(pred & truth)
    f1 = 2 * hits / (len(pred) + len(truth))
    e = expected_f1(len(pool), len(truth), len(pred))
    if e >= 1.0:
        return 1.0 if f1 >= 1.0 else 0.0
    return max(0.0, (f1 - e) / (1.0 - e))


def grade(submission, answers):
    if isinstance(submission, str):
        submission = pd.read_csv(submission)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    for col in ("row_id", "moa_classes"):
        if col not in submission.columns:
            raise InvalidSubmissionError(
                f"submission missing required column '{col}'")
    for col in ("row_id", "candidate_pool", "moa_classes"):
        if col not in answers.columns:
            raise InvalidSubmissionError(
                f"answers missing required column '{col}'")

    # A duplicated row_id is a conflict the grader will not resolve: every row
    # for it is discarded, so it can never score as if correct.
    ids = submission["row_id"].astype(str)
    dup = set(ids[ids.duplicated(keep=False)])
    submitted = {}
    for rid, val in zip(ids, submission["moa_classes"]):
        if rid not in dup:
            submitted[rid] = val

    total, n = 0.0, 0
    for row in answers.itertuples(index=False):
        rid = str(row.row_id)
        n += 1
        if rid not in submitted:
            continue
        total += score_row(parse_set(submitted[rid]),
                           parse_set(row.moa_classes),
                           parse_set(row.candidate_pool))

    if n == 0:
        raise InvalidSubmissionError("no gradable rows in answers")
    return total / n


def _cli():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sub = args[0] if args else "submission.csv"
    ans = None
    for i, a in enumerate(sys.argv):
        if a == "--answers" and i + 1 < len(sys.argv):
            ans = sys.argv[i + 1]
    if ans is None:
        ans = args[1] if len(args) > 1 else "dataset/private/answers.csv"
    print(f"{grade(sub, ans):.6f}")


if __name__ == "__main__":
    _cli()
