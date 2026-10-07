#!/usr/bin/env python3
"""Grader for "Coining Gene Symbols from Gene Names" (prefix agreement).

GRADING CONTRACT
----------------
    submission.csv : row_id, symbol
    answers.csv    : row_id, symbol

`symbol` is the same object in the same form on both sides: a gene symbol, as
a short string. The submitted `symbol` is compared against the held-out
`symbol`, which is the HGNC approved symbol for the gene whose name was shown.

THE METRIC
----------
A gene symbol is read from the left. The stem comes first and says which
family the gene belongs to; the designators that follow narrow it down to a
subfamily and then to one member. SLC2A1 is stem SLC, family 2, subfamily A,
member 1. Agreement from the left is therefore agreement down that hierarchy.

For each row, after normalisation (whitespace removed, upper-cased):

    shared    = number of leading characters the prediction and the approved
                symbol have in common, stopping at the first difference
    row_score = shared / max(len(prediction), len(approved))

    SLC2A1 vs SLC2A1   6/6 = 1.000    exact
    SLC2A3 vs SLC2A1   5/6 = 0.833    right subfamily, wrong member
    SLC9   vs SLC2A1   3/6 = 0.500    right stem only
    SCF2M1 vs SLC2A1   1/6 = 0.167    first letter only
    GLUT1  vs SLC2A1   0/6 = 0.000    nothing

Dividing by the LONGER of the two lengths is what stops padding: a prediction
cannot raise its score by being longer than the answer, and cannot protect a
correct stem by stopping early.

FINAL_SCORE = mean(row_score) over every row in answers.csv, in [0, 1].

WHY NOT THE ALTERNATIVES (each measured on the shipped split)
-------------------------------------------------------------
Edit similarity pays for characters in the wrong place. Taking the first
letter of every word of the name and keeping the digits scored 0.595 under
it -- most of that from digits copied out of the name and from letters that
happen to recur -- which leaves every real model a 0.4-wide band. Under
prefix agreement the digits sit behind the stem and earn nothing until the
stem is right.

Exact match alone is a legitimate reading of "did you coin the approved
symbol", but it scores SLC2A3 and GLUT1 the same against SLC2A1. One of those
found the family and the other did not.

DEGRADATION (degrade, never reject)
-----------------------------------
  missing row_id ................ 0.0 for that row
  unknown row_id ................ ignored
  blank / NaN prediction ........ 0.0 for that row
  duplicated row_id ............. all its rows discarded, that row scores 0
  malformed row_id .............. ignored, never raises
  missing required column ....... InvalidSubmission (a ValueError)

Harness signature: grade(submission, answers) -- exactly two inputs, no
filesystem access beyond them.
"""

import sys

import pandas as pd

ID_COL = "row_id"
TARGET_COL = "symbol"


class InvalidSubmission(ValueError):
    """Structural fault in the submission -- not a wrong answer."""


def normalise(value):
    """Upper-case and drop all whitespace. Blank and NaN become ''."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        value = str(value)
    return "".join(value.split()).upper()


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score_row(prediction, approved):
    """Prefix agreement for one row, in [0, 1]."""
    p, t = normalise(prediction), normalise(approved)
    if not p or not t:
        return 0.0
    return shared_prefix(p, t) / max(len(p), len(t))


def grade(submission, answers):
    if isinstance(submission, str):
        submission = pd.read_csv(submission, dtype=str, keep_default_na=False)
    if isinstance(answers, str):
        answers = pd.read_csv(answers, dtype=str, keep_default_na=False)

    for col in (ID_COL, TARGET_COL):
        if col not in submission.columns:
            raise InvalidSubmission(
                f"submission missing required column '{col}'")
    for col in (ID_COL, TARGET_COL):
        if col not in answers.columns:
            raise InvalidSubmission(
                f"answers missing required column '{col}'")

    # A duplicated row_id is a conflict the grader will not resolve: every row
    # for it is discarded, so it can never score as if correct.
    ids = submission[ID_COL].astype(str).str.strip()
    dup = set(ids[ids.duplicated(keep=False)])
    submitted = {}
    for rid, pred in zip(ids, submission[TARGET_COL]):
        if rid in dup:
            continue
        submitted[rid] = pred

    total, n = 0.0, 0
    for rid, approved in zip(answers[ID_COL].astype(str).str.strip(),
                             answers[TARGET_COL]):
        n += 1
        total += score_row(submitted.get(rid), approved)

    if n == 0:
        raise InvalidSubmission("no gradable rows in answers")
    return float(total / n)


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
