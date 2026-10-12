"""Grader for Observing Conditions.

Deterministic and pure-CSV: the hidden model is never run at grading time. Every
number it can produce was measured once, at dataset build, and stored. Grading
is a lookup, so two runs of this file on the same submission always agree.

    python grade.py submission.csv prepared/private/answers.csv
"""
import sys
import numpy as np
import pandas as pd

NOPS = 20
NPAIR = NOPS * NOPS
RCOLS = [f"r{i}" for i in range(NPAIR)]


class InvalidSubmission(Exception):
    pass


def _check(sub, answers):
    for c in ("row_id", "op1", "op2"):
        if c not in sub.columns:
            raise InvalidSubmission(f"missing required column '{c}'")
    if sub["row_id"].duplicated().any():
        d = sub.loc[sub["row_id"].duplicated(), "row_id"].tolist()[:5]
        raise InvalidSubmission(f"duplicate row_id(s): {d}")
    want = set(answers["row_id"])
    got = set(sub["row_id"])
    if got != want:
        miss, extra = sorted(want - got)[:5], sorted(got - want)[:5]
        raise InvalidSubmission(
            f"row_id mismatch: {len(want-got)} missing {miss}, "
            f"{len(got-want)} unexpected {extra}")
    for c in ("op1", "op2"):
        v = pd.to_numeric(sub[c], errors="coerce")
        if v.isna().any():
            raise InvalidSubmission(f"column '{c}' has non-integer values")
        if not np.all(v == v.astype(int)):
            raise InvalidSubmission(f"column '{c}' must be integers")
        v = v.astype(int)
        if v.min() < 0 or v.max() >= NOPS:
            raise InvalidSubmission(
                f"column '{c}' out of range: values must be in [0,{NOPS-1}], "
                f"saw [{v.min()},{v.max()}]")
        sub[c] = v
    return sub


def grade(submission, answers):
    """Mean measured response of the hidden model at each submitted pair.

    Each row's 400 responses are min-max normalised at build time, so 1.0 means
    the submitted pair is the single best of the 400 for that galaxy+target and
    0.0 means it is the worst. Row scores are averaged unweighted.
    """
    sub = pd.read_csv(submission) if isinstance(submission, str) else submission.copy()
    ans = pd.read_csv(answers) if isinstance(answers, str) else answers
    sub = _check(sub, ans)

    m = ans.merge(sub, on="row_id", how="left", validate="one_to_one")
    R = m[RCOLS].to_numpy(np.float32)
    pick = m["op1"].to_numpy(int) * NOPS + m["op2"].to_numpy(int)
    return float(np.round(R[np.arange(len(m)), pick].mean(), 6))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    try:
        print(f"{grade(sys.argv[1], sys.argv[2]):.6f}")
    except InvalidSubmission as e:
        print(f"INVALID SUBMISSION: {e}")
        raise SystemExit(1)
