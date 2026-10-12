"""Grader: block-closure accuracy on flattened Python files.

    grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Contract. The submission's `depths` column and the answers' `depths` column are the same
object in the same form: for one file, a space-separated list of non-negative integers,
one per logical line, giving that line's block depth (0 = module level). `answers` also
carries `lines_ref`, the file's logical lines as a JSON list (identical to test.csv's
`lines`), which the grader needs to tell which lines are scored. Rows join on `file_id`.

Scoring. A line is eligible when the line before it is not a block header (does not end
with a colon), does not begin with return, raise, break, continue or pass, and the line
itself does not begin with elif, else, except, finally, case, def, class, async or @.
An eligible line is scored when its true depth differs from the true depth of the line
before it, or its predicted depth differs from the predicted depth of the line before it.
A scored line is correct when the predicted change of depth equals the true change. A
file's score is the fraction of its scored lines that are correct; the result is the mean
over the answer files, in [0, 1].

Degrade, never reject: a missing file, a blank or unparsable value, a value of the wrong
length, a negative depth, or a file_id given more than once scores 0 for that file.
Unknown file_ids and extra columns are ignored. Only a missing `file_id` or `depths`
column raises InvalidSubmission (a ValueError).
"""
import json
import re

import pandas as pd


class InvalidSubmission(ValueError):
    pass


TERMINAL = {"return", "raise", "break", "continue", "pass"}
OPENERS = {"elif", "else", "except", "finally", "case", "def", "class", "async", "@"}


def first_word(t):
    m = re.match(r"[A-Za-z_]\w*|\S", t)
    return m.group(0) if m else ""


def is_header(t):
    return t.rstrip().endswith(":")


def eligible(T, i):
    return (not is_header(T[i - 1]) and first_word(T[i - 1]) not in TERMINAL
            and first_word(T[i]) not in OPENERS)


def file_score(pred, true, T):
    S = [i for i in range(1, len(true)) if eligible(T, i)
         and (true[i] != true[i - 1] or pred[i] != pred[i - 1])]
    if not S:
        return 1.0 if pred == true else 0.0
    return sum(pred[i] - pred[i - 1] == true[i] - true[i - 1] for i in S) / len(S)


def _parse(v, n):
    if not isinstance(v, str):
        return None
    try:
        d = [int(x) for x in v.split()]
    except ValueError:
        return None
    if len(d) != n or (d and min(d) < 0):
        return None
    return d


def grade(submission, answers):
    for c in ("file_id", "depths"):
        if c not in submission.columns:
            raise InvalidSubmission(f"submission is missing required column '{c}'")
    sub = submission[["file_id", "depths"]].copy()
    sub["file_id"] = sub["file_id"].astype(str)
    counts = sub["file_id"].value_counts()
    dup = set(counts[counts > 1].index)
    pred = {k: v for k, v in zip(sub["file_id"], sub["depths"]) if k not in dup}
    scores = []
    for fid, dstr, lref in zip(answers["file_id"].astype(str), answers["depths"], answers["lines_ref"]):
        T = json.loads(lref)
        true = [int(x) for x in dstr.split()]
        p = _parse(pred.get(fid), len(true))
        scores.append(0.0 if p is None else file_score(p, true, T))
    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--answers", default="dataset/private/answers.csv")
    a = ap.parse_args()
    print(f"{grade(pd.read_csv(a.submission, dtype=str, keep_default_na=False), pd.read_csv(a.answers, dtype=str, keep_default_na=False)):.4f}")
