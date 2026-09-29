#!/usr/bin/env python3
"""Grader for "The Vanished Clause" (gap-anchored changed-token F1).

GRADING CONTRACT
----------------
    submission.csv : row_id, text_before
    answers.csv    : row_id, shown_text, text_before

`text_before` is the same object in the same form on both sides: the wording
of the regulation section BEFORE the recorded amendment, as free text. The
submitted `text_before` is compared against the held-out `text_before`.

`shown_text` is the amended wording the solver was given (`text_after` in
test.csv, carried here under a different name so no column name is shared
between the public test table and the answer key). The grader needs it to
decide WHICH tokens the amendment changed -- it is never itself an answer.

THE METRIC
----------
Consecutive versions of a regulation share most of their text. Scoring whole-
string similarity would pay a submission for the ~90% of tokens that an
amendment never touched, and copying the input through unchanged would score
near the maximum. So only the changed tokens count.

For each row:

  1. Tokenize `shown_text` (S), the true `text_before` (B), and the
     prediction (P).
  2. Align S to B with an LCS backtrace. The positions of B that are NOT
     matched through from S are the CHANGED POSITIONS -- the tokens the
     amendment removed, i.e. what the solver has to recover.
  3. Each changed token is keyed by its GAP: which gap between two shown
     tokens it sits in. Align S to P the same way to get what the prediction
     proposes, keyed the same way. A proposed token counts only if it is the
     right token in the right gap.

     changed  = multiset of (gap, token) for B's unmatched tokens
     proposed = multiset of (gap, token) for P's unmatched tokens
     hits     = |changed n proposed|         (multiset intersection)
     row_score = F1 of hits against |changed| and |proposed|

  4. A row whose true wording has no changed token (possible only if the
     filter in prepare.py let one through) scores 1.0 for an unchanged
     prediction and 0.0 otherwise.

WHY THE GAP IS PART OF THE KEY
------------------------------
Without it, matching is position-free, and most changed tokens are common
ones ("," "." "the" "(" "of"). Measured on the shipped split: appending the
60 most frequent removed tokens from train to the end of the shown text --
reading nothing -- scored 0.129 position-free, well above every lookup
baseline. Keyed by gap, the same dump lands in one gap at the end of the
section and scores 0.012. The oracle still scores 1.0 and
copy-through still scores 0.

WHY F1 AND NOT RECALL ALONE
---------------------------
Recall alone -- hits / |changed| -- was measured and REJECTED. Under it, a
submission that appends a candidate vocabulary of plausible regulatory words
to the shown text recovers every changed token by brute force and scores
1.000 without reading anything. Measured on a worked example: the shown text
plus a 25-word vocabulary dump scored 1.000, identical to the oracle.

Balancing recall against precision over the ADDED tokens closes this: a dump
adds far more than the amendment removed, so |proposed| explodes, precision
collapses and the F1 falls back near zero. A solver is therefore scored on
proposing the right prior wording, not on proposing everything.

FINAL_SCORE = mean(row_score) over every row in answers.csv, in [0, 1].

Copying the shown text through therefore scores 0: it adds nothing, so
`proposed` is empty. Submitting a dictionary does not help either: it makes
`proposed` enormous, precision collapses, and tokens in the wrong gap never
count as hits.

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

import difflib
import re
import sys
from collections import Counter

import pandas as pd

TOKEN = re.compile(r"\w+|[^\w\s]")
ANCHOR = 8   # shared run length treated as a fixed alignment anchor


class InvalidSubmissionError(ValueError):
    """Structural fault in the submission -- not a wrong answer."""


def tokenize(t):
    if t is None:
        return []
    if not isinstance(t, str):
        if pd.isna(t):
            return []
        t = str(t)
    return TOKEN.findall(t)


def _changed_in_gaps(shown_toks, other_toks):
    """Tokens of `other` not matched through from `shown`, with their GAP.

    LCS backtrace. Every unmatched token of `other` is returned as
    (gap, token), where gap = how many shown tokens precede it in the
    alignment -- i.e. which gap between two shown tokens it was inserted into.
    """
    # Long runs the two sides share (>= ANCHOR tokens, found by difflib) are
    # matched outright; the exact LCS runs only in the short stretches between
    # them. Consecutive versions share ~97% of their tokens, so this keeps a
    # full grade to about a second. Deterministic, and both the truth and the
    # prediction go through the same procedure, so the oracle scores 1.0.
    sm = difflib.SequenceMatcher(None, shown_toks, other_toks, autojunk=False)
    out, si, oi = [], 0, 0
    for a0, b0, size in sm.get_matching_blocks():
        if size < ANCHOR and size != 0:
            continue
        for g, t in _lcs_gaps(shown_toks[si:a0], other_toks[oi:b0]):
            out.append((g + si, t))
        si, oi = a0 + size, b0 + size
    return out


def _lcs_gaps(shown_toks, other_toks):
    n, m = len(shown_toks), len(other_toks)
    if m == 0:
        return []
    if n == 0:
        return [(0, t) for t in other_toks]
    prev = [0] * (m + 1)
    choices = []
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        row = bytearray(m + 1)
        si = shown_toks[i - 1]
        for j in range(1, m + 1):
            if si == other_toks[j - 1]:
                cur[j] = prev[j - 1] + 1
                row[j] = 1
            elif prev[j] >= cur[j - 1]:
                cur[j] = prev[j]
                row[j] = 2
            else:
                cur[j] = cur[j - 1]
                row[j] = 3
        choices.append(row)
        prev = cur
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        c = choices[i - 1][j]
        if c == 1:
            i -= 1
            j -= 1
        elif c == 2:
            i -= 1
        else:
            out.append((i, other_toks[j - 1]))
            j -= 1
    while j > 0:
        out.append((0, other_toks[j - 1]))
        j -= 1
    return out


def score_row(prediction, shown, truth):
    """Gap-anchored changed-token F1 for one row, in [0, 1]."""
    s_toks = tokenize(shown)
    b_toks = tokenize(truth)
    p_toks = tokenize(prediction)

    truth_changed = _changed_in_gaps(s_toks, b_toks)
    if not truth_changed:
        # No recorded change: leaving the text alone is correct.
        return 1.0 if p_toks == s_toks else 0.0
    if not p_toks:
        return 0.0

    # What the amendment removed, keyed by WHERE it sat in the shown text.
    changed = Counter(truth_changed)
    # What the prediction proposes, found by the same alignment. Deriving
    # both sides the same way is what lets the true prior wording score
    # exactly 1.0.
    proposed = Counter(_changed_in_gaps(s_toks, p_toks))

    hits = sum(min(c, proposed.get(k, 0)) for k, c in changed.items())
    if hits == 0:
        return 0.0
    recall = hits / sum(changed.values())
    precision = hits / sum(proposed.values())
    return 2 * precision * recall / (precision + recall)


def grade(submission, answers):
    if isinstance(submission, str):
        submission = pd.read_csv(submission)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    for col in ("row_id", "text_before"):
        if col not in submission.columns:
            raise InvalidSubmissionError(
                f"submission missing required column '{col}'")
    for col in ("row_id", "shown_text", "text_before"):
        if col not in answers.columns:
            raise InvalidSubmissionError(
                f"answers missing required column '{col}'")

    # A duplicated row_id is a conflict the grader will not resolve: every row
    # for it is discarded, so it can never score as if correct.
    ids = submission["row_id"].astype(str)
    dup = set(ids[ids.duplicated(keep=False)])
    submitted = {}
    for rid, pred in zip(ids, submission["text_before"]):
        if rid in dup:
            continue
        submitted[rid] = pred

    total, n = 0.0, 0
    for row in answers.itertuples(index=False):
        rid = str(row.row_id)
        n += 1
        pred = submitted.get(rid)
        if pred is None or (not isinstance(pred, str) and pd.isna(pred)):
            continue
        total += score_row(str(pred), str(row.shown_text), str(row.text_before))

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
