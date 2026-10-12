#!/usr/bin/env python3
"""Grader for the Annotation Depth Track Prediction challenge.

TASK. Sequence in, sequence out. Given a DNA sequence of length L, emit a
DEPTH TRACK of the same length L: a string of digits where character i says how
many of an annotator's overlapping calls cover base i.

Why a track and not an interval. Tandem Repeats Finder does not report a real
tandem array once. It reports it several times over -- period 2 and period 4,
eight copies and twelve, a trimmed core and a frayed extension -- so its output
over a window is not a span but an integer profile: the core of an array sits at
depth 3-4, its shoulders at depth 2, its frayed edges at depth 1, and the flanks
at 0. Predicting that profile is predicting where an annotator is confident and
where it hedges.

METRIC. AGREEMENT -- the soft (Dice) overlap of the submitted track with the
annotator's:

    AGREEMENT = 2 * sum_s sum_i min( t_true[i], t_pred[i] )
                / sum_s ( sum_i t_true[i] + sum_i t_pred[i] )

over the evaluation sequences s and the positions i of each. Continuous in
[0, 1], higher is better, and 1.0 means every track is exactly right.

There is no matching threshold anywhere in it. Every unit of depth placed on a
position that has it adds credit, and every unit placed anywhere else subtracts
it, so the score moves smoothly with the quality of the prediction rather than
snapping at a cutoff. The incentives are exact: raising t_pred[i] while it is
below t_true[i] always helps, and raising it anywhere else always hurts.

SUBMISSION. Per sequence, either

  * a DEPTH TRACK -- L digits, e.g. "0000111222211100..." (the native form), or
  * an INTERVAL LIST -- pipe-separated half-open "start-end" ranges, which the
    grader stacks into a track. The ranges may overlap, and where k of them
    overlap the stacked track reads k. This is a shorthand for the same object:
    a track that is piecewise constant. It is what the released
    sample_submission.csv uses.

Leniency: a track shorter than L is zero-padded and one longer is truncated; a
malformed, out-of-range or zero-length interval token is skipped; an unusable
cell contributes 0 credit while its annotated weight still counts in the
denominator. At most MAX_INTERVALS ranges are read from an interval list -- past
that a submission is over-firing anyway, which the denominator already punishes.
"""

import json
import sys

import numpy as np
import pandas as pd

MAX_INTERVALS = 16               # ranges read per sequence in the interval form
MAX_DEPTH = 9                    # a track is digits, so depth saturates at 9

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
    pass


def _track_from_digits(s, length):
    """A digit string IS the track. Zero-pad if short, truncate if long."""
    a = np.frombuffer(s.encode(), dtype=np.uint8).astype(np.int32) - 48
    if len(a) < length:
        a = np.concatenate([a, np.zeros(length - len(a), dtype=np.int32)])
    return a[:length]


def _track_from_intervals(s, length):
    """Stack up to MAX_INTERVALS half-open ranges into a track, via a
    difference array. Ranges MAY overlap -- overlap is how the interval form
    expresses depth. Malformed / out-of-range / zero-length tokens are skipped
    leniently."""
    diff = np.zeros(length + 1, dtype=np.int32)
    kept = 0
    for tok in s.split("|"):
        tok = tok.strip()
        if not tok or "-" not in tok:
            continue
        a, _, b = tok.partition("-")
        try:
            a, b = int(a), int(b)
        except ValueError:
            continue
        if not (0 <= a < b <= length):
            continue
        diff[a] += 1
        diff[b] -= 1
        kept += 1
        if kept >= MAX_INTERVALS:
            break
    return np.cumsum(diff[:-1])


def _submitted_track(cell, length):
    """Read one submission cell as a depth track, in either accepted form."""
    if cell is None:
        return np.zeros(length, dtype=np.int32)
    s = "".join(str(cell).split())          # tolerate stray whitespace
    if s == "" or s.lower() == "nan":
        return np.zeros(length, dtype=np.int32)
    if s.isdigit():
        return _track_from_digits(s, length)
    return _track_from_intervals(s, length)


def _true_track(trues_json, length):
    diff = np.zeros(length + 1, dtype=np.int32)
    for t in json.loads(trues_json):
        diff[int(t["s"])] += 1
        diff[int(t["e"])] -= 1
    return np.cumsum(diff[:-1])


def grade(submission, answers):
    """submission, answers: DataFrame or path to CSV. Returns float in [0,1]."""
    if isinstance(submission, str):
        submission = pd.read_csv(submission, dtype=str)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    if "seq_id" not in submission.columns:
        raise InvalidSubmissionError("submission missing 'seq_id' column")
    if "depth_track" in submission.columns:
        col = "depth_track"
    elif "intervals" in submission.columns:
        col = "intervals"
    else:
        raise InvalidSubmissionError(
            "submission missing 'depth_track' column (an 'intervals' column is "
            "also accepted)")
    if submission["seq_id"].duplicated().any():
        raise InvalidSubmissionError("submission has duplicate seq_id rows")

    sub = dict(zip(submission["seq_id"].astype(str), submission[col]))

    num = 0.0
    den = 0.0
    for r in answers.itertuples(index=False):
        length = int(r.length)
        t_true = _true_track(r.trues_json, length)
        t_pred = np.clip(_submitted_track(sub.get(str(r.seq_id), ""), length),
                         0, MAX_DEPTH)

        num += 2.0 * float(np.minimum(t_true, t_pred).sum())
        den += float(t_true.sum()) + float(t_pred.sum())

    if den == 0:
        raise InvalidSubmissionError("no gradable sequences in answers")
    return num / den


# expose bounds as attributes on the grade callable too, and as a helper --- #
grade.minimum = 0.0
grade.maximum = 1.0
grade.min_score = 0.0
grade.max_score = 1.0
grade.direction = "maximize"


def score_bounds():
    """Return the declared (min, max, direction) of the AGREEMENT metric."""
    return {"min": 0.0, "max": 1.0, "direction": "maximize"}


def _cli():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sub = args[0] if len(args) > 0 else "submission.csv"
    ans = None
    for i, a in enumerate(sys.argv):
        if a == "--answers" and i + 1 < len(sys.argv):
            ans = sys.argv[i + 1]
    if ans is None:
        ans = args[1] if len(args) > 1 else "dataset/private/answers.csv"
    print(f"{grade(sub, ans):.6f}")


if __name__ == "__main__":
    _cli()
