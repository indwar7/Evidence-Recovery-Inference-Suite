#!/usr/bin/env python3
"""Grader for Repeat-Pattern Span Detection (sequence-to-sequence, continuous IoU).

Domain: sequence-to-sequence. The submission is a DNA SUBSTRING per window;
the grader locates its best occurrence inside the window and scores that
occurrence by overlap ratio (IoU) against the hidden reference span.

GRADING CONTRACT
----------------
    submission.csv : seq_id, prediction
    answers.csv    : seq_id, sequence, target, reference_start, reference_end

`prediction` (submitted) and `target` (held out) are the SAME object in the
SAME form: the DNA substring of the dominant repeated-pattern stretch. That
correspondence is the grading contract -- one submitted column, one answer
column, both DNA substrings over ACGT, compared to each other.

`prediction == target` is a perfect answer and scores exactly 1.0. The span
coordinates (reference_start, reference_end) exist so that a NEAR answer is
not thrown away: rather than scoring string equality 1/0, the grader locates
where the predicted substring actually sits in the window and measures how
tightly its edges agree with the reference span. `sequence` is carried in the
answers file so the grader can do that location step while keeping the
harness signature grade(submission, answers) -- exactly two inputs, no third
file.

Harness signature: grade(submission, answers)  -- exactly two inputs.

For window s with input sequence_s and hidden span
reference_s = [reference_start, reference_end):

    score_s = max over every contiguous occurrence O of prediction_s in
              sequence_s of IoU(O, reference_s)

Empty / non-ACGT / non-substring prediction -> score_s = 0.

    FINAL_SCORE = mean(score_s over all windows),  continuous in [0, 1]

Missing / unknown seq_id -> 0 for that window. Duplicate seq_id -> invalid.
"""

import sys

import pandas as pd

ALPHABET = set("ACGT")


class InvalidSubmissionError(Exception):
    pass


def _iou(a, b):
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _best_occurrence_iou(prediction, sequence, reference):
    if not prediction or not set(prediction) <= ALPHABET:
        return 0.0
    plen = len(prediction)
    if plen > len(sequence):
        return 0.0
    best, start = 0.0, 0
    while True:
        idx = sequence.find(prediction, start)
        if idx == -1:
            break
        best = max(best, _iou((idx, idx + plen), reference))
        start = idx + 1
    return best


def grade(submission, answers):
    if isinstance(submission, str):
        submission = pd.read_csv(submission)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    if "seq_id" not in submission.columns:
        raise InvalidSubmissionError("submission missing 'seq_id' column")
    if "prediction" not in submission.columns:
        raise InvalidSubmissionError("submission missing 'prediction' column")
    if submission["seq_id"].duplicated().any():
        raise InvalidSubmissionError("submission has duplicate seq_id rows")

    # `target` is the held-out answer in the submission's own form; the span
    # coordinates turn an exact/near string match into a continuous score.
    required = {"seq_id", "sequence", "target",
                "reference_start", "reference_end"}
    missing = required - set(answers.columns)
    if missing:
        raise InvalidSubmissionError(
            "answers missing columns: " + ", ".join(sorted(missing))
        )

    submitted = dict(zip(submission["seq_id"].astype(str), submission["prediction"]))

    total, n = 0.0, 0
    for row in answers.itertuples(index=False):
        sid = str(row.seq_id)
        reference = (int(row.reference_start), int(row.reference_end))
        n += 1
        if sid not in submitted:
            continue
        pred = submitted[sid]
        if pred is None or pd.isna(pred):
            continue
        total += _best_occurrence_iou(str(pred), str(row.sequence), reference)

    if n == 0:
        raise InvalidSubmissionError("no gradable windows in answers")
    return total / n


def _cli():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    submission_path = args[0] if len(args) > 0 else "submission.csv"
    answers_path = None
    for i, a in enumerate(sys.argv):
        if a == "--answers" and i + 1 < len(sys.argv):
            answers_path = sys.argv[i + 1]
    if answers_path is None:
        answers_path = args[1] if len(args) > 1 else "private/answers.csv"
    print(f"{grade(submission_path, answers_path):.6f}")


if __name__ == "__main__":
    _cli()
