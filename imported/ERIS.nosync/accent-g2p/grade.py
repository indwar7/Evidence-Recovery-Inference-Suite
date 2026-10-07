"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: CHANGE-SEGMENT ACCURACY. For each row, align the source transcription
to the true target transcription and identify the segments that differ (the
accent-specific rewrites). Score the prediction ONLY on those positions:

    per-row score = (# changed segments the prediction gets right)
                    / (# changed segments)

Rows where source == target (11.7% of the shipped test set) have no changed
segments, so the WHOLE prediction is compared against the target: it scores 1
only if it equals the target exactly after normalisation, and 0 otherwise.
Knowing NOT to rewrite is part of the task. The final score is the mean over
rows.

Why not plain edit similarity: ~80% of a transcription survives an accent
change untouched, so edit similarity awards ~0.8 to a model that copies the
source and does nothing. Measured: copy-source scored 0.795 under edit
similarity, leaving a 0.2 band to distinguish every real model. Scoring only
the changed segments sends copy-source to exactly 0 and spends the full [0,1]
range on the one thing the task asks: do you know how this accent rewrites
this sound in this context.

Alignment is Levenshtein on NFC-normalised strings with slashes, stress,
length marks, syllable dots and parentheses removed from all three strings.

answers.csv carries three columns: row_id, phonemes (the target), and
src_phonemes. src_phonemes is NOT a second target and is NOT a
public input: the source transcription is deliberately withheld from
test.csv, which ships only row_id, word, src_accent and tgt_accent. It is
held privately here so the grader is self-contained and can align source to
target when deciding which segments the accent rewrote. The grader never reads any file other
than the two frames it is given, and never changes what it measures based on
the environment.

Degradation, not rejection: missing or blank rows score 0. Only a submission
missing required columns raises. Only row_ids in `answers` are scored.
"""
import unicodedata

import pandas as pd

REQUIRED_SUBMISSION_COLUMNS = {"row_id", "phonemes"}
REQUIRED_ANSWER_COLUMNS = {"row_id", "phonemes", "src_phonemes"}
_STRIP = set("/ˈˌ.ːˑ ()[]")


class InvalidSubmission(ValueError):
    pass


def _norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFC", str(s)) if ch not in _STRIP)


def _align(a, b):
    """Levenshtein traceback -> list of (a_char|'' , b_char|'') ops."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((a[i - 1], "")); i -= 1
        else:
            ops.append(("", b[j - 1])); j -= 1
    return ops[::-1]


def row_score(src, pred, true):
    src, pred, true = _norm(src), _norm(pred), _norm(true)
    if src == true:                         # nothing should change
        return 1.0 if pred == src else 0.0
    st = _align(src, true)
    changed_true_pos = []
    tj = 0
    for a, b in st:
        if b:
            if a != b:
                changed_true_pos.append(tj)
            tj += 1
    if not changed_true_pos:
        return 1.0 if pred == true else 0.0
    pt = _align(pred, true)
    pred_at_true = {}
    tj = 0
    for a, b in pt:
        if b:
            pred_at_true[tj] = a
            tj += 1
    hit = sum(1 for k in changed_true_pos if pred_at_true.get(k, "") == true[k])
    return hit / len(changed_true_pos)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {REQUIRED_ANSWER_COLUMNS}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=sorted(REQUIRED_SUBMISSION_COLUMNS))
    missing = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing:
        raise InvalidSubmission(f"submission missing required columns: {missing}")
    # A duplicated row_id gives two answers for one graded row. Keeping the
    # first would let a self-contradicting submission score as if correct, so
    # every row carrying a duplicated id is discarded; those rows fall through
    # to the "" default below and score 0. The rest is graded normally.
    _ids = submission["row_id"].astype(str)
    _dup = set(_ids[_ids.duplicated()])
    _kept = submission[~_ids.isin(_dup)]
    pred = dict(zip(_kept["row_id"].astype(str), _kept["phonemes"].astype(str)))
    scores = []
    for row in answers.itertuples(index=False):
        p = pred.get(str(row.row_id), "")
        if p in ("nan", "None"):
            p = ""
        scores.append(row_score(row.src_phonemes, p, row.phonemes))
    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import sys
    print(grade(pd.read_csv(sys.argv[1], keep_default_na=False),
                pd.read_csv(sys.argv[2], keep_default_na=False)))
