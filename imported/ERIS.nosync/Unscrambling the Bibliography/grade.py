"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-unit position accuracy. For each test unit, the fraction
of its four reference cards that the submission places at the card's true
reading-order position. Averaged over all test units.

SHAPE. Every shipped unit has EXACTLY 4 references (see prepare.py's
"FIXED POOL SIZE" docstring section). One row per unit in both
submission.csv and answers.csv:
    submission:  unit_id, rank_A, rank_B, rank_C, rank_D
    answers:     unit_id, rank_A, rank_B, rank_C, rank_D
`rank_A` .. `rank_D` are each a plain integer in {0, 1, 2, 3}: the
0-indexed position that label's underlying citation truly occupies in the
paragraph's [CITE] reading order (rank_X = 0 means the citation shown
under label X is the FIRST one actually cited in the paragraph). None of
rank_A..rank_D appear in test.csv (test.csv carries only the shuffled
cards, never their true rank), so there is no shared-column-name leak risk
between test.csv and answers.csv.

THE METRIC, PER UNIT

    score_unit = (number of labels X in {A, B, C, D} with pred rank_X ==
                  true rank_X) / 4

    all four right                 4/4 = 1.00
    two right, other two swapped   2/4 = 0.50
    one right, other three cycled  1/4 = 0.25
    none right                     0/4 = 0.00

Exactly three right is impossible: with a permutation, three fixed cards
force the fourth. A submitted 4-tuple that is not a permutation of
{0, 1, 2, 3} scores 0.0 for that unit. The permutation requirement is
enforced, not assumed: a tuple such as (0, 0, 0, 0) would otherwise
collect credit for one card without committing to the other three.

CHANCE FLOOR. A uniformly random permutation of four cards places each
card at its true position with probability exactly 1/4, so the expected
score of a random guess is exactly 0.25, independent of the data.
Verified by exhaustive enumeration over all 24 permutations: 1 scores
1.0, 6 score 0.5, 8 score 0.25, 9 score 0.0 (the derangements); mean
(1.0 + 3.0 + 2.0) / 24 = 0.25.

REJECTED OUTRIGHT (raises InvalidSubmission, a ValueError): a required
column missing from the submission. Nothing else raises.

DEGRADES, NOT REJECTS, for everything else -- the unit scores 0.0 and the
rest of the submission is still graded:
  - a unit missing from the submission;
  - a unit_id appearing MORE THAN ONCE (all conflicting rows are dropped;
    the grader does not pick one);
  - a (rank_A, rank_B, rank_C, rank_D) tuple that is not a permutation of
    {0, 1, 2, 3} (a non-integer value such as 1.5 -- rejected, never
    truncated -- a value outside 0..3, or a repeated value).

ZERO, NOT THE CHANCE FLOOR, FOR MALFORMED ROWS. A missing, blank, or
malformed row is not a random guess, it is a non-attempt, and it scores
0.0 rather than the 0.25 an honest random guess would earn in
expectation. The gap between those two numbers is the incentive to
attempt every row.

SLICE SAFETY. Only unit_ids present in `answers` are scored; extra
unit_ids in the submission are ignored, so a row-wise public/private split
of answers.csv grades correctly. Every unit is exactly one row in both
test.csv and answers.csv, so a row-wise leaderboard slice cannot fragment
a graded unit.
"""
import pandas as pd

RANK_COLS = ["rank_A", "rank_B", "rank_C", "rank_D"]
N_CARDS = len(RANK_COLS)
VALID_RANKS = list(range(N_CARDS))  # [0, 1, 2, 3]
REQUIRED_SUBMISSION_COLUMNS = {"unit_id", *RANK_COLS}
REQUIRED_ANSWER_COLUMNS = {"unit_id", *RANK_COLS}

MISSING_SCORE = 0.0  # non-attempt penalty; see module docstring


class InvalidSubmission(ValueError):
    pass


def _as_exact_int(v):
    """Return v as an int only if it is exactly integral: 2, "2", 2.0 and
    "2.0" pass; 1.5, "1.5", "2a", True and blanks return None. A
    non-integer rank is rejected (the unit scores 0), never truncated."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f.is_integer() else None


def _parse_ranks(row) -> list:
    """Read (rank_A, rank_B, rank_C, rank_D) off a row-like object and
    validate it's a permutation of {0, 1, 2, 3}. Returns None if malformed
    -- the caller treats None as a non-attempt, scored at MISSING_SCORE."""
    vals = []
    for col in RANK_COLS:
        v = getattr(row, col, None)
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        iv = _as_exact_int(v)
        if iv is None:
            return None
        vals.append(iv)
    if sorted(vals) != VALID_RANKS:
        return None
    return vals


def _position_accuracy(true_ranks: list, pred_ranks: list) -> float:
    """Fraction of the four labelled cards whose predicted position equals
    the true position. Both inputs are validated permutations of
    {0, 1, 2, 3} over the SAME labels (A, B, C, D)."""
    n = len(true_ranks)
    hits = sum(1 for t, p in zip(true_ranks, pred_ranks) if t == p)
    return hits / n if n else 0.0


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if answers is None or not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {sorted(REQUIRED_ANSWER_COLUMNS)}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=sorted(REQUIRED_SUBMISSION_COLUMNS))
    missing = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing:
        raise InvalidSubmission(f"submission missing required columns: {sorted(missing)}")

    sub_ids = submission["unit_id"].astype(str).str.strip()
    dup_ids = set(sub_ids[sub_ids.duplicated()])
    sub_indexed = submission.copy()
    sub_indexed["unit_id"] = sub_ids
    pred_by_unit = {}
    for row in sub_indexed.itertuples(index=False):
        uid = row.unit_id
        if uid in dup_ids:
            continue
        pred_by_unit[uid] = row

    scores = []
    for row in answers.itertuples(index=False):
        uid = str(row.unit_id).strip()
        true_ranks = _parse_ranks(row)
        if true_ranks is None:
            continue  # malformed answer key row; never happens on shipped data

        if uid not in pred_by_unit:
            scores.append(MISSING_SCORE)
            continue
        pred_ranks = _parse_ranks(pred_by_unit[uid])
        if pred_ranks is None:
            scores.append(MISSING_SCORE)
            continue
        scores.append(_position_accuracy(true_ranks, pred_ranks))

    return float(sum(scores) / len(scores)) if scores else MISSING_SCORE


if __name__ == "__main__":
    import sys

    sub = pd.read_csv(sys.argv[1])
    ans = pd.read_csv(sys.argv[2])
    print(grade(sub, ans))
