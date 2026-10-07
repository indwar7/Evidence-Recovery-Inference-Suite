"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-unit rescaled Kendall's tau rank correlation between the
submitted reference order and the true citation order, averaged over all
test units.

SHAPE. Every shipped unit has EXACTLY 3 references (see prepare.py's
"FIXED POOL SIZE" docstring section). One row per unit in both
submission.csv and answers.csv:
    submission:  unit_id, rank_A, rank_B, rank_C
    answers:     unit_id, rank_A, rank_B, rank_C
`rank_A`/`rank_B`/`rank_C` are each a plain integer in {0, 1, 2}: the
0-indexed position that label's underlying citation truly occupies in the
paragraph's [CITE] reading order (rank_X = 0 means the citation shown
under label X is the FIRST one actually cited in the paragraph). This
replaces an earlier one-column, space-separated-string encoding
(`predicted_order` / `true_order`, e.g. "C A B") that the platform's
Target Recoverability check rejected ("the submission and answer columns
do not identify a supported target" -- see failures.md). Three
independent integer columns carry the identical information the string
did, but each individually is a supported scalar target type. None of
rank_A/B/C appear in test.csv (test.csv carries only the shuffled cards,
never their true rank), so there is no shared-column-name leak risk
between test.csv and answers.csv to guard against here.

WHY RESCALED KENDALL'S TAU, NOT EXACT-MATCH OR PER-COLUMN ACCURACY.
- Exact-match-on-the-whole-triple is all-or-nothing: a prediction that
  swaps only two of three ranks is scored identically to one that reverses
  all three. Kendall's tau counts concordant vs discordant PAIRS, so a
  near-miss still earns most of its credit.
- Per-column accuracy (treating rank_A/B/C as three independent
  classification targets) breaks the constraint that ranks must form a
  permutation of {0,1,2} and does not penalise a submission that ignores
  that constraint (e.g. predicting rank_A=rank_B=0); pairwise concordance
  is computed over ALL THREE columns jointly and is invariant to how the
  answer happens to be split across columns.
- A random permutation of n items has an EXACT, n-independent expected
  rescaled tau of 0.5 (verified by exhaustive enumeration for n=3..8, see
  DESIGN.md), so a "did not read the paragraph, just guessed" baseline is
  centered at a known, fixed floor -- the same reasoning the task-design
  guide gives for using a rank-correlation metric on any ordering-recovery
  task.

REJECTED OUTRIGHT (raises InvalidSubmission, a ValueError): a required
column missing from the submission. Nothing else raises.

DEGRADES, NOT REJECTS, for everything else -- the unit scores 0.0 and the
rest of the submission is still graded:
  - a unit missing from the submission;
  - a unit_id appearing MORE THAN ONCE (all conflicting rows are dropped;
    the grader does not pick one);
  - a (rank_A, rank_B, rank_C) triple that is not a permutation of
    {0, 1, 2} (a non-integer value such as 1.5 -- rejected, never
    truncated -- a value outside 0..2, or a repeated value).

ZERO, NOT THE METRIC'S OWN CHANCE FLOOR, FOR MALFORMED ROWS. A random
permutation of the true ranks scores 0.5 in EXPECTATION under rescaled
Kendall's tau (verified by exhaustive enumeration for n=3..8, see
DESIGN.md) -- but a missing, blank, or malformed row is not a random
guess, it is a non-attempt, and the standard degrade rule used throughout
this account's challenges scores a non-attempt at min_score (0.0), not at
the metric's own informed-guess floor. Concretely: submitting nothing
scores 0.000; submitting an honest random permutation for every unit
scores ~0.5 in expectation. The gap between those two numbers is the
incentive to attempt the task at all.

SLICE SAFETY. Only unit_ids present in `answers` are scored; extra
unit_ids in the submission are ignored, so a row-wise public/private split
of answers.csv grades correctly. Every unit is exactly one row in both
test.csv and answers.csv, so a row-wise leaderboard slice cannot fragment
a graded unit.
"""
import pandas as pd

RANK_COLS = ["rank_A", "rank_B", "rank_C"]
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
    """Read (rank_A, rank_B, rank_C) off a row-like object and validate
    it's a permutation of {0, 1, 2}. Returns None if malformed -- the
    caller treats None as a non-attempt, scored at MISSING_SCORE."""
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
    if sorted(vals) != [0, 1, 2]:
        return None
    return vals


def _rescaled_kendall_tau(true_ranks: list, pred_ranks: list) -> float:
    """Rescaled Kendall's tau-b between two rank triples over the SAME 3
    labels (A, B, C), computed directly over pairwise concordance so no
    external stats dependency is required."""
    n = len(true_ranks)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_sign = true_ranks[i] - true_ranks[j]
            pred_sign = pred_ranks[i] - pred_ranks[j]
            if true_sign * pred_sign > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else 0.0
    return (tau + 1.0) / 2.0


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
        scores.append(_rescaled_kendall_tau(true_ranks, pred_ranks))

    return float(sum(scores) / len(scores)) if scores else MISSING_SCORE


if __name__ == "__main__":
    import sys

    sub = pd.read_csv(sys.argv[1])
    ans = pd.read_csv(sys.argv[2])
    print(grade(sub, ans))
