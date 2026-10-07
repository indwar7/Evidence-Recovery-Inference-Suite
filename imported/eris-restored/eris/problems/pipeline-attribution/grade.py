"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-bag Adjusted Rand Index (ARI), clipped to [0, 1].

Why ARI, not raw pairwise F1: an earlier version of this grader used raw
pairwise co-membership F1 (matches a lesson from a prior challenge's review
about avoiding matched-assignment/Hungarian scoring). Measurement caught a
DIFFERENT problem with that choice: raw pairwise F1 structurally rewards
"lump everything into one group" -- it gets perfect recall on every true
same-batch pair for free, and only loses on precision, which for bags with a
handful of groups isn't enough of a penalty. Measured: with the current bag
composition, "everyone in one group" scored 0.25+ under raw pairwise F1 --
comfortably beating a properly trained embedding-based reference. Adjusted
Rand Index specifically corrects for this: it is chance-normalized, so both
degenerate strategies (one giant group, or every snippet its own singleton)
score ~0 by construction, and only genuine above-chance agreement with the
true partition earns credit. Re-measured on the same data: "lump together"
and "all singleton" both hit exactly 0.0 under ARI, while a real trained
encoder scores clearly above both. This still requires no group-label
alignment step (ARI is computed directly from the two label assignments,
same avoid-Hungarian-matching property as pairwise F1 had).

Degradation, not rejection (REJECTION-REGISTER.md 1.13): a bag missing from
the submission scores 0 for that bag rather than voiding the whole run.
Missing individual snippet rows within a bag are each given a unique
placeholder label (so they can't accidentally "help" the score by matching
each other) -- they still count in ARI as effectively wrong.

Slice-safety (REJECTION-REGISTER.md 1.9): only bag_ids present in `answers`
are scored; extra bag_ids in the submission are ignored.
"""
from collections import defaultdict

import pandas as pd
from sklearn.metrics import adjusted_rand_score

REQUIRED_SUBMISSION_COLUMNS = {"row_id", "bag_id", "snippet_id", "group_label"}
REQUIRED_ANSWER_COLUMNS = {"row_id", "bag_id", "snippet_id", "batch_id"}


class InvalidSubmission(ValueError):
    pass


def _bag_ari(true_groups: dict, pred_groups: dict, snippet_ids: list) -> float:
    true_labels = [true_groups[sid] for sid in snippet_ids]
    pred_labels = [
        pred_groups[sid] if sid in pred_groups else f"__missing__{sid}"
        for sid in snippet_ids
    ]
    if len(set(true_labels)) < 2:
        # a bag with a single true group can't be scored discriminatively;
        # shouldn't occur by construction (min 3 groups/bag) but degrade
        # gracefully rather than raise if it ever does
        return 1.0 if len(set(pred_labels)) == 1 else 0.0
    ari = adjusted_rand_score(true_labels, [str(p) for p in pred_labels])
    return max(0.0, float(ari))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {REQUIRED_ANSWER_COLUMNS}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=list(REQUIRED_SUBMISSION_COLUMNS))
    missing_cols = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing_cols:
        raise InvalidSubmission(f"submission missing required columns: {missing_cols}")

    submission = submission.drop_duplicates(subset=["bag_id", "snippet_id"], keep="first")

    true_by_bag = defaultdict(dict)
    for row in answers.itertuples(index=False):
        true_by_bag[row.bag_id][row.snippet_id] = row.batch_id

    pred_by_bag = defaultdict(dict)
    for row in submission.itertuples(index=False):
        label = row.group_label
        if pd.isna(label):
            continue  # treated as missing -> unique placeholder in _bag_ari
        # namespace by bag_id so reusing "0"/"1" across bags can never
        # accidentally create cross-bag effects (ARI is per-bag anyway, but
        # keep this defensive in case grading logic ever changes)
        pred_by_bag[row.bag_id][row.snippet_id] = (row.bag_id, label)

    bag_ids_to_grade = sorted(true_by_bag.keys())
    if not bag_ids_to_grade:
        return 0.0

    scores = []
    for bag_id in bag_ids_to_grade:
        true_groups = true_by_bag[bag_id]
        pred_groups = pred_by_bag.get(bag_id, {})
        snippet_ids = list(true_groups.keys())
        if len(snippet_ids) < 2:
            continue
        scores.append(_bag_ari(true_groups, pred_groups, snippet_ids))

    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import sys

    sub = pd.read_csv(sys.argv[1])
    ans = pd.read_csv(sys.argv[2])
    print(grade(sub, ans))
