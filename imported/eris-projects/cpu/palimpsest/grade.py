"""Palimpsest grader.

Metric: immediate-parent accuracy. For every snapshot in the hidden test items,
the submission names the snapshot it was edited FROM (or ROOT if it is the
earliest surviving revision of its document). The score is the fraction named
correctly, micro-averaged over all test snapshots.

A submission must describe a genuine forest: parents stay inside the item, no
snapshot is its own parent, and no cycles. These are format rules, checked
before scoring, and `validate_submission.py` in public/ applies exactly the
same checks so a solver can self-check offline.
"""
import sys
import pandas as pd

ROOT = "ROOT"
REQUIRED = ["item_id", "snapshot_id", "parent_id"]


class InvalidSubmission(ValueError):
    pass


def _forest_check(parent_of, item_id):
    """Raise if the parent map for one item is not a forest."""
    seen_state = {}                       # 0 = visiting, 1 = done
    for start in parent_of:
        path = []
        node = start
        while True:
            state = seen_state.get(node)
            if state == 1:
                break
            if state == 0:
                cycle = " -> ".join(path[path.index(node):] + [node])
                raise InvalidSubmission(
                    f"item {item_id}: parent pointers form a cycle ({cycle}). "
                    "Submissions must describe a forest.")
            seen_state[node] = 0
            path.append(node)
            nxt = parent_of.get(node, ROOT)
            if nxt == ROOT:
                break
            node = nxt
        for n in path:
            seen_state[n] = 1


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Score a submission against ground truth. Returns parent accuracy in [0,1]."""
    for col in REQUIRED:
        if col not in submission.columns:
            raise InvalidSubmission(f"submission is missing required column '{col}'")

    sub = submission[REQUIRED].copy()
    for c in REQUIRED:
        sub[c] = sub[c].astype(str).str.strip()
    ans = answers[REQUIRED].copy()
    for c in REQUIRED:
        ans[c] = ans[c].astype(str).str.strip()

    if sub.duplicated(["item_id", "snapshot_id"]).any():
        dup = sub[sub.duplicated(["item_id", "snapshot_id"], keep=False)].head(3)
        raise InvalidSubmission(f"duplicate (item_id, snapshot_id) rows, e.g.\n{dup}")

    need = set(map(tuple, ans[["item_id", "snapshot_id"]].values))
    got = set(map(tuple, sub[["item_id", "snapshot_id"]].values))
    missing = need - got
    if missing:
        ex = sorted(missing)[:3]
        raise InvalidSubmission(
            f"submission is missing {len(missing)} required snapshot(s), e.g. {ex}")

    # Rows beyond the answer key are NOT an error: the platform scores public and
    # private portions separately, passing a subset of the answers each time
    # while the submission legitimately covers the whole test set. Grading is
    # restricted to the rows the answer key actually contains.

    # parent must be ROOT or a snapshot of the SAME item; no self-parents.
    # Membership is the union of the ids seen in the submission and the answer
    # key for that item, so a parent pointing at a snapshot that happens to sit
    # in the other scoring portion is still legal.
    members = ans.groupby("item_id")["snapshot_id"].apply(set).to_dict()
    sub_members = sub.groupby("item_id")["snapshot_id"].apply(set).to_dict()
    for item_id, grp in sub.groupby("item_id"):
        if item_id not in members:
            continue                      # item not in this scoring portion
        allowed = members[item_id] | sub_members.get(item_id, set())
        for sid, pid in zip(grp["snapshot_id"], grp["parent_id"]):
            if pid == sid:
                raise InvalidSubmission(f"item {item_id}: {sid} is its own parent")
            if pid != ROOT and pid not in allowed:
                raise InvalidSubmission(
                    f"item {item_id}: parent '{pid}' of '{sid}' is not a snapshot "
                    f"of this item (use a snapshot_id from this item, or 'ROOT')")
        _forest_check(dict(zip(grp["snapshot_id"], grp["parent_id"])), item_id)

    merged = ans.merge(sub, on=["item_id", "snapshot_id"], how="left",
                       suffixes=("_true", "_pred"))
    if len(merged) != len(ans):
        raise InvalidSubmission("submission rows could not be aligned with the answer key")
    return float((merged["parent_id_true"] == merged["parent_id_pred"]).mean())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python grade.py submission.csv answers.csv")
        raise SystemExit(2)
    s = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    a = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    print(f"{grade(s, a):.6f}")
