"""Offline submission checker for Palimpsest.

Applies exactly the structural checks the grader applies, so you can confirm a
submission is well formed before spending a scoring attempt. It does NOT tell
you your score -- it only says whether the submission is valid.

    python validate_submission.py submission.csv test.csv
"""
import sys
import pandas as pd

ROOT = "ROOT"
REQUIRED = ["item_id", "snapshot_id", "parent_id"]


def validate(sub: pd.DataFrame, test: pd.DataFrame):
    problems = []
    for col in REQUIRED:
        if col not in sub.columns:
            problems.append(f"missing required column '{col}'")
    if problems:
        return problems

    sub = sub[REQUIRED].copy()
    for c in REQUIRED:
        sub[c] = sub[c].astype(str).str.strip()
    test = test.copy()
    for c in ["item_id", "snapshot_id"]:
        test[c] = test[c].astype(str).str.strip()

    if sub.duplicated(["item_id", "snapshot_id"]).any():
        n = int(sub.duplicated(["item_id", "snapshot_id"]).sum())
        problems.append(f"{n} duplicate (item_id, snapshot_id) row(s)")

    need = set(map(tuple, test[["item_id", "snapshot_id"]].values))
    got = set(map(tuple, sub[["item_id", "snapshot_id"]].values))
    if need - got:
        problems.append(f"{len(need - got)} test snapshot(s) missing, "
                        f"e.g. {sorted(need - got)[:3]}")
    if got - need:
        problems.append(f"{len(got - need)} row(s) not in the test set, "
                        f"e.g. {sorted(got - need)[:3]}")

    members = test.groupby("item_id")["snapshot_id"].apply(set).to_dict()
    for item_id, grp in sub.groupby("item_id"):
        allowed = members.get(item_id)
        if allowed is None:
            continue
        parent_of = {}
        for sid, pid in zip(grp["snapshot_id"], grp["parent_id"]):
            parent_of[sid] = pid
            if pid == sid:
                problems.append(f"item {item_id}: '{sid}' is its own parent")
            elif pid != ROOT and pid not in allowed:
                problems.append(f"item {item_id}: parent '{pid}' of '{sid}' is not a "
                                f"snapshot of this item")
        state = {}
        for start in parent_of:
            path, node = [], start
            while True:
                st = state.get(node)
                if st == 1:
                    break
                if st == 0:
                    problems.append(f"item {item_id}: cycle through '{node}'")
                    break
                state[node] = 0
                path.append(node)
                nxt = parent_of.get(node, ROOT)
                if nxt == ROOT:
                    break
                node = nxt
            for n in path:
                state[n] = 1
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    s = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    t = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    issues = validate(s, t)
    if not issues:
        print(f"VALID — {len(s)} rows, {s['item_id'].nunique()} items")
    else:
        print(f"INVALID — {len(issues)} problem(s):")
        for p in issues[:25]:
            print("  -", p)
        raise SystemExit(1)
