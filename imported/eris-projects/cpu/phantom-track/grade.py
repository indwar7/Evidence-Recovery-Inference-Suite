"""Grader for Phantom Track — appearance-free identity re-linking.

metric = link accuracy = fraction of (scene_id, a_id) query objects for which the
submitted frame-B index equals the true frame-B index, over all evaluation scenes.
Deterministic; range [0, 1]; higher is better.

Usage:
    python grade.py submission.csv --answers prepared/private/answers.csv
"""
import argparse
import pandas as pd


def _load(x):
    return x if isinstance(x, pd.DataFrame) else pd.read_csv(x)


def grade(submission, answers):
    ans = _load(answers)[["scene_id", "a_id", "b_id"]].copy()
    ans.columns = ["scene_id", "a_id", "true_b"]
    total = len(ans)
    if total == 0:
        return 0.0

    sub = _load(submission)
    need = {"scene_id", "a_id", "b_id"}
    if not need.issubset(sub.columns):
        raise ValueError(f"submission must have columns {sorted(need)}")
    sub = sub[["scene_id", "a_id", "b_id"]].copy()
    # coerce ids to integers where possible; malformed rows drop out and score 0
    for c in ("a_id", "b_id"):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["a_id", "b_id"])
    sub["a_id"] = sub["a_id"].astype(int)
    sub["b_id"] = sub["b_id"].astype(int)
    # last write wins on duplicate (scene_id, a_id)
    sub = sub.drop_duplicates(subset=["scene_id", "a_id"], keep="last")

    merged = ans.merge(sub, on=["scene_id", "a_id"], how="left")
    correct = (merged["b_id"] == merged["true_b"]).sum()
    return float(correct) / float(total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--answers", required=True)
    a = ap.parse_args()
    print(f"{grade(a.submission, a.answers):.6f}")
