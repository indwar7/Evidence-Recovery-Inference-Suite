"""Offline structural validator for submission.csv.

Checks the shape of a submission WITHOUT needing the answer key, and
mirrors the grader's own tolerance: only a missing required column is a
hard error; everything else the grader degrades to 0.0 for that unit is
reported here as a warning so you can see it before you submit.

Usage:
    python validate_submission.py submission.csv dataset/public/test.csv
"""
import csv
import sys

RANK_COLS = ["rank_A", "rank_B", "rank_C"]
REQUIRED_COLUMNS = ["unit_id"] + RANK_COLS


def _check_ranks(row):
    """Returns (ok, reason). ok=False means the grader scores this unit
    0.0."""
    vals = []
    for col in RANK_COLS:
        v = row.get(col)
        if v is None or str(v).strip() == "":
            return False, f"{col} is blank"
        try:
            vals.append(int(v))
        except ValueError:
            return False, f"{col}={v!r} is not an integer"
    if sorted(vals) != [0, 1, 2]:
        return False, f"(rank_A, rank_B, rank_C)={tuple(vals)} is not a permutation of (0, 1, 2)"
    return True, None


def main(submission_path, test_path):
    errors, warnings = [], []

    with open(test_path, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    expected_ids = [r["unit_id"] for r in test_rows]

    try:
        with open(submission_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = list(reader)
    except OSError as e:
        print(f"ERROR: could not read {submission_path}: {e}")
        return 1

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        errors.append(f"missing required column(s): {missing_cols} "
                      f"-- the grader rejects this submission outright")

    if not errors:
        ids = [str(r["unit_id"]).strip() for r in rows]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            warnings.append(f"{len(dupes)} unit_id(s) appear more than once "
                            f"(e.g. {dupes[:3]}) -- each scores 0.0; the grader "
                            f"does not pick one of the duplicates")

        missing = [q for q in expected_ids if q not in set(ids)]
        if missing:
            warnings.append(f"{len(missing)} units from test.csv are absent "
                            f"(e.g. {missing[:3]}) -- each scores 0.0")

        extra = sorted(set(ids) - set(expected_ids))
        if extra:
            warnings.append(f"{len(extra)} unit_id(s) are not in test.csv "
                            f"(e.g. {extra[:3]}) -- ignored by the grader")

        malformed = []
        for r in rows:
            uid = r.get("unit_id")
            if uid not in set(expected_ids):
                continue
            ok, reason = _check_ranks(r)
            if not ok:
                malformed.append((uid, reason))
        if malformed:
            q, reason = malformed[0]
            warnings.append(f"{len(malformed)} row(s) have invalid rank_A/rank_B/rank_C "
                            f"(e.g. {q}: {reason}) -- each scores 0.0")

    if errors:
        print("FAILED structural validation:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    if warnings:
        print("Structural validation passed, with warnings "
              "(these cost points, they do not block submission):")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print(f"Structural validation passed: {len(rows)} rows, one per unit, "
              f"every (rank_A, rank_B, rank_C) a valid permutation of (0, 1, 2).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python validate_submission.py submission.csv dataset/public/test.csv")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
