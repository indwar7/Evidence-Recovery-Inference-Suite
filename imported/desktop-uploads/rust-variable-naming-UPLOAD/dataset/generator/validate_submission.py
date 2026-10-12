"""Standalone sanity check for a submission.csv before grading.

Usage: python3 validate_submission.py path/to/submission.csv path/to/test.csv
"""
import csv
import sys


def validate(submission_path, test_path):
    with open(test_path) as f:
        test_rows = list(csv.DictReader(f))
    test_ids = {r["id"] for r in test_rows}
    num_vars_by_id = {r["id"]: int(r["num_vars"]) for r in test_rows}

    with open(submission_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "names"]:
            print(f"FAIL: columns must be exactly ['id', 'names'], got {reader.fieldnames}")
            return False
        sub_rows = list(reader)

    sub_ids = [r["id"] for r in sub_rows]
    ok = True
    if len(set(sub_ids)) != len(sub_ids):
        print("WARN: duplicated ids present (they will score 0 at grading time)")
    missing = test_ids - set(sub_ids)
    if missing:
        print(f"WARN: {len(missing)} test ids missing from submission (they will score 0)")
    for r in sub_rows:
        if r["id"] not in num_vars_by_id:
            continue
        expected_n = num_vars_by_id[r["id"]]
        got_n = len(r["names"].split("|")) if r["names"] else 0
        if got_n != expected_n:
            print(f"WARN: id={r['id']} expected {expected_n} names, got {got_n} (will score 0 for this row)")
    print("validation complete" + (" (see warnings above)" if not ok else ""))
    return True


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
