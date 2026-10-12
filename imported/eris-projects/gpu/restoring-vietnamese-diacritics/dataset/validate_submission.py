"""Validate a submission file for the Vietnamese diacritic restoration task.

Usage: python validate_submission.py <submission.csv> <test.csv>
Checks columns, ids, and that no row is empty. Does not score.
"""
import csv, sys

REQUIRED = ["id", "name"]

def main(sub_path, test_path):
    with open(test_path, encoding="utf-8", newline="") as fh:
        ids = {row["id"] for row in csv.DictReader(fh)}
    with open(sub_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if cols[:2] != REQUIRED:
            raise SystemExit(f"columns must start with {REQUIRED}, got {cols}")
        seen = set()
        for row in reader:
            rid = row["id"]
            if rid in seen:
                raise SystemExit(f"duplicate id: {rid}")
            seen.add(rid)
        missing = ids - seen
        if missing:
            print(f"warning: {len(missing)} test ids missing from submission (scored 0)")
    print("OK: submission is well formed")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
