#!/usr/bin/env python3
"""Offline structural checker for a submission to "The Vanished Clause".
Needs no answer key — it checks shape, not correctness.

    python validate_submission.py submission.csv --test dataset/public/test.csv

Exit status 0 if the submission is structurally gradable, 1 otherwise.
Warnings do not fail the check: a missing row scores 0 for that row rather
than voiding the submission.
"""

import argparse
import csv
import sys

csv.field_size_limit(10_000_000)

REQUIRED = ["row_id", "text_before"]


def validate(submission_path, test_path=None):
    errors, warnings = [], []

    try:
        with open(submission_path, newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"FAIL: cannot read {submission_path}: {e}")
        return 1

    if not rows:
        print("FAIL: submission has no data rows")
        return 1

    cols = list(rows[0].keys())
    for c in REQUIRED:
        if c not in cols:
            errors.append(f"missing required column '{c}' (found: {cols})")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    ids = [r["row_id"] for r in rows]
    seen, dupes = set(), set()
    for i in ids:
        if i in seen:
            dupes.add(i)
        seen.add(i)
    if dupes:
        errors.append(
            f"{len(dupes)} duplicated row_id value(s), e.g. "
            f"{sorted(dupes)[:3]} — every row for a duplicated id is "
            f"discarded and scores 0")

    blank = sum(1 for r in rows if not (r.get("text_before") or "").strip())
    if blank:
        warnings.append(f"{blank} row(s) have an empty text_before (score 0)")

    if test_path:
        try:
            with open(test_path, newline="", encoding="utf8") as f:
                test_ids = [r["row_id"] for r in csv.DictReader(f)]
        except Exception as e:
            warnings.append(f"could not read test file: {e}")
        else:
            t, s = set(test_ids), set(ids)
            missing, unknown = t - s, s - t
            if missing:
                warnings.append(
                    f"{len(missing)} test row_id(s) absent from the "
                    f"submission (each scores 0)")
            if unknown:
                warnings.append(
                    f"{len(unknown)} row_id(s) not in the test set (ignored)")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"FAIL: {e}")
    if errors:
        return 1
    print(f"OK: {len(rows)} rows, columns {cols}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission", nargs="?", default="submission.csv")
    p.add_argument("--test", default=None)
    a = p.parse_args()
    sys.exit(validate(a.submission, a.test))


if __name__ == "__main__":
    main()
