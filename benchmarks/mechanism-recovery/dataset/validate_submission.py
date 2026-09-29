#!/usr/bin/env python3
"""Offline structural checker for a submission to "What Does It Do In There?".
Needs no answer key — it checks shape, not correctness.

    python validate_submission.py submission.csv --test dataset/public/test.csv

Exit 0 if the submission is structurally gradable, 1 otherwise. Warnings do
not fail the check: a missing row scores 0 for that row rather than voiding
the submission.
"""

import argparse
import csv
import sys

csv.field_size_limit(10_000_000)

REQUIRED = ["row_id", "moa_classes"]
SEP = "|"


def parse_set(v):
    return {p.strip() for p in (v or "").split(SEP) if p.strip()}


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
    missing_cols = [c for c in REQUIRED if c not in cols]
    if missing_cols:
        for c in missing_cols:
            print(f"FAIL: missing required column '{c}' (found: {cols})")
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

    blank = sum(1 for r in rows if not parse_set(r.get("moa_classes")))
    if blank:
        warnings.append(f"{blank} row(s) select nothing (score 0)")

    if test_path:
        try:
            with open(test_path, newline="", encoding="utf8") as f:
                test_rows = list(csv.DictReader(f))
        except Exception as e:
            warnings.append(f"could not read test file: {e}")
        else:
            pools = {r["row_id"]: parse_set(r.get("candidates"))
                     for r in test_rows}
            t, s = set(pools), set(ids)
            if t - s:
                warnings.append(
                    f"{len(t - s)} test row_id(s) absent from the submission "
                    f"(each scores 0)")
            if s - t:
                warnings.append(
                    f"{len(s - t)} row_id(s) not in the test set (ignored)")
            off = 0
            for r in rows:
                pool = pools.get(r["row_id"])
                if pool and (parse_set(r.get("moa_classes")) - pool):
                    off += 1
            if off:
                warnings.append(
                    f"{off} row(s) name a class outside that row's own "
                    f"candidate pool (those names are dropped before scoring)")

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
