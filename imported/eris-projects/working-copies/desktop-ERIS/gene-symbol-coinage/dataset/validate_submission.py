#!/usr/bin/env python3
"""Offline structural check for a submission. Needs no answers file.

Usage:
    python validate_submission.py submission.csv [test.csv]

Checks the things that would cost score or void a run, without knowing any
answer:

  - the file parses as CSV and has exactly the columns row_id, symbol
  - every row_id in test.csv is present exactly once
  - no row_id is duplicated (the grader discards every copy of a duplicate)
  - no prediction is blank (a blank scores 0 for that row)
  - predictions look like symbols: short, no whitespace

Warnings do not stop a submission from being graded. Only a missing required
column does.
"""

import csv
import sys
from pathlib import Path

REQUIRED = ["row_id", "symbol"]
MAX_SYMBOL_CHARS = 25


def find_test(explicit=None):
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for p in (here / "test.csv", Path("/data/test.csv"),
              Path("dataset/public/test.csv"), here / "public" / "test.csv"):
        if p.exists():
            return p
    return None


def validate(sub_path, test_path=None):
    errors, warnings = [], []
    with open(sub_path, encoding="utf8", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)

    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        errors.append(f"missing required column(s): {missing}")
        return errors, warnings
    extra = [c for c in cols if c not in REQUIRED]
    if extra:
        warnings.append(f"unexpected column(s), ignored by the grader: {extra}")

    ids = [r["row_id"].strip() for r in rows]
    seen, dups = set(), set()
    for i in ids:
        if i in seen:
            dups.add(i)
        seen.add(i)
    if dups:
        warnings.append(f"{len(dups)} duplicated row_id(s); every copy is "
                        f"discarded and the row scores 0")

    blank = sum(1 for r in rows if not (r["symbol"] or "").strip())
    if blank:
        warnings.append(f"{blank} blank prediction(s); each scores 0")
    spaced = sum(1 for r in rows if len((r["symbol"] or "").split()) > 1)
    if spaced:
        warnings.append(f"{spaced} prediction(s) contain whitespace; it is "
                        f"removed before scoring")
    long_ = sum(1 for r in rows
                if len((r["symbol"] or "").strip()) > MAX_SYMBOL_CHARS)
    if long_:
        warnings.append(f"{long_} prediction(s) exceed {MAX_SYMBOL_CHARS} "
                        f"characters; length beyond the answer lowers the score")

    tp = find_test(test_path)
    if tp is None:
        warnings.append("test.csv not found; row_id coverage not checked")
    else:
        with open(tp, encoding="utf8", newline="") as f:
            want = [r["row_id"].strip() for r in csv.DictReader(f)]
        absent = set(want) - seen
        unknown = seen - set(want)
        if absent:
            warnings.append(f"{len(absent)} row_id(s) from test.csv are "
                            f"missing; each scores 0")
        if unknown:
            warnings.append(f"{len(unknown)} row_id(s) are not in test.csv; "
                            f"ignored by the grader")
    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    errors, warnings = validate(sys.argv[1],
                                sys.argv[2] if len(sys.argv) > 2 else None)
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        return 1
    print("OK" if not warnings else "OK (with warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
