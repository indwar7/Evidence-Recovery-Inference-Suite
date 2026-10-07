"""Offline structural validation for submission.csv -- run this before
submitting. Mirrors the grader's own leniency: this only checks structure
(columns, coverage), never correctness, since ground truth is private.

Usage: python validate_submission.py submission.csv test.csv
"""
import sys

import pandas as pd

REQUIRED_COLUMNS = {"row_id", "bag_id", "snippet_id", "group_label"}


def validate(submission_path: str, test_path: str) -> None:
    errors = []
    warnings = []

    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        print(f"ERROR: could not read {submission_path}: {e}")
        sys.exit(1)

    test = pd.read_csv(test_path)

    missing_cols = REQUIRED_COLUMNS - set(sub.columns)
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    if not errors:
        expected_pairs = set(zip(test["bag_id"], test["snippet_id"]))
        got_pairs = set(zip(sub["bag_id"], sub["snippet_id"]))

        missing_pairs = expected_pairs - got_pairs
        extra_pairs = got_pairs - expected_pairs
        dup_pairs = sub.duplicated(subset=["bag_id", "snippet_id"]).sum()

        if missing_pairs:
            warnings.append(
                f"{len(missing_pairs)} (bag_id, snippet_id) pairs from test.csv are "
                f"missing from your submission -- these will score as unrecoverable "
                f"for their bag, not cause a rejection. Example: {list(missing_pairs)[:3]}"
            )
        if extra_pairs:
            warnings.append(
                f"{len(extra_pairs)} rows in your submission don't match any "
                f"(bag_id, snippet_id) in test.csv -- these will be ignored."
            )
        if dup_pairs:
            warnings.append(
                f"{dup_pairs} duplicate (bag_id, snippet_id) rows -- only the first "
                f"occurrence of each will be used."
            )
        null_labels = sub["group_label"].isna().sum() if "group_label" in sub.columns else 0
        if null_labels:
            warnings.append(
                f"{null_labels} rows have a missing group_label -- these will be "
                f"treated as their own unrecoverable singleton group."
            )

    if errors:
        print("FAILED structural validation:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    if warnings:
        print("Structural validation passed, with warnings (these degrade your score, they do not block submission):")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("Structural validation passed: all (bag_id, snippet_id) pairs covered, no duplicates, no nulls.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python validate_submission.py submission.csv test.csv")
        sys.exit(1)
    validate(sys.argv[1], sys.argv[2])
