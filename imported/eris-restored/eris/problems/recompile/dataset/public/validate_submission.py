"""Check a submission offline, the way the grader will read it.

    python validate_submission.py submission.csv test.csv

Two levels of message:

  ERROR    the grader cannot score this at all -- exits 1.
  WARNING  the grader will score it, but the flagged items count as "no amendment
           applied", i.e. zero.  Exits 0.

The grader is deliberately forgiving about structure: an entry that omits an item,
leaves one blank or repeats a row is *scored low*, not thrown out.  So a warning
here is a silent loss of points rather than a failed run, and is still worth
fixing.
"""
import sys
import pandas as pd

MAX_CHARS = 200_000


def main(sub_path, test_path):
    try:
        sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    except Exception as e:                                     # noqa: BLE001
        print(f"ERROR: could not read {sub_path} as CSV: {e}")
        print("  after_text is long prose full of commas and quotation marks;"
              " write the file with pandas or the csv module, never by hand.")
        return 1
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)

    errors, warnings = [], []
    for col in ("item_id", "after_text"):
        if col not in sub.columns:
            errors.append(f"missing column '{col}'; found {list(sub.columns)}")
    if errors:
        print("ERROR:\n  " + "\n  ".join(errors))
        return 1

    if not set(sub["item_id"]) & set(test["item_id"]):
        print("ERROR:\n  no submitted item_id appears in the test set; "
              f"expected ids like {sorted(test['item_id'])[:3]}")
        return 1

    if sub["item_id"].duplicated().any():
        dup = sub.loc[sub["item_id"].duplicated(), "item_id"].unique()[:5]
        warnings.append(f"duplicate item_id(s) {list(dup)} -- the first row for "
                        "an item is used and the rest are ignored")

    missing = set(test["item_id"]) - set(sub["item_id"])
    if missing:
        warnings.append(f"{len(missing)} test item(s) absent, e.g. "
                        f"{sorted(missing)[:5]} -- each scores 0")

    extra = set(sub["item_id"]) - set(test["item_id"])
    if extra:
        warnings.append(f"{len(extra)} item_id(s) not in the test set, e.g. "
                        f"{sorted(extra)[:5]} -- ignored")

    empty = sub.loc[sub["after_text"].str.strip() == "", "item_id"]
    if len(empty):
        warnings.append(f"empty after_text for {len(empty)} item(s), e.g. "
                        f"{list(empty[:3])} -- each scores 0")

    long = sub.loc[sub["after_text"].str.len() > MAX_CHARS, "item_id"]
    if len(long):
        warnings.append(f"after_text over {MAX_CHARS} characters for "
                        f"{len(long)} item(s), e.g. {list(long[:3])} -- truncated")

    if warnings:
        print("WARNING:\n  " + "\n  ".join(warnings))
        print(f"\nScorable: {len(sub)} rows, "
              f"{len(set(sub['item_id']) & set(test['item_id']))}"
              f"/{len(test)} test items covered.")
        return 0
    print(f"OK: {len(sub)} rows, {len(test)} test items covered.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
