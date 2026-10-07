#!/usr/bin/env python3
"""prepare.py - split the telemetry dataset into public/private data for
the "Babel Gauge: Legacy Telemetry Log Canonicalization" task.

Accepts either of the two raw layouts, detected automatically:

  (A) generator output -- a single combined file
        telemetry_lines.csv  line_id, raw_line, canonical, format_family, split
        sites.csv            site,    utc_offset_min
      Every line carries its own 'split' label, so the split is read from
      the data rather than recomputed.

  (B) pre-split upload -- the files already separated
        train.csv            line_id, raw_line, canonical
        test.csv             line_id, raw_line[, canonical]
        sites.csv            site,    utc_offset_min
        answers.csv          line_id, canonical      (optional if test.csv
                                                      carries 'canonical')
        sample_submission.csv                        (optional; rebuilt)

Either way it writes exactly the same thing:

  public/                (given to solvers)
    train.csv              line_id, raw_line, canonical   (60,000 pairs)
    test.csv               line_id, raw_line              (8,000 lines; answer withheld)
    sites.csv              site,    utc_offset_min         (36 sites)
    sample_submission.csv  line_id, output                (format template, constant REJECT)
  private/               (grading key; never shown to solvers)
    answers.csv            line_id, canonical             (8,000 test answers)

Deterministic: byte-identical reruns. Requires only pandas (Kaggle Python
Docker image).

Usage:
    python prepare.py [--raw dataset/raw] [--public prepared/public]
                      [--private prepared/private]
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REJECT = "REJECT"
PUBLIC_TRAIN_COLS = ["line_id", "raw_line", "canonical"]
PUBLIC_TEST_COLS = ["line_id", "raw_line"]


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV as pure strings with NA-detection disabled, so raw log
    lines and canonical records survive exactly: a value that looks like
    'NA', 'null' or an empty corrupted line is never turned into NaN."""
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def _load_raw(raw: Path):
    """Return (train, test, answers) from whichever raw layout is present."""
    combined = raw / "telemetry_lines.csv"
    split_train = raw / "train.csv"

    # ---- layout A: single combined file carrying a 'split' column ----
    if combined.exists():
        lines = _read(combined)
        for col in ("line_id", "raw_line", "canonical", "split"):
            if col not in lines.columns:
                raise ValueError(
                    f"'{combined.name}' is missing the '{col}' column "
                    f"(found: {list(lines.columns)}).")
        train = lines[lines["split"] == "train"]
        test = lines[lines["split"] == "test"]
        if len(train) == 0 or len(test) == 0:
            raise ValueError(
                f"Expected both train and test rows via the 'split' column; "
                f"got {len(train)} train / {len(test)} test.")
        return train, test, test[["line_id", "canonical"]]

    # ---- layout B: files already split ----
    if split_train.exists():
        train = _read(split_train)
        test = _read(raw / "test.csv")
        answers_path = raw / "answers.csv"
        if answers_path.exists():
            answers = _read(answers_path)[["line_id", "canonical"]]
            # keep the answer key aligned to the published test order
            answers = (answers.set_index("line_id")
                              .reindex(test["line_id"])
                              .reset_index())
        elif "canonical" in test.columns:
            answers = test[["line_id", "canonical"]].copy()
        else:
            raise FileNotFoundError(
                "No test answers found: expected answers.csv next to "
                "train.csv, or a 'canonical' column inside test.csv.")
        return train, test, answers

    raise FileNotFoundError(
        f"No recognised raw dataset in '{raw}'. Expected either "
        f"'telemetry_lines.csv' (generator output) or 'train.csv' + "
        f"'test.csv' (pre-split upload).")


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train, test, answers = _load_raw(raw)
    sites = _read(raw / "sites.csv")

    if answers["canonical"].eq("").any() or answers["canonical"].isna().any():
        raise ValueError(
            "The answer key has blank entries; every test line must have a "
            "canonical record or the literal REJECT.")

    # ---- public files (row order preserved from the raw dataset) ----
    train[PUBLIC_TRAIN_COLS].to_csv(public / "train.csv", index=False)
    # never leak the answer: solvers see only line_id + raw_line
    test[PUBLIC_TEST_COLS].to_csv(public / "test.csv", index=False)
    sites.to_csv(public / "sites.csv", index=False)
    # required output format, rebuilt from test ids and filled with REJECT
    sample = pd.DataFrame({"line_id": test["line_id"].values,
                           "output": REJECT})
    sample.to_csv(public / "sample_submission.csv", index=False)

    # ---- private answer key ----
    answers.to_csv(private / "answers.csv", index=False)

    n_reject = (answers["canonical"] == REJECT).mean()
    print(f"public/train.csv              {len(train):,} rows")
    print(f"public/test.csv               {len(test):,} rows")
    print(f"public/sites.csv              {len(sites):,} rows")
    print(f"public/sample_submission.csv  {len(sample):,} rows")
    print(f"private/answers.csv           {len(answers):,} rows "
          f"({n_reject:.2%} REJECT)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    parser.add_argument("--public",
                        default=os.path.join(here, "prepared", "public"))
    parser.add_argument("--private",
                        default=os.path.join(here, "prepared", "private"))
    args = parser.parse_args()
    prepare(Path(args.raw), Path(args.public), Path(args.private))


if __name__ == "__main__":
    main()
