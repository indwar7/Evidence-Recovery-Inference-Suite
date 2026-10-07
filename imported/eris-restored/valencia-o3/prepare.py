#!/usr/bin/env python3
"""prepare.py - build the public/split for "The Channel That Lies".

The task is anomaly detection: flag the hours in which the low-cost node's
ozone channel cannot be trusted.

An hour is labelled anomalous when the node's divergence from the co-located
analyser falls in the worst quarter of its own 14-day neighbourhood. The
comparison is local and rank-based, not a fixed global cut, which matters: the
sensor drifts further from truth in cold air, so a single absolute threshold
would mark 10 percent of summer hours and 23 percent of winter hours and the
base rate itself would carry the season. Ranking inside a moving window holds
the base rate at 25 percent across the whole record, so a solver's own
held-out validation stays informative about the test period.

Each window is scored independently and uses only hours inside it, so no test
information reaches a training label.

Raw input (as downloaded from Zenodo record 17044034, hourly aggregation):

    gva_60.csv   regulatory analyser at Moli del Sol, Valencia
    n1_60.csv    low-cost IoT node 1, co-located
    n2_60.csv    low-cost IoT node 2, co-located

The split is chronological, not random: every hour up to and including
2024-11-09 is training, everything after is test. That puts the whole of the
cold season in the test period and none of it in training, which is the
distribution shift the task is built around. Shuffling would destroy it.

Outputs:

    public/train.csv             timestamp + 22 node channels + is_anomalous
    public/test.csv              timestamp + 22 node channels
    public/sample_submission.csv timestamp + score, constant (no skill)
    private/answers.csv          timestamp + is_anomalous for the test hours

Deterministic. pandas only.

Usage:
    python prepare.py [--raw dataset/raw] [--public prepared/public]
                      [--private prepared/private]
"""

import argparse
import os
from pathlib import Path

import pandas as pd

SPLIT_AT = pd.Timestamp("2024-11-10")   # first test hour
TARGET_SRC = "O3"                       # reference analyser column
REF_O3 = "_ref_o3"                      # intermediate, never published
LABEL = "is_anomalous"
WINDOW = "14D"                          # neighbourhood the divergence is ranked in
WORST_FRACTION = 0.25                   # top quarter of each window is anomalous
ID = "timestamp"

# The node module exposes these 11 channels. Both nodes report all of them.
NODE_CHANNELS = ["Temp", "HR", "PM1", "PM2_5", "PM10", "VOC", "CH2O",
                 "CO2", "CO", "O3", "NO2"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[ID] = pd.to_datetime(df["rounded_datetime"], unit="ns")
    return df.drop(columns=["rounded_datetime"]).set_index(ID)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    ref = _load(raw / "gva_60.csv")
    n1 = _load(raw / "n1_60.csv")[NODE_CHANNELS].add_prefix("n1_")
    n2 = _load(raw / "n2_60.csv")[NODE_CHANNELS].add_prefix("n2_")

    if TARGET_SRC not in ref.columns:
        raise ValueError(f"reference file has no '{TARGET_SRC}' column; "
                         f"found {list(ref.columns)}")

    df = (ref[[TARGET_SRC]].rename(columns={TARGET_SRC: REF_O3})
          .join(n1, how="inner").join(n2, how="inner"))

    # The analyser has gaps (calibration cycles, maintenance). An hour with no
    # reference value cannot be scored, so it is dropped from both splits
    # rather than imputed. Node gaps do not occur in this record.
    before = len(df)
    df = df.dropna().sort_index()
    print(f"hours with complete reference + both nodes: {len(df):,} "
          f"({before - len(df):,} dropped for missing reference)")

    # how far the node's own ozone channel sits from the analyser
    df["_err"] = (df["n1_O3"] - df[REF_O3]).abs()

    # rank that divergence within its own 14-day neighbourhood; the worst
    # quarter of each window is anomalous. Local ranking keeps the base rate
    # flat across seasons instead of letting it track the weather.
    rank = df.groupby(pd.Grouper(freq=WINDOW))["_err"].rank(pct=True)
    df[LABEL] = (rank > (1.0 - WORST_FRACTION)).astype(int)

    train = df[df.index < SPLIT_AT].copy()
    test = df[df.index >= SPLIT_AT].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty split at {SPLIT_AT.date()}: "
                         f"{len(train)} train / {len(test)} test")
    print(f"label: worst {WORST_FRACTION:.0%} of divergence within each "
          f"{WINDOW} window")

    feature_cols = list(n1.columns) + list(n2.columns)

    train.reset_index()[[ID] + feature_cols + [LABEL]].to_csv(
        public / "train.csv", index=False)
    # labels are withheld from the test split
    test.reset_index()[[ID] + feature_cols].to_csv(
        public / "test.csv", index=False)

    # a valid-but-no-skill template: the same score for every hour
    sample = pd.DataFrame({ID: test.index, "score": 0.5})
    sample.to_csv(public / "sample_submission.csv", index=False)

    test.reset_index()[[ID, LABEL]].to_csv(private / "answers.csv",
                                           index=False)

    print(f"public/train.csv              {len(train):,} rows  "
          f"{train.index.min().date()} .. {train.index.max().date()}")
    print(f"public/test.csv               {len(test):,} rows  "
          f"{test.index.min().date()} .. {test.index.max().date()}")
    print(f"public/sample_submission.csv  {len(sample):,} rows")
    print(f"private/answers.csv           {len(test):,} rows")
    print(f"anomalous hours: train {train[LABEL].mean():.1%}  "
          f"test {test[LABEL].mean():.1%}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    p.add_argument("--public", default=os.path.join(here, "prepared", "public"))
    p.add_argument("--private",
                   default=os.path.join(here, "prepared", "private"))
    a = p.parse_args()
    prepare(Path(a.raw), Path(a.public), Path(a.private))


if __name__ == "__main__":
    main()
