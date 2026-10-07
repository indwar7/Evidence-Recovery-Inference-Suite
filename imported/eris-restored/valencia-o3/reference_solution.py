#!/usr/bin/env python3
"""reference_solution.py - reference solver for "The Channel That Lies".

Flags the hours in which the low-cost node's ozone channel cannot be trusted.

A gradient-boosted classifier over every channel from both nodes. The choice
is reachable from the training data alone: a chronological hold-out ranks it
first, and it also ranks first on the real test period.

The thing to discover is which inputs carry the signal. The node's own ozone
channel - the very reading whose reliability is in question - is almost
worthless as a suspicion score (0.26 against a 0.25 base rate), and so is
disagreement between the two nodes. What predicts unreliability is the
surrounding context: humidity, temperature and the co-pollutant channels that
drive the cell's cross-sensitivity.

Reads  ./dataset/public/{train,test}.csv
Writes ./working/submission.csv

Deterministic (fixed seed); pandas, numpy and scikit-learn only.
"""

import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 20260801
ID = "timestamp"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pub = os.path.join(here, "dataset", "public")
    if not os.path.isdir(pub):
        pub = os.path.join(here, "prepared", "public")

    train = pd.read_csv(os.path.join(pub, "train.csv"), parse_dates=[ID])
    test = pd.read_csv(os.path.join(pub, "test.csv"), parse_dates=[ID])

    feats = [c for c in test.columns if c.startswith(("n1_", "n2_"))]
    print(f"{len(feats)} node channels, {len(train):,} training hours "
          f"({train.is_anomalous.mean():.1%} labelled anomalous)")

    model = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    model.fit(train[feats], train["is_anomalous"])
    score = model.predict_proba(test[feats])[:, 1]

    work = os.path.join(here, "working")
    os.makedirs(work, exist_ok=True)
    out = pd.DataFrame({ID: test[ID], "score": score})
    out.to_csv(os.path.join(work, "submission.csv"), index=False)
    print(f"wrote {len(out):,} rows to working/submission.csv")


if __name__ == "__main__":
    main()
