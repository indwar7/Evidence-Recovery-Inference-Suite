#!/usr/bin/env python3
"""prepare.py - build the public/private split for "Say It In English".

Raw input: the Research Organization Registry (ROR) data dump, as published on
Zenodo. One JSON array, one record per organization. Each record carries a
`names` list in which every entry is tagged `ror_display`, `label`, `acronym`
or `alias`, and labels carry a language code.

The task pairs each non-English label with that organization's English display
name. Both sides are registry content: nothing is generated, translated or
altered here.

The split is by ORGANIZATION, not by pair. Every label belonging to one
organization lands on the same side, so a solver can never see an
organization's English name in training and be asked for it again at test
time. Without that, the task would reduce to memorisation.

Outputs:

    public/train.csv             pair_id, source_name, lang, target_name
    public/test.csv              pair_id, source_name, lang
    public/sample_submission.csv pair_id, output (source copied through)
    private/answers.csv          pair_id, target_name

Deterministic: the organization shuffle uses a fixed seed. pandas only.

Usage:
    python prepare.py [--raw dataset/raw] [--public prepared/public]
                      [--private prepared/private]
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

SEED = 20260801
TRAIN_FRACTION = 0.80
DISPLAY = "ror_display"
LABEL = "label"


def _extract(raw: Path):
    """Yield (org_id, source_name, lang, target_name) for every non-English
    label whose organization also has an English display name."""
    files = sorted(raw.glob("*ror-data.json")) or sorted(raw.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No ROR JSON dump found in '{raw}'. Expected the file published "
            f"in the Zenodo record, e.g. v2.0-2025-12-16-ror-data.json")
    records = json.loads(files[0].read_text())

    out = []
    for rec in records:
        names = rec.get("names", [])
        display = next((n["value"] for n in names
                        if DISPLAY in n.get("types", [])), None)
        if not display:
            continue
        for n in names:
            if LABEL not in n.get("types", []):
                continue
            lang = n.get("lang")
            if lang in (None, "en"):
                continue
            out.append((rec["id"], n["value"], lang, display))
    return out


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rows = _extract(raw)
    print(f"pairs extracted: {len(rows):,}")
    identical = sum(1 for _, s, _, t in rows if s == t)
    print(f"  of which source == target: {identical:,} "
          f"({identical / len(rows):.1%})")

    orgs = sorted({o for o, *_ in rows})
    random.Random(SEED).shuffle(orgs)
    train_orgs = set(orgs[:int(len(orgs) * TRAIN_FRACTION)])
    print(f"organizations: {len(orgs):,} "
          f"({len(train_orgs):,} train / {len(orgs) - len(train_orgs):,} test)")

    # stable, split-independent ids: sorted by (org, label) before numbering
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    df = pd.DataFrame(rows, columns=["org_id", "source_name", "lang",
                                     "target_name"])
    df.insert(0, "pair_id", [f"P{i + 1:06d}" for i in range(len(df))])

    train = df[df.org_id.isin(train_orgs)]
    test = df[~df.org_id.isin(train_orgs)]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty split: {len(train)} train / {len(test)} test")

    train[["pair_id", "source_name", "lang", "target_name"]].to_csv(
        public / "train.csv", index=False)
    # the English name is withheld from the test split
    test[["pair_id", "source_name", "lang"]].to_csv(
        public / "test.csv", index=False)

    # a valid-but-unskilled template: hand the source straight back
    pd.DataFrame({"pair_id": test.pair_id,
                  "output": test.source_name}).to_csv(
        public / "sample_submission.csv", index=False)

    test[["pair_id", "target_name"]].to_csv(private / "answers.csv",
                                            index=False)

    print(f"public/train.csv              {len(train):,} rows")
    print(f"public/test.csv               {len(test):,} rows")
    print(f"public/sample_submission.csv  {len(test):,} rows")
    print(f"private/answers.csv           {len(test):,} rows")
    print(f"languages in test: {test.lang.nunique()}  "
          f"top: {', '.join(test.lang.value_counts().head(6).index)}")


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
