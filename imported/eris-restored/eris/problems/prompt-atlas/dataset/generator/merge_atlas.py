#!/usr/bin/env python3
"""merge_atlas.py -- join the MPS and CPU workers' output into one atlas.

The two workers walk the same article list from opposite ends, so their outputs
are disjoint except where they met. Articles are kept only if a worker recorded
the complete 256-pair surface; a partly-measured article is dropped rather than
shipped with holes.

Writes dataset/raw/atlas.parquet (the merged atlas) and the matching
dataset/raw/atlas_articles.csv, then export_raw.py turns those into the three
shipped CSVs.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
sys.path.insert(0, HERE)
import gen_atlas as G                                          # noqa: E402


def load(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    df["article_id"] = df["article_id"].astype(str)
    n = df.groupby("article_id").size()
    return df[df["article_id"].isin(set(n[n == 256].index))]


def main():
    mps = load(os.path.join(RAW, "atlas.parquet"))
    cpu = load(os.path.join(RAW, "atlas_cpu.parquet"))
    parts = [p for p in (mps, cpu) if p is not None and len(p)]
    if not parts:
        sys.exit("no worker output found")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["article_id", "comp_a", "comp_b"],
                                    keep="first")
    ids = sorted(merged["article_id"].unique())
    print(f"mps {0 if mps is None else mps['article_id'].nunique()} + "
          f"cpu {0 if cpu is None else cpu['article_id'].nunique()} "
          f"-> {len(ids)} complete articles")

    arts = G.sample_articles(119)
    arts["article_id"] = arts["data_id"].astype(str)
    arts = arts[arts["article_id"].isin(set(ids))]

    merged.sort_values(["article_id", "comp_a", "comp_b"]).to_parquet(
        os.path.join(RAW, "atlas.parquet"), index=False)
    arts.drop(columns=["article_id"]).to_csv(
        os.path.join(RAW, "atlas_articles.csv"), index=False)
    print(f"merged atlas: {len(merged):,} rows, {len(arts)} articles")


if __name__ == "__main__":
    main()
