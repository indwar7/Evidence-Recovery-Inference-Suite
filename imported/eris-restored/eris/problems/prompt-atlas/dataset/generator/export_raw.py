#!/usr/bin/env python3
"""export_raw.py -- turn the generator's output into the three shipped files.

Reads   dataset/raw/atlas.parquet, dataset/raw/atlas_articles.csv, mnds.csv
Writes  dataset/raw/atlas.csv, dataset/raw/articles.csv,
        dataset/raw/components.csv

Only articles with a COMPLETE 256-row surface are exported, so a generation run
that was stopped part way still yields a consistent dataset.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
sys.path.insert(0, HERE)
from components import CATALOG                                # noqa: E402
from render import BODY_CHARS                                 # noqa: E402


def main():
    atlas = pd.read_parquet(os.path.join(RAW, "atlas.parquet"))
    atlas["article_id"] = atlas["article_id"].astype(str)

    counts = atlas.groupby("article_id").size()
    complete = set(counts[counts == 256].index)
    atlas = atlas[atlas["article_id"].isin(complete)]
    print(f"{len(complete)} articles with a complete 256-pair surface")

    arts = pd.read_csv(os.path.join(RAW, "atlas_articles.csv"))
    arts["article_id"] = arts["data_id"].astype(str)
    # a resumed run can list an article twice; keep one row per article
    arts = arts.drop_duplicates(subset="article_id", keep="first")
    arts = arts[arts["article_id"].isin(complete)].copy()

    meta = pd.read_csv(os.path.join(RAW, "mnds.csv"),
                       usecols=["data_id", "source", "date"])
    meta["article_id"] = meta["data_id"].astype(str)
    # data_id repeats in the corpus; one metadata row per article
    meta = meta.drop_duplicates(subset="article_id", keep="first")
    arts = arts.merge(meta[["article_id", "source", "date"]],
                      on="article_id", how="left")

    out = pd.DataFrame({
        "article_id": arts["article_id"],
        "title": arts["title"],
        # exactly the text the classifier was shown
        "text": arts["content"].str.slice(0, BODY_CHARS).str.strip(),
        "true_category": arts["category_level_1"],
        "source": arts["source"],
        "date": arts["date"],
    }).sort_values("article_id")

    comps = pd.DataFrame(CATALOG, columns=["component_id", "name",
                                           "description"])

    atlas.sort_values(["article_id", "comp_a", "comp_b"]).to_csv(
        os.path.join(RAW, "atlas.csv"), index=False, float_format="%.6g")
    out.to_csv(os.path.join(RAW, "articles.csv"), index=False)
    comps.to_csv(os.path.join(RAW, "components.csv"), index=False)
    print(f"atlas.csv {len(atlas):,} rows   articles.csv {len(out)} rows")


if __name__ == "__main__":
    main()
