#!/usr/bin/env python3
"""prepare.py -- build the public and private files for "Prompt Response Atlas".

Reads the three raw files shipped with the dataset:

    atlas.csv       measured response of the hidden model to every one of the
                    256 ordered component pairs, for every sampled article
    articles.csv    the sampled news articles (title, text, true category)
    components.csv  the sixteen prompt components

and writes

    prepared/public/train.csv
    prepared/public/test.csv
    prepared/public/train_atlas.csv
    prepared/public/sample_submission.csv
    prepared/private/answers.csv

The split is ARTICLE-DISJOINT: no article contributes rows to both sides, so a
solver cannot memorise an article's response surface and must generalise the
behaviour of the components to text it has never seen.

Deterministic: fixed seed, no randomness beyond it, byte-identical on re-run.

Usage:  python3 prepare.py [--raw dataset/raw] [--out prepared]
"""

import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd

SEED = 20260803
TRAIN_FRACTION = 0.62          # of articles
TARGETS_PER_ARTICLE = 3
TOP_K = 5                      # size of the helpful / harmful component sets

CATEGORIES = [
    "arts, culture, entertainment and media",
    "conflict, war and peace",
    "crime, law and justice",
    "disaster, accident and emergency incident",
    "economy, business and finance",
    "education",
    "environment",
    "health",
    "human interest",
    "labour",
    "lifestyle and leisure",
    "politics",
    "religion and belief",
    "science and technology",
    "society",
    "sport",
    "weather",
]


def slug(c):
    return "p_" + c.split(",")[0].split(" ")[0].lower().replace("-", "_")


SLUGS = [slug(c) for c in CATEGORIES]
CAT_OF_SLUG = dict(zip(SLUGS, CATEGORIES))

PROMPT_TEXT = (
    "A hidden text classifier answers a seventeen-way news category question. "
    "Its prompt is assembled from a catalog of sixteen components, applied as "
    "an ordered pair: the first component, then the second. Given the article "
    "and a target category, describe how that hidden classifier responds "
    "across the 256 possible pairs, and name the pair that pushes the target "
    "category highest."
)

ANSWER_FORMAT = {
    "chosen_pair": {"first": "component id", "second": "component id"},
    "best_first_component": "component id",
    "helpful_components": ["component id", "up to five, unordered"],
    "harmful_components": ["component id", "up to five, unordered"],
    "default_category": "category name",
    "top_competitor": "category name",
    "target_reachable": "yes or no",
    "order_sensitive": "yes or no",
    "peak_probability_bin": "very_low, low, medium or high",
}


# ---- truth derivation ----------------------------------------------------- #

def prob_bin(x):
    if x < 0.05:
        return "very_low"
    if x < 0.15:
        return "low"
    if x < 0.40:
        return "medium"
    return "high"


def derive_answer(mat, pairs, comps, target_slug):
    """mat: (256, 17) probabilities for one article, rows aligned with `pairs`.
    Returns the measured answer record for one (article, target) item."""
    ti = SLUGS.index(target_slug)
    v = mat[:, ti]
    pmin, pmax = float(v.min()), float(v.max())

    idx = {p: i for i, p in enumerate(pairs)}

    # per-component means, split by position and overall
    first_mean, all_mean = {}, {}
    for c in comps:
        as_first = [idx[(c, b)] for b in comps]
        as_second = [idx[(a, c)] for a in comps]
        first_mean[c] = float(v[as_first].mean())
        all_mean[c] = float(v[sorted(set(as_first + as_second))].mean())

    ranked = sorted(comps, key=lambda c: (-all_mean[c], c))
    helpful = sorted(ranked[:TOP_K])
    harmful = sorted(ranked[-TOP_K:])
    best_first = min(comps, key=lambda c: (-first_mean[c], c))

    best_pair = pairs[int(np.argmax(v))]

    # what the classifier says when nobody is steering it
    argmax_cats = mat.argmax(axis=1)
    default_i = int(np.bincount(argmax_cats, minlength=17).argmax())
    default_cat = CATEGORIES[default_i]

    col_mean = mat.mean(axis=0)
    comp_order = [i for i in np.argsort(-col_mean) if i != ti]
    top_competitor = CATEGORIES[comp_order[0]]

    reachable = "yes" if (argmax_cats == ti).any() else "no"

    gaps = [abs(v[idx[(a, b)]] - v[idx[(b, a)]])
            for i, a in enumerate(comps) for b in comps[i + 1:]]
    span = pmax - pmin
    order_sensitive = "yes" if span > 0 and (np.mean(gaps) / span) > 0.10 else "no"

    answer = {
        "chosen_pair": {"first": best_pair[0], "second": best_pair[1]},
        "best_first_component": best_first,
        "helpful_components": helpful,
        "harmful_components": harmful,
        "default_category": default_cat,
        "top_competitor": top_competitor,
        "target_reachable": reachable,
        "order_sensitive": order_sensitive,
        "peak_probability_bin": prob_bin(pmax),
    }
    return answer, v, pmin, pmax


def prepare(raw, public, private):
    """Platform entry point. raw/public/private are directories (str or Path).

    Reads atlas.csv, articles.csv and components.csv from `raw`; writes
    train.csv, test.csv, train_atlas.csv and sample_submission.csv to `public`
    and answers.csv to `private`. Deterministic.
    """
    raw, pub, priv = str(raw), str(public), str(private)

    atlas = pd.read_csv(os.path.join(raw, "atlas.csv"))
    arts = pd.read_csv(os.path.join(raw, "articles.csv"))
    comps_df = pd.read_csv(os.path.join(raw, "components.csv"))
    comps = list(comps_df["component_id"])

    atlas["article_id"] = atlas["article_id"].astype(str)
    arts["article_id"] = arts["article_id"].astype(str)
    keep = set(atlas["article_id"]) & set(arts["article_id"])
    arts = arts[arts["article_id"].isin(keep)].reset_index(drop=True)
    atlas = atlas[atlas["article_id"].isin(keep)]

    pairs = [(x, y) for x in comps for y in comps]
    pair_pos = {p: i for i, p in enumerate(pairs)}

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(arts))
    n_train = int(round(len(arts) * TRAIN_FRACTION))
    split = {}
    for rank, i in enumerate(order):
        split[arts.loc[i, "article_id"]] = "train" if rank < n_train else "test"
    arts["split"] = arts["article_id"].map(split)

    by_article = {}
    for aid, g in atlas.groupby("article_id"):
        m = np.zeros((len(pairs), 17), dtype=np.float64)
        for r in g.itertuples(index=False):
            m[pair_pos[(r.comp_a, r.comp_b)]] = [getattr(r, s) for s in SLUGS]
        by_article[aid] = m

    catalog = [{"component_id": r.component_id, "name": r.name,
                "description": r.description}
               for r in comps_df.itertuples(index=False)]

    rows = {"train": [], "test": []}
    answers = []
    for r in arts.sort_values("article_id").itertuples(index=False):
        mat = by_article[r.article_id]
        default_i = int(np.bincount(mat.argmax(axis=1), minlength=17).argmax())
        # targets are drawn from the categories the classifier does NOT already
        # default to, so every item genuinely requires steering
        pool = [i for i in range(17) if i != default_i]
        # crc32, not hash(): Python string hashing is salted per process and
        # would make the target choice differ between runs
        seed_i = SEED + zlib.crc32(r.article_id.encode())
        pick = np.random.default_rng(seed_i).choice(
            pool, size=TARGETS_PER_ARTICLE, replace=False)
        for k, ti in enumerate(sorted(pick.tolist())):
            item_id = f"atl_{r.article_id}_{k}"
            ans, v, pmin, pmax = derive_answer(
                mat, pairs, comps, SLUGS[ti])
            packet = {
                "headline": r.title,
                "article": r.text,
                "true_category": r.true_category,
                "target_category": CATEGORIES[ti],
                "categories": CATEGORIES,
                "components": catalog,
                "n_pairs": len(pairs),
                "max_set_size": TOP_K,
            }
            row = {
                "id": item_id,
                "article_id": r.article_id,
                "prompt": PROMPT_TEXT,
                "packet_json": json.dumps(packet, ensure_ascii=False),
                "answer_format_json": json.dumps(ANSWER_FORMAT,
                                                 ensure_ascii=False),
                "answer_json": json.dumps(ans, ensure_ascii=False),
            }
            rows[r.split].append(row)
            if r.split == "test":
                arec = {"id": item_id, "p_min": pmin, "p_max": pmax,
                        "answer_json": json.dumps(ans, ensure_ascii=False)}
                for p, i in pair_pos.items():
                    arec[f"{p[0]}_{p[1]}"] = float(v[i])
                answers.append(arec)

    os.makedirs(pub, exist_ok=True)
    os.makedirs(priv, exist_ok=True)

    tr = pd.DataFrame(rows["train"]).sort_values("id").reset_index(drop=True)
    te = pd.DataFrame(rows["test"]).sort_values("id").reset_index(drop=True)
    cols = ["id", "article_id", "prompt", "packet_json", "answer_format_json"]
    tr[cols + ["answer_json"]].to_csv(
        os.path.join(pub, "train.csv"), index=False)
    te[cols].to_csv(os.path.join(pub, "test.csv"), index=False)

    train_ids = set(tr["article_id"])
    atlas[atlas["article_id"].isin(train_ids)].sort_values(
        ["article_id", "comp_a", "comp_b"]).to_csv(
        os.path.join(pub, "train_atlas.csv"), index=False)

    weak = {
        "chosen_pair": {"first": comps[0], "second": comps[0]},
        "best_first_component": comps[0],
        "helpful_components": comps[:TOP_K],
        "harmful_components": comps[-TOP_K:],
        "default_category": "society",
        "top_competitor": "politics",
        "target_reachable": "no",
        "order_sensitive": "no",
        "peak_probability_bin": "low",
    }
    pd.DataFrame({"id": te["id"],
                  "answer_json": json.dumps(weak, ensure_ascii=False)}).to_csv(
        os.path.join(pub, "sample_submission.csv"), index=False)

    pd.DataFrame(answers).sort_values("id").to_csv(
        os.path.join(priv, "answers.csv"), index=False)

    print(f"articles {len(arts)}  train {len(train_ids)}  "
          f"test {len(set(te['article_id']))}")
    print(f"rows: train {len(tr)}  test {len(te)}")
    print(f"train_atlas rows {len(atlas[atlas['article_id'].isin(train_ids)]):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--out", default="prepared")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    raw = a.raw if os.path.isabs(a.raw) else os.path.join(here, a.raw)
    out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    prepare(raw, os.path.join(out, "public"), os.path.join(out, "private"))


if __name__ == "__main__":
    main()
