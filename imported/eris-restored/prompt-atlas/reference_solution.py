#!/usr/bin/env python3
"""reference_solution.py -- reference solver for "Prompt Response Atlas".

The whole task reduces to one question: what does the hidden classifier's
response surface look like for an article it has never been shown? The public
training atlas gives that surface exactly for the training articles, so the
job is to learn the map from article text (plus the target category) onto the
surface, and then read every required field off the predicted surface.

That is what this does:

  1. Rebuild each training item's measured 256-value surface from
     train_atlas.csv, normalised to [0, 1] within the item.
  2. Featurise the article with a character and word TF-IDF, concatenated with
     a one-hot of the target category.
  3. Fit one multi-output ridge from features to the 256 normalised values.
  4. For a test item, predict the surface and derive EVERY answer field from
     it -- the chosen pair, the helpful and harmful component sets, the best
     first component -- so the fields stay mutually consistent.
  5. The four fields that describe the classifier rather than the surface
     (default category, top competitor, reachability, order sensitivity,
     peak bin) get their own small logistic models over the same features.

Everything is learned from the public files only. No pretrained weights, no
external data, no calls to any hidden model.

Reads  ./dataset/public/*.csv   (falls back to ./prepared/public/)
Writes ./working/submission.csv
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

SEED = 20260803
TOP_K = 5


def _pub(here):
    for cand in ("dataset/public", "prepared/public", "."):
        p = os.path.join(here, cand)
        if os.path.isfile(os.path.join(p, "test.csv")):
            return p
    raise SystemExit("cannot find public files")


def prob_bin(x):
    if x < 0.05:
        return "very_low"
    if x < 0.15:
        return "low"
    if x < 0.40:
        return "medium"
    return "high"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pub = _pub(here)

    train = pd.read_csv(os.path.join(pub, "train.csv"))
    test = pd.read_csv(os.path.join(pub, "test.csv"))
    atlas = pd.read_csv(os.path.join(pub, "train_atlas.csv"))
    atlas["article_id"] = atlas["article_id"].astype(str)

    pk0 = json.loads(train.loc[0, "packet_json"])
    cats = pk0["categories"]
    comps = [c["component_id"] for c in pk0["components"]]
    pairs = [(a, b) for a in comps for b in comps]
    pair_i = {p: i for i, p in enumerate(pairs)}
    slugs = ["p_" + c.split(",")[0].split(" ")[0].lower().replace("-", "_")
             for c in cats]

    # ---- measured surfaces for the training articles --------------------- #
    surf = {}
    for aid, g in atlas.groupby("article_id"):
        m = np.zeros((len(pairs), len(cats)))
        for r in g.itertuples(index=False):
            m[pair_i[(r.comp_a, r.comp_b)]] = [getattr(r, s) for s in slugs]
        surf[aid] = m

    def unpack(df):
        rows = []
        for r in df.itertuples(index=False):
            pk = json.loads(r.packet_json)
            rows.append({"id": r.id, "article_id": str(r.article_id),
                         "text": pk["headline"] + " \n " + pk["article"],
                         "target": pk["target_category"]})
        return pd.DataFrame(rows)

    tr = unpack(train)
    te = unpack(test)

    # ---- features -------------------------------------------------------- #
    word = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                           sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=40000, sublinear_tf=True, min_df=2)
    Xw = word.fit_transform(tr["text"])
    Xc = char.fit_transform(tr["text"])
    cat_i = {c: i for i, c in enumerate(cats)}

    def onehot(series):
        m = np.zeros((len(series), len(cats)))
        for i, c in enumerate(series):
            m[i, cat_i[c]] = 1.0
        return csr_matrix(m)

    Xtr = hstack([Xw, Xc, onehot(tr["target"])]).tocsr()
    Xte = hstack([word.transform(te["text"]), char.transform(te["text"]),
                  onehot(te["target"])]).tocsr()

    # ---- targets --------------------------------------------------------- #
    Y = np.zeros((len(tr), len(pairs)))
    aux = {k: [] for k in ("default_category", "top_competitor",
                           "target_reachable", "order_sensitive", "peak_bin")}
    for i, r in enumerate(tr.itertuples(index=False)):
        m = surf[r.article_id]
        v = m[:, cat_i[r.target]]
        span = v.max() - v.min()
        Y[i] = (v - v.min()) / span if span > 0 else 0.0

        am = m.argmax(axis=1)
        aux["default_category"].append(cats[int(np.bincount(
            am, minlength=len(cats)).argmax())])
        col = m.mean(axis=0)
        aux["top_competitor"].append(
            cats[[j for j in np.argsort(-col) if j != cat_i[r.target]][0]])
        aux["target_reachable"].append(
            "yes" if (am == cat_i[r.target]).any() else "no")
        gaps = [abs(v[pair_i[(a, b)]] - v[pair_i[(b, a)]])
                for k, a in enumerate(comps) for b in comps[k + 1:]]
        aux["order_sensitive"].append(
            "yes" if span > 0 and np.mean(gaps) / span > 0.10 else "no")
        aux["peak_bin"].append(prob_bin(float(v.max())))

    print(f"{len(tr)} train items, {len(te)} test items, "
          f"{len(surf)} measured surfaces")

    # ---- fit ------------------------------------------------------------- #
    ridge = Ridge(alpha=2.0, random_state=SEED)
    ridge.fit(Xtr, Y)
    Yhat = ridge.predict(Xte)

    heads = {}
    for k, labels in aux.items():
        labels = np.asarray(labels)
        if len(set(labels)) < 2:
            heads[k] = ("const", labels[0])
            continue
        clf = LogisticRegression(max_iter=2000, C=4.0, random_state=SEED)
        clf.fit(Xtr, labels)
        heads[k] = ("model", clf.predict(Xte))

    def head(k, i):
        kind, val = heads[k]
        return val if kind == "const" else val[i]

    # ---- read every field off the predicted surface ---------------------- #
    out = []
    for i, r in enumerate(te.itertuples(index=False)):
        v = Yhat[i]
        best = pairs[int(np.argmax(v))]

        first_mean, all_mean = {}, {}
        for c in comps:
            fi = [pair_i[(c, b)] for b in comps]
            si = [pair_i[(a, c)] for a in comps]
            first_mean[c] = v[fi].mean()
            all_mean[c] = v[sorted(set(fi + si))].mean()
        ranked = sorted(comps, key=lambda c: (-all_mean[c], c))

        rec = {
            "chosen_pair": {"first": best[0], "second": best[1]},
            "best_first_component": min(comps,
                                        key=lambda c: (-first_mean[c], c)),
            "helpful_components": sorted(ranked[:TOP_K]),
            "harmful_components": sorted(ranked[-TOP_K:]),
            "default_category": str(head("default_category", i)),
            "top_competitor": str(head("top_competitor", i)),
            "target_reachable": str(head("target_reachable", i)),
            "order_sensitive": str(head("order_sensitive", i)),
            "peak_probability_bin": str(head("peak_bin", i)),
        }
        out.append({"id": r.id, "answer_json": json.dumps(rec,
                                                          ensure_ascii=False)})

    work = os.path.join(here, "working")
    os.makedirs(work, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(work, "submission.csv"), index=False)
    print(f"wrote {len(out)} rows to working/submission.csv")


if __name__ == "__main__":
    main()
