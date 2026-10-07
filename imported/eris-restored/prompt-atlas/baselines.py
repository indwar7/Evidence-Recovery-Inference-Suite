#!/usr/bin/env python3
"""baselines.py -- measure the difficulty band with the official grader.

Runs a ladder of strategies a solver would actually try, from no-skill up to
the reference, and prints each one's score plus a per-term breakdown of the
strongest. Everything is fitted on the public files only.

Usage:  python3 baselines.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "prepared", "public")
PRIV = os.path.join(HERE, "prepared", "private")
sys.path.insert(0, HERE)
from grade import grade, WEIGHTS, _row_score          # noqa: E402

FIELDS = ["default_category", "top_competitor", "target_reachable",
          "order_sensitive", "peak_probability_bin"]


def load():
    train = pd.read_csv(os.path.join(PUB, "train.csv"))
    test = pd.read_csv(os.path.join(PUB, "test.csv"))
    atlas = pd.read_csv(os.path.join(PUB, "train_atlas.csv"))
    atlas["article_id"] = atlas["article_id"].astype(str)
    ans = pd.read_csv(os.path.join(PRIV, "answers.csv"))
    return train, test, atlas, ans


def emit(test, recs, name):
    df = pd.DataFrame({"id": test["id"],
                       "answer_json": [json.dumps(r) for r in recs]})
    p = os.path.join(HERE, "working", f"sub_{name}.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)
    return p


def main():
    train, test, atlas, ans = load()
    pk0 = json.loads(train.loc[0, "packet_json"])
    cats = pk0["categories"]
    comps = [c["component_id"] for c in pk0["components"]]
    pairs = [(a, b) for a in comps for b in comps]
    pair_i = {p: i for i, p in enumerate(pairs)}
    slugs = ["p_" + c.split(",")[0].split(" ")[0].lower().replace("-", "_")
             for c in cats]

    tr_truth = [json.loads(x) for x in train["answer_json"]]
    tr_pk = [json.loads(x) for x in train["packet_json"]]
    te_pk = [json.loads(x) for x in test["packet_json"]]

    # ---- normalised training surfaces ---------------------------------- #
    surf = {}
    for aid, g in atlas.groupby("article_id"):
        m = np.zeros((len(pairs), len(cats)))
        for r in g.itertuples(index=False):
            m[pair_i[(r.comp_a, r.comp_b)]] = [getattr(r, s) for s in slugs]
        surf[aid] = m

    cat_i = {c: i for i, c in enumerate(cats)}
    norm = np.zeros((len(train), len(pairs)))
    for i, (aid, pk) in enumerate(zip(train["article_id"].astype(str), tr_pk)):
        v = surf[aid][:, cat_i[pk["target_category"]]]
        sp = v.max() - v.min()
        norm[i] = (v - v.min()) / sp if sp > 0 else 0.0

    global_best = pairs[int(np.argmax(norm.mean(axis=0)))]

    per_target = {}
    tgt = [pk["target_category"] for pk in tr_pk]
    for c in cats:
        rows = [i for i, t in enumerate(tgt) if t == c]
        if rows:
            per_target[c] = pairs[int(np.argmax(norm[rows].mean(axis=0)))]

    maj = {f: pd.Series([t[f] for t in tr_truth]).mode()[0] for f in FIELDS}
    maj_by_target = {}
    for c in cats:
        rows = [i for i, t in enumerate(tgt) if t == c]
        if rows:
            maj_by_target[c] = {
                f: pd.Series([tr_truth[i][f] for i in rows]).mode()[0]
                for f in FIELDS}

    # component ranking pooled over all training items
    first_rank, all_rank = {}, {}
    for c in comps:
        fi = [pair_i[(c, b)] for b in comps]
        si = [pair_i[(a, c)] for a in comps]
        first_rank[c] = norm[:, fi].mean()
        all_rank[c] = norm[:, sorted(set(fi + si))].mean()
    pooled = sorted(comps, key=lambda c: (-all_rank[c], c))
    pooled_help, pooled_harm = sorted(pooled[:5]), sorted(pooled[-5:])
    pooled_first = min(comps, key=lambda c: (-first_rank[c], c))

    rng = np.random.default_rng(7)
    results = {}

    # 1. sample_submission
    ss = pd.read_csv(os.path.join(PUB, "sample_submission.csv"))
    results["sample_submission"] = grade(ss, ans)

    # 2. random everything
    recs = []
    for _ in range(len(test)):
        p = pairs[int(rng.integers(len(pairs)))]
        recs.append({
            "chosen_pair": {"first": p[0], "second": p[1]},
            "best_first_component": comps[int(rng.integers(16))],
            "helpful_components": sorted(rng.choice(comps, 5, replace=False)
                                         .tolist()),
            "harmful_components": sorted(rng.choice(comps, 5, replace=False)
                                         .tolist()),
            "default_category": cats[int(rng.integers(len(cats)))],
            "top_competitor": cats[int(rng.integers(len(cats)))],
            "target_reachable": ["yes", "no"][int(rng.integers(2))],
            "order_sensitive": ["yes", "no"][int(rng.integers(2))],
            "peak_probability_bin": ["very_low", "low", "medium", "high"][
                int(rng.integers(4))],
        })
    results["random"] = grade(pd.read_csv(emit(test, recs, "random")), ans)

    # 3. one global answer for everything
    recs = []
    for _ in range(len(test)):
        r = {"chosen_pair": {"first": global_best[0], "second": global_best[1]},
             "best_first_component": pooled_first,
             "helpful_components": pooled_help,
             "harmful_components": pooled_harm}
        r.update({f: maj[f] for f in FIELDS})
        recs.append(r)
    results["global_constant"] = grade(
        pd.read_csv(emit(test, recs, "global")), ans)

    # 4. conditioned on the target category only, no article text
    recs = []
    for pk in te_pk:
        t = pk["target_category"]
        p = per_target.get(t, global_best)
        r = {"chosen_pair": {"first": p[0], "second": p[1]},
             "best_first_component": pooled_first,
             "helpful_components": pooled_help,
             "harmful_components": pooled_harm}
        r.update(maj_by_target.get(t, maj))
        recs.append(r)
    results["target_conditioned"] = grade(
        pd.read_csv(emit(test, recs, "target")), ans)

    # 5. the reference solution
    subprocess.run([sys.executable, os.path.join(HERE, "reference_solution.py")],
                   check=True, cwd=HERE)
    ref = pd.read_csv(os.path.join(HERE, "working", "submission.csv"))
    results["reference"] = grade(ref, ans)

    print()
    print(f"{'strategy':24s} score")
    print("-" * 34)
    for k, v in results.items():
        print(f"{k:24s} {v:.4f}")

    print("\nper-term breakdown (reference):")
    lookup = dict(zip(ref["id"].astype(str), ref["answer_json"]))
    agg = {k: [] for k in WEIGHTS}
    for _, row in ans.iterrows():
        raw = lookup.get(str(row["id"]))
        if raw is None:
            continue
        _, parts = _row_score(json.loads(raw), row)
        for k, v in parts.items():
            agg[k].append(v)
    for k in WEIGHTS:
        m = float(np.mean(agg[k])) if agg[k] else 0.0
        print(f"  {k:22s} raw {m:.3f}   weighted {WEIGHTS[k] * m:.4f}")


if __name__ == "__main__":
    main()
