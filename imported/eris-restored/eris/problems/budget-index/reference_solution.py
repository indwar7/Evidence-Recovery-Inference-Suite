#!/usr/bin/env python3
"""Reference solution for BudgetIndex.

Pipeline (CPU-only, seconds, pandas + stdlib):
1. For each evaluation question, work out WHICH ATTRIBUTE it asks about and
   WHICH COUNTRY it asks about.
2. Find the sentences in that document that state that attribute for that
   country, and pick the true one by reconciling the value each states against
   the shipped reference table.
3. Per document, run a budgeted dynamic program choosing at most budget_k
   non-overlapping windows (each <= 60 tokens) that answer the most questions.
4. Write submission.csv.

Why this is not a regex
-----------------------
A question and the sentence that answers it word the attribute INDEPENDENTLY.
A question may ask for the "number of residents" while the sentence that answers
it records a "resident headcount". There is no lexical overlap to match on. Some
wordings in the evaluation set never appear in the training split at all, so a
lookup table harvested from the labelled data cannot cover them.

This solution therefore maps a wording to an attribute by CONTENT-WORD OVERLAP
against a small set of prototypes: a wording mentioning residents, people or
inhabitants is the population; one mentioning land, territory or expanse is the
area; and so on. That generalises to wordings it has never seen, because unseen
wordings reuse the same content words. Any method with that property works —
token overlap, edit distance, embeddings, or a classifier trained on the
labelled split. The score ranks how well a solution does this, which is the
point of the challenge.

Referring expressions are resolved the same way, structurally rather than by
template: a sentence carrying both a known capital and a known country name is a
capital-to-country link; one carrying a capital and an ISO code is the first hop
of a chain; one carrying an ISO code and a country name resolves that code.
"""

import argparse
import json
import os
import re

import pandas as pd

MAX_CHUNK_TOKENS = 60

# Content words that characterise each attribute. Deliberately small: the point
# is that unseen wordings reuse these stems, not that every wording is listed.
ATTR_PROTOTYPES = {
    "population": {"population", "people", "resident", "residents", "inhabitant",
                   "inhabitants", "headcount", "populace", "living"},
    "area": {"area", "land", "landmass", "territory", "territorial", "expanse",
             "surface", "extent", "size", "km"},
    "num_borders": {"border", "borders", "bordering", "neighbour", "neighbours",
                    "neighbor", "neighbors", "neighbouring", "adjoining",
                    "adjacent", "frontier"},
    "num_languages": {"language", "languages", "tongue", "tongues", "official",
                      "linguistic", "recognised", "recognized", "designated"},
}

RE_WORD = re.compile(r"[a-z]+")
RE_CODE = re.compile(r"\b([A-Z]{3})\b")

RE_MILLION = re.compile(r"([\d,]+(?:\.\d+)?)\s*million", re.I)
RE_THOUSAND = re.compile(r"([\d,]+(?:\.\d+)?)\s*thousand", re.I)
RE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def load_kb(public):
    with open(os.path.join(public, "real_facts.json"), encoding="utf-8") as f:
        rows = json.load(f)
    by_name = {r["name"]: r for r in rows}
    by_code = {r["cca3"]: r for r in rows}
    by_capital = {}
    for r in rows:
        by_capital.setdefault(r["capital"], r)
    return rows, by_name, by_code, by_capital


def stated_value(sentence):
    """The number a sentence denotes, handling scaled and grouped forms."""
    m = RE_MILLION.search(sentence)
    if m:
        return float(m.group(1).replace(",", "")) * 1e6
    m = RE_THOUSAND.search(sentence)
    if m:
        return float(m.group(1).replace(",", "")) * 1e3
    m = RE_NUMBER.search(sentence)
    if m:
        return float(m.group(0).replace(",", ""))
    return None


def attribute_of(text):
    """Which attribute does this wording denote? None if no prototype matches.

    Scored by how many characteristic content words the text shares with each
    prototype. This is what generalises to wordings never seen in training.
    """
    words = set(RE_WORD.findall(text.lower()))
    best, best_hits = None, 0
    for key, proto in ATTR_PROTOTYPES.items():
        hits = len(words & proto)
        if hits > best_hits:
            best, best_hits = key, hits
    return best


def resolve_entity(question, sents, by_name, by_code, by_capital):
    """(country_name, link_sentence_ids) or None.

    Structural, not template-driven: the question names a country, an ISO code,
    or a capital, and any linking sentences are recognised by what they contain
    rather than by matching a fixed phrasing.
    """
    # 1. the question names the country outright
    named = [n for n in by_name if n in question]
    if named:
        return max(named, key=len), []

    # 2. the question carries an ISO code -> find the sentence resolving it
    for code in RE_CODE.findall(question):
        if code not in by_code:
            continue
        for i, s in enumerate(sents):
            if code in s:
                here = [n for n in by_name if n in s]
                if here:
                    return max(here, key=len), [i]
        return None

    # 3. the question carries a capital
    caps = [c for c in by_capital if c in question]
    if not caps:
        return None
    cap = max(caps, key=len)
    # 3a. one hop: a sentence holding the capital and a country name
    for i, s in enumerate(sents):
        if cap in s:
            here = [n for n in by_name if n in s]
            if here:
                return max(here, key=len), [i]
    # 3b. two hops: capital -> ISO code, then code -> country
    for i, s in enumerate(sents):
        if cap not in s:
            continue
        for code in RE_CODE.findall(s):
            if code not in by_code:
                continue
            for j, t in enumerate(sents):
                if j == i or code not in t:
                    continue
                here = [n for n in by_name if n in t]
                if here:
                    return max(here, key=len), [i, j]
    return None


def predict(question, sents, by_name, by_code, by_capital):
    """(required_sent_ids, poison_sent_ids) or None."""
    attr = attribute_of(question)
    if attr is None:
        return None
    res = resolve_entity(question, sents, by_name, by_code, by_capital)
    if res is None:
        return None
    entity, link = res
    row = by_name.get(entity)
    if row is None:
        return None
    true_val = float(row[attr])

    cands = []
    for i, s in enumerate(sents):
        if entity not in s:
            continue
        if attribute_of(s) != attr:
            continue
        v = stated_value(s)
        if v is not None:
            cands.append((abs(v - true_val), i))
    if not cands:
        return None
    cands.sort()
    pick = cands[0][1]
    return sorted(set(link + [pick])), [i for _, i in cands[1:]]


def budget_windows(tok, queries, cap, K):
    """At most K non-overlapping <=cap-token windows maximising answered
    queries, with backtracking."""
    n = len(tok)
    pref = [0]
    for t in tok:
        pref.append(pref[-1] + t)
    qinfo = [(min(r), max(r), set(p)) for r, p in queries]

    def gain(j, i):
        return sum(1 for lo, hi, poi in qinfo
                   if j <= lo and hi < i and not any(j <= p < i for p in poi))

    g = [[0] * (n + 1) for _ in range(K + 1)]
    bk = [[None] * (n + 1) for _ in range(K + 1)]
    for k in range(1, K + 1):
        for i in range(1, n + 1):
            best, arg = g[k][i - 1], ("skip", i - 1)
            if g[k - 1][i] > best:
                best, arg = g[k - 1][i], ("drop", i)
            j = i - 1
            while j >= 0 and pref[i] - pref[j] <= cap:
                v = g[k - 1][j] + gain(j, i)
                if v > best:
                    best, arg = v, ("win", j)
                j -= 1
            g[k][i], bk[k][i] = best, arg
    spans, k, i = [], K, n
    while k > 0 and i > 0:
        kind, j = bk[k][i]
        if kind == "win":
            spans.append((j, i))
            k -= 1
            i = j
        elif kind == "drop":
            k -= 1
        else:
            i = j
    return sorted(spans)


def main(public="prepared/public", out="submission.csv"):
    docs = pd.read_csv(os.path.join(public, "documents.csv"))
    test = pd.read_csv(os.path.join(public, "queries_test.csv"))
    _, by_name, by_code, by_capital = load_kb(public)

    sents_by, tok_by, budget_by = {}, {}, {}
    for did, grp in docs.groupby("doc_id", sort=True):
        grp = grp.sort_values("sent_id")
        sents_by[did] = grp["text"].tolist()
        tok_by[did] = grp["n_tokens"].tolist()
        budget_by[did] = int(grp["budget_k"].iloc[0])

    pred_by = {d: [] for d in sents_by}
    hit = 0
    for r in test.itertuples(index=False):
        p = predict(r.question, sents_by[r.doc_id], by_name, by_code, by_capital)
        if p is not None:
            pred_by[r.doc_id].append(p)
            hit += 1
    print(f"located evidence for {hit}/{len(test)} questions")

    rows = []
    for did in sents_by:
        spans = budget_windows(tok_by[did], pred_by[did], MAX_CHUNK_TOKENS,
                               budget_by[did])
        rows.append({"doc_id": did,
                     "windows": "|".join(f"{a}-{b}" for a, b in spans)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} documents)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default="prepared/public")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    main(a.public, a.out)
