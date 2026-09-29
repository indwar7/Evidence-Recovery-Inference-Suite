#!/usr/bin/env python3
"""Reference solution — The Vanished Clause.

CPU-only, deterministic, no external data. This is the honest floor a solver
should beat, not a strong system.

THE APPROACH
------------
An amendment usually rewrites one passage of a long section and leaves the
rest untouched, so the task decomposes into two questions:

  1. WHICH passage did the amendment touch?
  2. WHAT did that passage say before?

This reference answers (1) with a supervised scorer over sentences, and (2)
with a substitution table mined from the training pairs.

Step 1 — mine an edit lexicon from train.
    For every training pair, align `text_after` to `text_before` and read off
    the contiguous runs that differ. Each run gives a substitution
        (after_phrase, before_phrase)
    in a left/right token context. Substitutions seen more than once are kept
    with their frequency; this is the model's entire knowledge of "what
    amendments do to this kind of language".

Step 2 — score each test sentence for how likely it is to have been amended.
    A logistic regression over character n-grams, trained on train sentences
    labelled by whether the amendment actually touched them. This is what
    locates the passage.

Step 3 — apply the highest-confidence substitution to the top-scoring
    sentence, and leave everything else exactly as shown.

Leaving the rest alone is not laziness: the metric's precision term punishes
every token a prediction adds that the amendment did not remove, so rewriting
untouched text costs score. The reference therefore makes one careful edit.

Usage:
    python reference_solution.py                      # uses dataset/public
    python reference_solution.py --data <dir> --out submission.csv
"""

import argparse
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 20260925
TOKEN = re.compile(r"\w+|[^\w\s]")
SENT = re.compile(r"(?<=[.;:])\s+")

MIN_SUB_COUNT = 2        # a phrase substitution must recur to be trusted
MIN_TOKEN_SUB_COUNT = 3  # a token swap must recur more, being far more common
MAX_SUB_TOKENS = 12      # ignore whole-paragraph rewrites


def tokenize(t):
    return TOKEN.findall(t or "")


def align_runs(a_toks, b_toks):
    """Contiguous (after_run, before_run) differences via an LCS backtrace."""
    n, m = len(a_toks), len(b_toks)
    if n == 0 or m == 0:
        return []
    prev = [0] * (m + 1)
    choices = []
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        row = bytearray(m + 1)
        ai = a_toks[i - 1]
        for j in range(1, m + 1):
            if ai == b_toks[j - 1]:
                cur[j] = prev[j - 1] + 1
                row[j] = 1
            elif prev[j] >= cur[j - 1]:
                cur[j] = prev[j]
                row[j] = 2
            else:
                cur[j] = cur[j - 1]
                row[j] = 3
        choices.append(row)
        prev = cur

    ops = []
    i, j = n, m
    while i > 0 and j > 0:
        c = choices[i - 1][j]
        if c == 1:
            ops.append(("=", a_toks[i - 1], b_toks[j - 1]))
            i -= 1
            j -= 1
        elif c == 2:
            ops.append(("-", a_toks[i - 1], None))
            i -= 1
        else:
            ops.append(("+", None, b_toks[j - 1]))
            j -= 1
    while i > 0:
        ops.append(("-", a_toks[i - 1], None))
        i -= 1
    while j > 0:
        ops.append(("+", None, b_toks[j - 1]))
        j -= 1
    ops.reverse()

    runs, cur_a, cur_b = [], [], []
    for op, av, bv in ops:
        if op == "=":
            if cur_a or cur_b:
                runs.append((cur_a, cur_b))
                cur_a, cur_b = [], []
        elif op == "-":
            cur_a.append(av)
        else:
            cur_b.append(bv)
    if cur_a or cur_b:
        runs.append((cur_a, cur_b))
    return runs


def mine_substitutions(train):
    """Mine substitutions at TWO granularities.

    Whole-phrase substitutions are precise but barely transfer: the split is
    part-disjoint, and an exact phrase mined from one CFR part rarely occurs
    verbatim in another. Measured on the shipped split, a phrase-only table
    scored 0.000 on test while clearly fitting train.

    So single-token substitutions are mined alongside them. A token swap
    ("shall" -> "must", "Division" -> "Administration", a year, a threshold)
    is the unit that actually recurs across parts, and it is what gives the
    reference any transfer at all.

    Returns (phrase_table, token_table); the phrase table is tried first
    because when it does fire it is far more specific.
    """
    phrases = defaultdict(Counter)
    tokens = defaultdict(Counter)
    for after, before in zip(train["text_after"], train["text_before"]):
        for a_run, b_run in align_runs(tokenize(after), tokenize(before)):
            if not a_run or not b_run:
                continue
            if len(a_run) > MAX_SUB_TOKENS or len(b_run) > MAX_SUB_TOKENS:
                continue
            phrases[" ".join(a_run)][" ".join(b_run)] += 1
            if len(a_run) == 1 and len(b_run) == 1:
                tokens[a_run[0]][b_run[0]] += 1

    def collapse(d, min_count):
        out = {}
        for k, counter in d.items():
            v, n = counter.most_common(1)[0]
            if n >= min_count and k != v:
                out[k] = (v, n)
        return out

    return (collapse(phrases, MIN_SUB_COUNT),
            collapse(tokens, MIN_TOKEN_SUB_COUNT))


def sentences(text):
    return [s for s in SENT.split(text or "") if s.strip()]


def build_sentence_model(train):
    """Label each train sentence: did the amendment touch it?"""
    X, y = [], []
    for after, before in zip(train["text_after"], train["text_before"]):
        b_sents = set(sentences(before))
        for s in sentences(after):
            X.append(s)
            y.append(0 if s in b_sents else 1)
    if len(set(y)) < 2:
        return None, None
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, max_features=60000, sublinear_tf=True)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=2.0, random_state=SEED)
    clf.fit(Xv, y)
    return vec, clf


MIN_SENT_PROB = 0.50     # do not edit unless the locator is actually confident
MIN_PHRASE_CHARS = 4     # a 1-2 character "substitution" is noise


def predict(text_after, tables, vec, clf):
    """One careful edit, and only when confident enough to be worth it.

    ABSTAINING IS A REAL STRATEGY HERE, not a cop-out. The metric's precision
    term divides by everything the prediction proposes, so a wrong edit scores
    0 for that row -- exactly what leaving the text alone scores. A wrong edit
    is therefore never better than no edit. Measured on the shipped split,
    editing every row scored 0.000; gating on locator confidence is what lets
    the reference score above the floor at all.
    """
    phrase_table, token_table = tables
    sents = sentences(text_after)
    if not sents or (not phrase_table and not token_table):
        return text_after

    if vec is not None:
        probs = clf.predict_proba(vec.transform(sents))[:, 1]
        order = sorted(range(len(sents)), key=lambda i: -probs[i])
    else:
        probs = [1.0] * len(sents)
        order = list(range(len(sents)))

    for idx in order:
        if probs[idx] < MIN_SENT_PROB:
            break                               # nothing looks amended

        s = sents[idx]

        # 1. exact phrase substitution -- specific, rarely fires across parts
        best = None
        for a_phrase, (b_phrase, n) in phrase_table.items():
            if len(a_phrase) < MIN_PHRASE_CHARS:
                continue
            if a_phrase in s:
                score = n * len(a_phrase)
                if best is None or score > best[0]:
                    best = (score, a_phrase, b_phrase)
        if best:
            sents[idx] = s.replace(best[1], best[2], 1)
            return " ".join(sents)

        # 2. single-token swap -- the unit that actually transfers
        toks = tokenize(s)
        best_t = None
        for i, t in enumerate(toks):
            if t in token_table and len(t) >= MIN_PHRASE_CHARS:
                b_tok, n = token_table[t]
                if best_t is None or n > best_t[0]:
                    best_t = (n, i, t, b_tok)
        if best_t:
            _, i, a_tok, b_tok = best_t
            sents[idx] = re.sub(r"\b" + re.escape(a_tok) + r"\b", b_tok, s, count=1)
            return " ".join(sents)

    return text_after


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.path.join(here, "dataset", "public"))
    p.add_argument("--out", default=os.path.join(here, "submission.csv"))
    a = p.parse_args()

    train = pd.read_csv(os.path.join(a.data, "train.csv"))
    test = pd.read_csv(os.path.join(a.data, "test.csv"))
    print(f"train {len(train)}  test {len(test)}")

    tables = mine_substitutions(train)
    print(f"substitutions: {len(tables[0])} phrase, {len(tables[1])} token")

    vec, clf = build_sentence_model(train)
    print("sentence model:", "trained" if vec is not None else "unavailable")

    preds = [predict(t, tables, vec, clf) for t in test["text_after"]]
    out = pd.DataFrame({"row_id": test["row_id"], "text_before": preds})
    out.to_csv(a.out, index=False)
    changed = sum(1 for t, p_ in zip(test["text_after"], preds) if t != p_)
    print(f"wrote {a.out}: {len(out)} rows, {changed} edited")


if __name__ == "__main__":
    main()
