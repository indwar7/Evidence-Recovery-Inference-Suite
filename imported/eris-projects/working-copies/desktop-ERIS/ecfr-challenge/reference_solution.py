#!/usr/bin/env python3
"""Reference solution for "The Vanished Clause".

Reads only the public data directory. Three stages, each a full submission:

  1. marks removed        the shown text with its `[GAP]` marks taken out
  2. bigram heuristic     no training labels: put each fragment where the
                          word before the mark is most often followed by the
                          fragment's first word, and the fragment's last word
                          by what comes next
  3. learned pair scorer  a logistic regression over (mark, fragment) pairs,
                          trained on train_slots.csv, then a one-to-one
                          assignment of fragments to marks per section

No pretrained model, no external data. Deterministic.

Every test row is predicted on its own. N-gram counts come from the TRAINING
text (both versions of every training section) and from the row's own shown
text -- never from test.csv as a corpus, and never from another test row.

    python reference_solution.py [--data dataset/public] [--out submission.csv]
                                 [--stage 3]
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

MARK = "[GAP]"
TOKEN = re.compile(r"\w+|[^\w\s]")
tok = TOKEN.findall


# --------------------------------------------------------------------------- #
# reading a row
# --------------------------------------------------------------------------- #
def parse(text_after):
    """Split on the marks. Returns (tokens of the clean text, token offset of
    every mark, the text pieces between marks)."""
    pieces = text_after.split(MARK)
    toks, slots = [], []
    for k, piece in enumerate(pieces):
        toks += tok(piece)
        if k < len(pieces) - 1:
            slots.append(len(toks))
    return toks, slots, pieces


def render(pieces, put):
    """Rebuild the text with `put[mark_index] = [fragment, ...]` inserted and
    every mark removed."""
    out = []
    for k, piece in enumerate(pieces):
        out.append(piece)
        if k < len(pieces) - 1:
            out.append(" " + " ".join(put.get(k, [])) + " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def rows(df):
    out = []
    for r in df.itertuples(index=False):
        toks, slots, pieces = parse(r.text_after)
        out.append({"row_id": r.row_id, "toks": toks, "slots": slots,
                    "pieces": pieces,
                    "frags": json.loads(r.removed_fragments),
                    "before": getattr(r, "text_before", None)})
    return out


# --------------------------------------------------------------------------- #
# features of one (mark, fragment) pair
# --------------------------------------------------------------------------- #
def shape(t):
    if t.isdigit():
        return "D"
    if re.fullmatch(r"[ivxlc]+", t):
        return "r"
    if t.isalpha():
        return "A" if t[0].isupper() else ("a1" if len(t) == 1 else "a")
    if t.isalnum():
        return "x"
    return t


def char_bigrams(s):
    s = s.lower()
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


class Counts:
    """Token n-gram counts over a set of token lists."""

    def __init__(self, texts=()):
        self.uni, self.big, self.tri = Counter(), Counter(), Counter()
        for t in texts:
            self.add(t)

    def add(self, t):
        self.uni.update(t)
        self.big.update(zip(t, t[1:]))
        self.tri.update(zip(t, t[1:], t[2:]))


class _Diff:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def get(self, k, d=0):
        return self.a.get(k, 0) - self.b.get(k, 0)


class Minus:
    """`total` with one fold's texts taken out, read-only."""

    def __init__(self, total, part):
        self.uni = _Diff(total.uni, part.uni)
        self.big = _Diff(total.big, part.big)
        self.tri = _Diff(total.tri, part.tri)


LM_KEYS = ("lm_in", "tri_in", "tri_in2", "lm_out", "tri_out", "lm_out0")


def pair_features(row, si, f, c):
    """Features of one pair: counts `c` from train, plus the same count
    features from the row's own shown text (suffix `_loc`)."""
    d = _features(row, si, f, c)
    if "local" not in row:
        row["local"] = Counts([row["toks"]])
    e = _features(row, si, f, row["local"])
    for k in LM_KEYS:
        if k in e:
            d[k + "_loc"] = e[k]
    return d


def _features(row, si, f, c):
    s, g = row["toks"], row["slots"][si]
    L1 = s[g - 1] if g > 0 else "<s>"
    L2 = s[g - 2] if g > 1 else "<s>"
    R1 = s[g] if g < len(s) else "</s>"
    R2 = s[g + 1] if g + 1 < len(s) else "</s>"
    F1, Fl, lb = f[0], f[-1], min(len(f), 6)
    d = {
        f"L1F1={L1}|{F1}": 1, f"L2L1F1={L2}|{L1}|{F1}": 1,
        f"FlR1={Fl}|{R1}": 1, f"F1R1={F1}|{R1}": 1,
        f"len={lb}": 1, f"lenR1={lb}|{R1}": 1, f"lenL1={lb}|{L1}": 1,
        f"sh={shape(F1)}|{shape(R1)}": 1, f"shL={shape(L1)}|{shape(F1)}": 1,
        f"slot={L1}|{R1}": 1, f"slotsh={shape(L1)}|{shape(R1)}": 1,
        f"F1={F1}|len{lb}": 1, f"eqF1R1={F1 == R1}": 1,
        f"eqFlL1={Fl == L1}": 1,
    }
    if len(f) > 1:
        d[f"F2R={f[1]}|{R2}"] = 1
        d["tri_in2"] = math.log1p(c.tri.get((L1, F1, f[1]), 0))
    # does the fragment read on from what precedes the mark?
    d["lm_in"] = (math.log1p(c.big.get((L1, F1), 0))
                  - 0.5 * math.log1p(c.uni.get(L1, 0)))
    d["tri_in"] = math.log1p(c.tri.get((L2, L1, F1), 0))
    # does the text read on from the fragment? The amendment may have put
    # new words right after the mark, so look a few tokens ahead as well.
    out2 = out3 = 0.0
    for k in range(10):
        if k < 8 and g + k < len(s):
            out2 = max(out2, math.log1p(c.big.get((Fl, s[g + k]), 0))
                       - 0.3 * k)
        if g + k + 1 < len(s):
            out3 = max(out3, math.log1p(
                c.tri.get((Fl, s[g + k], s[g + k + 1]), 0)) - 0.2 * k)
    d["lm_out"], d["tri_out"] = out2, out3
    d["lm_out0"] = math.log1p(c.big.get((Fl, R1), 0))
    # a substitution leaves the replacement where the fragment was, and the
    # replacement tends to look like it ("vi" -> "vii", "Internet" -> "internet")
    nxt = s[g:g + max(1, len(f))]
    if nxt:
        a, b = char_bigrams(" ".join(f)), char_bigrams(" ".join(nxt))
        j = len(a & b) / max(1, len(a | b))
        d["jac"] = j
        d[f"jacb={int(j * 5)}|{min(len(f), 4)}"] = 1
        d[f"lowereq={F1.lower() == nxt[0].lower()}"] = 1
        d[f"pfx={F1[:3].lower() == nxt[0][:3].lower()}|{min(len(F1), 4)}"] = 1
    # an amendment's edits sit together
    sl = row["slots"]
    nd = min([abs(g - sl[j]) for j in (si - 1, si + 1) if 0 <= j < len(sl)]
             or [999])
    nb = min(int(math.log2(nd + 1)), 7)
    d[f"nd={nb}"] = 1
    d[f"nd_len={nb}|{min(len(f), 4)}"] = 1
    d[f"relpos={int(10 * g / max(1, len(s)))}"] = 1
    return d


def assign(row, score_matrix):
    """One fragment per mark, maximising the total score."""
    ri, ci = linear_sum_assignment(-score_matrix)
    return render(row["pieces"],
                  {int(k): [row["frags"][i]] for i, k in zip(ri, ci)})


# --------------------------------------------------------------------------- #
# the three stages
# --------------------------------------------------------------------------- #
def stage1(row):
    return render(row["pieces"], {})


def make_stage2(c):
    def predict(row):
        fl = [tok(f) for f in row["frags"]]
        P = np.zeros((len(fl), len(row["slots"])))
        for i, f in enumerate(fl):
            for k in range(len(row["slots"])):
                d = pair_features(row, k, f, c)
                P[i, k] = (d["lm_in"] + d["lm_out"]
                           + d["lm_in_loc"] + d["lm_out_loc"])
        return assign(row, P)
    return predict


N_FOLDS = 5


def train_counts(train_rows):
    """Counts over all training text, and the same with each fold left out.

    A training row's own prior wording contains its fragments in place, so
    counts that include it make every true pair look better than it will at
    test time. Training features therefore use counts with the row's fold
    removed. train.csv is in source order, so contiguous folds keep the
    successive versions of one section together.
    """
    total, folds = Counts(), [Counts() for _ in range(N_FOLDS)]
    fold_of = {}
    for i, row in enumerate(train_rows):
        k = min(N_FOLDS - 1, i * N_FOLDS // len(train_rows))
        fold_of[row["row_id"]] = k
        for t in (row["toks"], tok(row["before"])):
            total.add(t)
            folds[k].add(t)
    return total, [Minus(total, f) for f in folds], fold_of


def make_stage3(train_rows, labels, c, oof, fold_of, C=1.0):
    hasher = FeatureHasher(n_features=2 ** 20)
    X, y = [], []
    for row in train_rows:
        c_row = oof[fold_of[row["row_id"]]]
        for fi, frag in enumerate(row["frags"]):
            f = tok(frag)
            for k in range(len(row["slots"])):
                X.append(pair_features(row, k, f, c_row))
                y.append(int(labels[row["row_id"]][fi] == k))
    clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced")
    clf.fit(hasher.transform(X), y)

    def predict(row):
        fl = [tok(f) for f in row["frags"]]
        P = np.array([clf.decision_function(hasher.transform(
            [pair_features(row, k, f, c) for k in range(len(row["slots"]))]))
            for f in fl])
        return assign(row, P)
    return predict


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--stage", type=int, default=3, choices=(1, 2, 3))
    a = ap.parse_args()
    data = a.data or next(
        p for p in ("/data", os.path.join(here, "dataset", "public"),
                    "dataset/public",
                    "/kaggle/input/ecfr-amendment-reversal/public")
        if os.path.exists(os.path.join(p, "test.csv")))

    train = pd.read_csv(os.path.join(data, "train.csv"))
    test = pd.read_csv(os.path.join(data, "test.csv"))
    slots = pd.read_csv(os.path.join(data, "train_slots.csv"))
    TR, TE = rows(train), rows(test)
    labels = defaultdict(dict)
    for r in slots.itertuples(index=False):
        labels[r.row_id][r.fragment_index] = r.gap_index

    # counts from the training text only; test.csv is never used as a corpus
    counts, oof, fold_of = train_counts(TR)

    if a.stage == 1:
        fn = stage1
    elif a.stage == 2:
        fn = make_stage2(counts)
    else:
        fn = make_stage3(TR, labels, counts, oof, fold_of)
    pd.DataFrame({"row_id": [r["row_id"] for r in TE],
                  "text_before": [fn(r) for r in TE]}).to_csv(a.out,
                                                              index=False)
    print(f"stage {a.stage}: wrote {a.out} ({len(TE)} rows)")


if __name__ == "__main__":
    main()
