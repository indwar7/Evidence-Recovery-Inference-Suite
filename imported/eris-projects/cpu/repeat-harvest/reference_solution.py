#!/usr/bin/env python3
"""Reference solution for Annotation Depth Track Prediction.

Two stages, matching the two things the metric actually rewards.

1. LOCALIZE the array, sharply. For each candidate period p the sequence gets a
   period-p match track (+1 where a base equals the base p to its left, -PEN
   where it does not) and the MAXIMAL-SCORING SEGMENT of that track is taken
   (Kadane). A maximal-scoring segment stops exactly where periodicity stops
   paying for itself, so its boundaries are sharp, unlike a density threshold
   which frays at the edges. The best-scoring period wins and its segment is
   emitted as the outermost call -- annotation depth 1.

2. RECONSTRUCT THE LAYERING. The annotator covers the core of an array with
   several overlapping calls and its edges with one, so the target is a per-base
   DEPTH field, not a single interval. A ridge regression is fitted ON THE
   TRAINING SPLIT ONLY, from smoothed period-match densities at four window
   widths to the observed depth, then applied inside the localized array. Its
   output, rounded and clipped, IS the emitted depth track; outside the array
   the track is 0.

Every constant below (GAIN, BIAS, MAXLEV, PEN) was fixed by a grid sweep on the
TRAINING split alone; the evaluation split is never consulted. On the evaluation
split it scores AGREEMENT 0.6574 -- against 0.2715 for the best answer-blind
fixed guess, 0.4644 for a flat depth-1 track over the located array, and 1.0 for
the annotator's own track. numpy only, about four seconds on CPU.

Usage: python reference_solution.py [train_csv] [test_csv] [out_submission.csv]
       (defaults: dataset/public/sequences_train.csv,
                  dataset/public/sequences_test.csv -> reference_submission.csv)
"""

import csv
import sys

import numpy as np

PERIODS = range(1, 25)       # candidate periods
WINS = (15, 31, 61, 121)     # smoothing widths for the depth features
PEN = 1.0                    # mismatch penalty in the period-match track
MINLEN = 20                  # shortest acceptable array
GAIN = 1.4                   # depth-field scale   ) all three swept on TRAIN
BIAS = 0.6                   # depth-field offset   ) only -- never on the
MAXLEV = 3                   # deepest layer emitted ) evaluation split
RIDGE = 1e-3


def boxsmooth(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(1, hi - lo)


def max_segment(x):
    """Maximal-scoring contiguous segment of x -> (start, end, score)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    gains = c[1:] - np.minimum.accumulate(c[:-1])
    e = int(np.argmax(gains))
    s = int(np.argmin(c[:e + 1]))
    return s, e + 1, float(gains[e])


def analyse(dna):
    """-> (per-base feature matrix, best (score, period, start, end) or None)."""
    b = np.frombuffer(dna.encode(), dtype=np.uint8)
    n = len(b)
    pw = {w: np.zeros(n) for w in WINS}
    best = None
    for p in PERIODS:
        if n <= p + 4:
            continue
        m = np.zeros(n)
        m[p:] = (b[p:] == b[:-p]).astype(float)
        for w in WINS:
            np.maximum(pw[w], boxsmooth(m, w), out=pw[w])
        if n > p + MINLEN:
            mm = np.full(n - p, -PEN)
            mm[b[p:] == b[:-p]] = 1.0
            s, e, sc = max_segment(mm)
            # the track compares base i+p against base i, so the array spans [s, e+p)
            if e - s >= MINLEN and (best is None or sc > best[0]):
                best = (sc, p, s, min(n, e + p))
    feats = np.stack([pw[w] for w in WINS] +
                     [pw[15] * pw[121], pw[61] ** 2,
                      np.minimum(pw[15], pw[121]), np.ones(n)], axis=1)
    return feats, best


def true_depth(spec, length):
    d = np.zeros(length, dtype=np.float64)
    if spec:
        for tok in spec.split("|"):
            a, b = tok.split("-")
            d[int(a):int(b)] += 1.0
    return d


def solve(train_csv, test_csv, out_csv):
    train = list(csv.DictReader(open(train_csv, newline="")))
    test = list(csv.DictReader(open(test_csv, newline="")))

    # ---- stage 2 model: fit the depth field on the TRAINING split only ---- #
    X, y = [], []
    for r in train:
        f, _ = analyse(r["sequence"])
        X.append(f)
        y.append(true_depth(r["true_intervals"], int(r["length"])))
    X = np.vstack(X)
    y = np.concatenate(y)
    w = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ y)

    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["seq_id", "depth_track"])
        for r in test:
            L = int(r["length"])
            feats, det = analyse(r["sequence"])
            track = np.zeros(L, dtype=int)
            if det is not None:
                _, _, s0, e0 = det
                track[s0:e0] = 1                      # the located array, depth 1
                inner = (feats @ w * GAIN + BIAS)[s0:e0]
                # the fitted depth field, rounded, is the track inside the array
                track[s0:e0] = np.clip(np.rint(inner), 1, MAXLEV).astype(int)
            wr.writerow([r["seq_id"], "".join(map(str, track.tolist()))])

    print(f"wrote {out_csv} ({len(test)} sequences)")


if __name__ == "__main__":
    tr = sys.argv[1] if len(sys.argv) > 1 else "dataset/public/sequences_train.csv"
    te = sys.argv[2] if len(sys.argv) > 2 else "dataset/public/sequences_test.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "reference_submission.csv"
    solve(tr, te, out)
