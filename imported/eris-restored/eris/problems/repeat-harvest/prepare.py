#!/usr/bin/env python3
"""Prepare public + private splits for the Tandem Repeat Harvesting challenge.

Reads the raw corpus (dataset/raw/sequences.csv) and writes:

  dataset/public/
    sequences_train.csv   seq_id, length, sequence, true_intervals, n_true
                          (the supervised split -- true repeat extents revealed
                           so you can build and calibrate a detector; the raw
                           corpus's `source` coordinate is deliberately NOT
                           released, see the writer below)
    sequences_test.csv    seq_id, length, sequence
                          (the evaluation split -- sequences only)
    sample_submission.csv seq_id, intervals   (a valid one-interval baseline:
                          a single fixed centre tile per test sequence)

  dataset/private/
    answers.csv           ONE ROW PER TEST SEQUENCE, keyed on seq_id (the FIRST
                          column -- the platform harness keys the submission on
                          it). Packs the sequence length, the total true weight
                          (the metric denominator) and the true extents as JSON.
                          Rows are ratio-spread then slice-balanced, so any
                          contiguous leaderboard slice carries the same ceiling
                          and the same share of the metric's denominator as the
                          full evaluation set.

Deterministic; no randomness beyond the raw file.
"""

import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PHI_INV = 0.6180339887498949
SAMPLE_TILE = 80             # sample-submission baseline tile size (one interval)
SLICE_SCALES = (2, 5, 10)    # slice counts a leaderboard split might use


def parse_intervals(s):
    out = []
    if s:
        for tok in s.split("|"):
            a, b = tok.split("-")
            out.append((int(a), int(b)))
    return out


def _balance_slices(rows, scales=SLICE_SCALES, max_passes=6):
    """Reorder rows so every contiguous slice mirrors the full set.

    A leaderboard split scores a contiguous run of these rows, so each run must
    carry the same ceiling (sum ceiling_weight / sum total_weight) and the same
    share of the denominator as the whole set -- otherwise a public/private
    split moves a submission's score for reasons that have nothing to do with
    the submission. Deterministic first-improvement pairwise swaps against a
    cost summed over several slice counts at once; a swap only disturbs the two
    slices holding the swapped positions, so each candidate is evaluated in
    O(len(scales)) rather than O(n).

    Structural only: this never looks at any solution's score, so it cannot tune
    the split toward a particular solver.
    """
    n = len(rows)
    W = [float(r["total_weight"]) for r in rows]
    C = [float(r["ceiling_weight"]) for r in rows]
    tot_w, tot_c = sum(W), sum(C)
    if n < 4 or tot_w <= 0:
        return rows
    gc = tot_c / tot_w

    def slice_of(scale, pos):
        return min(pos // (n // scale), scale - 1)

    # per (scale, slice) running sums
    sums = {k: [[0.0, 0.0] for _ in range(k)] for k in scales}
    for k in scales:
        for pos in range(n):
            b = sums[k][slice_of(k, pos)]
            b[0] += W[pos]; b[1] += C[pos]

    def bucket_cost(k, b):
        if b[0] <= 0:
            return 0.0
        return (b[0] / tot_w * k - 1.0) ** 2 + ((b[1] / b[0] - gc) / gc) ** 2

    cost = sum(bucket_cost(k, b) for k in scales for b in sums[k])
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                dw, dc = W[j] - W[i], C[j] - C[i]
                if dw == 0.0 and dc == 0.0:
                    continue
                touched, delta = [], 0.0
                for k in scales:
                    si, sj = slice_of(k, i), slice_of(k, j)
                    if si == sj:
                        continue
                    bi, bj = sums[k][si], sums[k][sj]
                    before = bucket_cost(k, bi) + bucket_cost(k, bj)
                    ni = [bi[0] + dw, bi[1] + dc]
                    nj = [bj[0] - dw, bj[1] - dc]
                    delta += bucket_cost(k, ni) + bucket_cost(k, nj) - before
                    touched.append((k, si, sj, ni, nj))
                if delta < -1e-12:
                    for k, si, sj, ni, nj in touched:
                        sums[k][si], sums[k][sj] = ni, nj
                    rows[i], rows[j] = rows[j], rows[i]
                    W[i], W[j] = W[j], W[i]
                    C[i], C[j] = C[j], C[i]
                    cost += delta
                    improved = True
        if not improved:
            break
    return rows


def _find_raw(raw_dir):
    """Locate sequences.csv given a raw dir (or a direct path to the file)."""
    if raw_dir and os.path.isfile(raw_dir):
        return raw_dir
    # Search every layout the corpus turns up in: the challenge tree
    # (dataset/raw/sequences.csv) and the uploaded dataset extracted to
    # <setup>/dataset, where the raw files sit FLAT next to public/ and private/.
    cwd = os.getcwd()
    for cand in (
        os.path.join(raw_dir or "", "sequences.csv"),
        os.path.join(raw_dir or "", "raw", "sequences.csv"),
        os.path.join(cwd, "dataset", "raw", "sequences.csv"),
        os.path.join(cwd, "dataset", "sequences.csv"),
        os.path.join(cwd, "sequences.csv"),
        os.path.join(_HERE, "dataset", "raw", "sequences.csv"),
        os.path.join(_HERE, "dataset", "sequences.csv"),
    ):
        if cand and os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"sequences.csv not found (raw_dir={raw_dir!r}, cwd={cwd!r}) -- looked in "
        "<raw_dir>/, <raw_dir>/raw/, ./dataset/raw/, ./dataset/, ./ and the "
        "challenge tree")


def prepare(raw_dir=None, public_dir=None, private_dir=None, **kwargs):
    """Entry point used by the platform harness.

    raw_dir: directory containing sequences.csv (or the file itself).
    public_dir / private_dir: output directories for the public and private
    artifacts. All default to the challenge-local layout when omitted.
    Tolerates common alternate keyword names from different harness versions."""
    raw_dir = raw_dir or kwargs.get("raw") or kwargs.get("input_dir") \
        or kwargs.get("data_dir") or kwargs.get("dataset_dir")
    public_dir = public_dir or kwargs.get("public") \
        or kwargs.get("output_public") or kwargs.get("public_output")
    private_dir = private_dir or kwargs.get("private") \
        or kwargs.get("output_private") or kwargs.get("private_output")
    raw_path = _find_raw(raw_dir if raw_dir is not None
                         else os.path.join(_HERE, "dataset", "raw"))
    # Platform layout: the harness extracts the dataset to <setup>/dataset and
    # its grade entrypoint reads <setup>/dataset/public/sample_submission.csv,
    # so the prepared splits MUST land in dataset/public and dataset/private.
    # Resolve relative to the CWD first (what the harness uses), falling back to
    # this file's directory when run from elsewhere.
    _base = os.getcwd() if os.path.isdir(os.path.join(os.getcwd(), "dataset")) \
        else _HERE
    PUB = public_dir or os.path.join(_base, "dataset", "public")
    PRIV = private_dir or os.path.join(_base, "dataset", "private")
    os.makedirs(PUB, exist_ok=True)
    os.makedirs(PRIV, exist_ok=True)

    rows = []
    with open(raw_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    # ---- public: training split (answers revealed) ----------------------- #
    # The raw corpus keeps a `source` coordinate per row so the build can be
    # audited against the reference assembly, but it is NOT released: publishing
    # exact coordinates would hand participants a direct route to the annotator's
    # own output for the evaluation sequences.
    with open(os.path.join(PUB, "sequences_train.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq_id", "length", "sequence", "true_intervals",
                     "n_true"])
        for r in train:
            wr.writerow([r["seq_id"], r["length"], r["sequence"],
                         r["true_intervals"], r["n_true"]])

    # ---- public: test split (sequences only) ----------------------------- #
    with open(os.path.join(PUB, "sequences_test.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq_id", "length", "sequence"])
        for r in test:
            wr.writerow([r["seq_id"], r["length"], r["sequence"]])

    # ---- public: sample submission (valid one-interval baseline) --------- #
    # A single fixed centre tile per sequence: a well-formed, low-scoring
    # baseline that shows the submission format (one interval is scored).
    with open(os.path.join(PUB, "sample_submission.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq_id", "intervals"])
        for r in test:
            L = int(r["length"])
            s = max(0, L // 2 - SAMPLE_TILE // 2)
            wr.writerow([r["seq_id"], f"{s}-{min(s + SAMPLE_TILE, L)}"])

    # ---- private: answers, class-stratified ------------------------------ #
    packed = []
    for r in test:
        ivs = parse_intervals(r["true_intervals"])
        W = sum(e - s for s, e in ivs)
        heaviest = max((e - s for s, e in ivs), default=0)   # per-seq ceiling
        ratio = (heaviest / W) if W > 0 else 0.0
        trues = [{"s": s, "e": e} for s, e in ivs]
        packed.append({
            "seq_id": r["seq_id"],
            "length": int(r["length"]),
            "total_weight": W,
            "ceiling_weight": heaviest,
            "ratio": ratio,
            "trues_json": json.dumps(trues, separators=(",", ":")),
        })

    # Order rows so that ANY contiguous slice (a public/private leaderboard
    # split) mirrors the full set on BOTH axes an ordering can control:
    #   * ratio  = heaviest/total, which sets that row's share of the ceiling
    #   * weight = total_weight, the metric's denominator (a few heavy rows
    #              otherwise dominate whichever slice they land in)
    # Sorting by ratio and re-keying (the golden-ratio low-discrepancy pass
    # below) spreads the first axis but leaves the second clumped, so a slice
    # could inherit a different ceiling than the full set. _balance_slices then
    # swaps rows until every contiguous slice carries the corpus ceiling and an
    # equal share of the denominator. Deterministic: no randomness anywhere,
    # ties broken on seq_id.
    packed.sort(key=lambda x: (x["ratio"], x["seq_id"]))
    for i, p in enumerate(packed):
        p["_key"] = ((i + 0.5) * PHI_INV) % 1.0
    packed.sort(key=lambda x: (x["_key"], x["seq_id"]))
    packed = _balance_slices(packed)

    with open(os.path.join(PRIV, "answers.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq_id", "length", "total_weight", "ceiling_weight",
                     "trues_json"])
        for p in packed:
            wr.writerow([p["seq_id"], p["length"], p["total_weight"],
                         p["ceiling_weight"], p["trues_json"]])

    tot_w = sum(p["total_weight"] for p in packed)
    tot_c = sum(p["ceiling_weight"] for p in packed)
    print(f"train seqs: {len(train)}  test seqs: {len(test)}")
    print(f"eval total_weight={tot_w:.0f}  ceiling={tot_c/tot_w:.4f}")
    return {"n_train": len(train), "n_test": len(test),
            "ceiling": tot_c / tot_w if tot_w else 0.0}


if __name__ == "__main__":
    # CLI: prepare.py [raw_dir] [public_dir] [private_dir]
    # Defaults match the platform layout: dataset/raw -> dataset/public + dataset/private
    a = sys.argv[1:]
    prepare(a[0] if len(a) > 0 else None,
            a[1] if len(a) > 1 else None,
            a[2] if len(a) > 2 else None)
