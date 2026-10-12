#!/usr/bin/env python3
"""Deterministic generator for the "Tandem Repeat Harvesting" benchmark.

REAL-DATA build. The sequences are REAL human genome (hg38) windows and the
true repeat extents are REAL Tandem Repeats Finder calls, taken from the UCSC
Genome Browser `simpleRepeat` track (Benson, NAR 1999). Nothing is synthesised:
the background carries genuine genomic structure (interspersed repeats, low-
complexity tracts, real GC skew) and the harvest targets are a real annotator's
output -- so a solution must actually DETECT tandem repeats, not reverse a
generator.

Pipeline
--------
  fetch_ucsc.py   fetches the hg38 DNA + simpleRepeat calls for a fixed list of
                  genomic regions from the UCSC REST API, clusters overlapping
                  TRF calls into loci, cuts a window around each cluster with
                  ASYMMETRIC random flanks (so a repeat sits at an unpredictable
                  offset and never fills more than ~20% of its window), keeps the
                  loci whose heaviest call is 25-70% of the locus's total called
                  extent, drops any window whose flanks carry a call from another
                  locus (so the released truth set is COMPLETE -- no real repeat
                  inside a window is missing from the key), and writes
                  raw_windows.json.  It is re-runnable with
                  network access; its output (raw_windows.json) is BUNDLED so
                  this generator is fully self-contained and deterministic.

  generate.py     (this file) reads raw_windows.json and writes the raw dataset:
                  sequences.csv, str_motifs.json, meta.json.

Determinism: raw_windows.json is a fixed byte-identical artifact; this script
only formats it. Two runs produce identical outputs.

Why the score band holds
------------------------
A tandem array is reported by TRF at several overlapping period/extent
interpretations, so the true extents at one locus OVERLAP. Only ONE interval is
scored per sequence (see grade.py, BUDGET_K=1), and one interval matches at most
one call, so a sequence is credited at most its HEAVIEST overlapping call while
the denominator counts them all: the ceiling on the metric is
sum(heaviest)/sum(all) -- 0.415 on this corpus (0.416 on the evaluation split), a
hard cap well under 0.5 that no submission can exceed. Tight localization against
real TRF boundaries under a 50% IoU bar keeps every answer-blind fixed guess at or
below 0.083 while the reference detector lands ~0.288.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_WINDOWS = os.path.join(HERE, "raw_windows.json")
# Write next to a sibling 'raw/' when run from a 'generator/' directory (the
# challenge layout); otherwise write beside this script (the flat dataset layout,
# where sequences.csv / str_motifs.json / meta.json sit next to generate.py).
if os.path.basename(HERE) == "generator":
    OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "raw"))
else:
    OUT_DIR = HERE

BUDGET_K = 1          # one interval is scored per sequence (see grade.py)
IOU_THRESHOLD = 0.50  # a submitted interval harvests a true iff IoU >= this


def load_windows():
    if not os.path.exists(RAW_WINDOWS):
        raise FileNotFoundError(
            f"{RAW_WINDOWS} missing -- run fetch_ucsc.py (needs network) to "
            "rebuild it from the UCSC Genome Browser.")
    return json.load(open(RAW_WINDOWS))


def main():
    data = load_windows()
    wins = data["windows"]
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- sequences.csv --------------------------------------------------- #
    tot_w = tot_ceil = 0.0
    eval_w = eval_ceil = 0.0      # the graded split's own ceiling
    period_hist = {}
    motif_count = {}
    n_train = n_test = 0
    with open(os.path.join(OUT_DIR, "sequences.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq_id", "split", "length", "sequence",
                     "true_intervals", "n_true"])
        for w in wins:
            dna = w["dna"]
            ivs = [(t["s"], t["e"]) for t in w["trf"]]
            tv = "|".join(f"{s}-{e}" for s, e in ivs)
            # NOTE: the window's real assembly coordinate stays in
            # raw_windows.json (the private build artifact) and is deliberately
            # NOT written here -- publishing it would point straight at the
            # annotator's own output for the evaluation sequences.
            wr.writerow([w["seq_id"], w["split"], len(dna), dna, tv, len(ivs)])
            tot_w += sum(e - s for s, e in ivs)
            tot_ceil += max((e - s for s, e in ivs), default=0)   # heaviest call
            if w["split"] == "test":
                eval_w += sum(e - s for s, e in ivs)
                eval_ceil += max((e - s for s, e in ivs), default=0)
            n_train += (w["split"] == "train")
            n_test += (w["split"] == "test")
            for t in w["trf"]:
                period_hist[t["period"]] = period_hist.get(t["period"], 0) + 1
                mot = t["consensus"]
                motif_count[mot] = motif_count.get(mot, 0) + 1

    ceiling = tot_ceil / tot_w if tot_w else 0.0
    eval_ceiling = eval_ceil / eval_w if eval_w else 0.0

    # ---- str_motifs.json: real observed motif statistics ----------------- #
    top_motifs = sorted(motif_count.items(), key=lambda x: -x[1])[:40]
    top_periods = sorted(period_hist.items(), key=lambda x: -x[1])[:15]
    with open(os.path.join(OUT_DIR, "str_motifs.json"), "w") as f:
        json.dump({
            "description": "Observed tandem-repeat motifs in the corpus -- the "
                           "actual consensus units reported by Tandem Repeats "
                           "Finder for these real hg38 windows (UCSC simpleRepeat "
                           "track). This is the empirical distribution, not a "
                           "curated set: it is dominated by common genome-wide "
                           "mono- and di-nucleotide microsatellites such as A, "
                           "AC, AG, AT, TG, exactly as a real reference genome "
                           "is. "
                           "No motif is planted or inserted.",
            "source": data["source"],
            "n_distinct_motifs": len(motif_count),
            "top_consensus_motifs": [{"motif": m, "count": c}
                                     for m, c in top_motifs],
            "period_histogram": [{"period": int(p), "count": c}
                                 for p, c in top_periods],
        }, f, indent=2)

    # ---- meta.json ------------------------------------------------------- #
    meta = {
        "source": data["source"],
        "assembly": data.get("assembly", "hg38"),
        "track": data.get("track", "simpleRepeat"),
        "n_seq": len(wins),
        "n_train": n_train,
        "n_test": n_test,
        "budget_k": BUDGET_K,
        "iou_threshold": IOU_THRESHOLD,
        "verified_ceiling": round(ceiling, 4),
        "verified_ceiling_eval_split": round(eval_ceiling, 4),
        "notes": "Ceiling = sum(heaviest call per sequence) / total called "
                 "weight -- a hard upper bound on the HARVEST metric. One "
                 "reported interval matches at most one call, so it can recover "
                 "at most the heaviest of a locus's overlapping real TRF calls; "
                 "the denominator counts them all, pinning the ceiling < 0.5.",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"windows: {len(wins)}  train: {n_train}  test: {n_test}")
    print(f"corpus ceiling (heaviest/total): {ceiling:.4f}  "
          f"eval-split ceiling: {eval_ceiling:.4f}")
    print(f"wrote {OUT_DIR}/sequences.csv, str_motifs.json, meta.json")


if __name__ == "__main__":
    main()
