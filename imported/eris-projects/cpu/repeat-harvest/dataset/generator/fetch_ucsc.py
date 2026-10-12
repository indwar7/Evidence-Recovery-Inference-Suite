#!/usr/bin/env python3
"""Rebuild raw_windows.json from the UCSC Genome Browser (needs network).

This documents the provenance of the dataset and lets anyone regenerate it. It
fetches real hg38 DNA and the real `simpleRepeat` (Tandem Repeats Finder) track
for a FIXED list of genomic regions via the UCSC REST API, then:

  1. clusters overlapping/adjacent TRF calls into loci (gap <= GAP);
  2. keeps loci where the heaviest single TRF call is a fraction H_LO..H_HI of
     the total called extent -- i.e. the locus carries real overlapping multi-
     period annotation (so a single reported interval can recover at most that
     heaviest fraction, which pins the score ceiling well under 0.5), but is not
     so fragmented that no dominant call exists;
  3. cuts a window around each locus: a total flank of [4.0, 8.0] x locus-span
     (so the repeat is <= ~20% of the window, defeating whole-window guesses)
     split at a uniform-ish left/right ratio (so the repeat sits at an
     unpredictable offset, defeating fixed centre/left/right guesses);
  4. DROPS any window whose flanks contain a TRF call from another locus, so the
     released truth set is COMPLETE: every tandem repeat the annotator reports
     inside a window is among that window's true extents. A solver is never
     punished for finding a real repeat that the answer key omits;
  5. deterministically assigns a stratified train/test split of exactly
     N_TRAIN / N_TEST windows (train >> test);
  6. writes raw_windows.json.

Everything is seeded, so re-running against the (static) hg38 assembly yields
byte-identical output. The heavy region fetches are cached under ./ucsc_cache/.
"""

import bisect
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ucsc_cache")
OUT = os.path.join(HERE, "raw_windows.json")
API = "https://api.genome.ucsc.edu/getData"

GLOBAL_SEED = 20260716
GAP = 20                     # merge TRF calls within this many bp into one locus
SPAN = (40, 1200)            # keep loci whose repeat span is in this range
H_LO, H_HI = 0.25, 0.70      # keep loci with heaviest-call / total-called in here
FLK_TOT_LO, FLK_TOT_HI = 4.0, 8.0   # total flank ~ U(.) * span (repeat <= ~20% win)
POS_LO, POS_HI = 0.05, 0.95  # locus placed at a near-uniform offset across the window
N_TRAIN, N_TEST = 1400, 300  # exact split sizes drawn from the filtered corpus

# Fixed genomic regions: genome-wide microsatellite-rich windows plus real
# disease-expansion loci (HTT / Huntington, C9orf72). hg38 coordinates.
REGIONS = [
    ("chr1", 2_000_000, 3_000_000), ("chr1", 20_000_000, 21_000_000),
    ("chr2", 30_000_000, 31_000_000), ("chr3", 50_000_000, 51_000_000),
    ("chr4", 3_000_000, 4_000_000), ("chr5", 10_000_000, 11_000_000),
    ("chr6", 20_000_000, 21_000_000), ("chr7", 5_000_000, 6_000_000),
    ("chr8", 40_000_000, 41_000_000), ("chr9", 27_000_000, 28_000_000),
    ("chr10", 15_000_000, 16_000_000), ("chr11", 2_000_000, 3_000_000),
    ("chr12", 6_000_000, 7_000_000), ("chr13", 25_000_000, 26_000_000),
    ("chr14", 30_000_000, 31_000_000), ("chr15", 40_000_000, 41_000_000),
    ("chr16", 8_000_000, 9_000_000), ("chr17", 7_000_000, 8_000_000),
    ("chr18", 12_000_000, 13_000_000), ("chr19", 1_000_000, 2_000_000),
    ("chr19", 5_000_000, 6_000_000), ("chr20", 5_000_000, 6_000_000),
    ("chrX", 10_000_000, 11_000_000),
    # second pass: mid-arm euchromatin across the karyotype, added to carry a
    # 1400 / 300 split after the wider flanks and clean-flank filter cut yield.
    ("chr1", 55_000_000, 56_000_000), ("chr1", 155_000_000, 156_000_000),
    ("chr2", 100_000_000, 101_000_000), ("chr2", 200_000_000, 201_000_000),
    ("chr3", 130_000_000, 131_000_000), ("chr4", 80_000_000, 81_000_000),
    ("chr5", 130_000_000, 131_000_000), ("chr6", 100_000_000, 101_000_000),
    ("chr7", 100_000_000, 101_000_000), ("chr8", 100_000_000, 101_000_000),
    ("chr9", 100_000_000, 101_000_000), ("chr10", 70_000_000, 71_000_000),
    ("chr11", 65_000_000, 66_000_000), ("chr12", 60_000_000, 61_000_000),
    ("chr13", 60_000_000, 61_000_000), ("chr14", 70_000_000, 71_000_000),
    ("chr15", 70_000_000, 71_000_000), ("chr16", 60_000_000, 61_000_000),
    ("chr17", 40_000_000, 41_000_000), ("chr18", 40_000_000, 41_000_000),
    ("chr19", 40_000_000, 41_000_000), ("chr20", 40_000_000, 41_000_000),
    ("chr21", 30_000_000, 31_000_000), ("chr22", 30_000_000, 31_000_000),
    ("chrX", 100_000_000, 101_000_000),
    # third pass: distal euchromatin, added to clear 1700 surviving windows.
    ("chr1", 90_000_000, 91_000_000), ("chr2", 150_000_000, 151_000_000),
    ("chr3", 170_000_000, 171_000_000), ("chr4", 150_000_000, 151_000_000),
    ("chr5", 90_000_000, 91_000_000), ("chr6", 150_000_000, 151_000_000),
    ("chr7", 140_000_000, 141_000_000), ("chr8", 130_000_000, 131_000_000),
    ("chr9", 130_000_000, 131_000_000), ("chr10", 120_000_000, 121_000_000),
    ("chr11", 110_000_000, 111_000_000), ("chr12", 110_000_000, 111_000_000),
]


def _get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 50:
        return json.load(open(path))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            json.dump(d, open(path, "w"))
            return d
        except Exception as e:                       # noqa: BLE001
            print(f"  retry {attempt} {os.path.basename(path)}: {e}")
    raise RuntimeError(f"failed to fetch {url}")


def cluster(calls):
    calls = sorted(calls)
    out, cur, ce = [], [], None
    for c in calls:
        if cur and c[0] <= ce + GAP:
            cur.append(c); ce = max(ce, c[1])
        else:
            if cur:
                out.append(cur)
            cur, ce = [c], c[1]
    if cur:
        out.append(cur)
    return out


def main():
    os.makedirs(CACHE, exist_ok=True)
    seqs, trk = {}, {}
    for chrom, s, e in REGIONS:
        tag = f"{chrom}_{s}_{e}"
        sd = _get(f"{API}/sequence?genome=hg38;chrom={chrom};start={s};end={e}",
                  os.path.join(CACHE, f"seq_{tag}.json"))
        seqs[(chrom, s, e)] = sd["dna"].upper()
        td = _get(f"{API}/track?genome=hg38;track=simpleRepeat;"
                  f"chrom={chrom};start={s};end={e}",
                  os.path.join(CACHE, f"sr_{tag}.json"))
        for r in td.get("simpleRepeat", []):
            trk.setdefault(chrom, []).append(
                (r["chromStart"], r["chromEnd"], r["period"],
                 round(float(r["copyNum"]), 1), r["sequence"]))

    def region_of(chrom, pos):
        for (c, s, e) in seqs:
            if c == chrom and s <= pos < e:
                return (c, s, e)
        return None

    # every TRF call per chromosome, sorted by start -- used to guarantee that a
    # window's flanks carry no call from a neighbouring locus (complete truth)
    all_calls = {c: sorted((x[0], x[1]) for x in v) for c, v in trk.items()}
    MAX_CALL = max((b - a) for v in all_calls.values() for a, b in v)

    def flank_clean(chrom, ws, we, own):
        """True iff no TRF call outside `own` intersects the window [ws, we)."""
        calls = all_calls[chrom]
        i = bisect.bisect_left(calls, (ws - MAX_CALL, -1))
        for a, b in calls[i:]:
            if a >= we:
                break
            if b > ws and (a, b) not in own:
                return False
        return True

    frng = np.random.default_rng(GLOBAL_SEED)
    W = []
    for chrom in sorted(trk):
        for cl in cluster(trk[chrom]):
            span = max(x[1] for x in cl) - min(x[0] for x in cl)
            if not (SPAN[0] <= span <= SPAN[1]) or len(cl) < 2:
                continue
            cs, ce = min(x[0] for x in cl), max(x[1] for x in cl)
            reg = region_of(chrom, cs)
            if reg is None or reg != region_of(chrom, ce - 1):
                continue
            _, rs, re = reg
            # total flank fixes the repeat at <= ~20% of the window; the split
            # places it at an unpredictable (roughly uniform) offset, so no fixed
            # position guess (centre / left / right) beats real detection.
            total_flk = frng.uniform(FLK_TOT_LO, FLK_TOT_HI) * span
            left_frac = frng.uniform(POS_LO, POS_HI)
            lf = int(max(60, left_frac * total_flk))
            rf = int(max(60, (1.0 - left_frac) * total_flk))
            ws, we = cs - lf, ce + rf
            if ws < rs or we > re:
                continue
            # complete-truth guard: a window may contain ONLY its own locus's
            # calls, so no real tandem repeat inside it is missing from the key
            if not flank_clean(chrom, ws, we, {(x[0], x[1]) for x in cl}):
                continue
            dna = seqs[reg][ws - rs:we - rs]
            if "N" in dna:
                continue
            trues = [{"s": a - ws, "e": b - ws, "period": p,
                      "copy": cn, "consensus": con}
                     for a, b, p, cn, con in cl]
            iv = [(t["s"], t["e"]) for t in trues]
            tot = sum(e - s for s, e in iv)
            heavy = max(e - s for s, e in iv)
            hfrac = heavy / tot if tot else 1.0
            if not (H_LO <= hfrac <= H_HI):
                continue
            W.append({"chrom": chrom, "abs_start": ws, "dna": dna,
                      "trues": trues, "ratio": hfrac})

    # deterministic stratified split (by overlap-ratio bucket) at EXACTLY
    # N_TRAIN / N_TEST windows; surplus windows are dropped bucket-proportionally
    # so both splits keep the corpus's harvestable-ratio mix.
    W.sort(key=lambda w: (w["chrom"], w["abs_start"]))
    srng = np.random.default_rng(GLOBAL_SEED ^ 0x5EED)
    need = N_TRAIN + N_TEST
    if len(W) < need:
        raise RuntimeError(
            f"only {len(W)} windows survived the filters, need {need} -- add "
            "regions to REGIONS and re-run")
    buckets = {}
    for i, w in enumerate(W):
        buckets.setdefault(int(w["ratio"] * 5), []).append(i)

    def allocate(total, sizes):
        """Largest-remainder split of `total` across buckets of given sizes."""
        exact = [total * s / sum(sizes) for s in sizes]
        base = [min(int(x), s) for x, s in zip(exact, sizes)]
        order = sorted(range(len(sizes)), key=lambda i: (-(exact[i] - base[i]), i))
        i = 0
        while sum(base) < total:
            k = order[i % len(order)]
            if base[k] < sizes[k]:
                base[k] += 1
            i += 1
        return base

    keys = sorted(buckets)
    quota = allocate(need, [len(buckets[k]) for k in keys])
    shuffled = {k: [buckets[k][j] for j in srng.permutation(len(buckets[k]))]
                for k in keys}
    picked = {k: shuffled[k][:q] for k, q in zip(keys, quota)}
    tr_quota = allocate(N_TRAIN, [len(picked[k]) for k in keys])

    keep, train_idx = [], set()
    for k, q in zip(keys, tr_quota):
        keep.extend(picked[k])
        train_idx.update(picked[k][:q])
    keep.sort()

    W = [W[i] for i in keep]
    train = {j for j, i in enumerate(keep) if i in train_idx}

    out = []
    for i, w in enumerate(W):
        out.append({"seq_id": f"seq_{i + 1:04d}", "chrom": w["chrom"],
                    "abs_start": w["abs_start"],
                    "split": "train" if i in train else "test",
                    "dna": w["dna"], "trf": w["trues"]})
    json.dump({"source": "UCSC Genome Browser hg38 - simpleRepeat track "
                         "(Tandem Repeats Finder, Benson NAR 1999)",
               "assembly": "hg38", "track": "simpleRepeat",
               "gap": GAP, "span": list(SPAN), "h_band": [H_LO, H_HI],
               "flank_total": [FLK_TOT_LO, FLK_TOT_HI], "pos": [POS_LO, POS_HI],
               "seed": GLOBAL_SEED,
               "windows": out}, open(OUT, "w"))
    n_tr = sum(1 for w in out if w["split"] == "train")
    print(f"wrote {OUT}: {len(out)} windows, train {n_tr}, test {len(out) - n_tr}")


if __name__ == "__main__":
    main()
