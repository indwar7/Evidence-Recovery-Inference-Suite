#!/usr/bin/env python3
"""Build the public/private split for "What Does It Do In There?".

RAW INPUT (dataset/generator/build_raw.py)
------------------------------------------
    raw/labels.jsonl          one record per drug label:
        record_id, set_id, generic_name, substance_name, route,
        manufacturer_name, pharm_class_moa, pharm_class_epc,
        pharmacology_masked, n_masked
    raw/mask_vocabulary.json  the corpus-wide mask vocabulary used

THE TASK
--------
The solver sees a label's Clinical Pharmacology prose with all mechanism
vocabulary masked, plus a per-row POOL of candidate mechanism-of-action
classes, and must select which of them FDA actually assigned to that drug.

    input   : pharmacology (masked), candidates (a pool, order shuffled)
    output  : moa_classes  (a subset of that row's pool)

WHY A PER-ROW POOL, AND NOT A FIXED LABEL SET
----------------------------------------------
With one global label set this would be multi-label classification over a
fixed vocabulary. Handing each row its own pool -- its true classes plus
sampled distractors, in shuffled order -- makes the submitted artifact a
SUBSET OF A GIVEN SET whose size is unknown and varies per row. The same
class is a correct answer for one row and a distractor for another, so
nothing can be memorised as a class prior.

THE POOL IS BUILT TO DEFEAT TWO SHORTCUTS, BOTH MEASURED
---------------------------------------------------------
1. Frequency. Distractors are sampled with probability proportional to how
   often a class appears as a TRUE class in train, so a common class is a
   common distractor too. Without this, "pick the most frequent candidate"
   is a strong strategy -- always naming the single most common class scored
   0.182 raw F1 on a sample before any of this was built.
2. Position. Pool order is shuffled with a per-row seed derived from the
   row id, so no position carries information. (Verified by the
   position-only probe in the baseline ladder.)

Pool size is fixed at POOL_SIZE so that pool size cannot itself signal how
many classes are true.

SPLIT
-----
By INGREDIENT GROUP, never by row. Many labels are the same drug from
different manufacturers, with near-identical pharmacology prose and identical
MoA classes; a row-wise split would put a generic and its brand-name twin on
opposite sides and leak the answer outright. Grouping by the exact ingredient
COMBINATION was not enough either: plain dexamethasone and a dexamethasone +
neomycin + polymyxin B product are different combinations of one substance
(measured: 30 of 165 test rows shared an ingredient with train). So every
ingredient is reduced to its moiety (salt/ester words dropped), labels that
share ANY ingredient are joined by union-find, and each resulting group lands
wholly in train or wholly in test. prepare.py asserts no test label shares an
ingredient with any train label.

Additionally:
  - Exact-duplicate masked texts are collapsed before splitting.
  - Near-duplicate texts (same 5-gram shingle signature) are collapsed too,
    because the same label is often resubmitted with trivial edits.
  - A class must have at least MIN_CLASS_SUPPORT labels to be kept at all,
    and must appear in BOTH splits, or it is dropped from pools and targets.

OUTPUTS
-------
    public/train.csv               row_id, route, pharmacology, candidates,
                                   moa_classes
    public/test.csv                row_id, route, pharmacology, candidates
    public/sample_submission.csv   row_id, moa_classes (first candidate)
    public/dataset_stats.json      split counts and build-time drops only --
                                   nothing derived from test labels
    public/validate_submission.py
    private/answers.csv            row_id, candidate_pool, moa_classes

    `candidate_pool` is test.csv's `candidates` under a DIFFERENT name: the
    grader needs the pool to compute the chance correction, and a shared
    column NAME between test.csv and answers.csv reads to the platform as a
    leaked target column. The two are asserted equal at build time.

No numeric column is published. Text length is derivable from the text, and
its long tail reads as an extreme-outlier column to the platform profiler.

Deterministic: fixed seed, sorted iteration, per-row RNG derived from row id.

Usage:
    python prepare.py [--raw dataset/raw] [--public dataset/public]
                      [--private dataset/private]
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260925
POOL_SIZE = 12            # fixed: pool size must not signal how many are true
MIN_CLASS_SUPPORT = 20    # a class needs this many labels to be kept
MIN_TEXT_CHARS = 800
SHINGLE_N = 5             # near-duplicate detection

csv.field_size_limit(10_000_000)
SEP = "|"
WORD = re.compile(r"[a-z0-9]+")


def load_raw(raw: Path):
    """Read every labels.jsonl under `raw`, de-duplicating by record_id."""
    cands = []
    direct = raw / "labels.jsonl"
    if direct.exists():
        cands.append(direct)
    for alt in sorted(raw.rglob("labels.jsonl")):
        if alt not in cands:
            cands.append(alt)
    if not cands:
        raise FileNotFoundError(
            f"labels.jsonl not found under {raw}. Run "
            f"dataset/generator/build_raw.py --out {raw} first.")
    by_id = {}
    for p in cands:
        with p.open(encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    by_id[r["record_id"]] = r
    return [by_id[k] for k in sorted(by_id)]


def shingle_sig(text, n=SHINGLE_N):
    """Order-sensitive signature over word n-grams, for near-duplicates."""
    w = WORD.findall(text.lower())
    if len(w) < n:
        return hashlib.sha256(" ".join(w).encode()).hexdigest()
    grams = {" ".join(w[i:i + n]) for i in range(0, len(w) - n + 1, 7)}
    top = sorted(grams)[:64]
    return hashlib.sha256("||".join(top).encode()).hexdigest()


SALT_WORDS = {
    "HYDROCHLORIDE", "HCL", "DIHYDROCHLORIDE", "HYDROBROMIDE", "SULFATE",
    "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "ACETATE", "ACETONIDE",
    "MALEATE", "BESYLATE", "MESYLATE", "TARTRATE", "BITARTRATE", "CITRATE",
    "PHOSPHATE", "SUCCINATE", "FUMARATE", "BROMIDE", "CHLORIDE", "NITRATE",
    "HYCLATE", "DIHYDRATE", "MONOHYDRATE", "TRIHYDRATE", "HEMIHYDRATE",
    "ANHYDROUS", "PROPIONATE", "DIPROPIONATE", "VALERATE", "BUTYRATE",
    "FUROATE", "LACTATE", "GLUCONATE", "TROMETHAMINE", "MEGLUMINE",
    "HEMIFUMARATE", "OXALATE", "TOSYLATE", "SUCCINATE", "ESTOLATE",
    "STEARATE", "PAMOATE", "DECANOATE", "ENANTHATE", "CYPIONATE",
}


def ingredients(rec):
    """Each active ingredient of a label, reduced to its moiety.

    Salt and ester words are dropped so DEXAMETHASONE SODIUM PHOSPHATE and
    DEXAMETHASONE are one ingredient. A name made only of salt words (e.g.
    SODIUM CHLORIDE) is kept as written.
    """
    out = set()
    for s in rec.get("substance_name") or []:
        words = [w for w in (s or "").upper().replace(",", " ").split()]
        core = [w for w in words if w not in SALT_WORDS]
        if words:
            out.add(" ".join(core or words))
    if not out:
        gn = (rec.get("generic_name") or "").strip().upper()
        out.add("GEN:" + gn if gn else "REC:" + rec["record_id"])
    return out


def ingredient_groups(records):
    """Union-find: labels sharing ANY ingredient belong to one group.

    Splitting on the exact ingredient combination is not enough -- a
    dexamethasone ophthalmic combination and plain dexamethasone are
    different combinations but the same substance. Measured on the previous
    build: 30 of 165 test rows shared an ingredient with some train row.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in records:
        ing = sorted(ingredients(r))
        for x in ing[1:]:
            ra, rb = find(ing[0]), find(x)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        find(ing[0])
    for r in records:
        r["_ing"] = ingredients(r)
        r["_sub"] = "GRP:" + find(sorted(r["_ing"])[0])


def prepare(raw, public, private):
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    records = load_raw(raw)
    print(f"raw labels: {len(records)}")

    # ---- filter + de-duplicate ----------------------------------------- #
    dropped = Counter()
    seen_exact, seen_shingle = set(), set()
    kept = []
    for r in records:
        text = (r.get("pharmacology_masked") or "").strip()
        moa = sorted(set(r.get("pharm_class_moa") or []))
        if not moa:
            dropped["no_moa"] += 1
            continue
        if len(text) < MIN_TEXT_CHARS:
            dropped["text_too_short"] += 1
            continue
        h = hashlib.sha256(text.encode("utf8")).hexdigest()
        if h in seen_exact:
            dropped["exact_duplicate_text"] += 1
            continue
        sig = shingle_sig(text)
        if sig in seen_shingle:
            dropped["near_duplicate_text"] += 1
            continue
        seen_exact.add(h)
        seen_shingle.add(sig)
        r["_text"] = text
        r["_moa"] = moa
        kept.append(r)
    print(f"kept {len(kept)} after dedup; dropped {dict(dropped)}")

    # ---- keep only classes with real support --------------------------- #
    support = Counter(c for r in kept for c in r["_moa"])
    classes = {c for c, n in support.items() if n >= MIN_CLASS_SUPPORT}
    for r in kept:
        r["_moa"] = [c for c in r["_moa"] if c in classes]
    kept = [r for r in kept if r["_moa"]]
    ingredient_groups(kept)
    print(f"classes with >= {MIN_CLASS_SUPPORT} support: {len(classes)}; "
          f"labels retained: {len(kept)}")
    if len(kept) < 300:
        raise SystemExit(
            f"only {len(kept)} usable labels -- let build_raw.py fetch more "
            f"before building")
    if len(classes) < POOL_SIZE:
        raise SystemExit(
            f"only {len(classes)} classes survive; need >= POOL_SIZE "
            f"({POOL_SIZE}) to build a pool")

    # ---- split by SUBSTANCE, targeting a ~20% test row ratio ----------- #
    by_sub = defaultdict(list)
    for r in kept:
        by_sub[r["_sub"]].append(r)
    # Target a test/train ROW ratio near 20%. Substances are very unevenly
    # sized, so a single greedy pass overshoots badly (measured: 33.7% on one
    # build, outside the 15-25% band). Instead, add substances one at a time
    # in a deterministic shuffled order and stop as soon as adding the next
    # one would take the ratio past the band -- then verify, and if the result
    # is still out of band, fall back to a strict ratio-driven pass.
    total = len(kept)
    rng = random.Random(SEED)
    order = sorted(by_sub, key=lambda s: (-len(by_sub[s]), s))
    rng.shuffle(order)

    def ratio_of(n_test):
        n_train = total - n_test
        return (n_test / n_train) if n_train > 0 else float("inf")

    TARGET, LO, HI = 0.20, 0.15, 0.25
    test_subs, acc = set(), 0
    for s in order:
        n = len(by_sub[s])
        if ratio_of(acc + n) <= HI:
            test_subs.add(s)
            acc += n
        if ratio_of(acc) >= TARGET:
            break

    if not (LO <= ratio_of(acc) <= HI):
        # Strict pass: smallest-first, stop the moment we enter the band.
        test_subs, acc = set(), 0
        for s in sorted(by_sub, key=lambda s: (len(by_sub[s]), s)):
            n = len(by_sub[s])
            if ratio_of(acc + n) > HI:
                continue
            test_subs.add(s)
            acc += n
            if ratio_of(acc) >= LO:
                break
    print(f"  test/train row ratio at split time: {ratio_of(acc):.1%}")
    train = [r for r in kept if r["_sub"] not in test_subs]
    test = [r for r in kept if r["_sub"] in test_subs]
    if not train or not test:
        raise SystemExit(f"empty split: {len(train)}/{len(test)}")

    # A class must be present in BOTH splits, or a test row could require a
    # class no training example ever demonstrates.
    tr_cls = {c for r in train for c in r["_moa"]}
    te_cls = {c for r in test for c in r["_moa"]}
    usable = tr_cls & te_cls
    if len(usable) < POOL_SIZE:
        raise SystemExit(
            f"only {len(usable)} classes appear in both splits; "
            f"need >= {POOL_SIZE}")
    for split in (train, test):
        for r in split:
            r["_moa"] = [c for c in r["_moa"] if c in usable]
    train = [r for r in train if r["_moa"]]
    test = [r for r in test if r["_moa"]]

    # Re-balance. The both-splits class filter above removes rows unevenly --
    # measured, it took a 24.2% split-time ratio to 33.7% final, outside the
    # band -- because the surviving classes are not distributed alike across
    # the two sides. Release whole test substances (largest first, so the
    # fewest substances move) back to train until the ratio is in band. Only
    # ever moves test -> train, so no train substance can reach test and the
    # substance-disjointness guarantee is preserved.
    def cur_ratio():
        return len(test) / len(train) if train else float("inf")

    if cur_ratio() > HI:
        test_by_sub = defaultdict(list)
        for r in test:
            test_by_sub[r["_sub"]].append(r)
        for s in sorted(test_by_sub, key=lambda x: (-len(test_by_sub[x]), x)):
            if cur_ratio() <= HI:
                break
            moving = test_by_sub[s]
            if len(test) - len(moving) < 10:
                break                      # keep test scoreable
            test = [r for r in test if r["_sub"] != s]
            train = train + moving
            test_subs.discard(s)
        train.sort(key=lambda r: r["record_id"])
        # Releasing substances can retire a class from test entirely; re-check.
        usable = ({c for r in train for c in r["_moa"]}
                  & {c for r in test for c in r["_moa"]})
        for split in (train, test):
            for r in split:
                r["_moa"] = [c for c in r["_moa"] if c in usable]
        train = [r for r in train if r["_moa"]]
        test = [r for r in test if r["_moa"]]
        if len(usable) < POOL_SIZE:
            raise SystemExit(
                f"after re-balancing only {len(usable)} classes appear in "
                f"both splits; need >= {POOL_SIZE}")

    ratio = cur_ratio()
    print(f"substances: {len(by_sub)} ({len(test_subs)} held out) | "
          f"classes in both splits: {len(usable)}")
    print(f"split: {len(train)} train / {len(test)} test (ratio {ratio:.1%})")
    if not (LO <= ratio <= HI):
        print(f"  WARNING: ratio {ratio:.1%} outside the {LO:.0%}-{HI:.0%} band")

    # ---- distractor sampling weights, from TRAIN ONLY ------------------ #
    # Weighted by true-class frequency so a common class is a common
    # distractor: this is what stops "pick the most frequent candidate".
    train_support = Counter(c for r in train for c in r["_moa"])
    pool_classes = sorted(usable)
    weights = [train_support.get(c, 1) for c in pool_classes]

    def build_pool(rec, rid):
        true = set(rec["_moa"])
        if len(true) >= POOL_SIZE:
            true = set(sorted(true)[:POOL_SIZE - 1])
        need = POOL_SIZE - len(true)
        # Per-row RNG from the row id: deterministic, and independent of
        # iteration order.
        r = random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:12], 16))
        avail = [(c, w) for c, w in zip(pool_classes, weights) if c not in true]
        picked = set()
        guard = 0
        while len(picked) < need and guard < 4000:
            guard += 1
            c = r.choices([a for a, _ in avail], [w for _, w in avail])[0]
            picked.add(c)
        if len(picked) < need:                      # tiny corpora
            for c in pool_classes:
                if len(picked) >= need:
                    break
                if c not in true:
                    picked.add(c)
        pool = sorted(true | picked)
        r.shuffle(pool)                             # position carries nothing
        return pool, sorted(true)

    def rid_for(i, split):
        return f"{split}_{i:05d}"

    for i, r in enumerate(sorted(train, key=lambda x: x["record_id"])):
        r["_rid"] = rid_for(i, "tr")
    for i, r in enumerate(sorted(test, key=lambda x: x["record_id"])):
        r["_rid"] = rid_for(i, "te")
    train.sort(key=lambda r: r["_rid"])
    test.sort(key=lambda r: r["_rid"])

    for r in train + test:
        r["_pool"], r["_true"] = build_pool(r, r["_rid"])

    # ---- asserts -------------------------------------------------------- #
    assert not ({r["_sub"] for r in train} & {r["_sub"] for r in test}), \
        "ingredient-group overlap across splits"
    tr_ing = set().union(*(r["_ing"] for r in train))
    assert not any(r["_ing"] & tr_ing for r in test), \
        "a test label shares an ingredient with train"
    assert not ({r["_text"] for r in train} & {r["_text"] for r in test}), \
        "identical pharmacology text across splits"
    for r in train + test:
        assert set(r["_true"]) <= set(r["_pool"]), "true class outside pool"
        assert len(r["_pool"]) == POOL_SIZE, "pool size not fixed"
        assert r["_true"], "row with no true class"

    route = lambda r: SEP.join(sorted(set(r.get("route") or []))) or "UNKNOWN"
    cols = ["row_id", "route", "pharmacology", "candidates"]

    with (public / "train.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["moa_classes"])
        w.writeheader()
        for r in train:
            w.writerow({"row_id": r["_rid"], "route": route(r),
                        "pharmacology": r["_text"],
                        "candidates": SEP.join(r["_pool"]),
                        "moa_classes": SEP.join(r["_true"])})

    with (public / "test.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["_rid"], "route": route(r),
                        "pharmacology": r["_text"],
                        "candidates": SEP.join(r["_pool"])})

    # answer-blind baseline: name the first candidate in the shuffled pool
    with (public / "sample_submission.csv").open(
            "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "moa_classes"])
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["_rid"], "moa_classes": r["_pool"][0]})

    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "candidate_pool",
                                          "moa_classes"])
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["_rid"],
                        "candidate_pool": SEP.join(r["_pool"]),
                        "moa_classes": SEP.join(r["_true"])})

    # ---- programmatic post-checks on the shipped files ------------------ #
    with (public / "test.csv").open(encoding="utf8") as f:
        t_rows = list(csv.DictReader(f))
    with (private / "answers.csv").open(encoding="utf8") as f:
        a_rows = list(csv.DictReader(f))
    assert len(t_rows) == len(a_rows)
    for t, a in zip(t_rows, a_rows):
        assert t["row_id"] == a["row_id"], "row_id misalignment"
        assert t["candidates"] == a["candidate_pool"], "pool mismatch"
    assert len({r["row_id"] for r in t_rows}) == len(t_rows), "ids not unique"
    shared = (set(t_rows[0]) & set(a_rows[0])) - {"row_id"}
    assert not shared, f"columns shared with answers.csv: {shared}"

    stats = {
        "n_train": len(train),
        "n_test": len(test),
        "test_train_ratio": round(ratio, 4),
        "pool_size": POOL_SIZE,
        "n_classes": len(usable),
        "n_ingredient_groups_train": len({r["_sub"] for r in train}),
        "n_ingredient_groups_test": len({r["_sub"] for r in test}),
        "split_unit": "ingredient_group",
        # Nothing here is derived from test labels: every count below is a
        # property of the inputs or of train only.
        "dropped_at_build": dict(dropped),
        "min_class_support": MIN_CLASS_SUPPORT,
        "metric": ("chance-corrected set F1: set F1 between predicted and "
                   "true classes, minus the expected F1 of a same-size "
                   "random draw from the row's pool, renormalised to [0,1]"),
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    here = Path(__file__).parent
    for cand in (here / "dataset" / "validate_submission.py",
                 here / "validate_submission.py"):
        if cand.exists():
            (public / "validate_submission.py").write_text(
                cand.read_text(encoding="utf8"), encoding="utf8")
            break

    print(f"public/train.csv              {len(train):,} rows")
    print(f"public/test.csv               {len(test):,} rows")
    print(f"public/sample_submission.csv  {len(test):,} rows")
    print(f"private/answers.csv           {len(test):,} rows")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    p.add_argument("--public", default=os.path.join(here, "dataset", "public"))
    p.add_argument("--private",
                   default=os.path.join(here, "dataset", "private"))
    a = p.parse_args()
    prepare(Path(a.raw), Path(a.public), Path(a.private))


if __name__ == "__main__":
    main()
