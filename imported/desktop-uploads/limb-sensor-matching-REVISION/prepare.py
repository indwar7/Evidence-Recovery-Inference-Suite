"""Build the task split from the raw FORTH-TRACE archive.

    python prepare.py [--raw dataset/raw] [--public dataset/public] [--private dataset/private]

Deterministic: every random choice comes from a fixed salt, so a re-run produces the same
bytes. Reads only dataset/raw/FORTH_TRACE_DATASET.zip.

What it does:
1. Reads the 75 device files (15 participants x 5 sensors). Each device has its own clock and
   drops whole buffers now and then, which shows as a jump in its timestamps.
2. Puts every device of a participant on the chest sensor's clock: exact per-sample times
   from each gap-free segment, then one clock offset per device, found from the device's own
   activity labels (coarse) and a gyro-energy envelope (fine). Samples that fall in a gap are
   missing.
3. Keeps a participant only if every device's own activity labels agree with the chest
   sensor's on at least MIN_AGREE of the shared timeline after alignment (12 of 15 pass).
4. Splits participants: N_TEST for test, the rest for train.
5. Cuts non-overlapping WINDOW-sample windows in which all five devices are present and all
   five label the same quiet activity (stand, sit, sit and talk). Each limb window is cut
   with its own clock offset of up to JITTER samples either side of the chest window, and
   every bag stream has each channel's mean over its window removed.
6. Groups windows into bags: K people (K in 3..5), all in the same activity, each from a
   different participant. The chest stream of every person in a bag is a candidate; every
   limb stream is a row to predict, with the chest stream of its wearer as the answer.
7. Writes train recordings (whole aligned sessions of the training participants, every
   activity), train and test bags, their signals, the answer key, the sample submission and
   a submission validator.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FS = 51.2
DT = 1000.0 / FS
WINDOW = 205                     # 4.0 s
JITTER = 102                     # +-1.99 s, the limb clock offset relative to the chest window
MIN_AGREE = 0.93
N_TEST = 5
K_CHOICES = (3, 4, 5)
CHUNK = 512                      # max samples per train_recordings row
SALT = "limb-sensor-matching/v1"
QUIET = {1: "stand", 2: "sit", 3: "sit_and_talk"}
SITES = ["left_wrist", "right_wrist", "chest", "right_thigh", "left_ankle"]   # device 1..5
LIMBS = [0, 1, 3, 4]
COLS = ["dev", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "t", "act"]
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
ACT_NAMES = {1: "stand", 2: "sit", 3: "sit_and_talk", 4: "walk", 5: "walk_and_talk",
             6: "stairs", 7: "stairs_and_talk", 8: "stand_to_sit", 9: "sit_to_stand",
             10: "stand_to_sit_and_talk", 11: "sit_and_talk_to_stand", 12: "stand_to_walk",
             13: "walk_to_stand", 14: "stand_to_stairs", 15: "stairs_to_walk",
             16: "stairs_and_talk_to_walk_and_talk"}

# written to public/validate_submission.py; kept inline so prepare.py needs no other file
VALIDATOR = '''"""Check a submission against test.csv.

    python validate_submission.py submission.csv [test.csv]

A submission has one row per test stream: stream_id, wearer. wearer must be one of the
row's candidates in test.csv.
"""
import sys
from pathlib import Path

import pandas as pd


def validate(sub_path, test_path):
    sub = pd.read_csv(sub_path, dtype=str)
    test = pd.read_csv(test_path, dtype=str)
    errors = []
    if list(sub.columns) != ["stream_id", "wearer"]:
        errors.append(f"columns must be stream_id,wearer; got {list(sub.columns)}")
        return errors
    if sub.stream_id.duplicated().any():
        errors.append(f"{int(sub.stream_id.duplicated().sum())} duplicated stream_id rows")
    missing = set(test.stream_id) - set(sub.stream_id)
    extra = set(sub.stream_id) - set(test.stream_id)
    if missing:
        errors.append(f"{len(missing)} test stream_ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} stream_ids not in test.csv, e.g. {sorted(extra)[:3]}")
    if sub.wearer.isna().any():
        errors.append(f"{int(sub.wearer.isna().sum())} empty wearer values")
    cand = dict(zip(test.stream_id, test.candidates.str.split()))
    bad = [s for s, w in zip(sub.stream_id, sub.wearer) if s in cand and w not in cand[s]]
    if bad:
        errors.append(f"{len(bad)} rows whose wearer is not among the row's candidates, e.g. {bad[:3]}")
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    test_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent / "test.csv"
    errs = validate(sys.argv[1], test_path)
    if errs:
        print("INVALID"); print("\\n".join("- " + e for e in errs)); sys.exit(1)
    print("OK")
'''


def h(*parts):
    return hashlib.sha256((SALT + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def rng_for(*parts):
    return np.random.default_rng(int(h(*parts)[:16], 16))


# ---------------------------------------------------------------- alignment
def sample_times(t):
    t = np.asarray(t, float)
    cut = np.where(np.diff(t) > 100.0)[0] + 1
    out = np.empty_like(t)
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        k = np.arange(e - s)
        out[s:e] = np.mean(t[s:e] - k * DT) + k * DT
    return out


def to_grid(times, grid):
    """index of the device sample at each grid time (within half a period), else -1"""
    idx = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    j = np.where(np.abs(grid - times[idx - 1]) <= np.abs(grid - times[idx]), idx - 1, idx)
    return np.where(np.abs(times[j] - grid) <= DT / 2, j, -1)


def envelope(df):
    g = np.linalg.norm(df[["gx", "gy", "gz"]].values, axis=1)
    c = np.convolve(g, np.ones(26) / 26, "same")
    return (c - c.mean()) / (c.std() + 1e-9)


def align(dfs):
    T = {d: sample_times(dfs[d].t.values) for d in range(5)}
    grid = np.arange(T[2][0], T[2][-1], DT)
    j_ref = to_grid(T[2], grid); ok_ref = j_ref >= 0
    a_ref = np.where(ok_ref, dfs[2].act.values[j_ref], -1)
    e_ref = np.where(ok_ref, envelope(dfs[2])[j_ref], 0.0)
    offsets = {2: 0.0}
    for d in LIMBS:
        acts, env = dfs[d].act.values, envelope(dfs[d])
        best, best_off = -1.0, 0.0
        for off in np.arange(-180000.0, 180001.0, 500.0):
            j = to_grid(T[d] - off, grid); m = ok_ref & (j >= 0)
            if m.sum() < 5000:
                continue
            s = np.mean(acts[j[m]] == a_ref[m])
            if s > best:
                best, best_off = s, off
        fine, fine_off = -1e18, best_off
        for off in best_off + np.arange(-1000.0, 1000.0 + 1e-9, DT):
            j = to_grid(T[d] - off, grid); m = ok_ref & (j >= 0)
            if m.sum() < 5000:
                continue
            s = np.mean(e_ref[m] * env[j[m]])
            if s > fine:
                fine, fine_off = s, off
        offsets[d] = fine_off
    X = np.full((5, len(grid), 6), np.nan, np.float64)
    A = np.full((5, len(grid)), -1, np.int64)
    R = np.full((5, len(grid)), -1, np.int64)              # row of the device file behind each grid sample
    for d in range(5):
        j = to_grid(T[d] - offsets[d], grid); m = j >= 0
        X[d, m] = dfs[d][CH].values[j[m]]
        A[d, m] = dfs[d].act.values[j[m]]
        R[d, m] = j[m]
    valid = (A >= 0).all(0)
    agree = {d: float(np.mean(A[d][valid] == A[2][valid])) if valid.any() else 0.0 for d in range(5)}
    return X, A, offsets, agree, R


def find_zip(raw):
    """The platform may pass a raw root one level above or below the shipped one."""
    raw = Path(raw)
    for root in (raw, raw / "raw", raw / "dataset" / "raw", raw.parent):
        if (root / "FORTH_TRACE_DATASET.zip").exists():
            return root / "FORTH_TRACE_DATASET.zip"
    hits = sorted(raw.rglob("FORTH_TRACE_DATASET.zip")) + sorted(raw.parent.rglob("FORTH_TRACE_DATASET.zip"))
    if not hits:
        raise SystemExit(f"FORTH_TRACE_DATASET.zip not found under {raw}")
    return hits[0]


def load_all(raw):
    z = zipfile.ZipFile(find_zip(raw))
    out = {}
    for p in range(15):
        dfs = {}
        for d in range(5):
            name = f"FORTH_TRACE_DATASET-master/part{p}/part{p}dev{d + 1}.csv"
            dfs[d] = pd.read_csv(io.BytesIO(z.read(name)), header=None, names=COLS)
        out[p] = align(dfs)
    return out


# ---------------------------------------------------------------- windows and bags
def quiet_windows(X, A, p):
    """non-overlapping windows (start on the chest clock) inside one quiet activity, with the
    jitter margin present on every device"""
    present = ~np.isnan(X).any(axis=(0, 2))
    out = []
    for a in QUIET:
        ok = present & (A == a).all(0)
        run = np.convolve(ok.astype(np.int64), np.ones(WINDOW + 2 * JITTER, np.int64), "valid")
        good = run == WINDOW + 2 * JITTER                 # window plus margin fully inside
        s = 0
        while s < len(good):
            if good[s]:
                out.append((a, s + JITTER))
                s += WINDOW + 2 * JITTER                  # next window starts after this margin
            else:
                s += 1
    return out


def make_bags(items, split):
    """items: list of (participant, activity, start). Bags of K distinct participants, same activity."""
    bags = []
    for a in QUIET:
        pool = {}
        for p, aa, s in items:
            if aa == a:
                pool.setdefault(p, []).append(s)
        r = rng_for("bags", split, a)
        for p in pool:
            r.shuffle(pool[p])
        n = 0
        while True:
            alive = [p for p in sorted(pool) if pool[p]]
            k = int(r.choice(K_CHOICES))
            if len(alive) < min(K_CHOICES):
                break
            k = min(k, len(alive))
            # take from the participants with the most windows left, ties broken by a salted hash
            alive.sort(key=lambda p: (-len(pool[p]), h("tie", split, a, n, p)))
            chosen = alive[:k]
            bags.append((a, [(p, pool[p].pop()) for p in chosen]))
            n += 1
    return bags


def demean(w):
    """remove each channel's mean over the window: the stream keeps its motion but not the
    sensor's orientation or its offset, which identify a recording session rather than a body"""
    return w - w.mean(0, keepdims=True)


def provenance(data, p, d, s):
    r = data[p][4][d, s:s + WINDOW]
    assert (r >= 0).all()
    contiguous = bool((np.diff(r) == 1).all())
    return [p, d + 1, int(r[0]), int(r[-1])] if contiguous else [p, d + 1, [int(x) for x in r]]


def emit(bags, data, split, prov):
    """streams + rows for one split; prov collects each stream's archive participant, device and rows"""
    rows, sig, ans = [], [], []
    for b_i, (a, members) in enumerate(bags):
        bag_id = f"{split[:2]}b{h('bag', split, b_i)[:8]}"
        r = rng_for("emit", split, b_i)
        chest_ids = {}
        for p, s in members:
            X = data[p][0]
            cid = "s" + h("chest", split, b_i, p, s)[:10]
            chest_ids[p] = cid
            sig.append((cid, demean(X[2, s:s + WINDOW])))
            prov[cid] = provenance(data, p, 2, s)
        cand = list(chest_ids.values()); r.shuffle(cand)
        for p, s in members:
            X = data[p][0]
            for d in LIMBS:
                off = int(r.integers(-JITTER, JITTER + 1))
                lid = "s" + h("limb", split, b_i, p, s, d)[:10]
                sig.append((lid, demean(X[d, s + off:s + off + WINDOW])))
                prov[lid] = provenance(data, p, d, s + off)
                rows.append({"stream_id": lid, "bag_id": bag_id, "activity": QUIET[a], "site": SITES[d],
                             "candidates": " ".join(cand), "wearer": chest_ids[p], "n_candidates": len(cand)})
    return rows, sig


def pack(col):
    """one channel of one stream as space-separated values, 4 decimals"""
    return " ".join(f"{v:.4f}" for v in np.round(col, 4))


def write_signals(sig, path):
    """one row per stream; each channel cell holds the stream's WINDOW samples in time order"""
    rows = []
    for sid, w in sorted(sig, key=lambda x: h("order", x[0])):
        rows.append({"stream_id": sid, **{c: pack(w[:, k]) for k, c in enumerate(CH)}})
    pd.DataFrame(rows, columns=["stream_id"] + CH).to_csv(path, index=False)


def prepare(raw, public, private):
    public, private = Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    data = load_all(raw)
    usable = [p for p in range(15) if min(data[p][3].values()) >= MIN_AGREE]
    order = sorted(usable, key=lambda p: h("split", p))
    test_p, train_p = sorted(order[:N_TEST]), sorted(order[N_TEST:])

    stats = {"participants": 15, "usable": len(usable), "dropped_alignment": [p for p in range(15) if p not in usable],
             "train_participants": len(train_p), "test_participants": len(test_p)}
    alias = {p: f"r{i + 1:02d}" for i, p in enumerate(sorted(train_p, key=lambda p: h("alias", p)))}

    # ---- train recordings: whole aligned sessions, every activity
    rec = []
    for p in sorted(train_p, key=lambda p: alias[p]):
        X, A = data[p][0], data[p][1]
        for d in range(5):
            m = ~np.isnan(X[d]).any(1)
            idx = np.where(m)[0]
            if not len(idx):
                continue
            # a chunk is a run of consecutive samples with one activity, at most CHUNK long
            brk = np.where((np.diff(idx) != 1) | (np.diff(A[d][idx]) != 0))[0] + 1
            for s0, e0 in zip(np.r_[0, brk], np.r_[brk, len(idx)]):
                for c0 in range(s0, e0, CHUNK):
                    j = idx[c0:min(c0 + CHUNK, e0)]
                    rec.append({"recording": alias[p], "site": SITES[d], "start_i": int(j[0]),
                                "activity": ACT_NAMES.get(int(A[d][j[0]]), "unknown"),
                                **{c: pack(X[d][j, k]) for k, c in enumerate(CH)}})
    pd.DataFrame(rec).to_csv(public / "train_recordings.csv", index=False)

    # ---- bags
    out = {}; prov = {}
    for split, plist in (("train", train_p), ("test", test_p)):
        items = [(p, a, s) for p in plist for a, s in quiet_windows(data[p][0], data[p][1], p)]
        bags = make_bags(items, split)
        rows, sig = emit(bags, data, split, prov)
        df = pd.DataFrame(rows)
        df = df.iloc[sorted(range(len(df)), key=lambda i: h("row", split, df.stream_id.iat[i]))].reset_index(drop=True)
        out[split] = (df, sig, bags)
        write_signals(sig, public / f"{split}_signals.csv")
        stats[f"{split}_bags"] = len(bags)
        stats[f"{split}_rows"] = len(df)
        stats[f"{split}_bags_by_k"] = {str(k): sum(len(m) == k for _, m in bags) for k in K_CHOICES}
        stats[f"{split}_bags_by_activity"] = {QUIET[a]: sum(aa == a for aa, _ in bags) for a in QUIET}

    tr, te = out["train"][0], out["test"][0]
    for df in (tr, te):                       # n_candidates in the answer key is the candidate count
        assert (df.candidates.str.split().str.len() == df.n_candidates).all()
        assert df.apply(lambda r: r.wearer in r.candidates.split(), axis=1).all()
    tr[["stream_id", "bag_id", "activity", "site", "candidates", "wearer"]].to_csv(public / "train.csv", index=False)
    te[["stream_id", "bag_id", "activity", "site", "candidates"]].to_csv(public / "test.csv", index=False)
    te[["stream_id", "wearer", "n_candidates"]].to_csv(private / "answers.csv", index=False)
    sub = te[["stream_id"]].copy(); sub["wearer"] = te.candidates.str.split().str[0]
    sub.to_csv(public / "sample_submission.csv", index=False)
    stats.update({"window_samples": WINDOW, "sample_rate_hz": FS, "jitter_samples": JITTER,
                  "channels": CH, "sites": SITES})
    (public / "dataset_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    (public / "validate_submission.py").write_text(VALIDATOR)
    # private: the participant split and the clock offsets, for audit
    (private / "split_manifest.json").write_text(json.dumps({
        "test_participants_archive_ids": test_p, "train_participants_archive_ids": train_p,
        "train_aliases": {alias[p]: p for p in train_p},
        "offsets_ms": {p: {SITES[d]: round(o, 1) for d, o in data[p][2].items()} for p in range(15)},
        "label_agreement": {p: {SITES[d]: round(a, 4) for d, a in data[p][3].items()} for p in range(15)},
        "streams": {sid: prov[sid] for sid in sorted(prov)},
    }) + "\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--public", default="dataset/public")
    ap.add_argument("--private", default="dataset/private")
    a = ap.parse_args()
    print(json.dumps(prepare(a.raw, a.public, a.private), indent=1))
