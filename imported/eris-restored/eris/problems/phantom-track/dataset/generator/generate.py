"""Deterministic generator for Phantom Track — appearance-free identity re-linking.

A *scene* shows N anonymous objects at two instants, frame A (time t) and frame B
(time t+15 s), as 2-D positions only. Each object's motion between the frames is a
REAL Porto-taxi 15-second displacement vector drawn from the bundled pool
(`motion_pool.npz`); only the placement of the objects is synthetic, which is what
lets us dial confusability to a target band and certify a ceiling.

Two kinds of objects per scene:
  * soft objects  — starts placed uniformly in a `box`x`box` metre square, each given
    a real displacement. Density (box) controls how often objects are confusable, and
    is what separates naive nearest-neighbour matching from global assignment.
  * hard pairs    — two objects sharing an identical start point (and each a real
    displacement). Their two frame-A points coincide, so *which* A maps to *which* B
    is information-theoretically a coin flip: ANY matcher, even one that knows the
    generative process, scores at most 1/2 on such a pair. A controlled number of hard
    pairs certifies a provable upper bound on achievable accuracy, < 1.

Ground truth is the identity permutation by construction; frame-B points are emitted
in a scene-shuffled order so position order carries no information.
"""
import json, os
import numpy as np

POOL = os.path.join(os.path.dirname(__file__), "..", "raw", "motion_pool.npz")

# ---- calibrated scene parameters (see meta.json anchors) -------------------
N          = 34        # objects per scene
BOX        = 2000.0    # metres; placement density dial for soft objects
N_PAIRS    = 4         # hard exchangeable pairs per scene (=> 8 hard objects)
N_TRAIN    = 4000      # supervised scenes (matches revealed)
N_TEST     = 2500      # evaluation scenes (matches hidden)
SEED       = 20260718


def _load_pool():
    d = np.load(POOL)
    return d["dxy"].astype(np.float64)  # (M, 2) real displacement vectors, metres


def _scene(dxy, rng):
    """Return (A, B, perm): A[i] frame-A pos of object i, B[perm[i]] its frame-B pos.

    We build objects in canonical order then emit B in shuffled order; `truth[a]` is
    the frame-B index holding object a.
    """
    n_hard = 2 * N_PAIRS
    n_soft = N - n_hard
    # real displacements for every object
    idx = rng.integers(0, len(dxy), N)
    d = dxy[idx]
    A = np.empty((N, 2))
    # soft objects: independent uniform starts
    A[:n_soft] = rng.uniform(0.0, BOX, (n_soft, 2))
    # hard pairs: each pair shares one start point -> coincident frame-A points
    for p in range(N_PAIRS):
        s = rng.uniform(0.0, BOX, 2)
        A[n_soft + 2 * p] = s
        A[n_soft + 2 * p + 1] = s
    B_canon = A + d                      # frame-B position of object i (canonical order)
    # emit B in shuffled order; truth[a] = shuffled index of object a
    order = rng.permutation(N)           # order[k] = canonical object at shuffled slot k
    B = B_canon[order]
    truth = np.empty(N, dtype=int)
    truth[order] = np.arange(N)          # inverse permutation
    return A, B, truth


def generate(seed=SEED, n_train=N_TRAIN, n_test=N_TEST):
    dxy = _load_pool()
    out = {}
    for split, n_scenes, base in [("train", n_train, 0), ("test", n_test, 1)]:
        rng = np.random.default_rng(seed + base)
        pts_rows, match_rows = [], []
        for s in range(n_scenes):
            sid = f"{split}_{s:05d}"
            A, B, truth = _scene(dxy, rng)
            for i in range(N):
                pts_rows.append((sid, "A", i, round(float(A[i, 0]), 2), round(float(A[i, 1]), 2)))
            for j in range(N):
                pts_rows.append((sid, "B", j, round(float(B[j, 0]), 2), round(float(B[j, 1]), 2)))
            for a in range(N):
                match_rows.append((sid, a, int(truth[a])))
        out[split] = (pts_rows, match_rows)
    return out


if __name__ == "__main__":
    import pandas as pd
    raw = os.path.join(os.path.dirname(__file__), "..", "raw")
    os.makedirs(raw, exist_ok=True)
    data = generate()
    for split, (pts, matches) in data.items():
        pd.DataFrame(pts, columns=["scene_id", "frame", "point_id", "x", "y"]).to_csv(
            os.path.join(raw, f"points_{split}.csv"), index=False)
        pd.DataFrame(matches, columns=["scene_id", "a_id", "b_id"]).to_csv(
            os.path.join(raw, f"matches_{split}.csv"), index=False)
    meta = dict(N=N, box=BOX, n_pairs=N_PAIRS, n_train=N_TRAIN, n_test=N_TEST, seed=SEED,
                source="kraina/porto_taxi (CC-BY-4.0); real 15s taxi displacements",
                provable_upper_bound=round((N - N_PAIRS) / N, 4))
    json.dump(meta, open(os.path.join(raw, "meta.json"), "w"), indent=2)
    print("generated:", {k: (len(v[0]), len(v[1])) for k, v in data.items()})
    print("provable upper bound = (N - n_pairs)/N =", (N - N_PAIRS) / N)
