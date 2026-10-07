"""prepare.py -- Blind Spot: transform the raw dataset into public/private splits.

Deterministic by construction:
  * the split is a fixed-seed permutation over sorted image ids
  * every threshold is a module-level constant
  * no wall-clock, no unseeded randomness, no dict-ordering dependence
Running it twice produces byte-identical outputs.

Row construction
----------------
A graded row is a pair (image, target_class). A row is kept only when
  * the hidden model classifies the clean image correctly -- otherwise there is
    no intact prediction to steer away from, and
  * the target class is REACHABLE: at least one of the 400 ordered operation
    pairs drives p_H(target) to >= REACH -- otherwise the row is unwinnable and
    would only add noise to the leaderboard.

The split is BY IMAGE, not by row, so no test image appears in training under a
different target class.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260731
REACH = 0.30          # a target must be attainable to at least this probability
TEST_FRAC = 0.40
N_OPS = 20


def _rows_for(P, meta, keep_ids):
    """Build (image_id, target_class) rows and their 400-length response vectors."""
    id_to_ix = {v: i for i, v in enumerate(meta["image_id"].tolist())}
    recs, vecs = [], []
    for iid in keep_ids:
        i = id_to_ix[iid]
        y = int(meta["true_class"].iloc[i])
        flat = P[i].reshape(N_OPS * N_OPS, 10).astype(np.float32)
        for t in range(10):
            if t == y:
                continue
            v = flat[:, t]
            if v.max() < REACH:
                continue
            recs.append((iid, y, t))
            vecs.append(v)
    return recs, (np.stack(vecs) if vecs else np.zeros((0, N_OPS * N_OPS), np.float32))


def _response_frame(row_ids, vecs):
    cols = {f"r{k}": np.round(vecs[:, k], 4) for k in range(N_OPS * N_OPS)}
    return pd.DataFrame({"row_id": row_ids, **cols})


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    (public / "images").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(raw / "meta.csv")
    ops = pd.read_csv(raw / "ops.csv")
    P = np.load(raw / "responses.npy")

    # only images the hidden model gets right when clean
    usable = meta[meta["clean_correct"] == 1]["image_id"].tolist()
    usable = sorted(usable)                       # deterministic ordering

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(usable))
    n_test = int(round(TEST_FRAC * len(usable)))
    test_ids = sorted(usable[i] for i in perm[:n_test])
    train_ids = sorted(usable[i] for i in perm[n_test:])

    tr_recs, tr_vecs = _rows_for(P, meta, train_ids)
    te_recs, te_vecs = _rows_for(P, meta, test_ids)

    tr_ids = [f"tr_{i:06d}" for i in range(len(tr_recs))]
    te_ids = [f"te_{i:06d}" for i in range(len(te_recs))]

    train = pd.DataFrame(tr_recs, columns=["image_id", "true_class", "target_class"])
    train.insert(0, "row_id", tr_ids)
    test = pd.DataFrame(te_recs, columns=["image_id", "true_class", "target_class"])
    test.insert(0, "row_id", te_ids)

    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    _response_frame(tr_ids, tr_vecs).to_csv(public / "train_response.csv", index=False)
    ops.to_csv(public / "ops.csv", index=False)

    # sample submission: the identity/identity pair, a deliberately weak default
    pd.DataFrame({"row_id": te_ids, "op1": 0, "op2": 0}).to_csv(
        public / "sample_submission.csv", index=False)

    _response_frame(te_ids, te_vecs).to_csv(private / "answers.csv", index=False)

    # only images actually referenced by a kept row -- an image whose every
    # target is unreachable contributes no rows and would otherwise ship unused
    import shutil
    referenced = sorted(set(train["image_id"]) | set(test["image_id"]))
    for iid in referenced:
        shutil.copyfile(raw / "images" / f"{iid}.png", public / "images" / f"{iid}.png")

    # report images that actually produced rows -- an image whose every target
    # is unreachable is assigned to a split but contributes nothing
    print(f"train rows {len(train)}  ({train['image_id'].nunique()} images "
          f"with rows, of {len(train_ids)} assigned)")
    print(f"test  rows {len(test)}   ({test['image_id'].nunique()} images "
          f"with rows, of {len(test_ids)} assigned)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    prepare(here / "dataset" / "raw", here / "prepared" / "public",
            here / "prepared" / "private")
