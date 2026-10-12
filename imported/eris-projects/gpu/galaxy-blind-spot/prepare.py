"""Turn the raw response atlas into the public/private split the platform ships.

Three filters, each closing a specific hole:

* Only galaxies the hidden model gets RIGHT when clean survive. Steering a model
  that is already wrong is not a well-posed question.
* Only (galaxy, target) rows where some pair actually reaches p_target >= 0.30
  survive, so every graded row is winnable rather than a coin flip on noise.
* The split is BY GALAXY, never by row. One galaxy contributes up to C-1 rows
  and those rows share an image, so a row-level split would put near-duplicate
  inputs on both sides and leak.

Row responses are min-max normalised here, at build time, so the grader is a
pure lookup and the metric cannot drift between grading runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REACHABLE = 0.30
TEST_FRACTION = 0.40
SEED = 20260731


def prepare(raw, public, private):
    raw, public, private = Path(raw), Path(public), Path(private)
    (public / "images").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(raw / "meta.csv")
    ops = pd.read_csv(raw / "ops.csv")
    R = np.load(raw / "responses.npy")              # [N, K, K, C] float16
    N, K, _, C = R.shape
    npair = K * K
    assert len(meta) == N, f"meta rows {len(meta)} != responses {N}"

    keep_img = meta["clean_correct"].to_numpy(bool)
    V = R.reshape(N, npair, C).astype(np.float32)

    rows = []
    for i in np.where(keep_img)[0]:
        true_c = int(meta["true_class"].iat[i])
        for c in range(C):
            if c == true_c:
                continue
            v = V[i, :, c]
            if v.max() < REACHABLE:
                continue
            lo, hi = float(v.min()), float(v.max())
            rows.append((int(i), meta["image_id"].iat[i], true_c, c,
                         (v - lo) / (hi - lo)))

    rows.sort(key=lambda r: (r[1], r[3]))
    print(f"galaxies {N} -> H-correct {keep_img.sum()} -> rows {len(rows)}")

    img_ids = sorted({r[1] for r in rows})
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(img_ids))
    n_test = int(round(TEST_FRACTION * len(img_ids)))
    test_imgs = {img_ids[j] for j in perm[:n_test]}

    rec, resp = [], []
    for k, (idx, iid, true_c, tgt_c, v) in enumerate(rows):
        rec.append(dict(row_id=k, image_id=iid, true_class=true_c,
                        target_class=tgt_c, split="test" if iid in test_imgs
                        else "train"))
        resp.append(v)
    df = pd.DataFrame(rec)
    Rn = np.stack(resp).astype(np.float32)
    rcols = [f"r{i}" for i in range(npair)]

    tr = df["split"] == "train"
    te = ~tr
    cols = ["row_id", "image_id", "true_class", "target_class"]

    df.loc[tr, cols].to_csv(public / "train.csv", index=False)
    df.loc[te, cols].to_csv(public / "test.csv", index=False)

    pd.concat([df.loc[tr, ["row_id"]].reset_index(drop=True),
               pd.DataFrame(np.round(Rn[tr.to_numpy()], 5), columns=rcols)],
              axis=1).to_csv(public / "train_response.csv", index=False)
    pd.concat([df.loc[te, ["row_id"]].reset_index(drop=True),
               pd.DataFrame(np.round(Rn[te.to_numpy()], 5), columns=rcols)],
              axis=1).to_csv(private / "answers.csv", index=False)

    sub = df.loc[te, ["row_id"]].copy()
    sub["op1"] = 0
    sub["op2"] = 0
    sub.to_csv(public / "sample_submission.csv", index=False)

    ops.to_csv(public / "ops.csv", index=False)
    for name in ("classes.csv", "catalog.py"):
        src = raw / name
        if src.exists():
            (public / name).write_bytes(src.read_bytes())

    used = set(df["image_id"])
    n_copied = 0
    for iid in sorted(used):
        src = raw / "images" / f"{iid}.png"
        (public / "images" / f"{iid}.png").write_bytes(src.read_bytes())
        n_copied += 1

    summary = dict(rows_train=int(tr.sum()), rows_test=int(te.sum()),
                   images_train=int(df.loc[tr, "image_id"].nunique()),
                   images_test=int(df.loc[te, "image_id"].nunique()),
                   images_copied=n_copied, n_pairs=npair, n_classes=int(C),
                   reachable_threshold=REACHABLE, seed=SEED)
    (public / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    prepare("dataset/raw", "prepared/public", "prepared/private")
