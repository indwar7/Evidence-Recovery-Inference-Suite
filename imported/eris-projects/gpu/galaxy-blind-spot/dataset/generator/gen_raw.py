"""Build dataset/raw/ -- the measured response atlas. Deterministic; seed 20260731.

For every released galaxy and every ordered pair (op1, op2) drawn from the
20-condition catalog, render the degraded image and record the hidden model's
full class-probability vector. 400 measurements per galaxy, built with 420 op
applications instead of 800 by reusing the first-stage render.

    python gen_raw.py --h5 Galaxy10.h5 --n 4500 --out ../raw

Emits:
  images/<image_id>.png   the clean galaxy, as shipped
  meta.csv                image_id, true_class, clean_correct, clean_p_true
  ops.csv                 the catalog with its calibrated strengths
  classes.csv             class_index -> Galaxy Zoo morphology name
  catalog.py              the op implementations, verbatim, for solvers
  responses.npy           [N, 20, 20, C] float16
  meta.json               provenance and generation parameters
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import catalog
from model import Net, splits, to_tensor, device, SEED

HERE = Path(__file__).resolve().parent
dev = device()


@torch.no_grad()
def probs(net, arr_uint8, mean, std, bs=400):
    out = []
    for i in range(0, len(arr_uint8), bs):
        out.append(net(to_tensor(arr_uint8[i:i + bs], mean, std).to(dev))
                   .softmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(HERE / "Galaxy10.h5"))
    ap.add_argument("--n", type=int, default=4500, help="0 = whole pool")
    ap.add_argument("--out", default=str(HERE.parent / "raw"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    _, (Xp, yp), kept, names = splits(a.h5)
    st = np.load(HERE / "norm_stats.npz")
    mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
    calib = json.load(open(HERE / "calib.json"))

    net = Net(len(names)).to(dev)
    net.load_state_dict(torch.load(HERE / "hidden.pt", map_location=dev))
    net.eval()

    idx = np.arange(len(Xp))
    if a.n:
        idx = np.sort(np.random.RandomState(SEED).choice(idx, a.n, replace=False))
    X, y = Xp[idx], yp[idx]
    cat = catalog.build(calib)
    K = len(cat)
    print(f"{len(X)} galaxies x {K}x{K} pairs = {len(X)*K*K:,} measured responses")

    clean = probs(net, X, mean, std)
    R = np.zeros((len(X), K, K, len(names)), np.float16)
    ids = []
    for i, gi in enumerate(idx):
        image_id = f"g{int(gi):06d}"
        ids.append(image_id)
        base = Image.fromarray(X[i])
        base.save(out / "images" / f"{image_id}.png")
        first = [fn(base, image_id) for _, fn in cat]
        stage2 = np.stack([np.asarray(fn(f1, image_id), dtype=np.uint8)
                           for f1 in first for _, fn in cat])
        R[i] = probs(net, stage2, mean, std).reshape(K, K, -1).astype(np.float16)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(X)}", flush=True)

    np.save(out / "responses.npy", R)

    import pandas as pd
    pd.DataFrame(dict(
        image_id=ids, true_class=y,
        clean_correct=(clean.argmax(1) == y),
        clean_p_true=np.round(clean[np.arange(len(y)), y], 5),
    )).to_csv(out / "meta.csv", index=False)

    pd.DataFrame(dict(
        op_index=range(K), op_name=catalog.NAMES,
        family=[catalog.FAMILY[n] for n in catalog.NAMES],
        strength_param=[round(calib[n], 6) for n in catalog.NAMES],
        description=[(catalog.FN[n].__doc__ or "").strip().split("\n")[0]
                     for n in catalog.NAMES],
    )).to_csv(out / "ops.csv", index=False)

    pd.DataFrame(dict(class_index=range(len(names)), morphology=names,
                      galaxy10_index=kept)).to_csv(out / "classes.csv", index=False)

    shutil.copy(HERE / "catalog.py", out / "catalog.py")

    (out / "meta.json").write_text(json.dumps(dict(
        seed=SEED, n_galaxies=int(len(X)), n_ops=K, n_pairs=K * K,
        n_classes=len(names), classes=names, galaxy10_indices=kept,
        source="Galaxy10 SDSS (astroNN), Zenodo 10844811, CC-BY-4.0",
        source_url="https://zenodo.org/records/10844811",
        hidden_model="4-block CNN, augmentation-hardened; weights withheld",
        clean_accuracy=float((clean.argmax(1) == y).mean()),
        calibration="each op bisected to an equal solo break rate",
        strengths=calib,
    ), indent=1))
    print(f"clean accuracy on released pool {float((clean.argmax(1)==y).mean()):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
