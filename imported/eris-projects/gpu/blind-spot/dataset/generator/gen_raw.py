"""Generate the RAW dataset artifacts for the Blind Spot challenge.

Writes, for a pool of images disjoint from the hidden model's training data:
  raw/images/<image_id>.png          the image
  raw/meta.csv                       image_id, true_class, clean_correct
  raw/responses.npy                  (n, 20, 20, 10) float16 -- p_H(class | pair)
  raw/ops.csv                        op_index, op_name, strength_param
  raw/meta.json                      provenance

prepare.py then turns these into public/ and private/ splits. Keeping the
expensive model evaluation here means prepare.py is fast and obviously
deterministic.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import Net, splits, to_tensor, device
import cat3

N_IMAGES = 6000
OUT = Path("../dataset/raw")


def main():
    params = json.load(open("cat3_params.json"))
    CAT = cat3.build(params)
    K = len(CAT)
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    hid, tr, te = splits()
    pool = tr + te                      # both disjoint from the hidden-model pool
    rng = np.random.RandomState(20260731)
    sel = rng.permutation(len(pool))[:N_IMAGES]
    files = [pool[i] for i in sorted(sel)]

    dev = device()
    net = Net().to(dev)
    net.load_state_dict(torch.load("hidden_robust.pt", map_location=dev))
    net.eval()

    n = len(files)
    P = np.zeros((n, K, K, 10), dtype=np.float16)
    clean = np.zeros((n, 10), dtype=np.float32)
    labels = np.array([c for _, c in files], dtype=np.int64)
    ids = []

    t0 = time.time()
    CH = 20
    for s in range(0, n, CH):
        chunk = files[s:s + CH]
        batch = np.zeros((len(chunk) * K * K, 64, 64, 3), dtype=np.uint8)
        cb = np.zeros((len(chunk), 64, 64, 3), dtype=np.uint8)
        w = 0
        for ci, (p, _) in enumerate(chunk):
            im = Image.open(p).convert("RGB")
            cb[ci] = np.asarray(im)
            iid = p.stem                       # e.g. Forest_864 -- stable, unique
            ids.append(iid)
            im.save(OUT / "images" / f"{iid}.png")
            for i, (_, f1) in enumerate(CAT):
                mid = f1(im, iid)
                for j, (_, f2) in enumerate(CAT):
                    batch[w] = np.asarray(f2(mid, iid).convert("RGB"))
                    w += 1
        with torch.no_grad():
            clean[s:s + len(chunk)] = torch.softmax(
                net(to_tensor(cb).to(dev)), 1).cpu().numpy()
            pr = np.zeros((len(batch), 10), dtype=np.float16)
            for i in range(0, len(batch), 1024):
                pr[i:i + 1024] = torch.softmax(
                    net(to_tensor(batch[i:i + 1024]).to(dev)), 1
                ).cpu().numpy().astype(np.float16)
        P[s:s + len(chunk)] = pr.reshape(len(chunk), K, K, 10)

        if s % 400 == 0:
            el = time.time() - t0
            r = (s + len(chunk)) / max(el, 1e-9)
            print(f"  {s+len(chunk)}/{n}  {el/60:.1f}m  {r:.1f}/s  "
                  f"eta {(n-s-len(chunk))/max(r,1e-9)/60:.1f}m", flush=True)

    np.save(OUT / "responses.npy", P)
    import pandas as pd
    pd.DataFrame({
        "image_id": ids,
        "true_class": labels,
        "clean_correct": (clean.argmax(1) == labels).astype(int),
        "clean_p_true": clean[np.arange(n), labels].round(6),
    }).to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({
        "op_index": range(K),
        "op_name": cat3.NAMES,
        "strength_param": [params[nm] for nm in cat3.NAMES],
    }).to_csv(OUT / "ops.csv", index=False)
    json.dump({
        "source": "EuroSAT RGB (Helber et al.), Sentinel-2, MIT licence",
        "source_url": "https://zenodo.org/records/7711810",
        "n_images": int(n),
        "n_ops": int(K),
        "hidden_model": "4-block CNN, augmentation-trained, 95.85% clean val acc",
        "op_calibration": "each op tuned to ~8% solo break rate on a held-out "
                          "calibration split from the hidden-model pool",
        "seed": 20260731,
    }, open(OUT / "meta.json", "w"), indent=1)
    print(f"done {n} images in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
