"""Reference solution for Observing Conditions.

Approach: model the response surface directly. A CNN reads the clean cutout,
concatenates an embedding of the target morphology, and regresses all 400
normalised responses at once. At inference we take the argmax of the predicted
surface. Supervising on the whole surface rather than on its argmax is what
makes this trainable from a few thousand rows -- each row carries 400 numbers of
signal instead of one.

Two details that matter more than the architecture:

* the validation split is grouped by image_id, because one galaxy contributes up
  to seven rows and a row-level split leaks;
* augmentation is flips only. A 90-degree rotation exchanges the trail_h and
  trail_v conditions, so rotating the input without permuting the 400-vector
  target corrupts the labels.

    python reference_solution.py --data prepared/public --out working/submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

NPAIR = 400
SEED = 20260731


def device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))


class Surrogate(nn.Module):
    def __init__(self, ncls, npair=NPAIR, width=48):
        super().__init__()
        def blk(i, o, pool=True):
            L = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            if pool:
                L.append(nn.MaxPool2d(2))
            return nn.Sequential(*L)
        w = width
        self.trunk = nn.Sequential(blk(3, w), blk(w, 2 * w), blk(2 * w, 4 * w),
                                   blk(4 * w, 4 * w, pool=False),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.emb = nn.Embedding(ncls, 64)
        self.head = nn.Sequential(nn.Linear(4 * w + 64, 512), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(512, npair))

    def forward(self, x, t):
        return self.head(torch.cat([self.trunk(x), self.emb(t)], 1))


def load_bank(data, ids):
    """One uint8 image per galaxy; rows index into it. Rows outnumber galaxies
    ~6:1, so a tensor per row would waste most of the memory budget."""
    order = {iid: k for k, iid in enumerate(ids)}
    bank = np.zeros((len(ids), 69, 69, 3), np.uint8)
    for iid, k in order.items():
        bank[k] = np.asarray(Image.open(Path(data) / "images" / f"{iid}.png")
                             .convert("RGB"), dtype=np.uint8)
    return bank, order


def batch(bank, idx, mean, std, dev):
    a = np.ascontiguousarray(bank[idx].transpose(0, 3, 1, 2)).astype(np.float32) / 255.
    x = torch.from_numpy(a)
    return ((x - mean) / std).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prepared/public")
    ap.add_argument("--out", default="working/submission.csv")
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=48)
    a = ap.parse_args()
    dev = device()
    data = Path(a.data)

    tr = pd.read_csv(data / "train.csv")
    te = pd.read_csv(data / "test.csv")
    resp = pd.read_csv(data / "train_response.csv")
    resp = resp.set_index("row_id").loc[tr["row_id"]]
    Y = resp.to_numpy(np.float32)
    ncls = int(pd.read_csv(data / "classes.csv")["class_index"].max()) + 1

    ids = sorted(set(tr["image_id"]) | set(te["image_id"]))
    bank, order = load_bank(data, ids)
    tr_img = tr["image_id"].map(order).to_numpy()
    te_img = te["image_id"].map(order).to_numpy()
    tr_tgt = tr["target_class"].to_numpy()
    te_tgt = te["target_class"].to_numpy()

    px = bank.astype(np.float32) / 255.
    mean = torch.tensor(px.mean((0, 1, 2))).view(1, 3, 1, 1)
    std = torch.tensor(px.std((0, 1, 2)) + 1e-6).view(1, 3, 1, 1)
    del px

    # galaxy-grouped holdout, for an honest self-estimate before submitting
    rng = np.random.RandomState(SEED)
    uniq = np.unique(tr_img)
    hold = set(rng.permutation(uniq)[:max(1, len(uniq) // 6)].tolist())
    is_val = np.array([i in hold for i in tr_img])
    print(f"{len(tr)} train rows / {len(uniq)} galaxies · "
          f"holdout {is_val.sum()} rows · {len(te)} test rows · device {dev}")

    Yt = torch.from_numpy(Y)
    preds_te, val_scores = [], []
    for s in range(a.seeds):
        torch.manual_seed(SEED + s)
        np.random.seed(SEED + s)
        net = Surrogate(ncls, width=a.width).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
        fit = np.where(~is_val)[0]
        BS = 128
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=2.5e-3,
            total_steps=a.epochs * int(np.ceil(len(fit) / BS)))
        for ep in range(a.epochs):
            net.train()
            for i in range(0, len(fit), BS):
                b = fit[i:i + BS]
                x = batch(bank, tr_img[b], mean, std, dev)
                if np.random.rand() < 0.5:
                    x = torch.flip(x, dims=[3])
                if np.random.rand() < 0.5:
                    x = torch.flip(x, dims=[2])
                p = net(x, torch.from_numpy(tr_tgt[b]).to(dev))
                y = Yt[b].to(dev)
                loss = F.mse_loss(torch.sigmoid(p), y) * 10.0 \
                    + F.kl_div(F.log_softmax(p, 1), F.softmax(y * 8, 1),
                               reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
            np.random.shuffle(fit)
        vi = np.where(is_val)[0]
        pv = infer(net, bank, tr_img[vi], tr_tgt[vi], mean, std, dev)
        v = float(Y[vi, pv.argmax(1)].mean())
        val_scores.append(v)
        print(f"  seed {s}: holdout {v:.4f}", flush=True)
        preds_te.append(infer(net, bank, te_img, te_tgt, mean, std, dev))

    P = np.mean(preds_te, 0)
    pick = P.argmax(1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(row_id=te["row_id"], op1=pick // 20, op2=pick % 20)
                 ).to_csv(out, index=False)
    print(f"mean holdout {np.mean(val_scores):.4f} over {a.seeds} seed(s)")
    print(f"wrote {out}")


@torch.no_grad()
def infer(net, bank, img, tgt, mean, std, dev, bs=256):
    net.eval()
    o = []
    for i in range(0, len(img), bs):
        sl = slice(i, i + bs)
        o.append(net(batch(bank, img[sl], mean, std, dev),
                     torch.from_numpy(tgt[sl]).to(dev)).cpu().numpy())
    return np.concatenate(o)


if __name__ == "__main__":
    main()
