"""Train a ROBUST hidden model H.

Augmentation covers the same *families* as the catalog (geometric, photometric,
blur, noise) but at strengths inside the catalog's range, so H ends up robust
to any single catalog op while retaining idiosyncratic, image-specific weak
points where two ops interact.

All augmentation is done as tensor ops so it stays fast on MPS.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from common import Net, splits, device, SEED, MEAN, STD

torch.manual_seed(SEED)
np.random.seed(SEED)
dev = device()
M = torch.tensor(MEAN).view(1, 3, 1, 1)
S = torch.tensor(STD).view(1, 3, 1, 1)


def load(files):
    a = np.zeros((len(files), 64, 64, 3), dtype=np.uint8)
    y = np.zeros(len(files), dtype=np.int64)
    for i, (p, c) in enumerate(files):
        a[i] = np.asarray(Image.open(p).convert("RGB"))
        y[i] = c
    return a, y


def gauss_kernel(sigma, dev):
    r = 2
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    return k


def augment(x):
    """x: (N,3,64,64) float in [0,1] on device. Returns augmented copy."""
    n = x.shape[0]
    # geometric: flips + rot90
    if True:
        f = torch.rand(n, device=x.device) < 0.5
        x[f] = torch.flip(x[f], dims=[3])
        f = torch.rand(n, device=x.device) < 0.5
        x[f] = torch.flip(x[f], dims=[2])
        for k in range(1, 4):
            m = (torch.rand(n, device=x.device) < 0.25)
            if m.any():
                x[m] = torch.rot90(x[m], k, dims=[2, 3])
    # photometric: brightness / contrast / saturation, per-sample
    b = (0.85 + 0.30 * torch.rand(n, 1, 1, 1, device=x.device))
    x = x * b
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    c = (0.80 + 0.40 * torch.rand(n, 1, 1, 1, device=x.device))
    x = (x - mean) * c + mean
    gray = x.mean(dim=1, keepdim=True)
    s = (0.70 + 0.60 * torch.rand(n, 1, 1, 1, device=x.device))
    x = gray + (x - gray) * s
    # channel gain (warm/cool family)
    g = (0.95 + 0.10 * torch.rand(n, 3, 1, 1, device=x.device))
    x = x * g
    # blur on a subset
    m = torch.rand(n, device=x.device) < 0.35
    if m.any():
        k = gauss_kernel(0.6, x.device)
        xb = x[m]
        kh = k.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        kv = k.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        xb = F.conv2d(F.pad(xb, (2, 2, 0, 0), mode="reflect"), kh, groups=3)
        xb = F.conv2d(F.pad(xb, (0, 0, 2, 2), mode="reflect"), kv, groups=3)
        x[m] = xb
    # noise
    m = torch.rand(n, device=x.device) < 0.35
    if m.any():
        x[m] = x[m] + torch.randn_like(x[m]) * 0.03
    return x.clamp(0, 1)


def main():
    hid, tr, te = splits()
    Xa, ya = load(hid)
    n_val = 2000
    Xv, yv = Xa[:n_val], ya[:n_val]
    Xt, yt = Xa[n_val:], ya[n_val:]

    net = Net().to(dev)
    EP = 30
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EP * ((len(Xt) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    Xt_t = torch.from_numpy(Xt.transpose(0, 3, 1, 2).copy()).float().div_(255.)
    Xv_t = torch.from_numpy(Xv.transpose(0, 3, 1, 2).copy()).float().div_(255.)
    Xv_t = ((Xv_t - M) / S).to(dev)
    yv_t = torch.from_numpy(yv).to(dev)
    Md, Sd = M.to(dev), S.to(dev)

    for ep in range(EP):
        net.train()
        perm = np.random.permutation(len(Xt_t))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            xb = Xt_t[b].to(dev).clone()
            xb = augment(xb)
            xb = (xb - Md) / Sd
            yb = torch.from_numpy(yt[b]).to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0 or ep == EP - 1:
            net.eval()
            with torch.no_grad():
                acc = 0
                for i in range(0, len(Xv_t), 512):
                    acc += (net(Xv_t[i:i + 512]).argmax(1) == yv_t[i:i + 512]).sum().item()
            print(f"  ep{ep + 1:2d}  loss {tot / len(perm):.4f}  val_acc {acc / len(Xv):.4f}",
                  flush=True)

    torch.save(net.state_dict(), "hidden_robust.pt")
    print("saved hidden_robust.pt")


if __name__ == "__main__":
    main()
