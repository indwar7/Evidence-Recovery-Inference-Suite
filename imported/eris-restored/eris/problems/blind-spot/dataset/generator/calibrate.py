"""Bisect each op's strength so all ops have equal BREAK RATE applied alone.

Break rate = fraction of clean-correct images the op alone pushes below
p_true < 0.5. Unlike mean damage it has a floor near zero, so the target is
meaningful. Equalising it removes the "which op is strongest" confound that
made the best pair predictable from the class alone.

Calibration uses images from the HIDDEN-model pool only -- never the challenge
test split -- so no test information reaches the catalog definition.
"""
import json
import numpy as np
import torch
from PIL import Image
from common import Net, splits, to_tensor, device
from cat3 import OPS, FN

TARGET = 0.08
NCAL = 400

dev = device()
net = Net().to(dev)
net.load_state_dict(torch.load("hidden_robust.pt", map_location=dev))
net.eval()

hid, tr, te = splits()
cal = hid[:NCAL]
imgs = [(Image.open(p).convert("RGB"), p.name, c) for p, c in cal]
ys = np.array([c for _, _, c in imgs])


def probs_for(fn, p):
    batch = np.zeros((len(imgs), 64, 64, 3), dtype=np.uint8)
    for i, (im, key, _) in enumerate(imgs):
        batch[i] = np.asarray(fn(im, key, p).convert("RGB"))
    with torch.no_grad():
        return torch.softmax(net(to_tensor(batch).to(dev)), 1).cpu().numpy()


# restrict to images the model gets right when clean
pc = probs_for(FN["identity"], 0.0)
ok = pc.argmax(1) == ys
print(f"clean-correct calibration images: {ok.sum()}/{len(ys)}")
print(f"identity break rate (floor): {(pc[np.arange(len(ys)), ys][ok] < 0.5).mean():.4f}")
print(f"target break rate: {TARGET}\n")


def break_rate(fn, p):
    pr = probs_for(fn, p)
    return float((pr[np.arange(len(ys)), ys][ok] < 0.5).mean())


params = {"identity": 0.0}
for name, fn in OPS:
    if name == "identity":
        continue
    hi_r = break_rate(fn, 1.0)
    if hi_r < TARGET:
        params[name] = 1.0
        print(f"  {name:>12s}  p=1.0000 (max)  break {hi_r:.4f}  <-- cannot reach target")
        continue
    lo, hi = 0.0, 1.0
    for _ in range(10):
        mid = (lo + hi) / 2
        if break_rate(fn, mid) < TARGET:
            lo = mid
        else:
            hi = mid
    p = round((lo + hi) / 2, 4)
    params[name] = p
    print(f"  {name:>12s}  p={p:.4f}  break {break_rate(fn, p):.4f}")

json.dump(params, open("cat3_params.json", "w"), indent=1)
print("\nwrote cat3_params.json")
