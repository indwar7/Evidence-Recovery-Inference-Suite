"""Hidden model H and the deterministic Galaxy10 pool split.

Standalone so the generator directory reproduces the dataset on its own. The
weights in hidden.pt are the artifact under study; they are NEVER shipped to
solvers, because the whole task is to infer this model's behaviour without
being able to query it.
"""
import numpy as np
import torch
import torch.nn as nn

SEED = 20260731
MIN_CLASS = 500      # drop morphologies too rare for H to learn at all
CAP = 2500           # and cap the common ones so H is not dominated by three

GALAXY10_NAMES = [
    "Disk, Face-on, No Spiral",          # 0
    "Smooth, Completely round",          # 1
    "Smooth, in-between round",          # 2
    "Smooth, Cigar shaped",              # 3  dropped (349 examples)
    "Disk, Edge-on, Rounded Bulge",      # 4
    "Disk, Edge-on, Boxy Bulge",         # 5  dropped (17 examples)
    "Disk, Edge-on, No Bulge",           # 6
    "Disk, Face-on, Tight Spiral",       # 7
    "Disk, Face-on, Medium Spiral",      # 8
    "Disk, Face-on, Loose Spiral",       # 9
]


def load_pool(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        X = np.array(f["images"], dtype=np.uint8)
        y = np.array(f["ans"], dtype=np.int64)
    counts = np.bincount(y, minlength=10)
    kept = [c for c in range(10) if counts[c] >= MIN_CLASS]
    remap = {c: i for i, c in enumerate(kept)}
    rng = np.random.RandomState(SEED)
    sel = []
    for c in kept:
        idx = np.sort(np.where(y == c)[0])
        if len(idx) > CAP:
            idx = np.sort(rng.permutation(idx)[:CAP])
        sel.append(idx)
    sel = np.sort(np.concatenate(sel))
    return (X[sel], np.array([remap[c] for c in y[sel]], np.int64), kept,
            [GALAXY10_NAMES[c] for c in kept])


def splits(h5_path):
    """Disjoint: the pool H trains on is never the pool the challenge ships."""
    X, y, kept, names = load_pool(h5_path)
    perm = np.random.RandomState(SEED).permutation(len(X))
    n_hid = int(0.45 * len(X))
    return ((X[perm[:n_hid]], y[perm[:n_hid]]),
            (X[perm[n_hid:]], y[perm[n_hid:]]), kept, names)


class Net(nn.Module):
    """Four-block CNN. 69 -> 34 -> 17 -> 8 -> 4 through the max-pools."""

    def __init__(self, ncls):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(256, ncls))

    def forward(self, x):
        return self.head(self.f(x))


def to_tensor(arr_uint8, mean, std):
    x = arr_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(((x - mean) / std).transpose(0, 3, 1, 2).copy())


def device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
