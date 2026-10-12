"""Shared pieces for the Blind Spot gate experiments.

Splits, the hidden classifier architecture, and the transform catalog.
Everything here is deterministic: fixed seeds, sorted file listings.
"""
import hashlib
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
RGB = ROOT / "EuroSAT_RGB"
SEED = 1234

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def all_files():
    """Every image, sorted, with integer label. Deterministic ordering."""
    out = []
    for ci, c in enumerate(CLASSES):
        for p in sorted((RGB / c).glob("*.jpg")):
            out.append((p, ci))
    return out


def splits():
    """Partition images into: hidden-model train, challenge train, challenge test.

    The hidden model is fit on `hid`; the challenge draws its public train and
    test images from disjoint pools so the solver never sees an image the
    hidden model memorised.
    """
    files = all_files()
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(files))
    n = len(files)
    n_hid = int(0.45 * n)
    n_tr = int(0.33 * n)
    hid = [files[i] for i in idx[:n_hid]]
    tr = [files[i] for i in idx[n_hid:n_hid + n_tr]]
    te = [files[i] for i in idx[n_hid + n_tr:]]
    return hid, tr, te


# ---------------------------------------------------------------- hidden model

class Net(nn.Module):
    """Small VGG-ish CNN. Big enough for ~97% on EuroSAT at 64px, small
    enough that the damage table is cheap to precompute."""

    def __init__(self, ncls=10):
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


MEAN = np.array([0.3444, 0.3803, 0.4078], dtype=np.float32)
STD = np.array([0.2027, 0.1369, 0.1156], dtype=np.float32)


def to_tensor(arr_uint8):
    """(N,64,64,3) uint8 -> normalised float tensor (N,3,64,64)."""
    x = arr_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


# ------------------------------------------------------------ transform catalog
# Design intent: every op is MILD -- on its own it should usually leave the
# hidden model correct. Damage is meant to come from image-specific interaction
# between the two ops, not from one universally destructive op. Mechanisms are
# deliberately diverse (blur / resolution / codec / noise / photometric /
# colour-geometry / spatial) so that different image content is hurt by
# different ops.

def _seed_for(name, key):
    h = hashlib.sha256(f"{name}|{key}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def t_identity(im, key):
    return im


def t_blur(im, key):
    return im.filter(ImageFilter.GaussianBlur(radius=0.9))


def t_downup(im, key):
    s = im.resize((26, 26), Image.BILINEAR)
    return s.resize((64, 64), Image.BILINEAR)


def t_jpeg(im, key):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=18)
    b.seek(0)
    return Image.open(b).convert("RGB")


def t_noise(im, key):
    rng = np.random.RandomState(_seed_for("noise", key))
    a = np.asarray(im).astype(np.float32)
    a += rng.normal(0, 11.0, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def t_bright_up(im, key):
    return ImageEnhance.Brightness(im).enhance(1.35)


def t_bright_dn(im, key):
    return ImageEnhance.Brightness(im).enhance(0.68)


def t_contrast_dn(im, key):
    return ImageEnhance.Contrast(im).enhance(0.55)


def t_desat(im, key):
    return ImageEnhance.Color(im).enhance(0.30)


def t_huerot(im, key):
    a = np.asarray(im).astype(np.uint8)
    hsv = np.asarray(Image.fromarray(a).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 28) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def t_chanswap(im, key):
    a = np.asarray(im)
    return Image.fromarray(a[..., [1, 2, 0]])


def t_gamma(im, key):
    a = np.asarray(im).astype(np.float32) / 255.0
    a = np.power(a, 1.7)
    return Image.fromarray((a * 255).astype(np.uint8))


def t_sharpen(im, key):
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=190, threshold=2))


def t_motion_h(im, key):
    k = ImageFilter.Kernel((3, 3), [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0], scale=1)
    return im.filter(k)


def t_motion_v(im, key):
    k = ImageFilter.Kernel((3, 3), [0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0], scale=1)
    return im.filter(k)


def t_rot90(im, key):
    return im.transpose(Image.ROTATE_90)


def t_flip(im, key):
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def t_zoom(im, key):
    c = im.crop((10, 10, 54, 54))
    return c.resize((64, 64), Image.BILINEAR)


def t_posterize(im, key):
    return ImageOps.posterize(im, 3)


def t_equalize(im, key):
    return ImageOps.equalize(im)


CATALOG = [
    ("identity", t_identity), ("blur", t_blur), ("downup", t_downup),
    ("jpeg", t_jpeg), ("noise", t_noise), ("bright_up", t_bright_up),
    ("bright_dn", t_bright_dn), ("contrast_dn", t_contrast_dn),
    ("desat", t_desat), ("huerot", t_huerot), ("chanswap", t_chanswap),
    ("gamma", t_gamma), ("sharpen", t_sharpen), ("motion_h", t_motion_h),
    ("motion_v", t_motion_v), ("rot90", t_rot90), ("flip", t_flip),
    ("zoom", t_zoom), ("posterize", t_posterize), ("equalize", t_equalize),
]
NAMES = [n for n, _ in CATALOG]
K = len(CATALOG)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
