"""Parameterised catalog.

Every op takes a scalar strength p in [0,1] that is MONOTONE in damage, so a
bisection can calibrate each op to equal mean solo damage. Removing the
strength confound is what makes the best pair depend on the image rather than
on which op happens to be most destructive.

`identity` is the null slot and has no parameter.
"""
import hashlib
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _seed_for(name, key):
    h = hashlib.sha256(f"{name}|{key}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _lerp(a, b, p):
    return a + (b - a) * p


def identity(im, key, p):
    return im


def blur(im, key, p):
    return im.filter(ImageFilter.GaussianBlur(radius=_lerp(0.15, 2.2, p)))


def downup(im, key, p):
    s = int(round(_lerp(56, 12, p)))
    return im.resize((s, s), Image.BILINEAR).resize((64, 64), Image.BILINEAR)


def jpeg(im, key, p):
    q = int(round(_lerp(85, 6, p)))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=max(1, q))
    b.seek(0)
    return Image.open(b).convert("RGB")


def noise(im, key, p):
    rng = np.random.RandomState(_seed_for("noise", key))
    a = np.asarray(im).astype(np.float32) + rng.normal(0, _lerp(1.0, 26.0, p), (64, 64, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def speckle(im, key, p):
    rng = np.random.RandomState(_seed_for("speckle", key))
    a = np.asarray(im).astype(np.float32) * (
        1.0 + rng.normal(0, _lerp(0.01, 0.26, p), (64, 64, 1)))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def bright_up(im, key, p):
    return ImageEnhance.Brightness(im).enhance(_lerp(1.02, 1.85, p))


def bright_dn(im, key, p):
    return ImageEnhance.Brightness(im).enhance(_lerp(0.98, 0.42, p))


def contrast_up(im, key, p):
    return ImageEnhance.Contrast(im).enhance(_lerp(1.03, 2.4, p))


def contrast_dn(im, key, p):
    return ImageEnhance.Contrast(im).enhance(_lerp(0.97, 0.30, p))


def desat(im, key, p):
    return ImageEnhance.Color(im).enhance(_lerp(0.95, 0.0, p))


def sat_up(im, key, p):
    return ImageEnhance.Color(im).enhance(_lerp(1.05, 3.2, p))


def warm(im, key, p):
    a = np.asarray(im).astype(np.float32)
    g = _lerp(1.01, 1.32, p)
    a[..., 0] *= g
    a[..., 2] *= 2.0 - g
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def cool(im, key, p):
    a = np.asarray(im).astype(np.float32)
    g = _lerp(1.01, 1.32, p)
    a[..., 2] *= g
    a[..., 0] *= 2.0 - g
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def gamma(im, key, p):
    a = (np.asarray(im).astype(np.float32) / 255.0) ** _lerp(1.03, 2.3, p)
    return Image.fromarray((a * 255).astype(np.uint8))


def sharpen(im, key, p):
    return im.filter(ImageFilter.UnsharpMask(
        radius=2, percent=int(round(_lerp(20, 330, p))), threshold=2))


def _box1d(L):
    """Continuous-width box kernel: a box of real width L centred on the middle
    tap, with end taps partially weighted. Continuity matters because the
    strength calibration bisects on this parameter."""
    L = max(1.0, float(L))
    n = int(np.ceil(L))
    if n % 2 == 0:
        n += 1
    c, half = n // 2, L / 2.0
    k = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = i - c - 0.5, i - c + 0.5
        k[i] = max(0.0, min(hi, half) - max(lo, -half))
    return k / k.sum()


def _motion(im, L, axis):
    """Directional box blur. PIL's Kernel only accepts 3x3/5x5, so use numpy."""
    from scipy.ndimage import convolve1d
    if L <= 1.0:
        return im
    a = np.asarray(im).astype(np.float32)
    a = convolve1d(a, _box1d(L), axis=axis, mode="reflect")
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def motion_h(im, key, p):
    return _motion(im, _lerp(1.0, 13.0, p), axis=1)


def motion_v(im, key, p):
    return _motion(im, _lerp(1.0, 13.0, p), axis=0)


def zoom(im, key, p):
    # PIL's resize accepts a float box, so the crop margin stays continuous
    m = _lerp(0.5, 20.0, p)
    return im.resize((64, 64), Image.BILINEAR, box=(m, m, 64 - m, 64 - m))


def posterize(im, key, p):
    lv = _lerp(64, 3, p)
    a = np.asarray(im).astype(np.float32) / 255.0
    a = np.round(a * lv) / lv
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


OPS = [
    ("identity", identity), ("blur", blur), ("downup", downup), ("jpeg", jpeg),
    ("noise", noise), ("speckle", speckle), ("bright_up", bright_up),
    ("bright_dn", bright_dn), ("contrast_up", contrast_up),
    ("contrast_dn", contrast_dn), ("desat", desat), ("sat_up", sat_up),
    ("warm", warm), ("cool", cool), ("gamma", gamma), ("sharpen", sharpen),
    ("motion_h", motion_h), ("motion_v", motion_v), ("zoom", zoom),
    ("posterize", posterize),
]
NAMES = [n for n, _ in OPS]
K = len(OPS)
FN = {n: f for n, f in OPS}


def build(params):
    """params: {name: p}. Returns [(name, fn(im,key))] with p bound."""
    out = []
    for n, f in OPS:
        p = params.get(n, 0.0)
        out.append((n, (lambda f=f, p=p: (lambda im, key: f(im, key, p)))()))
    return out
