"""Observational-systematics catalog for the Galaxy10 build.

Every entry is a real effect that an imaging survey actually suffers: atmosphere
(seeing, extinction, cirrus), telescope (guiding trail, coadd misregistration),
detector (read noise, shot noise, saturation, nonlinearity, bleed columns,
cosmic rays), sky (background pedestal, scattered-light gradient), and pipeline
(pixel binning, cutout scale, neighbour deblending residual, JPEG delivery).
That is the difference between this catalog and a generic corruption suite: an
answer here is a statement about *observing conditions*, not about image codecs.

Contract shared with the rest of the build:

* signature `(im: PIL.Image, key: str, p: float) -> PIL.Image`, p in [0,1];
* p is MONOTONE in damage so `calibrate.py` can bisect each op to an equal solo
  break rate -- that removes the "which op is strongest" confound, which is what
  makes the winning pair depend on the galaxy rather than on its class;
* any op that injects structure (cosmic rays, a blended star, a bad column)
  draws its geometry from `seed_for(name, key)`, so the realisation is fixed per
  image and the response table is reproducible.

Scale note: strengths are quoted in pixels. Galaxy10 SDSS cutouts are 69x69; at
the SDSS native plate scale of 0.396"/pixel a sigma of 2.6 px is a FWHM of about
2.4", i.e. a genuinely poor night rather than a synthetic extreme.

This module is standalone on purpose -- numpy, scipy and Pillow only, no torch,
no project imports. It ships verbatim to solvers as catalog.py so that anyone can
render the exact degraded image any (op1, op2) pair produces. Being able to
reproduce the *stimulus* is fair game; what stays hidden is the model's
*response* to it.
"""
import hashlib
import io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, shift as ndshift


def seed_for(name, key):
    """Deterministic per-(op, image) seed. `key` is the public image_id, so the
    realisation of a random op is reproducible by anyone holding the image."""
    h = hashlib.sha256(f"{name}|{key}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _l(a, b, p):
    return a + (b - a) * p


def _f(im):
    """PIL -> float32 [0,1] array, H x W x 3."""
    return np.asarray(im).astype(np.float32) / 255.0


def _im(a):
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- atmosphere

def identity(im, key, p):
    """Null slot: the observation as delivered."""
    return im


def seeing(im, key, p):
    """Atmospheric turbulence broadens the PSF. sigma 0.15 -> 2.6 px."""
    a = _f(im)
    return _im(gaussian_filter(a, sigma=(_l(0.15, 2.6, p), _l(0.15, 2.6, p), 0)))


def trail_h(im, key, p):
    """Guiding/tracking error elongates the PSF along x. An edge-on disk trailed
    along its major axis survives; trailed across it, it stops being edge-on."""
    a = _f(im)
    return _im(gaussian_filter1d(a, sigma=_l(0.1, 4.2, p), axis=1, mode="reflect"))


def trail_v(im, key, p):
    """Same, along y. Kept as a separate op precisely because the better of the
    two depends on the galaxy's position angle, not on its morphology class."""
    a = _f(im)
    return _im(gaussian_filter1d(a, sigma=_l(0.1, 4.2, p), axis=0, mode="reflect"))


def airmass_red(im, key, p):
    """Differential atmospheric extinction at high airmass: blue is attenuated
    more than red, so the source reddens as well as dims."""
    a = _f(im)
    t = np.array([_l(0.995, 0.86, p), _l(0.99, 0.62, p), _l(0.985, 0.40, p)],
                 np.float32)
    return _im(a * t)


def cirrus_gray(im, key, p):
    """Thin cirrus: grey (achromatic) transmission loss."""
    return _im(_f(im) * _l(0.985, 0.34, p))


# ----------------------------------------------------------------------- sky

def sky_bkg(im, key, p):
    """Bright sky (moon, twilight) adds a flat pedestal, crushing contrast."""
    return _im(_f(im) + _l(0.004, 0.34, p))


def sky_grad(im, key, p):
    """Scattered light from a bright neighbour: a linear ramp across the stamp.
    Direction is fixed per image."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("sky_grad", key))
    th = rng.uniform(0, 2 * np.pi)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx - w / 2) * np.cos(th) + (yy - h / 2) * np.sin(th)) / (w / 2)
    return _im(a + (_l(0.01, 0.42, p) * r)[..., None])


# ------------------------------------------------------------------ detector

def read_noise(im, key, p):
    """Additive Gaussian detector read noise."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("read_noise", key))
    return _im(a + rng.normal(0, _l(0.003, 0.115, p), a.shape))


def shot_noise(im, key, p):
    """Photon shot noise: dispersion scales with the square root of the flux, so
    unlike read noise it bites hardest in the bright galaxy core."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("shot_noise", key))
    return _im(a + rng.normal(0, 1, a.shape) * _l(0.01, 0.30, p) * np.sqrt(a + 1e-3))


def saturate(im, key, p):
    """Detector wells fill and the bright core clips flat."""
    return _im(np.minimum(_f(im), _l(0.995, 0.30, p)))


def nonlinear(im, key, p):
    """Detector response departs from linear at high counts, compressing the
    bright end smoothly rather than clipping it."""
    return _im(_f(im) ** _l(1.02, 2.7, p))


def cosmic_ray(im, key, p):
    """Cosmic-ray hits: short, sharp, near-saturated streaks."""
    a = _f(im).copy()
    rng = np.random.RandomState(seed_for("cosmic_ray", key))
    h, w = a.shape[:2]
    n = int(round(_l(0.0, 20.0, p)))
    for _ in range(n):
        y, x = rng.randint(2, h - 2), rng.randint(2, w - 2)
        L = rng.randint(1, 4)
        dy, dx = rng.randint(-1, 2), rng.randint(-1, 2)
        for t in range(L):
            yy, xx = min(max(y + dy * t, 0), h - 1), min(max(x + dx * t, 0), w - 1)
            a[yy, xx, :] = 1.0
    return _im(a)


def bleed_col(im, key, p):
    """A hot/dead column plus the charge bleed around it.

    Both the column's width AND how saturated it is ramp with p. Ramping width
    alone makes damage discontinuous at p->0 -- a hairline column is already
    fully white and already breaks the model -- which a bisection cannot
    calibrate: it converges to the jump and lands on either a no-op or a large
    effect, never the 8% target.
    """
    a = _f(im).copy()
    rng = np.random.RandomState(seed_for("bleed_col", key))
    h, w = a.shape[:2]
    c = rng.randint(int(0.15 * w), int(0.85 * w))
    half = _l(0.3, 3.2, p)
    amp = _l(0.0, 1.0, p)
    if amp <= 1e-4:
        return _im(a)
    lo, hi = int(np.floor(c - half)), int(np.ceil(c + half))
    for x in range(max(lo, 0), min(hi + 1, w)):
        wgt = amp * float(np.clip(half - abs(x - c) + 0.5, 0.0, 1.0))
        a[:, x, :] = a[:, x, :] * (1 - wgt) + 1.0 * wgt
    return _im(a)


# ---------------------------------------------------------------- contaminants

def star_blend(im, key, p):
    """A foreground star lands near the galaxy. Its PSF wing is exactly the kind
    of feature a morphology classifier can read as a bulge or a companion."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("star_blend", key))
    h, w = a.shape[:2]
    cy = rng.uniform(0.25 * h, 0.75 * h)
    cx = rng.uniform(0.25 * w, 0.75 * w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))
    tint = np.array([1.0, rng.uniform(0.85, 1.0), rng.uniform(0.75, 1.0)], np.float32)
    return _im(a + _l(0.0, 1.15, p) * g[..., None] * tint)


def deblend_resid(im, key, p):
    """Residual flux from a neighbour the deblender only partly removed: a soft,
    extended blob offset from the centre."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("deblend_resid", key))
    h, w = a.shape[:2]
    th = rng.uniform(0, 2 * np.pi)
    cy = h / 2 + 0.36 * h * np.sin(th)
    cx = w / 2 + 0.36 * w * np.cos(th)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy, sx = rng.uniform(4, 8), rng.uniform(4, 8)
    g = np.exp(-(((yy - cy) ** 2 / (2 * sy ** 2)) + ((xx - cx) ** 2 / (2 * sx ** 2))))
    return _im(a + _l(0.0, 0.55, p) * g[..., None])


# ------------------------------------------------------------------- pipeline

def binning(im, key, p):
    """Coarser effective pixel scale: on-chip binning or a resampled coadd."""
    w = im.size[0]
    s = max(4, int(round(w / _l(1.03, 4.3, p))))
    return im.resize((s, s), Image.BILINEAR).resize((w, w), Image.BILINEAR)


def stamp_scale(im, key, p):
    """The cutout service picks the postage-stamp size from a catalogue radius.
    When that radius is wrong the galaxy is delivered at the wrong scale."""
    w = im.size[0]
    m = _l(0.4, w * 0.31, p)
    return im.resize((w, w), Image.BILINEAR, box=(m, m, w - m, w - m))


def coadd_misreg(im, key, p):
    """Dithered exposures stacked with a small astrometric error: the coadd is a
    partial-weight double image."""
    a = _f(im)
    rng = np.random.RandomState(seed_for("coadd_misreg", key))
    th = rng.uniform(0, 2 * np.pi)
    d = _l(0.0, 4.0, p)
    wt = _l(0.0, 0.5, p)
    b = ndshift(a, (d * np.sin(th), d * np.cos(th), 0), order=1, mode="reflect")
    return _im(a * (1 - wt) + b * wt)


def jpeg(im, key, p):
    """Galaxy Zoo classified JPEG cutouts, not FITS. Delivery compression is a
    real part of that measurement chain, so it belongs in the catalog."""
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=max(1, int(round(_l(92, 4, p)))))
    b.seek(0)
    return Image.open(b).convert("RGB")


OPS = [
    ("identity", identity),
    ("seeing", seeing), ("trail_h", trail_h), ("trail_v", trail_v),
    ("airmass_red", airmass_red), ("cirrus_gray", cirrus_gray),
    ("sky_bkg", sky_bkg), ("sky_grad", sky_grad),
    ("read_noise", read_noise), ("shot_noise", shot_noise),
    ("saturate", saturate), ("nonlinear", nonlinear),
    ("cosmic_ray", cosmic_ray), ("bleed_col", bleed_col),
    ("star_blend", star_blend), ("deblend_resid", deblend_resid),
    ("binning", binning), ("stamp_scale", stamp_scale),
    ("coadd_misreg", coadd_misreg), ("jpeg", jpeg),
]
NAMES = [n for n, _ in OPS]
K = len(OPS)
FN = dict(OPS)

# Families, used by the dataset description and by the gate's shortcut checks.
FAMILY = {
    "identity": "none", "seeing": "atmosphere", "trail_h": "telescope",
    "trail_v": "telescope", "airmass_red": "atmosphere",
    "cirrus_gray": "atmosphere", "sky_bkg": "sky", "sky_grad": "sky",
    "read_noise": "detector", "shot_noise": "detector", "saturate": "detector",
    "nonlinear": "detector", "cosmic_ray": "detector", "bleed_col": "detector",
    "star_blend": "contaminant", "deblend_resid": "contaminant",
    "binning": "pipeline", "stamp_scale": "pipeline",
    "coadd_misreg": "telescope", "jpeg": "pipeline",
}


def build(params):
    """Freeze each op at its calibrated strength -> list of (name, fn(im, key))."""
    return [(n, (lambda f=f, p=params.get(n, 0.0): (lambda im, key: f(im, key, p)))())
            for n, f in OPS]
