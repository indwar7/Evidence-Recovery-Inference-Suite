"""Re-derive the bundled real-motion pool from the public Porto Taxi dataset.

Source: kraina/porto_taxi on the Hugging Face Hub (CC-BY-4.0), the ECML/PKDD-2015
Porto taxi trajectories (442 taxis, GPS pings every ~15 s). We extract every
consecutive same-taxi ping pair spaced 14-16 s apart, giving a *real* 2-D motion
vector (dx, dy) in metres together with its start position and instantaneous speed.
A deterministic subsample of these real vectors is bundled as `motion_pool.npz`, so
the scene generator is fully self-contained and needs no network at build time.

Usage:
    python fetch_porto.py                 # downloads from HF (or uses --cache)
    python fetch_porto.py --cache PATH    # use a local porto_raw.parquet
"""
import argparse, io, os, ssl, urllib.request
import numpy as np, pandas as pd

RAW_URL = ("https://huggingface.co/datasets/kraina/porto_taxi/resolve/main/"
           "data/raw/porto_taxi.parquet")
LAT0 = 41.16
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))
POOL_SIZE = 200_000
SEED = 20260718
OUT = os.path.join(os.path.dirname(__file__), "..", "raw", "motion_pool.npz")


def load_raw(cache):
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache, columns=["timestamp", "taxi_id", "longitude", "latitude"])
    ssl._create_default_https_context = ssl._create_unverified_context
    print("downloading Porto raw parquet from HF ...", flush=True)
    raw = urllib.request.urlopen(RAW_URL, timeout=600).read()
    return pd.read_parquet(io.BytesIO(raw), columns=["timestamp", "taxi_id", "longitude", "latitude"])


def build(df):
    df = df.copy()
    df["x"] = df.longitude * M_PER_DEG_LON
    df["y"] = df.latitude * M_PER_DEG_LAT
    df = df.sort_values(["taxi_id", "timestamp"])
    recs = []
    for _, sub in df.groupby("taxi_id", sort=False):
        ts = sub.timestamp.values
        x = sub.x.values
        y = sub.y.values
        dt = np.diff(ts)
        ok = np.where((dt >= 14) & (dt <= 16))[0]
        for i in ok:
            recs.append((x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i]))
    pool = np.asarray(recs, dtype=np.float64)
    disp = np.hypot(pool[:, 2], pool[:, 3])
    pool = pool[disp > 5.0]  # keep genuinely moving objects
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool), size=min(POOL_SIZE, len(pool)), replace=False)
    return pool[np.sort(idx)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    pool = build(load_raw(a.cache))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # store as start (x,y) and displacement (dx,dy), metres
    np.savez_compressed(OUT, xy=pool[:, :2].astype(np.float32), dxy=pool[:, 2:].astype(np.float32))
    disp = np.hypot(pool[:, 2], pool[:, 3])
    print(f"pool={len(pool)} disp(m) median={np.median(disp):.1f} "
          f"p25={np.percentile(disp,25):.1f} p75={np.percentile(disp,75):.1f} -> {OUT}")
