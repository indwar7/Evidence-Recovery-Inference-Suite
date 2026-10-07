"""Reference solution for Phantom Track.

The correct primitive is a *global* one-to-one assignment, not per-object nearest
neighbour: greedily linking each frame-A point to its closest frame-B point double-
books popular targets and cascades errors. We instead solve, per scene, the minimum-
cost bipartite matching between frame-A and frame-B points with Euclidean cost, via
the Hungarian algorithm. This lands mid-band (~0.60); it cannot reach 1.0 because
objects that are mutually confusable (notably the coincident hard pairs) are
information-theoretically unresolvable from position alone.

Usage:
    python reference_solution.py --points prepared/public/points_test.csv --out submission.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def solve(points):
    rows = []
    for sid, g in points.groupby("scene_id", sort=False):
        A = g[g.frame == "A"].sort_values("point_id")
        B = g[g.frame == "B"].sort_values("point_id")
        Axy = A[["x", "y"]].to_numpy(float)
        Bxy = B[["x", "y"]].to_numpy(float)
        aid = A["point_id"].to_numpy(int)
        bid = B["point_id"].to_numpy(int)
        D = np.linalg.norm(Axy[:, None, :] - Bxy[None, :, :], axis=2)
        r, c = linear_sum_assignment(D)
        pred = np.empty(len(aid), dtype=int)
        pred[r] = c
        for i in range(len(aid)):
            rows.append((sid, int(aid[i]), int(bid[pred[i]])))
    return pd.DataFrame(rows, columns=["scene_id", "a_id", "b_id"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    solve(pd.read_csv(a.points)).to_csv(a.out, index=False)
    print("wrote", a.out)
