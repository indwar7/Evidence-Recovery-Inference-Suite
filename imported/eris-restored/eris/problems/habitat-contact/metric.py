"""Re-implementation of the challenge metric for local tuning."""
import numpy as np
from scipy.optimize import linear_sum_assignment

GAP = 0.02


def iou_mat(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A = np.asarray(A, float); B = np.asarray(B, float)
    x0 = np.maximum(A[:, None, 0], B[None, :, 0])
    y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2])
    y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aA = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aA[:, None] + aB[None, :] - inter, 1e-12)


def box_gap(a, b):
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def edges_from_boxes(boxes, ids, thr=GAP):
    e = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_gap(boxes[i], boxes[j]) <= thr:
                e.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
    return e


def match(gt, pred, thr=0.5):
    """Return list of (gi, pi) pairs."""
    M = iou_mat(gt, pred)
    if M.size == 0:
        return []
    ri, ci = linear_sum_assignment(-M)
    return [(int(a), int(b)) for a, b in zip(ri, ci) if M[a, b] >= thr]


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 1.0 if d == 0 else 2 * tp / d


def score(gts, preds, topo_mode='mean'):
    """gts/preds: list per image of (boxes Nx4, edge set over indices)."""
    det = {'empty': [], 'single': [], 'multi': []}
    topo = {'with': [], 'without': []}
    for (gb, ge), (pb, pe) in zip(gts, preds):
        m = match(gb, pb)
        tp = len(m); fp = len(pb) - tp; fn = len(gb) - tp
        k = 'empty' if len(gb) == 0 else ('single' if len(gb) == 1 else 'multi')
        det[k].append(f1(tp, fp, fn))
        if len(gb) < 2:
            continue
        p2g = {p: g for g, p in m}
        mapped = {tuple(sorted((p2g[a], p2g[b])))
                  for (a, b) in pe if a in p2g and b in p2g}
        gset = {tuple(sorted(e)) for e in ge}
        etp = len(mapped & gset)
        ef1 = f1(etp, len(mapped) - etp, len(gset) - etp)
        cov = len(m) / len(gb)
        t = 0.5 * (ef1 + cov) if topo_mode == 'mean' else ef1 * cov
        topo['with' if gset else 'without'].append(t)
    ds = np.mean([np.mean(det[k]) for k in ['empty', 'single', 'multi'] if det[k]])
    ts = np.mean([np.mean(topo[k]) for k in ['with', 'without'] if topo[k]])
    return dict(score=float(np.sqrt(ds * ts)), detection=float(ds), topology=float(ts),
                empty=float(np.mean(det['empty'])), single=float(np.mean(det['single'])),
                multi=float(np.mean(det['multi'])),
                topo_with=float(np.mean(topo['with'])),
                topo_without=float(np.mean(topo['without'])))
