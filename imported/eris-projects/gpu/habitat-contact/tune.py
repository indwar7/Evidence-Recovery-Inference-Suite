"""Cache raw val detections, then sweep decision thresholds against the metric."""
import sys, json, os, numpy as np, pandas as pd
from ultralytics import YOLO
import metric as M

D = '/Users/abhayindwar/Downloads/public-3/'
HERE = os.path.dirname(os.path.abspath(__file__))


def raw_predict(weights, paths, imgsz=768, conf=0.01, iou=0.6, tta=False, batch=8):
    model = YOLO(weights)
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device='mps',
                            verbose=False, augment=tta, max_det=100)
        for r in res:
            b = r.boxes.xyxyn.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            s = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
            o = np.argsort(-s)
            out.append((b[o].astype(float), s[o].astype(float)))
    return out


def load_gt(split_csv='split.csv', which='val'):
    df = pd.read_csv(D + 'train.csv')
    sp = pd.read_csv(f'{HERE}/{split_csv}')
    ids = set(sp[sp.split == which]['id'])
    df = df[df['id'].isin(ids)].reset_index(drop=True)
    gts, paths = [], []
    for _, r in df.iterrows():
        insts = json.loads(r['instances'])
        boxes = [i['bbox'] for i in insts]
        idx = {i['instance_id']: k for k, i in enumerate(insts)}
        e = set()
        for i in insts:
            for c in i['contacts']:
                if c in idx:
                    a, b = idx[i['instance_id']], idx[c]
                    e.add((min(a, b), max(a, b)))
        gts.append((np.array(boxes, float).reshape(-1, 4), e))
        paths.append(D + r['image_path'])
    return df, gts, paths


def decide(raw, t_img, t_box, min_area=0.0):
    """Two-threshold gate: image kept only if a box clears t_img."""
    out = []
    for b, s in raw:
        if len(s) == 0 or s[0] < t_img:
            out.append(np.zeros((0, 4)))
            continue
        k = s >= t_box
        bb = b[k]
        if min_area > 0 and len(bb):
            a = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
            bb = bb[a >= min_area]
        out.append(bb)
    return out


def to_pred(boxes_list, gap):
    return [(bb, M.edges_from_boxes(bb, list(range(len(bb))), gap)) for bb in boxes_list]


def sweep(raw, gts, grid_img=None, grid_box=None, gaps=(0.02,)):
    grid_img = grid_img if grid_img is not None else np.arange(0.10, 0.75, 0.025)
    grid_box = grid_box if grid_box is not None else np.arange(0.05, 0.70, 0.025)
    best = None
    for ti in grid_img:
        for tb in grid_box:
            if tb > ti:
                continue
            bl = decide(raw, ti, tb)
            for g in gaps:
                r = M.score(gts, to_pred(bl, g))
                if best is None or r['score'] > best[0]['score']:
                    best = (r, dict(t_img=float(ti), t_box=float(tb), gap=float(g)))
    return best


if __name__ == '__main__':
    w = sys.argv[1]
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    tta = len(sys.argv) > 3 and sys.argv[3] == 'tta'
    df, gts, paths = load_gt()
    tag = os.path.basename(os.path.dirname(os.path.dirname(w))) + f'_{imgsz}' + ('_tta' if tta else '')
    cache = f'{HERE}/rawval_{tag}.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); raw = list(z['raw'])
    else:
        raw = raw_predict(w, paths, imgsz=imgsz, tta=tta)
        np.savez(cache, raw=np.array(raw, dtype=object))
    best = sweep(raw, gts, gaps=(0.015, 0.02, 0.025, 0.03))
    print(tag, json.dumps(best[1]))
    print({k: round(v, 4) for k, v in best[0].items()})
