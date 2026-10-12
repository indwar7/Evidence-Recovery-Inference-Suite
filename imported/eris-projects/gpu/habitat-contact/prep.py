"""Build YOLO-format dataset with a session-proxy (cluster-grouped) val split."""
import os, json, shutil, numpy as np, pandas as pd

D = '/Users/abhayindwar/Downloads/public-3/'
OUT = '/Users/abhayindwar/Projects/eris/habitat-contact/yolo'

df = pd.read_csv(D + 'train.csv')
df['inst'] = df['instances'].apply(json.loads)
df['n'] = df['inst'].apply(len)
clus = np.load('/Users/abhayindwar/Projects/eris/habitat-contact/trainclus.npy')
df['clus'] = clus

# hold out whole clusters (~20%) keeping stratum mix reasonable
rng = np.random.RandomState(0)
best = None
for seed in range(300):
    r = np.random.RandomState(seed)
    order = r.permutation(20)
    val = []
    for c in order:
        if sum((df.clus.isin(val + [c])).sum() for _ in [0]) > 0.24 * len(df):
            continue
        val.append(c)
    m = df.clus.isin(val)
    frac = m.mean()
    if not (0.16 < frac < 0.24):
        continue
    sub = df[m]
    # want all three strata present in decent numbers
    e, s, mu = (sub.n == 0).sum(), (sub.n == 1).sum(), (sub.n >= 2).sum()
    tot = e + s + mu
    score = abs(e / tot - 0.29) + abs(s / tot - 0.41) + abs(mu / tot - 0.30)
    if best is None or score < best[0]:
        best = (score, val, frac, (e, s, mu))
print('val clusters', best[1], 'frac', round(best[2], 3), 'strata', best[3])
valc = set(best[1])
df['split'] = np.where(df.clus.isin(valc), 'val', 'train')
df[['id', 'image_path', 'split', 'clus', 'n']].to_csv(
    '/Users/abhayindwar/Projects/eris/habitat-contact/split.csv', index=False)

for sp in ['train', 'val']:
    for k in ['images', 'labels']:
        os.makedirs(f'{OUT}/{k}/{sp}', exist_ok=True)

for _, r in df.iterrows():
    sp = r['split']
    src = D + r['image_path']
    dst = f"{OUT}/images/{sp}/{r['id']}.jpg"
    if not os.path.exists(dst):
        os.link(src, dst) if not os.path.islink(dst) else None
    lines = []
    for ins in r['inst']:
        x0, y0, x1, y1 = ins['bbox']
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, x1), min(1.0, y1)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(f"{OUT}/labels/{sp}/{r['id']}.txt", 'w') as f:
        f.write('\n'.join(lines))

with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: organism\n")
print('train', (df.split == 'train').sum(), 'val', (df.split == 'val').sum())
