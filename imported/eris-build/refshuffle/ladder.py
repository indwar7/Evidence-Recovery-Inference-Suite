"""Re-measure the whole ladder under BOTH metrics: v1 rescaled tau and v2
position accuracy (the shipped grade.py). Reference weights are re-tuned on
train under the v2 metric. Writes working/submission.csv (reference)."""
import os, re, sys, json, importlib.util
from collections import defaultdict, Counter
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

PKG = os.path.expanduser("~/Desktop/ERIS.nosync/Unscrambling the Bibliography")
os.chdir(PKG)
spec = importlib.util.spec_from_file_location("g", "grade.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

train = pd.read_csv("dataset/public/train.csv"); test = pd.read_csv("dataset/public/test.csv")
answers = pd.read_csv("dataset/private/answers.csv")
LABELS = ["A", "B", "C"]

def tau(t, p):
    c = d = 0
    for i in range(3):
        for j in range(i + 1, 3):
            c += 1 if (t[i]-t[j])*(p[i]-p[j]) > 0 else 0
            d += 0 if (t[i]-t[j])*(p[i]-p[j]) > 0 else 1
    return ((c - d) / 3 + 1) / 2

def to_df(pred):  # pred: uid -> [rA,rB,rC]
    return pd.DataFrame([{"unit_id": u, "rank_A": r[0], "rank_B": r[1], "rank_C": r[2]} for u, r in pred.items()])

def score(pred):
    v2 = g.grade(to_df(pred), answers)
    v1 = np.mean([tau([r.rank_A, r.rank_B, r.rank_C], pred[r.unit_id]) if r.unit_id in pred else 0.0 for r in answers.itertuples(index=False)])
    return v1, v2

def pa_train(pred):  # v2 metric on train, for tuning
    return np.mean([np.mean([a == b for a, b in zip([r.rank_A, r.rank_B, r.rank_C], pred[r.unit_id])]) for r in train.itertuples(index=False)])

def parse_card(s):
    d = {}
    for p in s.split(" | "):
        if ": " in p:
            k, v = p.split(": ", 1); d[k] = v
    return d
YEAR_RE = re.compile(r"\d{4}")
def rank_from_scores(sc):
    order = sorted(range(3), key=lambda i: sc[i]); r = [0, 0, 0]
    for pos, li in enumerate(order): r[li] = pos
    return r
WORD_RE = re.compile(r"[a-zA-Z]{3,}")
STOP = set("the a an of in on at to for and or but with from by as is was were are be been being this that these those his her their its our your which who whom what when where why how not no also had has have into after before during while over under between more most some such than then out about up down off again further once here there all any both each few other same so than too very".split())
words = lambda s: {w.lower() for w in WORD_RE.findall(s)} - STOP
card_words = lambda c: words(c.get("title", "")) | words(c.get("venue", ""))
def author_words(c):
    a = c.get("author", ""); return set() if (not a or "uncredited" in a) else words(a)
def card_year(c):
    m = YEAR_RE.search(c.get("year", "")); return int(m.group()) if m else None

def blended(row, aw, yw, use_overlap=True):
    cards = [parse_card(getattr(row, f"ref_{L}_card")) for L in LABELS]
    cw = [card_words(c) for c in cards]; au = [author_words(c) for c in cards]; ys = [card_year(c) for c in cards]
    parts = row.text.split("[CITE]"); tw = words(row.text)
    S = np.zeros((3, 3))
    for i in range(3):
        ctx = words(parts[i][-2000:])
        for j in range(3):
            if use_overlap: S[i, j] = len(ctx & cw[j])
            if au[j] and (au[j] & tw): S[i, j] += aw
    ky = [y for y in ys if y is not None]
    if len(ky) >= 2:
        for i in range(3):
            for j in range(3):
                if ys[j] is not None: S[i, j] += yw * (3 - i) * (1.0 / (1 + abs(ys[j] - min(ky))))
    ri, ci = linear_sum_assignment(-S); r = [0, 0, 0]
    for mp, li in zip(ri, ci): r[li] = mp
    return r

ladder = []
def add(name, pred): v1, v2 = score(pred); ladder.append((name, v1, v2)); return v2

add("blank / non-attempt", {})
add("pool order (A=0,B=1,C=2)", {u: [0, 1, 2] for u in test.unit_id})
add("reversed pool order", {u: [2, 1, 0] for u in test.unit_id})
rs = []
for seed in range(5):
    rng = np.random.default_rng(seed); p = {}
    for u in test.unit_id:
        perm = [0, 1, 2]; rng.shuffle(perm); p[u] = perm
    rs.append(score(p))
ladder.append(("random permutation (mean of 5 seeds)", np.mean([a for a, _ in rs]), np.mean([b for _, b in rs])))
add("year-sort ascending", {r.unit_id: rank_from_scores([(card_year(parse_card(getattr(r, f"ref_{L}_card"))) or 0) for L in LABELS]) for r in test.itertuples(index=False)})
add("title-length sort", {r.unit_id: rank_from_scores([len(parse_card(getattr(r, f"ref_{L}_card")).get("title", "")) for L in LABELS]) for r in test.itertuples(index=False)})
vf = defaultdict(list)
for r in train.itertuples(index=False):
    for L in LABELS: vf[parse_card(getattr(r, f"ref_{L}_card")).get("venue", "")].append(getattr(r, f"rank_{L}") / 2.0)
va = {k: float(np.mean(v)) for k, v in vf.items() if len(v) >= 3}; ga = float(np.mean([x for v in vf.values() for x in v]))
add("adversarial: global venue-position, fit on train", {r.unit_id: rank_from_scores([va.get(parse_card(getattr(r, f"ref_{L}_card")).get("venue", ""), ga) for L in LABELS]) for r in test.itertuples(index=False)})
add("author-surname bonus only (no overlap)", {r.unit_id: blended(r, 2.0, 0.0, use_overlap=False) for r in test.itertuples(index=False)})
add("word overlap only, Hungarian", {r.unit_id: blended(r, 0, 0) for r in test.itertuples(index=False)})
# tune on TRAIN under the v2 metric
best = None
for aw in [0, 1, 2, 4, 8]:
    for yw in [0, 0.05, 0.1, 0.2, 0.4]:
        s = pa_train({r.unit_id: blended(r, aw, yw) for r in train.itertuples(index=False)})
        if best is None or s > best[1]: best = ((aw, yw), s)
AW, YW = best[0]
ref = {r.unit_id: blended(r, AW, YW) for r in test.itertuples(index=False)}
add(f"REFERENCE: overlap + author(w={AW}) + year(w={YW}), Hungarian", ref)
add("oracle", {r.unit_id: [r.rank_A, r.rank_B, r.rank_C] for r in answers.itertuples(index=False)})

print(f"tuned on train (v2 metric): weights={best[0]} train_score={best[1]:.4f}")
print(f"\n{'rung':58s} {'v1 tau':>8s} {'v2 pos-acc':>10s}")
for n, v1, v2 in ladder: print(f"{n:58s} {v1:8.4f} {v2:10.4f}")
# per-unit distribution of the reference under v2
per = []
for r in answers.itertuples(index=False):
    p = ref[r.unit_id]; per.append(sum(a == b for a, b in zip([r.rank_A, r.rank_B, r.rank_C], p)) / 3)
print("\nreference per-unit v2 distribution:", {f"{k:.3f}": v for k, v in sorted(Counter(per).items())})
# public/private slice check: halves of answers
h = len(answers) // 2
for name, sl in [("first half", answers.iloc[:h]), ("second half", answers.iloc[h:])]:
    print(f"  reference on {name}: {g.grade(to_df(ref), sl):.4f}   pool-order: {g.grade(to_df({u:[0,1,2] for u in test.unit_id}), sl):.4f}")
os.makedirs("working", exist_ok=True); to_df(ref).to_csv("working/submission.csv", index=False)
to_df({u: [0, 1, 2] for u in test.unit_id}).to_csv("working/submission_0_pool_order.csv", index=False)
json.dump({"weights": best[0], "train_score": best[1], "ladder": ladder}, open(os.path.expanduser("~/eris-build/refshuffle/ladder.json"), "w"), indent=1)
