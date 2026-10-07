"""Full v3 ladder (4 cards) scored with the SHIPPED grade.py."""
import re, sys, json
from collections import defaultdict, Counter
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, ".")
from grade import grade, RANK_COLS, N_CARDS
N = N_CARDS; L = [c.split("_")[1] for c in RANK_COLS]
train = pd.read_csv("dataset/public/train.csv"); test = pd.read_csv("dataset/public/test.csv")
answers = pd.read_csv("dataset/private/answers.csv"); sample = pd.read_csv("dataset/public/sample_submission.csv")
def to_sub(pred): return pd.DataFrame([{"unit_id": u, **{f"rank_{x}": r[i] for i, x in enumerate(L)}} for u, r in pred.items()])
def G(pred): return grade(to_sub(pred), answers)
def parse_card(s):
    d = {}
    for p in s.split(" | "):
        if ": " in p: k, v = p.split(": ", 1); d[k] = v
    return d
YEAR_RE = re.compile(r"\d{4}"); WORD_RE = re.compile(r"[a-zA-Z]{3,}")
STOP = set("the a an of in on at to for and or but with from by as is was were are be been being this that these those his her their its our your which who whom what when where why how not no also had has have into after before during while over under between more most some such than then out about up down off again further once here there all any both each few other same so than too very".split())
words = lambda s: {w.lower() for w in WORD_RE.findall(s)} - STOP
cw = lambda c: words(c.get("title", "")) | words(c.get("venue", ""))
def aw_(c):
    a = c.get("author", ""); return set() if (not a or "uncredited" in a) else words(a)
def cy(c):
    m = YEAR_RE.search(c.get("year", "")); return int(m.group()) if m else None
def rfs(sc):
    o = sorted(range(N), key=lambda i: sc[i]); r = [0] * N
    for pos, li in enumerate(o): r[li] = pos
    return r
def blended(row, aw, yw, overlap=True):
    cards = [parse_card(getattr(row, f"ref_{x}_card")) for x in L]
    CW = [cw(c) for c in cards]; AU = [aw_(c) for c in cards]; YS = [cy(c) for c in cards]
    parts = row.text.split("[CITE]"); tw = words(row.text); S = np.zeros((N, N))
    for i in range(N):
        ctx = words(parts[i][-2000:])
        for j in range(N):
            if overlap: S[i, j] = len(ctx & CW[j])
            if AU[j] and (AU[j] & tw): S[i, j] += aw
    ky = [y for y in YS if y is not None]
    if len(ky) >= 2:
        for i in range(N):
            for j in range(N):
                if YS[j] is not None: S[i, j] += yw * (N - i) * (1 / (1 + abs(YS[j] - min(ky))))
    ri, ci = linear_sum_assignment(-S); r = [0] * N
    for mp, li in zip(ri, ci): r[li] = mp
    return r
rows = list(test.itertuples(index=False)); out = {}
out["blank_non_attempt"] = grade(pd.DataFrame(columns=["unit_id"] + RANK_COLS), answers)
out["sample_submission"] = grade(sample, answers)
out["pool_order"] = G({u: list(range(N)) for u in test.unit_id})
out["reversed_pool_order"] = G({u: list(range(N))[::-1] for u in test.unit_id})
rs = []
for seed in range(5):
    rng = np.random.default_rng(seed); p = {}
    for u in test.unit_id: perm = list(range(N)); rng.shuffle(perm); p[u] = perm
    rs.append(G(p))
out["random_permutation"] = float(np.mean(rs))
out["year_sort_ascending"] = G({r.unit_id: rfs([(cy(parse_card(getattr(r, f"ref_{x}_card"))) or 0) for x in L]) for r in rows})
out["title_length_sort"] = G({r.unit_id: rfs([len(parse_card(getattr(r, f"ref_{x}_card")).get("title", "")) for x in L]) for r in rows})
vf = defaultdict(list)
for r in train.itertuples(index=False):
    for x in L: vf[parse_card(getattr(r, f"ref_{x}_card")).get("venue", "")].append(getattr(r, f"rank_{x}") / (N - 1))
va = {k: float(np.mean(v)) for k, v in vf.items() if len(v) >= 3}; ga = float(np.mean([x for v in vf.values() for x in v]))
out["adversarial_global_venue_position"] = G({r.unit_id: rfs([va.get(parse_card(getattr(r, f"ref_{x}_card")).get("venue", ""), ga) for x in L]) for r in rows})
out["author_surname_bonus_only"] = G({r.unit_id: blended(r, 1, 0, overlap=False) for r in rows})
out["word_overlap_only_hungarian"] = G({r.unit_id: blended(r, 0, 0) for r in rows})
# sweep on TRAIN only
train_key = {r.unit_id: [getattr(r, f"rank_{x}") for x in L] for r in train.itertuples(index=False)}
def train_score(pred):
    return float(np.mean([sum(a == b for a, b in zip(train_key[u], pred[u])) / N for u in train_key]))
sweep = {}
for aw in [0, 1, 2, 4, 8]:
    for yw in [0, 0.05, 0.1, 0.2, 0.4]:
        sweep[(aw, yw)] = train_score({r.unit_id: blended(r, aw, yw) for r in train.itertuples(index=False)})
best = max(sweep, key=sweep.get)
ref = {r.unit_id: blended(r, *best) for r in rows}
out["reference_word_overlap_author_year"] = G(ref)
out["oracle"] = grade(answers.copy(), answers)
# diagnostics
ans_key = {r.unit_id: [getattr(r, f"rank_{x}") for x in L] for r in answers.itertuples(index=False)}
per = Counter(sum(a == b for a, b in zip(ans_key[u], ref[u])) / N for u in ans_key)
half = len(answers) // 2
ha, hb = answers.iloc[:half], answers.iloc[half:]
halves = {"ref": (grade(to_sub(ref), ha), grade(to_sub(ref), hb)),
          "pool": (grade(to_sub({u: list(range(N)) for u in test.unit_id}), ha), grade(to_sub({u: list(range(N)) for u in test.unit_id}), hb))}
train_venues = {parse_card(getattr(r, f"ref_{x}_card")).get("venue", "") for r in train.itertuples(index=False) for x in L}
recur = sum(parse_card(c).get("venue", "") in train_venues for c in test.ref_A_card)
print(f"N={N} train {len(train)} ({train.unit_id.nunique()}) test {len(test)}")
for k, v in out.items(): print(f"  {k:38s} {v:.4f}")
print("  best weights (author, year):", best, "train score", round(sweep[best], 4), "| overlap-only train", round(sweep[(0, 0)], 4))
print("  per-unit ref:", dict(sorted(per.items())))
print("  half-split:", halves)
print("  venue recurrence ref_A_card test->train:", recur, "/", len(test))
print("  margin ref - strongest probe:", round(out["reference_word_overlap_author_year"] - max(v for k, v in out.items() if k not in ("reference_word_overlap_author_year", "word_overlap_only_hungarian", "oracle")), 4))
json.dump({"anchors": out, "best": best, "sweep": {f"{k[0]},{k[1]}": v for k, v in sweep.items()}, "per_unit": {str(k): v for k, v in per.items()}, "halves": halves, "venue_recur": recur}, open("/Users/abhayindwar/eris-build/refshuffle/ladder4.json", "w"), indent=1)
