import os, re, sys, itertools
from collections import defaultdict, Counter
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
N=int(sys.argv[1]); D=sys.argv[2]
L=[chr(65+i) for i in range(N)]
train=pd.read_csv(f"{D}/dataset/public/train.csv"); test=pd.read_csv(f"{D}/dataset/public/test.csv"); ans=pd.read_csv(f"{D}/dataset/private/answers.csv")
tr_ans={r.unit_id:[getattr(r,f"rank_{x}") for x in L] for r in ans.itertuples(index=False)}
def pa(pred, key):  # position accuracy over a dict of true ranks
    s=[]
    for u,t in key.items():
        p=pred.get(u)
        s.append(0.0 if (p is None or sorted(p)!=list(range(N))) else sum(a==b for a,b in zip(t,p))/N)
    return float(np.mean(s))
train_key={r.unit_id:[getattr(r,f"rank_{x}") for x in L] for r in train.itertuples(index=False)}
def parse_card(s):
    d={}
    for p in s.split(" | "):
        if ": " in p: k,v=p.split(": ",1); d[k]=v
    return d
YEAR_RE=re.compile(r"\d{4}"); WORD_RE=re.compile(r"[a-zA-Z]{3,}")
STOP=set("the a an of in on at to for and or but with from by as is was were are be been being this that these those his her their its our your which who whom what when where why how not no also had has have into after before during while over under between more most some such than then out about up down off again further once here there all any both each few other same so than too very".split())
words=lambda s:{w.lower() for w in WORD_RE.findall(s)}-STOP
cw=lambda c: words(c.get("title",""))|words(c.get("venue",""))
def aw_(c):
    a=c.get("author",""); return set() if (not a or "uncredited" in a) else words(a)
def cy(c):
    m=YEAR_RE.search(c.get("year","")); return int(m.group()) if m else None
def rfs(sc):
    o=sorted(range(N),key=lambda i:sc[i]); r=[0]*N
    for pos,li in enumerate(o): r[li]=pos
    return r
def blended(row,aw,yw,overlap=True):
    cards=[parse_card(getattr(row,f"ref_{x}_card")) for x in L]
    CW=[cw(c) for c in cards]; AU=[aw_(c) for c in cards]; YS=[cy(c) for c in cards]
    parts=row.text.split("[CITE]"); tw=words(row.text); S=np.zeros((N,N))
    for i in range(N):
        ctx=words(parts[i][-2000:])
        for j in range(N):
            if overlap: S[i,j]=len(ctx&CW[j])
            if AU[j] and (AU[j]&tw): S[i,j]+=aw
    ky=[y for y in YS if y is not None]
    if len(ky)>=2:
        for i in range(N):
            for j in range(N):
                if YS[j] is not None: S[i,j]+=yw*(N-i)*(1/(1+abs(YS[j]-min(ky))))
    ri,ci=linear_sum_assignment(-S); r=[0]*N
    for mp,li in zip(ri,ci): r[li]=mp
    return r
rows=list(test.itertuples(index=False)); out=[]
def add(n,pred): out.append((n,pa(pred,tr_ans)))
add("pool order",{u:list(range(N)) for u in test.unit_id})
add("reversed",{u:list(range(N))[::-1] for u in test.unit_id})
rs=[]
for seed in range(5):
    rng=np.random.default_rng(seed); p={}
    for u in test.unit_id: perm=list(range(N)); rng.shuffle(perm); p[u]=perm
    rs.append(pa(p,tr_ans))
out.append(("random (5 seeds)",float(np.mean(rs))))
add("year-sort",{r.unit_id:rfs([(cy(parse_card(getattr(r,f"ref_{x}_card"))) or 0) for x in L]) for r in rows})
add("title-length sort",{r.unit_id:rfs([len(parse_card(getattr(r,f"ref_{x}_card")).get("title","")) for x in L]) for r in rows})
vf=defaultdict(list)
for r in train.itertuples(index=False):
    for x in L: vf[parse_card(getattr(r,f"ref_{x}_card")).get("venue","")].append(getattr(r,f"rank_{x}")/(N-1))
va={k:float(np.mean(v)) for k,v in vf.items() if len(v)>=3}; ga=float(np.mean([x for v in vf.values() for x in v]))
add("adversarial venue",{r.unit_id:rfs([va.get(parse_card(getattr(r,f"ref_{x}_card")).get("venue",""),ga) for x in L]) for r in rows})
add("word overlap only",{r.unit_id:blended(r,0,0) for r in rows})
best=None
for aw in [0,1,2,4,8]:
    for yw in [0,0.05,0.1,0.2,0.4]:
        s=pa({r.unit_id:blended(r,aw,yw) for r in train.itertuples(index=False)},train_key)
        if best is None or s>best[1]: best=((aw,yw),s)
ref={r.unit_id:blended(r,*best[0]) for r in rows}
add(f"REFERENCE w={best[0]} (train {best[1]:.3f})",ref)
print(f"\n=== N={N}: train {len(train)} / test {len(test)} | chance 1/{N}={1/N:.3f} ===")
for n,s in out: print(f"  {n:38s} {s:.4f}")
per=[sum(a==b for a,b in zip(tr_ans[u],ref[u]))/N for u in tr_ans]
print("  reference per-unit:",{f"{k:.2f}":v for k,v in sorted(Counter(per).items())})
