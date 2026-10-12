"""Is 'which slice did the edit move' a REAL target?
For each op, compute per-slice movement vs the 'none' baseline.
If an op moves all slices equally, the slice label is unlearnable noise.
If it moves them unequally, the slice is a genuine target."""
import json, re
G=json.load(open("probe/ops_grid.json"))
SL=["factual","ambiguous","unanswerable","reasoning"]
def vec(o):
    w=len(o.split())
    return (w, float(o.strip().startswith("{")),
            float(bool(re.search(r"^\s*[-*•]|^\s*\d+\.",o,re.M))),
            float(bool(re.search(r"don't know|do not know|cannot|unsure",o,re.I))))
def avg(outs):
    V=[vec(o) for o in outs]
    return [sum(x[i] for x in V)/len(V) for i in range(4)]
base={s:avg(G["none"][s]) for s in SL}
print(f"{'op':13s}" + "".join(f"{s[:9]:>11s}" for s in SL) + "   spread  argmax")
for op in G:
    if op=="none": continue
    d={}
    for s in SL:
        a=avg(G[op][s]); b=base[s]
        # normalised movement: length change in relative terms + flag changes
        dl=abs(a[0]-b[0])/max(b[0],1)
        df=abs(a[1]-b[1])+abs(a[2]-b[2])+abs(a[3]-b[3])
        d[s]=dl+df
    mx=max(d.values()); mn=min(d.values())
    top=max(d,key=d.get)
    print(f"{op:13s}" + "".join(f"{d[s]:11.2f}" for s in SL) + f"{mx-mn:9.2f}  {top}")
