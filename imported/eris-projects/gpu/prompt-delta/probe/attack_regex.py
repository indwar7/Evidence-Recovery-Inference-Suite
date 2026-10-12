"""ATTACK MY OWN DATA (rule 2.6): how far does a dumb regex classifier get
at naming the op from the output alone? Measured, not argued."""
import json, re
from collections import defaultdict
G=json.load(open("probe/ops_grid.json"))
def feats(o):
    w=o.split()
    return dict(n=len(w), json=o.strip().startswith("{"),
        bullet=bool(re.search(r"^\s*[-*•]|^\s*\d+\.",o,re.M)),
        idk=bool(re.search(r"don't know|do not know|cannot|unsure",o,re.I)))
def rule(f):
    if f["json"]: return "fmt_json"
    if f["n"]<=3: return "len_brief"
    if f["bullet"] and f["n"]>40: return "cot"
    if f["bullet"]: return "fmt_bullets"
    if f["idk"]: return "hedge"
    if f["n"]>=45: return "len_expand"
    if f["n"]<=5: return "persona"
    return "none"
tot=hit=0; per=defaultdict(lambda:[0,0])
for op,sl in G.items():
    for s,outs in sl.items():
        for o in outs:
            p=rule(feats(o)); tot+=1; per[op][1]+=1
            if p==op: hit+=1; per[op][0]+=1
print(f"REGEX-ONLY OP ACCURACY: {hit}/{tot} = {hit/tot:.3f}   (chance={1/len(G):.3f})")
print("\nper-op recall (how free is each op?):")
for op,(h,n) in sorted(per.items(), key=lambda x:-x[1][0]/x[1][1]):
    print(f"  {op:13s} {h}/{n} = {h/n:.2f}")
