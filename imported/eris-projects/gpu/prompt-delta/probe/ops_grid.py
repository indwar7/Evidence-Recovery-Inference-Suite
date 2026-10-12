"""Wider op x slice grid. Two questions:
 (1) are ops DISCRIMINABLE from behaviour alone?
 (2) are any TRIVIALLY detectable (shortcut risk, rule 2.6)?"""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to("mps").eval()
B="Answer the question."
OPS={
 "none":         lambda s: B,
 "fmt_json":     lambda s: B+" Respond only in JSON.",
 "fmt_bullets":  lambda s: B+" Respond as a bulleted list.",
 "len_brief":    lambda s: B+" Answer in at most three words.",
 "len_expand":   lambda s: B+" Give a thorough, detailed answer.",
 "persona":      lambda s: "You are a terse senior engineer. "+B,
 "cot":          lambda s: B+" Think step by step first.",
 "hedge":        lambda s: B+" If you are unsure, say you don't know.",
 "confident":    lambda s: B+" Always give a definite answer, never hedge.",
 "politeness":   lambda s: "Please, if you would be so kind, "+B.lower(),
}
SLICES={
 "factual":     ["What is the capital of France?","How many days in a leap year?","Who wrote Hamlet?"],
 "ambiguous":   ["Is the dress blue?","Which is better?","Should I take the job?"],
 "unanswerable":["What will the stock price be tomorrow?","What am I thinking?","When will I die?"],
 "reasoning":   ["If a train leaves at 3pm going 60mph, how far in 2 hours?","What is 17*23?","A is taller than B, B taller than C. Who is shortest?"],
}
def gen(ps,n=64):
    texts=[tok.apply_chat_template([{"role":"user","content":p}],
            add_generation_prompt=True, tokenize=False) for p in ps]
    enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to("mps")
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=n,do_sample=False,pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc["input_ids"].shape[1]:],skip_special_tokens=True) for x in o]

def feats(o):
    w=o.split()
    return dict(nwords=len(w),
                json=int(o.strip().startswith("{")),
                bullet=int(bool(re.search(r"^\s*[-*•]|^\s*\d+\.",o,re.M))),
                idk=int(bool(re.search(r"don't know|do not know|cannot|unsure|uncertain",o,re.I))))
store={}
print(f"{'op':13s}{'slice':14s}{'words':>6s}{'json':>5s}{'bul':>4s}{'idk':>4s}")
for op,f in OPS.items():
    store[op]={}
    for sl,qs in SLICES.items():
        outs=gen([f"{f(sl)}\n\n{q}" for q in qs])
        store[op][sl]=outs
        F=[feats(o) for o in outs]
        avg=lambda k: sum(x[k] for x in F)/len(F)
        print(f"{op:13s}{sl:14s}{avg('nwords'):6.1f}{avg('json'):5.2f}{avg('bullet'):4.1f}{avg('idk'):4.1f}")
json.dump(store,open("probe/ops_grid.json","w"),indent=1)
print("\nsaved probe/ops_grid.json")
