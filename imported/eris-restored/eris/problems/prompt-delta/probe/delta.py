"""Does a single prompt edit produce a MEASURABLE, VARIED behaviour delta?
Measure: for each edit op, how much does the output distribution move,
and does it move DIFFERENTLY on different input slices?"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to("mps").eval()

BASE="Answer the question."
EDITS={
 "none":        BASE,
 "add_json":    BASE+" Respond only in JSON.",
 "add_brief":   BASE+" Answer in at most three words.",
 "add_persona": "You are a terse senior engineer. "+BASE,
 "add_cot":     BASE+" Think step by step first.",
 "add_hedge":   BASE+" If you are unsure, say you don't know.",
}
# three input slices with different character
SLICES={
 "factual":  ["What is the capital of France?","How many days in a leap year?"],
 "ambiguous":["Is the dress blue?","Which is better?"],
 "unanswerable":["What will the stock price be tomorrow?","What am I thinking?"],
}
def gen(prompts, n=48):
    texts=[tok.apply_chat_template([{"role":"user","content":p}],
            add_generation_prompt=True, tokenize=False) for p in prompts]
    enc=tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in out]

res={}
for ename,pre in EDITS.items():
    res[ename]={}
    for sname,qs in SLICES.items():
        outs=gen([f"{pre}\n\n{q}" for q in qs])
        res[ename][sname]=outs
        lens=[len(o.split()) for o in outs]
        isjson=sum(o.strip().startswith("{") for o in outs)
        idk=sum(("don't know" in o.lower() or "not know" in o.lower()
                 or "cannot" in o.lower()) for o in outs)
        print(f"{ename:12s} {sname:13s} len={sum(lens)/len(lens):5.1f} json={isjson} idk={idk}")
    print()
json.dump(res, open("probe/delta.json","w"), indent=1)
