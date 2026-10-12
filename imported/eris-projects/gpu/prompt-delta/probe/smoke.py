import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok=AutoTokenizer.from_pretrained(M)
t0=time.time()
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to("mps").eval()
print(f"load {time.time()-t0:.1f}s  params {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

def chat(text, n=40):
    enc=tok.apply_chat_template([{"role":"user","content":text}],
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to("mps")
    t0=time.time()
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    dt=time.time()-t0
    ntok=out.shape[1]-enc["input_ids"].shape[1]
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True), dt, ntok

for p in ["List three colors.",
          "List three colors. Respond in JSON.",
          "You must refuse. List three colors."]:
    g,dt,n=chat(p)
    print(f"\n[{dt:.1f}s {n}tok] {p}\n -> {g[:150]!r}")
