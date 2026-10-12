import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to("mps").eval()

prompts=[f"Summarize the benefit of item {i} in one sentence." for i in range(32)]
for bs in (1,4,8,16):
    batch=prompts[:bs]
    texts=[tok.apply_chat_template([{"role":"user","content":p}],
            add_generation_prompt=True, tokenize=False) for p in batch]
    enc=tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
    torch.mps.synchronize(); t0=time.time()
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=48, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    torch.mps.synchronize(); dt=time.time()-t0
    newtok=(out.shape[1]-enc["input_ids"].shape[1])*bs
    print(f"bs={bs:2d}  {dt:5.1f}s  {newtok/dt:6.1f} tok/s  {dt/bs:.2f}s/prompt")
