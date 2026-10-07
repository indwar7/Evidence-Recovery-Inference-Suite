#!/usr/bin/env python3
"""worker_cpu.py -- second measurement worker, running on the CPU cores.

Identical measurement to gen_atlas.py, but it walks the SAME article list from
the far end backwards and writes to its own parquet. The two workers therefore
meet in the middle and between them cover the list roughly twice as fast as
either alone. merge_atlas.py joins the two outputs and drops any article both
happened to finish.

Usage:  python3 worker_cpu.py --articles 119 --batch 16
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from components import PAIRS                                   # noqa: E402
from render import CATEGORIES, LETTERS, render                 # noqa: E402
import gen_atlas as G                                          # noqa: E402

RAW = os.path.join(HERE, "..", "raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=119)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RAW, "atlas_cpu.parquet"))
    a = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(G.MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(G.MODEL_ID,
                                                 dtype=torch.float32)
    model.to("cpu").eval()
    lids = G.letter_ids(tok)

    arts = G.sample_articles(a.articles)
    arts = arts.iloc[::-1].reset_index(drop=True)      # walk from the far end
    print(f"cpu worker: {len(arts)} articles, walking backwards", flush=True)

    rng = np.random.default_rng(G.SEED)
    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    rows = []
    for ai, art in enumerate(arts.itertuples()):
        t0 = time.time()
        wrong = [c for c in CATEGORIES if c != art.category_level_1]
        least = wrong[int(rng.integers(len(wrong)))]

        prompts, maps = [], []
        for (ca, cb) in PAIRS:
            txt, l2c = render(art.title, art.content, ca, cb,
                              least_likely=least)
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": txt}], tokenize=False,
                add_generation_prompt=True))
            maps.append(l2c)

        ids = [tok(t, truncation=True, max_length=1024)["input_ids"]
               for t in prompts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        probs = np.zeros((len(PAIRS), 17), dtype=np.float32)
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            enc = tok.pad({"input_ids": [ids[i] for i in idx]},
                          return_tensors="pt")
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
            p = torch.softmax(lg.index_select(1, lids).float(), -1).numpy()
            for k, i in enumerate(idx):
                probs[i] = p[k]

        folded = np.zeros_like(probs)
        for j, l2c in enumerate(maps):
            for li, L in enumerate(LETTERS):
                folded[j, cat_index[l2c[L]]] = probs[j, li]
        for j, (ca, cb) in enumerate(PAIRS):
            rows.append((art.data_id, ca, cb, *folded[j].tolist()))

        print(f"    cpu art {ai+1} done in {time.time()-t0:.1f}s", flush=True)
        if (ai + 1) % 2 == 0:
            pd.DataFrame(rows, columns=["article_id", "comp_a", "comp_b"]
                         + G.SLUGS).to_parquet(a.out, index=False)

    pd.DataFrame(rows, columns=["article_id", "comp_a", "comp_b"]
                 + G.SLUGS).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
