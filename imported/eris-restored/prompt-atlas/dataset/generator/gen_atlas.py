#!/usr/bin/env python3
"""gen_atlas.py - measure the hidden model's response to every prompt variant.

For each sampled article and each of the 256 ordered component pairs, render
the prompt, run ONE forward pass, and read the model's next-token distribution
restricted to the 17 option letters. The letter probabilities are folded back
onto categories (the option order is itself one of the interventions), so every
row of the atlas is a 17-vector over categories, not over letters.

Output: dataset/raw/atlas.parquet with columns
    article_id, comp_a, comp_b, p_<category slug> x 17

Usage:
    python3 gen_atlas.py --articles 600 --batch 32
    python3 gen_atlas.py --bench          # time 256 prompts and exit
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
from components import PAIRS                                  # noqa: E402
from render import CATEGORIES, LETTERS, render                # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RAW = os.path.join(HERE, "..", "raw")
SEED = 20260803
CKPT = 2          # articles between checkpoints
HEARTBEAT = True  # log every article


def slug(c):
    return "p_" + c.split(",")[0].split(" ")[0].lower().replace("-", "_")


SLUGS = [slug(c) for c in CATEGORIES]
assert len(set(SLUGS)) == 17, SLUGS


def load_model():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if dev == "mps" else torch.float32)
    model.to(dev).eval()
    return tok, model, dev


def letter_ids(tok):
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False)
        ids.append(enc[0])
    assert len(set(ids)) == 17, "letter tokens must be distinct"
    return torch.tensor(ids)


def sample_articles(n, seed=SEED):
    df = pd.read_csv(os.path.join(RAW, "mnds.csv"),
                     usecols=["data_id", "title", "content", "category_level_1"])
    df = df.dropna(subset=["title", "content", "category_level_1"])
    df = df[df["content"].str.len() >= 400]
    df = df[df["title"].str.len().between(20, 200)]
    # stratified by category so all 17 classes are represented in both splits
    per = max(1, n // len(CATEGORIES))
    picked = []
    for cat in sorted(df["category_level_1"].unique()):
        g = df[df["category_level_1"] == cat]
        picked.append(g.sample(min(len(g), per), random_state=seed))
    out = pd.concat(picked, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default=os.path.join(RAW, "atlas.parquet"))
    a = ap.parse_args()

    torch.manual_seed(SEED)
    arts = sample_articles(a.articles)
    print(f"{len(arts)} articles, {len(PAIRS)} pairs "
          f"-> {len(arts) * len(PAIRS):,} forward passes", flush=True)

    tok, model, dev = load_model()
    lids = letter_ids(tok).to(dev)
    print(f"device={dev}", flush=True)

    # 'least likely' hint for C14: a fixed wrong category per article, chosen
    # deterministically so the component is reproducible and never leaks truth.
    rng = np.random.default_rng(SEED)

    # ---- resume ------------------------------------------------------- #
    rows = []
    done_ids = set()
    if os.path.isfile(a.out):
        prev = pd.read_parquet(a.out)
        prev["article_id"] = prev["article_id"].astype(str)
        n = prev.groupby("article_id").size()
        done_ids = set(n[n == len(PAIRS)].index)
        prev = prev[prev["article_id"].isin(done_ids)]
        rows = list(prev.itertuples(index=False, name=None))
        print(f"resuming: {len(done_ids)} articles already measured", flush=True)

    t0 = time.time()
    todo = arts.itertuples()
    n_done = 0
    for ai, art in enumerate(todo):
        if str(art.data_id) in done_ids:
            continue
        t_art = time.time()
        wrong = [c for c in CATEGORIES if c != art.category_level_1]
        least = wrong[int(rng.integers(len(wrong)))]

        prompts, maps = [], []
        for (ca, cb) in PAIRS:
            txt, l2c = render(art.title, art.content, ca, cb, least_likely=least)
            msgs = [{"role": "user", "content": txt}]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            maps.append(l2c)

        # Batch length-sorted. The 256 variants differ a lot in length (a
        # three-shot prompt is several times a title-only one), so sorting
        # first cuts the padding waste and roughly halves the wall clock.
        ids = [tok(t, truncation=True, max_length=1024)["input_ids"]
               for t in prompts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))

        probs = np.zeros((len(PAIRS), 17), dtype=np.float32)
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            enc = tok.pad({"input_ids": [ids[i] for i in idx]},
                          return_tensors="pt").to(dev)
            with torch.no_grad():
                # logits_to_keep=1 -> only the final position goes through the
                # 152k-wide lm_head, which is what keeps this inside GPU memory
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            sel = logits.index_select(1, lids).float()
            p = torch.softmax(sel, dim=-1).cpu().numpy()
            for k, i in enumerate(idx):
                probs[i] = p[k]

        # fold letter probs onto categories
        cat_index = {c: i for i, c in enumerate(CATEGORIES)}
        folded = np.zeros_like(probs)
        for j, l2c in enumerate(maps):
            for li, L in enumerate(LETTERS):
                folded[j, cat_index[l2c[L]]] = probs[j, li]

        for j, (ca, cb) in enumerate(PAIRS):
            rows.append((str(art.data_id), ca, cb, *folded[j].tolist()))

        n_done += len(PAIRS)
        if HEARTBEAT:
            print(f"    art {ai+1} done in {time.time()-t_art:.1f}s", flush=True)
        if a.bench and ai == 0:
            dt = time.time() - t0
            print(f"bench: {len(PAIRS)} prompts in {dt:.1f}s "
                  f"= {len(PAIRS)/dt:.1f}/s -> "
                  f"{a.articles*len(PAIRS)/(len(PAIRS)/dt)/60:.0f} min total")
            return
        if (ai + 1) % CKPT == 0:
            el = time.time() - t0
            rate = n_done / el
            left = (len(arts) - ai - 1) * len(PAIRS) / rate
            # checkpoint: the atlas on disk is always a complete, usable atlas
            # over the articles finished so far, so the run can be stopped at
            # any point without losing work
            _flush(rows, arts.head(ai + 1), a.out)
            print(f"  {ai+1}/{len(arts)} articles  {rate:.1f} prompts/s  "
                  f"eta {left/60:.0f} min", flush=True)

    _flush(rows, arts, a.out)
    print(f"wrote {len(rows):,} rows -> {a.out}")


def _flush(rows, arts, out_path):
    df = pd.DataFrame(rows, columns=["article_id", "comp_a", "comp_b"] + SLUGS)
    df["article_id"] = df["article_id"].astype(str)   # stable across resumes
    df.to_parquet(out_path, index=False)
    arts.to_csv(os.path.join(RAW, "atlas_articles.csv"), index=False)


if __name__ == "__main__":
    main()
