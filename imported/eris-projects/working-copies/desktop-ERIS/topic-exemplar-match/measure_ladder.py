"""
Measures the full baseline ladder against the shipped dataset/public + dataset/private:
  - chance (fixed slot guess)
  - keyword-overlap adversarial baseline (no model, bag-of-words)
  - trained reference (pretrained sentence-embedding cosine similarity)
  - oracle ceiling (1.0 by construction)

Not part of the shipped challenge package; a local measurement tool only.
"""
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from grade import grade

STOP = set(
    "the a an of to in and is are was were for on with by this that these those we "
    "our it its as at from be can has have been which or not using used study studies "
    "results result also however between among within each both more most such than "
    "then but if while over under about into out may might could would should will "
    "can these two one three based found show shown new high low effect effects data "
    "method methods approach analysis significant significantly compared group groups".split()
)


def toks(s):
    return set(w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP and len(w) > 2)


def main():
    test = pd.read_csv("dataset/public/test.csv")
    answers = pd.read_csv("dataset/private/answers.csv")
    n = len(test)
    print(f"test rows: {n}")

    # --- chance: always slot 0 ---
    chance_sub = pd.DataFrame({"work_id": test["work_id"], "slot": 0})
    chance_score = grade(chance_sub, answers)
    print(f"chance (fixed slot 0):        {chance_score:.4f}")

    # --- keyword overlap adversary ---
    preds = []
    for _, row in test.iterrows():
        q = toks(row["query_abstract"])
        best_slot, best_overlap = 0, -1
        for slot in range(6):
            ex_toks = toks(row[f"candidate_{slot}_exemplar"])
            overlap = len(q & ex_toks)
            if overlap > best_overlap:
                best_overlap, best_slot = overlap, slot
        preds.append(best_slot)
    overlap_sub = pd.DataFrame({"work_id": test["work_id"], "slot": preds})
    overlap_score = grade(overlap_sub, answers)
    print(f"keyword-overlap adversary:    {overlap_score:.4f}")

    # --- trained reference: pretrained sentence embedding, cosine to exemplar centroid ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    preds = []
    for _, row in test.iterrows():
        q_emb = model.encode(row["query_abstract"])
        best_slot, best_sim = 0, -1e9
        for slot in range(6):
            ex_emb = model.encode(row[f"candidate_{slot}_exemplar"])
            sim = float(np.dot(q_emb, ex_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ex_emb) + 1e-9))
            if sim > best_sim:
                best_sim, best_slot = sim, slot
        preds.append(best_slot)
    ref_sub = pd.DataFrame({"work_id": test["work_id"], "slot": preds})
    ref_score = grade(ref_sub, answers)
    print(f"trained reference (MiniLM):   {ref_score:.4f}")

    print(f"oracle ceiling:                1.0000 (by construction)")
    print()
    print(f"margin (reference - adversary): {ref_score - overlap_score:.4f}")


if __name__ == "__main__":
    main()
