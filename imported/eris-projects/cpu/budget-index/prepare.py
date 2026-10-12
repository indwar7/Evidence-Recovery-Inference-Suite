#!/usr/bin/env python3
"""prepare.py -- transform the raw "Budget-Constrained Index Construction"
dataset into the public / private splits.

Split design
------------
* PUBLIC  documents.csv       : the full corpus (doc_id, sent_id, n_tokens,
                                text, budget_k). budget_k is the per-document
                                index budget -- the max number of windows you
                                may build for that document.
* PUBLIC  queries_train.csv   : development queries WITH supervision --
                                required_sent_ids and poison_sent_ids (and the
                                construction class) revealed.
* PUBLIC  queries_test.csv    : evaluation queries -- question text and doc_id
                                only.
* PUBLIC  sample_submission.csv : a valid budget-respecting baseline (the first
                                budget_k short windows of each document).
* PRIVATE answers.csv         : ONE ROW PER DOCUMENT, keyed by `doc_id` as the
                                FIRST column (the platform keys submissions on
                                the first answers column). Each row packs the
                                document's per-sentence token counts, its index
                                budget, and its evaluation queries as JSON, so
                                every row is a complete, independently gradeable
                                unit -- any row-wise slice (the leaderboard
                                split) grades correctly. Rows are ordered so a
                                contiguous slice is a representative mix.

Determinism: pure selection / projection, no RNG. Only pandas is used.

Usage:
    python prepare.py [--raw-dir dataset/raw] [--out-dir prepared]
"""

import argparse
import json
import os
import shutil

import pandas as pd

MAX_CHUNK_TOKENS = 60
PHI_INV = 0.61803398874989
KFRAC = 0.5            # budget_k = max(BUDGET_MIN, round(KFRAC * #test-queries))
BUDGET_MIN = 2        # must match dataset/generator/generate.py exactly


def budget_k_for(n_queries):
    """Per-document index budget -- identical to the generator's definition.
    Used to reconstruct budget_k when the raw documents.csv does not carry the
    column (verified to reproduce the generator's values exactly)."""
    return max(BUDGET_MIN, int(round(KFRAC * n_queries)))


def prepare(raw_dir, public_dir, private_dir):
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    docs = pd.read_csv(os.path.join(raw_dir, "documents.csv"))
    queries = pd.read_csv(os.path.join(raw_dir, "queries.csv"),
                          keep_default_na=False)

    assert set(queries["split"]) == {"train", "test"}
    assert queries["query_id"].is_unique
    assert set(queries["doc_id"]) <= set(docs["doc_id"])

    train_q = queries[queries["split"] == "train"][
        ["query_id", "doc_id", "question", "qtype", "cls",
         "required_sent_ids", "poison_sent_ids"]]
    test_q = queries[queries["split"] == "test"][
        ["query_id", "doc_id", "question"]]

    # test queries per document -- built first, because it is also the source
    # from which budget_k is reconstructed when the raw corpus omits the column
    test_full = queries[queries["split"] == "test"]
    q_by_doc = {}
    for r in test_full.itertuples(index=False):
        req = [int(x) for x in r.required_sent_ids.split("|")]
        poi = ([int(x) for x in str(r.poison_sent_ids).split("|")]
               if str(r.poison_sent_ids).strip() else [])
        q_by_doc.setdefault(r.doc_id, []).append({"r": req, "p": poi})

    # budget_k: use the column if the raw corpus carries it; otherwise
    # reconstruct it from the test-query workload with the generator's exact
    # formula (verified to reproduce every value). This keeps prepare robust to
    # a raw documents.csv that does not ship the budget_k column.
    has_budget = "budget_k" in docs.columns
    tok_by_doc, budget_by_doc = {}, {}
    for doc_id, grp in docs.groupby("doc_id", sort=True):
        grp = grp.sort_values("sent_id")
        assert grp["sent_id"].tolist() == list(range(len(grp)))
        tok_by_doc[doc_id] = grp["n_tokens"].tolist()
        if has_budget:
            budget_by_doc[doc_id] = int(grp["budget_k"].iloc[0])
        else:
            budget_by_doc[doc_id] = budget_k_for(len(q_by_doc.get(doc_id, [])))

    # order docs by answerable-fraction, then golden-ratio re-key so any
    # contiguous slice (the leaderboard split) holds a representative mix
    def answerable_frac(doc_id):
        toks, qs = tok_by_doc[doc_id], q_by_doc.get(doc_id, [])
        if not qs:
            return 0.0
        ok = 0
        for q in qs:
            lo, hi = min(q["r"]), max(q["r"])
            if (sum(toks[lo:hi + 1]) <= MAX_CHUNK_TOKENS
                    and not any(lo <= p <= hi for p in q["p"])):
                ok += 1
        return ok / len(qs)

    all_docs = sorted(tok_by_doc)
    fracs = sorted((answerable_frac(d), d) for d in all_docs)
    keyed = sorted(((rank * PHI_INV) % 1.0, d)
                   for rank, (_, d) in enumerate(fracs))
    doc_order = [d for _, d in keyed]

    ans_rows = []
    for doc_id in doc_order:
        qs = q_by_doc.get(doc_id, [])
        ans_rows.append({
            "doc_id": doc_id,
            "budget_k": budget_by_doc[doc_id],
            "tok_json": json.dumps(tok_by_doc[doc_id], separators=(",", ":")),
            "queries_json": json.dumps(qs, separators=(",", ":")),
            "n_test_queries": len(qs),
        })
    answers = pd.DataFrame(ans_rows, columns=[
        "doc_id", "budget_k", "tok_json", "queries_json", "n_test_queries"])
    assert list(answers.columns)[0] == "doc_id"
    assert answers["n_test_queries"].sum() == len(test_full)

    # sample submission: the first budget_k short (<=3-sentence, cap-safe)
    # windows of each document -- valid but budget-blind
    sample_rows = []
    for doc_id, grp in docs.groupby("doc_id", sort=True):
        toks = grp.sort_values("sent_id")["n_tokens"].tolist()
        n = len(toks)
        K = budget_by_doc[doc_id]
        spans, s = [], 0
        while s < n and len(spans) < K:
            e, tk = s, 0
            while e < n and e - s < 3 and tk + toks[e] <= MAX_CHUNK_TOKENS:
                tk += toks[e]
                e += 1
            if e == s:
                e = s + 1
            spans.append(f"{s}-{e}")
            s = e
        sample_rows.append({"doc_id": doc_id, "windows": "|".join(spans)})
    sample = pd.DataFrame(sample_rows)

    assert "required_sent_ids" not in test_q.columns
    assert "poison_sent_ids" not in test_q.columns
    assert set(sample["doc_id"]) == set(answers["doc_id"])

    if not has_budget:                        # materialize reconstructed budget
        docs["budget_k"] = docs["doc_id"].map(budget_by_doc)
    docs[["doc_id", "sent_id", "n_tokens", "text", "budget_k"]].to_csv(
        os.path.join(public_dir, "documents.csv"), index=False)
    train_q.to_csv(os.path.join(public_dir, "queries_train.csv"), index=False)
    test_q.to_csv(os.path.join(public_dir, "queries_test.csv"), index=False)
    sample.to_csv(os.path.join(public_dir, "sample_submission.csv"),
                  index=False)
    answers.to_csv(os.path.join(private_dir, "answers.csv"), index=False)

    # ship the real-world knowledge base publicly: poison sentences carry a
    # different country's real value and have no textual tell, so a solver must
    # cross-reference each stated value against the entity's true value here.
    # (Satisfies the platform's "no external knowledge required" constraint.)
    kb_src = os.path.join(raw_dir, "real_facts.json")
    if os.path.exists(kb_src):
        shutil.copyfile(kb_src, os.path.join(public_dir, "real_facts.json"))

    print(f"documents.csv        : {len(docs):,} rows, "
          f"{docs['doc_id'].nunique()} documents")
    print(f"queries_train.csv    : {len(train_q)} rows")
    print(f"queries_test.csv     : {len(test_q)} rows")
    print(f"sample_submission.csv: {len(sample)} rows")
    print(f"answers.csv          : {len(answers)} rows (one per document)")


def main(raw_dir, out_dir):
    prepare(raw_dir,
            os.path.join(out_dir, "public"),
            os.path.join(out_dir, "private"))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=os.path.join(here, "dataset", "raw"))
    parser.add_argument("--out-dir", default=os.path.join(here, "prepared"))
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
