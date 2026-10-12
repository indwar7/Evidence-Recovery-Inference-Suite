#!/usr/bin/env python3
"""prepare.py - build the public/private split for "Predicting Retrieval
Sufficiency in RAG Pipelines".

RAW INPUT (produced by generator/gen_raw.py, run once on a GPU):

    raw/retrieval.json   for every ITEM (a query shown with its top-k
                          retrieved passages, k varying per item -- see
                          gen_raw.py's ITEM MULTIPLIER comment): the
                          MEASURED gold passage rank in the query's own
                          topic, the released top-k passage ids and their
                          real cosine similarity scores, the measured
                          best-own-topic and best-other-topic similarity
                          scores, and a numeric behaviour fingerprint of
                          the generator's actual answer over that item's
                          context. No response text or passage text is
                          released in this raw file's numeric fields.
    raw/grid.json          the hand-written topic/passage/query corpus
                          that produced those measurements.

THE ITEM
--------
Each row of this dataset is one item: a query, the topic's real retrieval
result truncated to that item's k, and the generator's real behaviour
fingerprint over that retrieved context. A solver is shown the query text,
the retrieved passage ids/scores/rank-adjacent signals, and the generation
fingerprint -- never the gold passage id, never the true rank, never the
raw response text.

THE SUBMITTED ARTIFACT
-----------------------
A joint 3-field decision trace, not a score, not a single label, not a
ranking:

    (diagnosis, action, reason_code)

  diagnosis   in {sufficient, insufficient}
  action      in {answer_as_is, reformulate, abstain}
  reason_code in {sufficient, gold_low_rank, gold_absent, query_ambiguous}

LABEL RULE (rule 2.6: measure shortcuts with a real attacker, then close them)
-------------------------------------------------------------------------
The gold trace is derived ENTIRELY from measured retrieval facts (never
from the generator's output, per an explicit design decision to keep
grading deterministic) using the item's MEASURED gold_rank_in_topic versus
its own k, not from the query's construction_intent directly:

    if construction_intent == "gold_absent":
        diagnosis=insufficient, action=abstain, reason_code=gold_absent
    elif gold_rank_in_topic is None or gold_rank_in_topic > k:
        diagnosis=insufficient
        if best_other_topic_score is within AMBIGUITY_MARGIN of
        best_own_topic_score:
            reason_code=query_ambiguous, action=reformulate
        else:
            reason_code=gold_low_rank, action=reformulate
    else:  # gold_rank_in_topic <= k
        diagnosis=sufficient, action=answer_as_is, reason_code=sufficient

An item's construction_intent is a hypothesis, not the label: an item
"intended" as gold_low_rank whose measured rank happens to fall inside its
k window is relabeled sufficient by its true measured facts, and an item
"intended" as query_ambiguous whose cross-topic score gap does not actually
measure as ambiguous is relabeled gold_low_rank. This mirrors the "Prompt
Edit Effect Attribution" build's discipline of never asserting a label from
construction intent when a real measurement is available.

MEASURED SHORTCUT (see STATE.md SHORTCUT PLAN for the risk that was flagged
BEFORE any data existed): a solver could threshold on
best_own_topic_score - best_other_topic_score alone -- both released
features -- to shortcut reason_code without reading the generation
fingerprint at all, since that gap is literally what the label rule uses to
split query_ambiguous from gold_low_rank. Measured attacker and any
mitigation applied are recorded below once real numbers exist (this
section is updated after the first real gen_raw.py run; do not trust the
docstring over STATE.md's FINAL LOCKED NUMBERS once that section exists).

THE SPLIT
---------
By TOPIC, never by row: every item for a given topic lands entirely in
train or entirely in test. Which topics land in test is chosen by exact
search over same-size topic combinations (deterministic, no randomness),
picking the combination that lands the item-count ratio nearest 20% (rule
4's 15-25% band) while minimizing a majority-vote-among-candidates shortcut
on the resulting test set -- same mechanism as the "Prompt Edit Effect
Attribution" build's by-family search.

OUTPUTS
-------
    public/train.csv              item_id, topic_id, query_text, k,
                                   retrieved_id_1..k, retrieved_score_1..k,
                                   fp__<field> x N, diagnosis, action,
                                   reason_code
    public/test.csv                same, WITHOUT diagnosis/action/reason_code
    public/sample_submission.csv   item_id, diagnosis, action, reason_code
                                    (a valid-but-unskilled baseline)
    private/answers.csv            item_id, diagnosis, action, reason_code

Deterministic: no randomness anywhere (the test-topic choice is an exact
search, not a shuffle), sorted iteration order throughout. pandas only.

Usage:
    python prepare.py [--raw raw] [--public dataset/public]
                      [--private dataset/private]
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import pandas as pd

TRAIN_FRACTION = 0.80        # target ~20% test (rule 4: 15-25% band)
RATIO_BAND = (0.15, 0.25)
AMBIGUITY_MARGIN = 0.03      # measured-score-gap threshold for query_ambiguous
                             # vs gold_low_rank; TBD/tune once real data exists
BALANCE_TOLERANCE = 1.15     # allow each (diagnosis,action,reason_code) combo
                             # up to 1.15x the even share within a topic
                             # before capping (see LABEL RULE docstring)


def _load_raw(raw: Path):
    ret_path = raw / "retrieval.json"
    grid_path = raw / "grid.json"
    if not ret_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            f"Expected '{ret_path}' and '{grid_path}'. Run "
            f"generator/gen_raw.py on a GPU first (see its --smoke / full "
            f"build usage) and point --raw at its --out directory.")
    payload = json.loads(ret_path.read_text())
    grid = json.loads(grid_path.read_text())
    return payload, grid


def _derive_trace(item):
    """The LABEL RULE (see module docstring): read gold trace off MEASURED
    retrieval facts, never off construction_intent directly."""
    intent = item["construction_intent"]
    rank = item["gold_rank_in_topic"]
    k = item["k"]

    if intent == "gold_absent":
        return "insufficient", "abstain", "gold_absent"

    if rank is None or rank > k:
        gap = item["best_own_topic_score"] - (item["best_other_topic_score"] or -1.0)
        if item["best_other_topic_score"] is not None and gap <= AMBIGUITY_MARGIN:
            return "insufficient", "reformulate", "query_ambiguous"
        return "insufficient", "reformulate", "gold_low_rank"

    return "sufficient", "answer_as_is", "sufficient"


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    payload, grid = _load_raw(raw)
    items_raw = payload["items"]
    fingerprints = payload["generation_fingerprints"]
    fp_fields = payload["fingerprint_fields"]
    topics = payload["topics"]

    print(f"raw: {len(topics)} topics, {payload['n_passages']} passages, "
          f"{payload['n_queries']} queries, {len(items_raw):,} items")

    rows = []
    for item_key, it in sorted(items_raw.items()):
        diagnosis, action, reason_code = _derive_trace(it)
        fp = fingerprints[item_key]
        row = {
            "topic_id": it["topic_id"],
            "query_index": it["query_index"],
            "query_text": it["query_text"],
            "k": it["k"],
        }
        for i, (pid, score) in enumerate(
                zip(it["top_k_ids"], it["top_k_scores"]), start=1):
            row[f"retrieved_id_{i}"] = pid
            row[f"retrieved_score_{i}"] = score
        # pad missing retrieved slots (k varies per item) so the CSV has a
        # fixed column set across all items, up to the max k observed
        row["_n_retrieved"] = len(it["top_k_ids"])
        for fld, val in zip(fp_fields, fp):
            row[f"fp__{fld}"] = val
        row["diagnosis"] = diagnosis
        row["action"] = action
        row["reason_code"] = reason_code
        row["_combo"] = f"{diagnosis}|{action}|{reason_code}"
        rows.append(row)

    max_k = max(r["_n_retrieved"] for r in rows)
    for row in rows:
        for i in range(row["_n_retrieved"] + 1, max_k + 1):
            row.setdefault(f"retrieved_id_{i}", "")
            row.setdefault(f"retrieved_score_{i}", "")

    df_all = pd.DataFrame(rows)

    # -- balance (diagnosis, action, reason_code) combos within each topic,
    # dropping the highest-margin (easiest, most lopsided score-gap) items
    # for over-represented combos first, same discipline as "Prompt Edit
    # Effect Attribution"'s per-family label balancing. -------------------
    kept = []
    for topic_id, grp in df_all.groupby("topic_id"):
        n_combos = grp["_combo"].nunique()
        even_share = len(grp) / max(n_combos, 1)
        cap = int(even_share * BALANCE_TOLERANCE) + 1
        for combo, sub in grp.groupby("_combo"):
            kept.append(sub.head(cap))
    df_all = pd.concat(kept, ignore_index=True) if kept else df_all

    per_topic_n = df_all.groupby("topic_id").size().to_dict()
    total_n = sum(per_topic_n.values())
    print(f"items after balancing: {len(df_all):,}")
    print(f"trace distribution: {df_all._combo.value_counts().to_dict()}")

    # -- choose test topics by exact search: land the ratio in-band AND
    # minimize the majority-vote-among-topics shortcut on the resulting
    # test set (same mechanism as the prior build's by-family search). ----
    n_test_topics = len(topics) - round(len(topics) * TRAIN_FRACTION)
    per_topic_rows = {t: df_all[df_all.topic_id == t] for t in topics}

    best = None
    for combo in itertools.combinations(sorted(topics), n_test_topics):
        test_n = sum(per_topic_n.get(t, 0) for t in combo)
        train_n = total_n - test_n
        if train_n == 0:
            continue
        ratio = test_n / train_n
        if not (RATIO_BAND[0] <= ratio <= RATIO_BAND[1]):
            continue
        train_labels = pd.concat(
            [per_topic_rows[t] for t in topics if t not in combo]
        )._combo.value_counts()
        if train_labels.empty:
            continue
        majority_label = train_labels.idxmax()
        test_df = pd.concat([per_topic_rows[t] for t in combo])
        maj_acc = (test_df._combo == majority_label).mean()
        key = (maj_acc, combo)
        if best is None or key < best[0]:
            best = (key, combo, ratio, maj_acc)

    if best is None:
        raise ValueError(
            "no topic combination lands the test/train ratio in "
            f"{RATIO_BAND}; widen RATIO_BAND or adjust BALANCE_TOLERANCE")
    test_topics = set(best[1])
    train_topics = set(topics) - test_topics
    print(f"topics: {len(topics)} total "
          f"({len(train_topics)} train / {len(test_topics)} test)")
    print(f"  train topics: {sorted(train_topics)}")
    print(f"  test  topics: {sorted(test_topics)}  "
          f"(ratio {best[2]:.1%}, majority-combo-shortcut {best[3]:.4f})")

    df = df_all.drop(columns=["_combo", "_n_retrieved"])
    df = df.sort_values(["topic_id", "query_index", "k"]).reset_index(drop=True)
    df.insert(0, "item_id", [f"I{i + 1:05d}" for i in range(len(df))])

    train = df[df.topic_id.isin(train_topics)].copy()
    test = df[df.topic_id.isin(test_topics)].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty split: {len(train)} train / {len(test)} test")

    label_cols = ["diagnosis", "action", "reason_code"]
    feature_cols = [c for c in df.columns
                    if c not in ["item_id"] + label_cols]

    train[["item_id"] + feature_cols + label_cols].to_csv(
        public / "train.csv", index=False)
    test[["item_id"] + feature_cols].to_csv(
        public / "test.csv", index=False)

    # valid-but-unskilled baseline: always guess "sufficient" (the modal
    # class in an unbalanced real-world retrieval setting)
    pd.DataFrame({
        "item_id": test.item_id,
        "diagnosis": "sufficient",
        "action": "answer_as_is",
        "reason_code": "sufficient",
    }).to_csv(public / "sample_submission.csv", index=False)

    test[["item_id"] + label_cols].to_csv(
        private / "answers.csv", index=False)

    ratio = len(test) / len(train)
    print(f"public/train.csv              {len(train):,} rows")
    print(f"public/test.csv               {len(test):,} rows")
    print(f"public/sample_submission.csv  {len(test):,} rows")
    print(f"private/answers.csv           {len(test):,} rows")
    print(f"test/train ratio: {ratio:.1%}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=os.path.join(here, "raw"))
    p.add_argument("--public", default=os.path.join(here, "public"))
    p.add_argument("--private", default=os.path.join(here, "private"))
    a = p.parse_args()
    prepare(Path(a.raw), Path(a.public), Path(a.private))


if __name__ == "__main__":
    main()
