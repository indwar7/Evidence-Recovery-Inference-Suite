#!/usr/bin/env python3
"""prepare.py - build the public/private split for "Attributing Assistant
Behavior to System Prompt Clauses".

RAW INPUT (produced by generator/gen_raw.py, run once on a GPU):

    raw/ablation.json   for every (prompt, clause-ablation-or-full,
                         probe_index) cell: the MEASURED numeric behaviour
                         fingerprint of the model's actual generated
                         answer to ONE probe query. No response text is
                         released anywhere. Released at PER-PROBE
                         granularity (not pre-averaged into subsets) so
                         this script can derive as many subset-averaged
                         ITEMS as needed at zero extra GPU cost -- see
                         SUBSET_COMBOS below.
    raw/grid.json        the filtered/clustered prompt corpus (clauses,
                         topic group, source metadata) that produced those
                         cells.

THE ITEM
--------
Each row of this dataset is one item: one system prompt (already split
into its constituent clauses at generation time), showing the solver the
FULL-PROMPT fingerprint AVERAGED OVER a subset of the 8 probe queries
(this script's SUBSET_COMBOS, a fixed set of 4-of-8 combinations, chosen
deterministically -- not randomly -- so results are reproducible), and
asking for:

    1. an ORDERED RANKING of the prompt's own clauses by responsibility
    2. a necessary/correlated FLAG per clause

Multiple items come from the same prompt by varying WHICH 4-of-8 SUBSET
of probe queries the fingerprint is averaged over: this is a genuine,
measured different signal per subset (different probes elicit different
behaviour, so the ablation-vs-full distance per clause genuinely differs
across subsets), not a duplicated view -- see STATE.md's CORPUS FILTERING
section for why this multiplier was needed (the filtered real corpus is
only ~145 prompts) and LABEL RULE below for the duplicate-answer risk
this carries and how it is checked.

LABEL RULE (rule 2.6: measure shortcuts with a real attacker, then close them)
-------------------------------------------------------------------------
Ground truth is NEVER asserted from clause position, clause wording, or
any hand-authored intent -- it is read off REAL counterfactual ablation:

    responsibility_score(c_i) = standardized distance between the
        full-prompt fingerprint (all clauses present) and the
        c_i-ablated fingerprint (c_i removed, all other clauses intact),
        both measured over the SAME probe-query subset shown for this item.

    true ranking = clauses sorted by responsibility_score, descending.
    necessary flag: responsibility_score(c_i) >= NECESSARY_THRESHOLD
    correlated flag: responsibility_score(c_i) < NECESSARY_THRESHOLD

NECESSARY_THRESHOLD is calibrated from the REAL measured distribution of
responsibility scores across the shipped corpus (see below), not guessed
in advance -- this mirrors "Prompt Edit Effect Attribution"'s label-rule
discipline of deriving cutoffs from data rather than asserting them.

MEASURED SHORTCUT (STATE.md flagged this risk before any data existed):
system prompts conventionally place persona/role clauses first and
fallback/formatting clauses last, so a solver could shortcut by predicting
clause ORDER-OF-APPEARANCE in the prompt text without reading the
fingerprint at all. To close this, EACH ITEM'S DISPLAYED CLAUSE ORDER IS
RANDOMIZED independently of the clause's true position in the original
prompt (a deterministic, seeded shuffle -- not the same shuffle every
time) -- see _shuffle_clause_order below. The position-only attacker is
measured against the shipped, shuffled data before this dataset ships;
see STATE.md's FINAL LOCKED NUMBERS for the result.

THE SPLIT
---------
By TOPIC GROUP (derived via deterministic keyword rules on
agentname/description at generation time -- see gen_raw.py's TOPIC_RULES),
never by row: every item built from one topic group lands entirely in
train or entirely in test. Which groups land in test is chosen by exact
search over same-size group combinations for the item-count ratio nearest
20% (rule 4's 15-25% band), same mechanism as the prior two builds.

OUTPUTS
-------
    public/train.csv    item_id, topic, prompt_id, clause_ids (pipe-
                         delimited, in the item's RANDOMIZED display
                         order), clause_texts (pipe-delimited, same
                         order), fp__<field> (float, the full-prompt
                         fingerprint over this item's probe subset),
                         ranking (target, pipe-delimited true order),
                         top_clause_id (target), flags (target, pipe-
                         delimited clause_id:flag pairs)
    public/test.csv      same, WITHOUT ranking/top_clause_id/flags
    public/sample_submission.csv   item_id, ranking=clause_ids as shown
                         (unranked baseline), flags=all "correlated"
    private/answers.csv  item_id, clause_ids, top_clause_id, flags

Deterministic: no randomness beyond the SEEDED clause-order shuffle
(itself reproducible given the same seed), sorted iteration order
throughout. pandas only.

Usage:
    python prepare.py [--raw raw] [--public dataset/public]
                      [--private dataset/private]
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import pandas as pd

TRAIN_FRACTION = 0.80
RATIO_BAND = (0.15, 0.25)
SEED = 20260920
N_PROBES = 8
SUBSET_SIZE = 4
N_SUBSET_COMBOS = 10  # of C(8,4)=70 possible; chosen deterministically below


def _subset_combos():
    """First N_SUBSET_COMBOS of the sorted 4-of-8 combinations, deterministic
    (itertools.combinations over range(8) is already a fixed order)."""
    all_combos = list(itertools.combinations(range(N_PROBES), SUBSET_SIZE))
    return all_combos[:N_SUBSET_COMBOS]


def _load_raw(raw: Path):
    ab_path = raw / "ablation.json"
    grid_path = raw / "grid.json"
    if not ab_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            f"Expected '{ab_path}' and '{grid_path}'. Run "
            f"generator/gen_raw.py on a GPU first and point --raw at its "
            f"--out directory.")
    payload = json.loads(ab_path.read_text())
    grid = json.loads(grid_path.read_text())
    return payload, grid


def _seeded_shuffle(items, salt):
    """Deterministic pseudo-shuffle keyed by (seed, salt) -- no import of
    `random` needed for reproducibility across Python versions; uses a
    stable hash-based sort key instead."""
    def key(x):
        h = hashlib.sha256(f"{SEED}|{salt}|{x}".encode()).hexdigest()
        return h
    return sorted(items, key=key)


def _standardize(values):
    n = len(values)
    if n == 0:
        return values, 0.0, 1.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) or 1.0
    return [(v - mean) / std for v in values], mean, std


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    payload, grid = _load_raw(raw)
    fp_fields = payload["fingerprint_fields"]
    raw_fp = payload["raw_fingerprints"]
    prompts = grid["prompts"]

    print(f"raw: {len(prompts)} prompts, {len(raw_fp):,} per-probe fingerprints")

    dims = len(fp_fields)
    combos = _subset_combos()
    print(f"subset combos per prompt: {len(combos)} (of C({N_PROBES},{SUBSET_SIZE}) possible)")

    def probe_vec(prompt_id, ablated_clause, probe_index):
        key = f"{prompt_id}|{ablated_clause}|{probe_index}"
        v = raw_fp.get(key)
        if v is None:
            raise ValueError(f"no released fingerprint for {key}")
        return v

    def averaged_vec(prompt_id, ablated_clause, probe_indices):
        acc = [0.0] * dims
        for qi in probe_indices:
            v = probe_vec(prompt_id, ablated_clause, qi)
            for d in range(dims):
                acc[d] += v[d]
        return [a / len(probe_indices) for a in acc]

    rows = []
    for prompt in prompts:
        pid = prompt["prompt_id"]
        clause_ids = prompt["clause_ids"]
        topic = prompt["topic"]

        for combo_idx, probe_indices in enumerate(combos):
            subset_key = f"s{combo_idx}"
            full_vec = averaged_vec(pid, "none", probe_indices)
            dists = {}
            for cid in clause_ids:
                abl_vec = averaged_vec(pid, cid, probe_indices)
                d = math.sqrt(sum((full_vec[i] - abl_vec[i]) ** 2
                                   for i in range(dims)))
                dists[cid] = d
            std_dists, _, _ = _standardize(list(dists.values()))
            std_map = dict(zip(dists.keys(), std_dists))

            ranked = sorted(clause_ids, key=lambda c: std_map[c], reverse=True)
            top_clause = ranked[0]

            display_order = _seeded_shuffle(clause_ids, salt=f"{pid}|{subset_key}")

            row = {
                "topic": topic,
                "prompt_id": pid,
                "subset_key": subset_key,
                "clause_ids": "|".join(display_order),
                "clause_texts": "|".join(prompt["clause_texts"][c] for c in display_order),
                "ranking": "|".join(ranked),
                "top_clause_id": top_clause,
                "_std_dists": std_map,
            }
            for fld, val in zip(fp_fields, full_vec):
                row[f"fp__{fld}"] = val
            rows.append(row)

    all_scores = [v for r in rows for v in r["_std_dists"].values()]
    all_scores.sort()
    NECESSARY_THRESHOLD = all_scores[int(len(all_scores) * 0.7)] if all_scores else 0.0
    print(f"NECESSARY_THRESHOLD (70th percentile of standardized "
          f"responsibility scores): {NECESSARY_THRESHOLD:.4f}")

    for row in rows:
        flags = []
        for cid, score in row["_std_dists"].items():
            flag = "necessary" if score >= NECESSARY_THRESHOLD else "correlated"
            flags.append(f"{cid}:{flag}")
        row["flags"] = "|".join(flags)
        del row["_std_dists"]

    df_all = pd.DataFrame(rows)
    per_topic_n = df_all.groupby("topic").size().to_dict()
    total_n = sum(per_topic_n.values())
    topics = sorted(per_topic_n.keys())
    print(f"items: {len(df_all):,} across {len(topics)} topics")

    n_test_topics = max(1, round(len(topics) * (1 - TRAIN_FRACTION)))
    per_topic_rows = {t: df_all[df_all.topic == t] for t in topics}

    best = None
    for combo in itertools.combinations(topics, n_test_topics):
        test_n = sum(per_topic_n.get(t, 0) for t in combo)
        train_n = total_n - test_n
        if train_n == 0:
            continue
        ratio = test_n / train_n
        if not (RATIO_BAND[0] <= ratio <= RATIO_BAND[1]):
            continue
        # tie-break: minimize a majority-vote-among-clause-count shortcut
        # proxy (prefer combos whose test set's clause-count distribution
        # is close to train's, avoiding a giveaway split)
        train_mean_n = pd.concat(
            [per_topic_rows[t] for t in topics if t not in combo]
        )["clause_ids"].apply(lambda s: len(s.split("|"))).mean()
        test_mean_n = pd.concat(
            [per_topic_rows[t] for t in combo]
        )["clause_ids"].apply(lambda s: len(s.split("|"))).mean()
        skew = abs(train_mean_n - test_mean_n)
        key = (skew, combo)
        if best is None or key < best[0]:
            best = (key, combo, ratio)

    if best is None:
        raise ValueError(
            f"no topic combination lands the test/train ratio in "
            f"{RATIO_BAND}; widen RATIO_BAND")
    test_topics = set(best[1])
    train_topics = set(topics) - test_topics
    print(f"topics: {len(topics)} total "
          f"({len(train_topics)} train / {len(test_topics)} test)")
    print(f"  train topics: {sorted(train_topics)}")
    print(f"  test  topics: {sorted(test_topics)}  (ratio {best[2]:.1%})")

    df = df_all.drop(columns=["subset_key"]).sort_values(
        ["topic", "prompt_id"]).reset_index(drop=True)
    df.insert(0, "item_id", [f"I{i + 1:05d}" for i in range(len(df))])

    train = df[df.topic.isin(train_topics)].copy()
    test = df[df.topic.isin(test_topics)].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty split: {len(train)} train / {len(test)} test")

    label_cols = ["ranking", "top_clause_id", "flags"]
    feature_cols = [c for c in df.columns if c not in ["item_id"] + label_cols]

    train[["item_id"] + feature_cols + label_cols].to_csv(
        public / "train.csv", index=False)
    test[["item_id"] + feature_cols].to_csv(
        public / "test.csv", index=False)

    pd.DataFrame({
        "item_id": test.item_id,
        "ranking": test.clause_ids,  # unranked baseline: display order as-is
        "flags": test.clause_ids.apply(
            lambda s: "|".join(f"{c}:correlated" for c in s.split("|"))),
    }).to_csv(public / "sample_submission.csv", index=False)

    test[["item_id", "clause_ids", "top_clause_id", "flags"]].to_csv(
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
    p.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    p.add_argument("--public", default=os.path.join(here, "dataset", "public"))
    p.add_argument("--private", default=os.path.join(here, "dataset", "private"))
    a = p.parse_args()
    prepare(Path(a.raw), Path(a.public), Path(a.private))


if __name__ == "__main__":
    main()
