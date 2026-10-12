#!/usr/bin/env python3
"""reference_solution.py - reference solver for "Say It In English".

Learns the transduction as a per-language word substitution table plus a
copy-through rule, both induced from the training pairs alone.

Three things have to be learned, and they pull in different directions:

  * Systematic vocabulary. "universiteit", "universite", "universidad" and
    "universitaet" all map to "university". These are learnable from aligned
    pairs and account for most of the gain.
  * When to leave the string alone. Around 38 percent of registry labels are
    already identical to the English display name, so a solver that translates
    everything is worse than one that knows when to stop.
  * Proper nouns. City and founder names usually carry over unchanged, and a
    rare token is far more likely to be one of those than a word with a known
    translation. The solver copies low-frequency tokens rather than forcing
    them through the table.

Reads  ./dataset/public/{train,test}.csv
Writes ./working/submission.csv

Deterministic; pandas only.
"""

import collections
import os
import re

import pandas as pd

RARE_BELOW = 3          # tokens seen fewer times than this are treated as names
MIN_EVIDENCE = 2        # a substitution needs this many aligned observations
TOKEN = re.compile(r"\w+", re.UNICODE)


def tokens(s):
    return TOKEN.findall(str(s).lower())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pub = os.path.join(here, "dataset", "public")
    if not os.path.isdir(pub):
        pub = os.path.join(here, "prepared", "public")
    rd = lambda f: pd.read_csv(os.path.join(pub, f), keep_default_na=False)

    train, test = rd("train.csv"), rd("test.csv")
    print(f"train {len(train):,} pairs, test {len(test):,} pairs")

    # --- aligned substitutions, learned per language and pooled ---
    per_lang = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    pooled = collections.defaultdict(collections.Counter)
    cooccur = collections.defaultdict(collections.Counter)
    freq = collections.Counter()

    for src, lang, tgt in zip(train.source_name, train.lang,
                              train.target_name):
        s, t = tokens(src), tokens(tgt)
        freq.update(s)
        if len(s) == len(t):                 # positionally alignable
            for a, b in zip(s, t):
                per_lang[lang][a][b] += 1
                pooled[a][b] += 1
        else:                                # fall back to bag co-occurrence
            for a in set(s):
                for b in set(t):
                    cooccur[a][b] += 1

    def best(counter, floor):
        top = counter.most_common(1)
        return top[0][0] if top and top[0][1] >= floor else None

    lang_tbl = {l: {w: best(c, MIN_EVIDENCE) for w, c in d.items()}
                for l, d in per_lang.items()}
    pooled_tbl = {w: best(c, MIN_EVIDENCE) for w, c in pooled.items()}
    cooc_tbl = {w: best(c, MIN_EVIDENCE + 1) for w, c in cooccur.items()}
    print(f"learned {sum(1 for v in pooled_tbl.values() if v):,} pooled "
          f"substitutions across {len(lang_tbl)} languages")

    def translate(src, lang):
        out = []
        for w in tokens(src):
            sub = (lang_tbl.get(lang, {}).get(w)
                   or pooled_tbl.get(w)
                   or (w if freq[w] < RARE_BELOW else cooc_tbl.get(w)))
            out.append(sub or w)          # unknown -> copy through
        return " ".join(out)

    outputs = [translate(s, l) for s, l in zip(test.source_name, test.lang)]

    work = os.path.join(here, "working")
    os.makedirs(work, exist_ok=True)
    pd.DataFrame({"pair_id": test.pair_id, "target_name": outputs}).to_csv(
        os.path.join(work, "submission.csv"), index=False)
    print(f"wrote {len(outputs):,} rows to working/submission.csv")


if __name__ == "__main__":
    main()
