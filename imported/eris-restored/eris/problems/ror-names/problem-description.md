# Cross-Lingual Organization Name Normalization

## Overview

A curated index of research institutions holds, for each one, a single
canonical English name, and also records what that institution calls itself in
its own language. "Universiteit Antwerpen" and "University of Antwerp" are the same
place. So are "Potsdam-Institut fur Klimafolgenforschung" and "Potsdam
Institute for Climate Impact Research".

Your task is to learn that mapping from 41,399 paired examples and apply it to
organizations you have never seen.

This is a string-to-string transduction problem. No pretrained models, no LLMs,
no external data - whatever you produce has to be induced from the pairs you
are given.

## Task

For each test row you get an organization's name in some language, plus the
language code. Output that organization's canonical English name.

The split is by **organization**, not by row. Every name belonging to one
institution sits entirely in train or entirely in test, so nothing can be
looked up: an organization whose English name you need at test time never had
that name shown to you in training. What transfers is the pattern, not the
entry.

## What makes this hard

**Three different behaviours share one dataset.** Some names need real
translation (Universidad de Chile becomes University of Chile). Some need
transliteration from another script. And about 38 percent need nothing at all -
the name is already identical to the canonical English one, and touching it
costs you. Deciding which case you are in is most of the work.

**Proper nouns must survive.** Potsdam stays Potsdam. A word you have never
seen before is far more likely to be a place or a founder's name than a
translatable term, so a system that pushes every token through a substitution
table damages the names it should have left alone.

**The vocabulary is systematic but the surface forms are not.** universiteit,
universite, universidad and universitaet all mean university, and there
are 112 languages in the test split with a long tail. Patterns learned from
French will not fire on Japanese.

Measured on this data with the official grader:

- Return an empty string: 0.000
- Copy the source name through unchanged: 0.428
- Aligned word substitution table: 0.475
- Bag co-occurrence table: 0.486
- Per-language table with copy-through for rare tokens: 0.494


The copy-through baseline is high because of the identity cases and the shared
proper nouns. Beating it means learning the vocabulary without destroying the
parts that were already correct.

## Data files

All public files are in ./dataset/public/

**train.csv** - 41,399 rows.

- pair_id (str): Unique row id.
- source_name (str): The organization's name in another language.
- lang (str): ISO language code of source_name, for example fr, ja, zh.
- target_name (str): Target. The canonical English name for that organization.


**test.csv** - 10,291 rows. Same columns without target_name.

**sample_submission.csv** - the required format, filled by copying
source_name through. It scores 0.428.

## Metric

**Mean token-level F1** between your output and the canonical English name,
averaged over all 10,291 test rows.

Both strings are lowercased and split into alphanumeric tokens. Precision is
the share of your tokens that appear in the target, recall is the share of
target tokens that appear in your output, and the row scores their harmonic
mean. Token counts are respected, so repeating a word earns nothing extra.

```python
from collections import Counter
import re

def token_f1(pred, target):
    p = Counter(re.findall(r"\w+", pred.lower()))
    t = Counter(re.findall(r"\w+", target.lower()))
    overlap = sum((p & t).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(t.values())
    return 2 * precision * recall / (precision + recall)
```

Range: 0.0 to 1.0, higher is better.

## Submission format

Write a CSV named submission.csv to ./working/ with exactly two columns:

```
pair_id,output
P000004,University of Antwerp
P000011,Potsdam Institute for Climate Impact Research
...
```

- One row for each pair_id in test.csv, 10,291 rows plus a header, in any
  order.
- output is your English name for that organization, as plain text.
- A pair_id you omit scores 0 for that row rather than voiding the
  submission; duplicated ids keep their first row and unknown ids are ignored.
  Only an unreadable file or a missing pair_id or output column is
  unscorable.

## Rules & environment

- Learn from the provided data only. No pretrained models, no LLMs, no
  translation APIs, and no external data of any kind.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data,
  learn the mapping, then write ./working/submission.csv
- Seed everything; your run should be reproducible.
