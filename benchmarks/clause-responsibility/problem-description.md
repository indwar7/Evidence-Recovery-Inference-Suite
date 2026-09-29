# Attributing Assistant Behavior to System Prompt Clauses

## Overview

An assistant is behaving oddly. Maybe it hedges on things it should just
answer, or refuses requests it should handle, or writes three paragraphs
when a sentence would do. Its system prompt is not one instruction, it is
several stacked together - a persona line, a scope restriction, a tone
rule, a fallback rule for uncertain cases, and so on. Which one of those
instructions is actually causing the odd behavior, and which ones are just
sitting there, uninvolved?

This dataset answers that question the only reliable way: by actually
removing each instruction, one at a time, from a real system prompt, and
running a real language model against real user queries to see what
changes. The result is real measured behavior, not a guess about which
clause "sounds like" it would matter.

Your task is to recover, from the released measurements alone, which
clause was responsible.

## Task

Each row shows one real system prompt (drawn from a public collection of
assistant prompts, then split into its individual instructions), together
with the numeric behavior of the model's actual answers when the FULL
prompt was in effect. You submit two things:

- ranking: the prompt's own clauses, ordered from most to least
  responsible for that behavior.
- flags: for every clause, whether it was necessary (removing it alone
  caused a real, sizeable shift in behavior) or merely correlated (it was
  present, but removing it alone did not meaningfully change anything).

The clauses shown are the SAME ones the prompt is actually built from -
there is no hidden edit to guess, no vocabulary of possible operations to
identify. What is hidden is which one of them is doing the causal work,
and that is only discoverable by comparing the full-prompt behavior
against what happens when each clause is individually taken away - the
released measurement here comes from that comparison, done for real, on
GPU, once per clause per prompt.

## What makes this hard

**Position is not a reliable shortcut.** System prompts conventionally put
a persona line first and a formatting or fallback rule last, so it might
seem like responsibility tracks position. It measurably does not: the
order in which clauses are shown to you has been reshuffled independently
for every row, breaking any link between a clause's position in the
original prompt and its position in what you see.

**The same prompt appears more than once.** Every prompt's full-prompt
behavior is measured over several different probe questions, and a
prompt's clauses are shown across MULTIPLE rows, each built from a
different subset of those probe questions. Which clause turns out to be
most responsible can genuinely differ from row to row for the SAME
prompt, because different probe questions expose different instructions -
a persona rule might dominate on a casual query and barely register on a
technical one. Learning to read the released behavior measurement, rather
than memorizing "this prompt's answer is always clause 3," is what
generalizes.

**Necessary and correlated are not the same as first and second.** A
clause can rank second in the ordering while still clearing the
"necessary" bar, and a clause can rank highest among the shown candidates
while still falling under it, if none of the clauses in that particular
row produced a large individual shift. The ranking is a relative ordering
among a prompt's own clauses; the flag is a threshold judgment, and the
two do not have to move together.

Measured on this data with the official grader (mean of the ranking
credit and the flag-match fraction; see Metric below):

- Always guess the first clause shown, all clauses "correlated": TBD
- A random valid ranking, random flags: TBD
- Guess based on clause position in the prompt text alone (ignoring the
  released behavior measurement): TBD
- A trained classifier using the released behavior measurements: TBD

(Anchor numbers above are filled in from the official grader once the
full GPU build completes - see STATE.md's FINAL LOCKED NUMBERS section
for the values used to build this dataset.)

## Data files

All public files are in ./dataset/public/

**train.csv** - one row per item.

- item_id (str): Unique row id.
- topic (str): Which training topic group this prompt belongs to (a
  domain cluster like coding_dev, writing_editing, health_wellness - see
  dataset-description.md).
- prompt_id (str): Which source system prompt this item was built from.
  The same prompt_id can appear on multiple rows (see What makes this
  hard).
- clause_ids (str): Pipe-delimited clause identifiers, in this row's
  RANDOMIZED display order - e.g. "p12__c4|p12__c1|p12__c3".
- clause_texts (str): Pipe-delimited clause text, in the SAME order as
  clause_ids.
- fp__\<field\> (float or int): Behavior measurements of the model's
  actual answers with the FULL prompt in effect, averaged over this row's
  probe-question subset - see dataset-description.md for the full field
  list.
- ranking (str): Target. clause_ids reordered from most- to
  least-responsible.
- top_clause_id (str): Target. The single most responsible clause (also
  recoverable as ranking's first entry, provided as a convenience).
- flags (str): Target. Pipe-delimited clause_id:flag pairs, e.g.
  "p12__c1:necessary|p12__c3:correlated|p12__c4:correlated".

**test.csv** - same columns without ranking, top_clause_id, flags.

**sample_submission.csv** - the required format, filled with the
unranked, all-correlated baseline.

## Metric

**Combined ranking + flag score**: for each test row, average a ranking
component (positional credit for where the true top clause landed in your
submitted ranking) with a flag component (fraction of clauses whose
necessary/correlated flag you got right), 50/50, then average over all
test rows.

```python
def ranking_credit(n_clauses, true_top_position_in_your_ranking):
    if n_clauses <= 1:
        return 1.0
    return 1.0 - true_top_position_in_your_ranking / (n_clauses - 1)

def flag_credit(true_flags, your_flags):
    correct = sum(1 for cid, flag in true_flags.items()
                  if your_flags.get(cid) == flag)
    return correct / len(true_flags)

item_score = 0.5 * ranking_credit(...) + 0.5 * flag_credit(...)
```

Range: 0.0 to 1.0, higher is better. A ranking that is not a valid
permutation of that row's own clause_ids scores 0 for the ranking
component; a flag value outside {necessary, correlated} scores 0 for that
clause, rather than voiding the row.

## Submission format

Write a CSV named submission.csv to ./working/ with exactly three
columns:

```
item_id,ranking,flags
I00001,p12_c1|p12_c4|p12_c3,p12_c1:necessary|p12_c3:correlated|p12_c4:correlated
I00002,p07_c2|p07_c1|p07_c3|p07_c4,p07_c1:correlated|p07_c2:necessary|p07_c3:correlated|p07_c4:correlated
...
```

- One row for each item_id in test.csv, in any order.
- ranking must be a pipe-delimited permutation of that row's own
  clause_ids from test.csv - not a fixed vocabulary, since every prompt
  has its own clause set.
- flags must cover every clause_id for that row, each mapped to exactly
  one of necessary or correlated.
- An item_id you omit scores 0 for that row rather than voiding the
  submission; duplicated ids keep their first row and unknown ids are
  ignored. Only an unreadable file or missing required columns are
  unscorable.

## Rules & environment

- Learn from the provided data only. No pretrained language models, no
  LLMs at solve time, and no external data of any kind - the released
  behavior measurements are the entire feature set.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data,
  fit whatever you fit, then write ./working/submission.csv
- Seed everything; your run should be reproducible.
