# Predicting Retrieval Sufficiency in RAG Pipelines

## Overview

A retrieval-augmented question-answering system is shown a query, a topic's
knowledge base of short policy passages, and the top few passages that a
real dense retriever actually returned for that query, ranked by real
cosine similarity. A real instruction-tuned language model then answers the
query using only those retrieved passages, and its actual answer is reduced
to a small set of numeric behaviour measurements (whether it hedges,
whether it says the context does not contain the answer, whether it cites
the context explicitly, and so on) - never the response text itself.

Retrieval genuinely succeeds on some queries and genuinely fails on others,
in three distinct, deliberately constructed ways: the answering passage is
present but ranked too low to make the shown window, the answering passage
does not exist anywhere in the topic's knowledge base at all, or the query
is worded generically enough that it plausibly matches content from a
different topic entirely.

Your task is to submit a structured diagnosis of what the retrieval system
should do next, not a final answer and not a single label.

## Task

Each row shows a query, the retrieved passage ids and their real similarity
scores, how many passages were shown, and the behaviour measurements of the
model's actual answer over that retrieved context. You submit three
jointly-scored fields:

- diagnosis: was the retrieval shown to the model sufficient to answer the
  query, or insufficient?
- action: given that diagnosis, what should the system do -
  answer_as_is, reformulate the query and retrieve again, or abstain
  entirely?
- reason_code: the specific reason behind the diagnosis - sufficient, the
  answering passage was retrieved but ranked too low to appear in the shown
  window (gold_low_rank), the answering passage does not exist in this
  topic's knowledge base (gold_absent), or the query is ambiguous enough to
  plausibly match a different topic (query_ambiguous).

These three fields are not independent: a valid trace never pairs
reason_code=gold_absent with action=answer_as_is, for example. Getting the
diagnosis right without the correct reason, or the reason right with an
inconsistent action, is only partial credit.

The retrieved passage ids and similarity scores are an input feature, never
the target - this task is not asking you to retrieve or to re-rank
anything. The retrieval has already happened; your job is to diagnose
whether it worked.

The split is by **topic**, not by row. Several topics - out of the full
set - are held out entirely for testing. Every measurement you train on
comes from a different set of topics than the ones you are tested on, so
nothing about a specific topic's own passages or phrasing can be
memorized and looked up.

## What makes this hard

**The reason behind an insufficient retrieval is not fully recoverable from
similarity scores alone.** Two of the three insufficient reasons -
gold_low_rank and query_ambiguous - can produce similar-looking similarity
score patterns: both involve a query whose top similarity score to its own
topic is not much higher than its similarity to something else. Telling
them apart reliably means reading the released behaviour measurements of
the model's actual answer, not just thresholding on the retrieval scores
themselves.

**The same query appears at multiple retrieval depths.** For queries whose
answering passage sits at a middling rank, the dataset shows that query at
several different numbers of retrieved passages - at a shallow depth the
retrieval is genuinely insufficient, at a deeper one it becomes genuinely
sufficient, because the answering passage enters the shown window. The
correct diagnosis for the same query text can differ from row to row
depending on how many passages were actually shown, so the row's k
(how many passages were retrieved) is part of what must be reasoned about,
not just the query in isolation.

**Sufficient retrieval is the majority case, but not by an overwhelming
margin.** A trivial always-guess-sufficient baseline scores well above
zero but well short of a skilled solution, because a real fraction of
items are deliberately constructed to fail retrieval in one of the three
ways above.

Measured on this data with the official grader (mean joint-field accuracy
across diagnosis, action and reason_code; see Metric below):

- Always predict "sufficient / answer_as_is / sufficient": TBD
- A random valid trace per row: TBD
- Threshold only on the released retrieval similarity scores (ignoring the
  generation behaviour measurements entirely): TBD
- A trained classifier using both retrieval scores and generation behaviour
  measurements: TBD

(Baseline numbers above are measured directly on the shipped dataset once
the raw generation run completes - see dataset-description.md "Measured
Difficulty" for the final locked numbers.)

## Data files

All public files are in ./dataset/public/

**train.csv** - one row per item.

- item_id (str): Unique row id.
- topic_id (str): Which training topic this item's query and passages come
  from.
- query_text (str): The user's question.
- k (int): How many top-ranked passages were retrieved and shown for this
  item.
- retrieved_id_1..retrieved_id_k (str): The ids of the passages actually
  retrieved, in descending similarity order. Columns beyond this row's own
  k are empty.
- retrieved_score_1..retrieved_score_k (float): The real cosine similarity
  score for each retrieved passage. Columns beyond this row's own k are
  empty.
- fp__\<field\> (float or int): Behaviour measurements of the model's
  actual answer over this item's retrieved context - see
  dataset-description.md for the full field list.
- diagnosis, action, reason_code (str): Target. The joint decision trace
  for this item.

**test.csv** - same columns without diagnosis, action, reason_code.

**sample_submission.csv** - the required format, filled with the
always-sufficient baseline.

## Metric

**Joint per-item field accuracy**: for each test row, average whether
diagnosis, action, and reason_code each exactly match the true trace
(1 point each, divided by 3), then average that per-item score over every
test row.

```python
FIELDS = ["diagnosis", "action", "reason_code"]

def item_score(pred_row, true_row):
    return sum(
        1 for f in FIELDS if pred_row[f].strip().lower() == true_row[f].strip().lower()
    ) / len(FIELDS)
```

Range: 0.0 to 1.0, higher is better. A missing, malformed, or
out-of-vocabulary field value scores 0 for that field rather than voiding
the row.

## Submission format

Write a CSV named submission.csv to ./working/ with exactly four columns:

```
item_id,diagnosis,action,reason_code
I00001,sufficient,answer_as_is,sufficient
I00002,insufficient,reformulate,gold_low_rank
...
```

- One row for each item_id in test.csv, in any order.
- diagnosis must be one of: sufficient, insufficient.
- action must be one of: answer_as_is, reformulate, abstain.
- reason_code must be one of: sufficient, gold_low_rank, gold_absent,
  query_ambiguous.
- An item_id you omit scores 0 for that row rather than voiding the
  submission; duplicated ids keep their first row and unknown ids are
  ignored. Only an unreadable file or missing required columns are
  unscorable.

## Rules & environment

- Learn from the provided data only. No pretrained language models, no
  LLMs at solve time, and no external data of any kind - the released
  retrieval scores and behaviour measurements are the entire feature set.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data,
  fit whatever you fit, then write ./working/submission.csv
- Seed everything; your run should be reproducible.
