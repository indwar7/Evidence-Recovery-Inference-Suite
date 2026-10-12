# Matching Abstracts by Shared Research Topic, Not by Name

A biologist reading an abstract about coral reef bleaching and one about
deep-sea hydrothermal vents can tell they belong to different subfields of
marine biology without ever being told the subfield's name — the shared
vocabulary, methods, and framing are enough. This challenge asks a model to
do the same: decide which of six candidate research areas a paper's abstract
belongs to, using only one example abstract from each area as the
description of what that area is. No area is ever named.

## Task

Each test row gives you one query paper's abstract, plus six
candidate groups labeled only by position (`candidate_0` through
`candidate_5`). Each candidate group is represented by one other real
paper's abstract — never by a topic name, field name, or any descriptive
label. Exactly one candidate shares the query paper's research topic; the
other five are drawn from different, but deliberately similar, topics. You
must output which slot (0-5) is the correct one.

## What you are actually being asked

This is not topic classification into a fixed vocabulary: the six slots are
reshuffled per row, drawn from a pool of over a thousand possible topics,
and the only thing identifying a slot is the one example abstract sitting
in it. The submitted artifact is a slot index that is meaningless outside
its own row — the same way a correct answer to "which of these six boxes
has the matching sock" doesn't generalize to a sock catalog. This also is
not information extraction (nothing is being pulled out of the query text
that is independently verifiable) and it is not retrieval against a fixed
corpus (the candidate pool itself changes composition every row). It is
closest to exemplar-based similarity judgment: given one example of a
category and one query item, decide whether the query belongs to that
category more than to any other candidate category.

**Why the distractors are hard.** The five wrong candidates per row are
not drawn at random and not merely from the same broad field as the true
topic. They are the true topic's nearest neighbours by embedding
similarity among more than a thousand topics, so every distractor is a
topic that is genuinely close to — and therefore easy to confuse with —
the query's real topic. A model that only separates obviously unrelated
fields will not do well here, because the test set contains no easy rows
of that kind by construction.

## Dataset

- `train.csv` — one row per query paper from a set of topics that is
  disjoint from the test topics. Columns: `work_id`, `query_abstract`, `candidate_0_exemplar` .. `candidate_5_exemplar`, and
  `slot` (the correct candidate index, 0-5). The six-candidate structure
  is identical to the test set, so this is a direct training signal.
- `test.csv` — same columns minus `slot`, for query papers from topics
  never seen in `train.csv`.
- `sample_submission.csv` — `work_id`, `slot` (template, all zeros).

Every topic is entirely in train or entirely in test. Within a row, no
candidate's exemplar text is ever the query's own abstract, no exemplar
text is any other row's query abstract, and the true exemplar of each row
is unique to that row. Rows are shuffled so that no two consecutive rows
share a topic.

## Rules

**Use only the data directory.** The papers in this dataset are real and
publicly indexed. Looking a query paper up by its title or abstract to
retrieve its catalogued subject is not solving the task — a submission
built that way is invalid regardless of its score, and so is any
submission that depends on data not shipped in the data directory. The
solver environment has no network access; your solution must run to
completion from the data directory alone.

**Do not use any pretrained classifier or lookup table trained on a
scholarly subject taxonomy** (for example, a model fine-tuned to predict
fields of study from abstracts). The task is designed to be solved from the
shipped abstracts' content alone; a model that has already memorized the
taxonomy this dataset draws from is not demonstrating the similarity
judgment being measured.

**Treat every test row on its own.** Each row's prediction must come from
that row's own query and six exemplars plus whatever was learned from the
training data. Comparing test rows with each other — counting how often a
text appears across rows, grouping rows, or using one row's answer to decide
another's — is not allowed.

- Learn from the provided data only, beyond a general-purpose pretrained
  text encoder (e.g. a sentence-embedding model trained on generic web or
  book text). No LLM calls at inference time, and no LLM-generated labels
  anywhere in your pipeline.
- Your notebook must run end to end, top to bottom, within the compute
  budget below.
- Seed everything; your run should be reproducible.

## Evaluation

**Score = number of test rows whose submitted `slot` equals the true slot, divided by the number of test rows.**

Plain accuracy over the six-way slot choice. Each row has exactly six
equally likely slots, so chance is 1/6 for every row and no
chance-correction term is needed.

A ranking-based metric (e.g. mean reciprocal rank) was considered and
rejected: the slots carry no ordering or relatedness to one another, so
"almost picking the right slot" is not a meaningful notion here — the
model either judges topical membership correctly or it does not.

## Submission

Submit `submission.csv` with exactly two columns: `work_id`, `slot`
(integer 0-5), one row per `work_id` in `test.csv`. A missing `work_id`
scores 0 for that row. A duplicated `work_id` has all its rows discarded
and scores 0. An out-of-range or non-integer `slot` scores 0 for that row.
The submission is only rejected outright if the `slot` column is missing.

## Compute

1x NVIDIA A10G, 1.5 hour wall-clock limit.
