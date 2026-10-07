# BudgetIndex: Choosing What a Search Index Can Afford to Hold

## Overview

A search system cannot index everything it stores. Embedding each passage costs money, every stored vector costs memory, and a larger index makes every query slower. Real systems therefore index only part of their content and give up on the rest.

**This challenge makes that choice the task.** For each document you are told how many passages you may index — usually two or three, never more than four. You choose which sentence ranges to index. You are then scored on how many questions from an unseen question set your choice makes answerable.

You do **not** answer the questions. You do **not** retrieve or rank anything. You submit one line per document listing the sentence ranges you chose to index, and nothing else.

Two things make the choice hard. The budget is far too small to index everything useful. And each document contains **deliberately planted false sentences** that are word-for-word identical to the true ones apart from the number they state — so if a range you index swallows a false sentence, every question it would otherwise have answered is lost.

A concrete example. A document has 34 sentences and allows 2 ranges. Sentences 11 and 12 both look like ordinary statements of the same country's population, but only one gives the real figure; the other is a planted twin. Sentences 20 to 22 hold three facts one range could cover at once. Indexing `11-12` scores nothing for those two questions, because the range contains the twin. Indexing `12-13` and `20-23` answers four questions with two ranges.

## What actually has to be figured out

To know which ranges are worth indexing, you have to work out which sentence answers which question. That is deliberately **not** a pattern-matching exercise.

**A question and the sentence that answers it describe the attribute in different words.** A question may ask for the "number of residents" while the sentence that answers it records a "resident headcount". There is no shared wording to match on: across the workload, **88.6%** of questions use a different wording from the sentence that answers them. The corpus contains **40 distinct attribute wordings** across four underlying attributes.

**Some wordings never appear in training at all.** 14 of the 40 are held out of the training split entirely, and they show up in **472 of the 799** evaluation questions. A lookup table harvested from the labelled data therefore cannot cover the evaluation set — a solution has to generalise across wordings rather than enumerate them.

**Sentence forms vary too.** The training split alone contains 103 distinct question shapes and over a thousand distinct fact-sentence shapes, so no finite set of templates covers the corpus.

This is where solutions separate. Matching by shared words, harvesting a lookup from training, or reasoning about what the wordings *mean* are all viable approaches, and they score very differently — see the measured anchors below.

## Task

You are given 228 documents. Each is a numbered list of sentences, each carrying a token count, plus a per-document limit `budget_k`.

For each document you submit at most `budget_k` **windows**. A window is a contiguous run of sentences. Three hard rules:

- no window may exceed **60 tokens**,
- windows within a document may **not overlap**,
- at most **`budget_k`** windows per document.

Any sentence not inside a window is not indexed and can answer nothing.

A question counts as **answered** when one single window of its document contains **all** of that question's evidence sentences and **none** of its planted false sentences. Two windows that jointly cover the evidence do not count; it must be one window.

`budget_k` ranges from 2 to 4 and averages 2.15, against an average of 3.50 hidden questions per document. The budget binds: you cannot cover everything, and choosing what to abandon is part of the task.

## Dataset

Five files are released, plus two scripts shipped for reproducibility.

**`documents.csv`** — the corpus. 228 documents, 7,884 sentences, averaging 34.6 sentences per document.

- `doc_id` (string) — document identifier.
- `sent_id` (integer) — 0-based position of the sentence within its document.
- `n_tokens` (integer) — whitespace token count. The 60-token window cap is measured in these units. Mean 10.77, maximum 18.
- `text` (string) — the sentence itself.
- `budget_k` (integer) — how many windows this document may receive. Constant within a document; ranges 2 to 4.

**`queries_train.csv`** — 599 labeled example questions, with the labels revealed so you can develop against them.

- `query_id` (string) — question identifier.
- `doc_id` (string) — the document this question is about.
- `question` (string) — the natural-language question.
- `qtype` (string) — how the question names its country: `direct`, `aliased`, `relational`, or `chained`.
- `cls` (string) — how the evidence is laid out: `easy`, `tight`, `cluster`, `conflict`, or `impossible`.
- `required_sent_ids` (string) — pipe-separated `sent_id`s that must all land in one window, for example `12|15`.
- `poison_sent_ids` (string) — pipe-separated `sent_id`s that must not land in that window.

**`queries_test.csv`** — the 799 questions you are scored on, with labels withheld.

- `query_id` (string) — question identifier.
- `doc_id` (string) — the document this question is about.
- `question` (string) — the natural-language question.

**`sample_submission.csv`** — a valid submission to copy the format from and score against. It indexes the first `budget_k` short windows of every document and scores 0.1164.

- `doc_id` (string) — document identifier, one row per document.
- `windows` (string) — pipe-separated window list, in the syntax below.

**`real_facts.json`** — the reference table needed to tell a true sentence from its planted twin. A JSON array of 214 country records.

- `name` (string) — country name, as it appears in questions and sentences.
- `cca3` (string) — ISO 3166-1 alpha-3 code, used by `aliased` and `chained` questions.
- `capital` (string) — capital city, used by `relational` and `chained` questions.
- `region` (string) — continental region. Used by no question or sentence; present only for completeness.
- `population` (integer) — an attribute a question may ask for.
- `area` (integer) — land area in square kilometers.
- `num_borders` (integer) — count of bordering countries.
- `num_languages` (integer) — count of official languages.

**`generate.py`** (Python script) — the deterministic corpus generator. Builds the documents and questions from a fixed seed, measures every score anchor, and refuses to emit a corpus that fails its built-in checks.

**`prepare.py`** (Python script) — splits the generated workload into the public training and evaluation files, withholds the evaluation labels, and sizes each document's `budget_k`.

**Two different counts that are easy to confuse.** The corpus has **228 documents**; the reference table separately has **214 country records**. Of those 214, **142** are used as subjects of questions and **72** exist only to supply plausible false values, so only 142 distinct countries are ever asked about.

## Window Syntax

A window is a half-open sentence range written `start-end`, so `7-10` covers sentences 7, 8, and 9. A document's windows are joined with `|`, for example `0-3|7-10`.

Order does not matter; the grader sorts windows before scoring. Each document's list must satisfy `0 <= start < end <= n_sentences`, hold at most `budget_k` windows, keep them mutually non-overlapping, and keep each one at or under 60 tokens. An empty cell is legal and indexes nothing.

## Submission Format

Submit `submission.csv` with a header row and exactly one row per `doc_id` in `documents.csv` — 228 data rows, 229 lines in total — and exactly two columns:

```
doc_id,windows
doc_0001,0-3|7-10
doc_0002,4-9
doc_0003,
```

- `doc_id` (string) — must match a `doc_id` in `documents.csv`.
- `windows` (string) — the window list in the syntax above, or empty.

A document whose windows are malformed, out of range, overlapping, over budget, or over the token cap scores 0 for that document's questions only; the rest of the submission is still graded. A submission is rejected outright only if it cannot be scored at all: missing columns, or duplicate `doc_id` values.

## Evaluation

```
answerability = answered_evaluation_questions / 799
```

A question is answered when some single submitted window of its document holds all of its evidence sentences and none of its planted false sentences, within the budget, overlap, and token-cap rules. The score runs from 0 to 1 and **higher is better**.

**A certified maximum of 0.7021.** Documents are independent, so the best possible submission can be computed exactly rather than estimated. A budgeted dynamic program does this at build time and reaches 0.7021. No submission can score higher, because some questions are deliberately placed where no legal window can reach them. The program that proves this ships with the dataset.

## Measured anchors

Each figure below comes from running that approach against the hidden labels on this exact dataset. The middle group differs **only** in how a question is matched to the sentence that answers it — everything else, including the window allocation, is held identical. That is the spread the challenge is built to produce.

- `sample_submission.csv`, which ignores the questions entirely — **0.1164**
- the best fixed-size tiling that ignores the questions — **0.1389**
- matching a question to a sentence by shared words — **0.4706**
- matching with a wording lookup harvested from the training split — **0.4105**
- mapping wordings to attributes by meaning, but never separating a true sentence from its twin — **0.3517**
- mapping wordings by meaning **and** reconciling values against the reference table, the reference solution — **0.6696**
- certified ceiling — **0.7021**

Two things are worth reading off this list. Shared-word matching and a training-harvested lookup both land near 0.41–0.47, and neither is close to the ceiling; the gap to 0.6696 is the value of generalising across wordings and then checking the values. And a solution that generalises across wordings but skips the reference table scores **0.3517**, worse than a cruder matcher that does check values — the two capabilities are not interchangeable.

Separately, holding evidence-finding perfect and varying only how the budget is spent: opening one minimal window per question scores **0.5770**, against **0.7021** for a solver that merges nearby evidence and concedes what it cannot reach. That is a different experiment from the list above and its number should not be added to those.

The reference solution locates evidence for 759 of the 799 evaluation questions, so even it does not reach the ceiling.

## Why guessing does not work

The planted twin of a true sentence uses the same country, the same attribute, the same sentence template, the same wording and the same number format. Once the number is masked, the two are character-for-character identical, and they sit near each other in random order. Three ways of guessing which is true are closed by construction — the generator checks each before writing anything and aborts the build if any fails.

- *Counting how often a value appears.* Each false value is fixed per country-and-attribute pair, and every true sentence is matched by exactly one false sentence for that pair, so occurrence counts are an exact tie, always.
- *Checking whether a value exists in the reference table.* False values come from the 72 countries that appear in `real_facts.json` but never appear as a question subject, so both candidates are genuine entries.
- *Guessing by position or size.* Measured on the shipped data: the false sentence follows the true one 49.6% of the time, the true value is the larger number 51.0% of the time, and the two numbers have equal digit length 98.4% of the time.

The generator additionally checks that every true/false pair stays separable from the released files, so no question is an unavoidable coin flip. 39.4% of true fact sentences write their value in a form other than plain digits; the training split reveals the evidence and false sentences of every labeled example, so the range of forms is recoverable by inspecting the released data.

**The budget forces real choices.** Some documents pack several facts together so one window answers them all; some interleave two questions so no legal set of windows answers both; a few place evidence beyond any window's reach. The evaluation set contains `easy` 407, `cluster` 194, `tight` 142, `conflict` 36, `impossible` 20. Question forms are `direct` 521, `chained` 99, `relational` 98, `aliased` 81. Both labels are given on the training split.

## How this differs from nearby tasks

- **Chunking and text splitting** cut a document so every token lands somewhere, and are judged by what happens downstream. Here you choose what to index at all, most of a document stays uncovered, and the choice itself is the graded answer.
- **Retrieval and ranking** produce an ordering per query. Here you produce no ordering and commit before any query is seen.
- **Supporting-fact question answering** hands you a context and asks which sentences justify an answer. Here you build the context, and the distractor differs from the truth in one number only.
- **Fact verification** outputs a judgment about a claim. Here the same reasoning is an input to a structural choice, not the output.

## Data Provenance

Every graded value is real. Each subject is a real country, each attribute value is that country's real figure, and each planted false value is a different real country's real figure for the same attribute. Only the connective filler prose between facts is synthetic, and no question tests it.

The reference table is built by `build_kb.py` from two public endpoints that need no API key:

- `name`, `cca3`, `capital`, `region`, `area`, `num_borders`, `num_languages` — from `mledoze/countries`, pinned to tag v4.0.0, licence ODbL-1.0:

      https://raw.githubusercontent.com/mledoze/countries/v4.0.0/countries.json

- `population` — from the World Bank World Development Indicators, indicator `SP.POP.TOTL`, reference year 2024, licence CC BY 4.0:

      https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&date=2024

A SHA-256 digest of each downloaded input ships in `real_facts.provenance.json`. Of 250 upstream rows, the 214 with a capital, a positive land area, at least one official language, and a World Bank population are kept; nothing is filled in or guessed. The derived dataset is ODbL-1.0 (share-alike); World Bank values are CC BY 4.0.

## Multi-Run Reporting

The reference solution is deterministic and produces an identical submission every run. Stochastic systems must be evaluated over at least five independent runs. The reported score is the arithmetic mean across runs, never the best run, and reports must include the sample standard deviation, minimum, maximum, and number of runs. Runs must use independently chosen seeds with the dataset, metric, and configuration held fixed.

## Not Allowed Methods

- Matching test text against online copies, cached source records, source identifiers, or answer keys.
- Internet access, remote APIs, hosted search services, or external corpora during prediction.
- Manual labeling or per-example inspection of the test set in order to write answers.
- Reading grader internals, answer files, or any file not provided to participants.
- Hard-coding document identifiers, window lists, or document-to-window lookup tables.
- Reconstructing withheld labels from anything other than the released public files.
- Writing any window list by hand, or having a model write one per document outside the submitted pipeline.

**On models, including LLMs.** Pretrained models are allowed, and that includes language models run locally. They may be used as components of a solution — to parse questions, map wordings to attributes, normalise values, rank candidate windows, or anything else. What is not allowed is bypassing the pipeline: every window in `submission.csv` must be produced by the notebook you submit, running end to end, deterministically and seeded, from the released files alone. A model may inform the selection; it may not stand in for a reproducible pipeline, and no answer may be authored by hand or carried in from a separate session.

You may train on `queries_train.csv` and build local lexical, dense, or hybrid indexes using only the supplied files.

## Environment

Solutions run CPU-only on 10 cores with 62 GB of RAM and a hard limit of 1.5 hours end to end. Only libraries in the Kaggle Python Docker image may be used. The reference solution finishes in seconds using pandas and the standard library. Your notebook must run start to finish, deterministically and seeded.
