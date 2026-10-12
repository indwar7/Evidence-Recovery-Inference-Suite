# Design record — Matching Abstracts by Shared Research Topic, Not by Name

## Task in one line

Given a query paper's abstract and six candidate groups (each shown only as
one other paper's abstract, never as a label), pick which group shares the
query's real OpenAlex-assigned topic.

## Why this shape

Candidate ideas killed before this one, and why:

- **OpenAlex citation-order reconstruction** (recover the true citation
  order within a paper's reference list from shuffled references). Killed
  at the measurement stage: `referenced_works` in the OpenAlex API is
  sorted by internal numeric work ID, not by any citation-order or
  appearance-order signal — confirmed by checking five sampled works and
  finding every `referenced_works` list already sorted ascending by ID.
  No real order exists to recover; building on this field would have
  reproduced the Crossref retraction-lag dead end already logged in the
  rejection register (section on abandoned sources).
- **Stack Exchange data dumps.** Ruled out before fetching: as of July
  2024 Stack Exchange restricted public dump access and the historical
  archive.org mirror is stale; reachability and reproducibility risk was
  judged too high for a 1.5h-runtime, no-network solver environment and a
  deterministic `prepare.py`.
- **Plain topic/field classification** (predict `primary_topic` from the
  abstract directly, from a fixed published vocabulary). Hard-rejected by
  design before any data was touched — reduces to single-label
  classification into a fixed vocabulary, explicitly disallowed.

## The label-leak finding (measured, not assumed)

The first viable design was: show the query abstract plus 8 candidate
**topic path strings** (domain > field > subfield > topic display name,
drawn from OpenAlex's own taxonomy), ask the solver to pick the correct
one.

Measured against a 400-row sample (`fetch_sample.py` / `sample.json`):

- A trivial keyword-overlap baseline — tokenize the abstract, tokenize each
  candidate's full topic-path string, count shared non-stopword tokens,
  pick the highest-overlap candidate — scored **0.632** accuracy on an
  8-way choice (chance = 0.125). Mean margin between the true candidate's
  overlap and the best wrong candidate's overlap was +0.73 tokens, i.e. the
  true label reliably shared more vocabulary with the abstract than any
  distractor.
- Restricting distractors to the **same field** as the true topic (a
  harder condition, since lexical overlap with the field itself is now
  shared across all 8 candidates) only brought the baseline down to
  **0.578** — still 4.6x chance.

Diagnosis: OpenAlex topic names are themselves descriptive noun phrases
("Meta-analysis and systematic reviews," "Magnetic confinement fusion
research") that are written to be human-readable summaries of the topic's
content — so they directly echo the abstract's own vocabulary by
construction. This is a structural property of the label field, not a
distractor-pool artifact, and is not fixable by harder negative sampling.
This matches the rejection register's section 2.6 standard (shortcut
opportunities must be measured, not argued) and would have failed exactly
the way the FDA mechanism-of-action and budget-index challenges nearly did.

**Fix applied**: never show any topic/field/domain name anywhere in
`dataset/public/`. Represent each candidate only by two *other* real
papers' abstracts that share that candidate's true topic. A solver must
judge "does the query read like these two examples" — topical similarity
between documents — rather than matching a descriptive label string
against the query's vocabulary.

**Re-measurement after the fix** (keyword overlap now computed between the
query abstract and the candidates' *exemplar abstracts*, not label
strings), on the same class of sample: 0.59 accuracy on a 6-way choice
(chance = 0.167), roughly 3.5x chance. This is NOT treated as a remaining
leak — unlike the label-string case, this reflects genuine topical
vocabulary overlap between documents that legitimately share a topic,
which is the actual signal a trained encoder is also meant to exploit, just
less precisely than a model that captures semantic similarity beyond raw
word overlap. The design is kept; the trained reference (sentence-embedding
cosine similarity) is measured against this keyword-overlap baseline as the
adversarial floor, per the standard methodology, with the required margin
established before shipping (see ladder below once the full-scale run
completes).

## Source

OpenAlex (CC0 1.0), verified from the primary documentation source —
`https://github.com/ourresearch/openalex-docs/blob/main/license.md`
(fetched 2026-10-09): "OpenAlex data is made available under the CC0
license. That means it's in the public domain, and free to use in any way
you like." This is distinct from the per-work open-access `license` field
OpenAlex also exposes (which varies per paper and is not used here) — the
license verified and declared is for OpenAlex's own metadata and topic
classification, which is what this dataset actually redistributes.

Source URL never used in this portfolio before (checked against
`references/sources.md`: Wikimedia, PLOS, Chronicling America, HuggingFace
model cards, CourtListener, FDA, eCFR, live Wikipedia API — no overlap with
`api.openalex.org`).

## Split

Group-disjoint by `topic_id`: every topic is entirely in train or entirely
in test. Required because a test query's candidate exemplars must never
include anything visible during training for that same topic, and because
splitting by row (rather than by topic) would let a solver see other
members of the test query's true topic during training — a direct
train/test group overlap of exactly the kind the rejection register's
section 4 warns about ("split must be disjoint by group, never by row").

Distractor topics for a query are sampled preferentially from the **same
domain** as the query's true topic (confirmed harder than cross-domain
distractors by the re-measured keyword-overlap baseline above), falling
back to other domains only when the same-domain pool is too small.

## Transductive-leakage check (rejection register section 2A)

Each test row's prediction depends only on: the query abstract, and the
exemplar abstracts inside that row's own six candidate slots. No row's
answer requires or benefits from looking at any other test row — exemplars
are sampled independently per query (with replacement across rows, since
exemplars are drawn from the training-visible pool for that topic, not
from other test rows), so there is no cross-row intersection or
corpus-consensus shortcut available. Verified: the candidate-construction
code (`prepare.py: _build_candidates`) only ever reads from
`topic_to_items`, built once from the full raw file, and explicitly
excludes the query's own `work_id` — it never references another test
row's identity or content.

## Difficulty tuning (measured, three iterations)

The first shipped design (2 exemplars/candidate, same-domain-preferred
random distractors) was measured, not assumed, to be far too easy:

| # | Exemplars/candidate | Distractor selection | Adversary | Reference (MiniLM) |
|---|---|---|---|---|
| 1 | 2 | same-domain-preferred, random | 0.550 | 0.798 |
| 2 | 1 | same-domain-preferred, random | 0.456 | 0.747 |
| 3 | 1 | hard-neighbor (embedding rank 2-30) | 0.345 | 0.521 |

Row 1->2 (cutting exemplar count in half) barely moved the reference,
confirming the distractor-selection method — not the exemplar count — was
the dominant source of the leak. Row 2->3 (switching to embedding-nearest-
neighbor distractors, computed by `dataset/generator/build_hard_distractors.py`
over topic-centroid embeddings) is what actually closed the gap: distractor
topics are now the true topic's own nearest neighbors (excluding the
closest 2 ranks, to avoid rows whose "distractor" is arguably within the
true topic's own boundary), so a solver must actually separate topically
adjacent research areas rather than lean on coarse domain membership.

Final measured ladder (3,979 train / 866 test rows):

- Chance (fixed slot): 0.1628
- Keyword-overlap adversary (bag-of-words, no model): 0.3453
- Trained reference (pretrained MiniLM, cosine similarity, no fine-tuning): 0.5208
- Oracle ceiling: 1.0000 (by construction)
- Margin (reference - adversary): 0.1755

Reference sits at 52% of the oracle ceiling with real headroom remaining,
is comfortably inside the 0.2-0.6 target band, and the margin — while
0.0045 under the 0.18 guideline — was treated as close enough to stop
tuning rather than risk overshooting into "too hard" (register section
3.6), per the explicit instruction to measure rather than keep iterating
blindly.

## A real data-quality bug caught during scale-up

At full scale (7,961 raw rows), two defects surfaced that were invisible
on the small feasibility sample:

1. **Duplicate `work_id`s.** The raw fetch originally filtered on
   `topics.id:{tid}` (any of a work's topics), not `primary_topic.id:{tid}`
   — so a work with multiple topic tags got fetched once per matching
   topic query, producing the same work under the same (correct)
   `primary_topic`, but as physically duplicate rows. `prepare.py`'s
   `_load_raw` now deduplicates by `work_id` before any split logic runs,
   and the generator itself was fixed to filter on `primary_topic.id`.
2. **Boilerplate abstracts.** A handful of distinct works shared verbatim
   journal front-matter text as their recorded "abstract" (e.g. a
   journal's own self-description, reused across every article in that
   issue) — carrying no real per-work topical signal. `_load_raw` now
   drops every row whose abstract text is shared by more than one distinct
   `work_id`.
3. **A redundant shipped column.** After cutting exemplars-per-candidate
   to 1, `candidate_N_exemplar_2` was always byte-identical to
   `candidate_N_exemplar_1` for every row — caught by an explicit equality
   check before packaging, not assumed; the column was removed and all
   docs updated to the single `candidate_N_exemplar` schema.

Both data bugs were caught by writing and running leak/correctness checks
against the full-scale build, not by inspection — the dry-run on a small
sample (26-44 rows) did not surface either, which is itself a finding:
small feasibility samples can pass every check and still hide bugs that
only appear at the row counts the platform actually requires.

## Platform failure round 1 — nested zip (fixed)

First upload failed at the prepare step: `FileNotFoundError` on
`works.jsonl`. The raw zip had been built as `dataset/raw/works.jsonl` and
`dataset/generator/...`, but the platform mounts the archive's top level as
`raw_root` and calls `prepare(raw_root, ...)` directly — exactly rejection
register rule 1.1 (zip must be flat) and 1.3. Two fixes, both verified:

1. **Zip rebuilt flat**: `works.jsonl`, `hard_neighbors.json`, and
   `generator/` now sit at the archive root.
2. **`prepare.py` made self-healing** (register rule 1.11): a
   `_find_raw_file` helper resolves each raw file whether it is at the
   root, under `raw/`, under `dataset/raw/`, or anywhere beneath
   `raw_dir`, so a future layout change cannot reproduce this failure.

The platform entrypoint path was then replayed locally from the
extracted flat zip — `prepare()` given the extraction root, not a
subfolder — and every generated file diffed byte-identical against the
shipped split.

## Review round 1 — cross-row exemplar shortcut (fixed, measured)

Reviewer (nxank4) found that with one shared exemplar per topic and
same-topic rows adjacent in `test.csv`, a no-model probe — pick the
candidate text that recurs in the three rows before and after — scored
**0.8291**, and two agent runs scored 1.0000. A real leak that the leak
suite did not test for: every per-row check passed while a cross-row
recurrence check was never run. Rejection register section 2A in a new
shape (row *order* and *text reuse* as the transductive channel).

Fix, in `prepare.py`:

- Each topic is halved into a **reserved exemplar pool** (never used as a
  query) and a **query set**. Query *i* gets reserved item *i* as its
  unique true exemplar, so no true-exemplar text appears in more than one
  row.
- Distractors are sampled **per row** from other topics' reserved pools,
  so no exemplar text (true or distractor) can ever equal any row's query.
- Rows are shuffled and reshuffled on a stepped seed until **no two
  consecutive rows share a topic**.
- `train.csv` now carries the true `slot` (reviewer's point 2); the
  answer and submission columns are both named `slot` (point 7);
  `dataset_stats.json` is no longer written to `public/` (point 6).

Measured on the rebuilt split (2,386 train / 519 test):

- Reviewer's neighbour-recurrence probe: 0.8291 -> **0.1503** (below chance)
- Global text-recurrence probe: **0.0597**
- True-exemplar texts used in more than one row: **0**
- Any exemplar equal to any train or test query abstract: **0**
- Consecutive same-topic rows: **0**
- Chance 0.1580 / adversary 0.3372 / **reference 0.5356** / oracle 1.0;
  margin **0.1985** (up from 0.1755 — now above the 0.18 guideline)

Also per review: the solver-facing page no longer names the source
catalogue anywhere (its labels are the answers), the measured-baseline
list, version history, and references to unshipped files were removed
from Evaluation (reference scores go to the private Discord channel), the
metric is stated in one line in a real code block, and Compute is cut to
the A10G tier and the time limit.

## Review round 2 — agents at 0.99–1.00 via the raw archive (fixed, measured)

After round 1 shipped, agent runs scored 0.99–1.00. Attacking the shipped
files found the path: `works.jsonl` paired every abstract with its
`topic_id`. Any agent that can read the data mount maps each candidate
text to its topic and picks the match — measured locally at exactly
**1.0**. Splitting the mapping into a separate `topic_map.json` keyed by
`work_id` did not help (still 1.0): `works.jsonl` still joins abstract to
`work_id`, so the lookup is two hops instead of one. `title` was a second
covert key (490/490 test titles present in raw).

The conclusion is structural: **any raw file that lets a candidate's text
reach its topic is an answer key, however it is split.** So the raw
archive now carries no topic information of any kind, and the grouping
`prepare.py` needs is embedded directly in `prepare.py` (written in by
`build_prepare_groups.py`; the platform accepts a single prepare file). `query_title` was
also dropped from the public split.

Rejected alternative, measured: recomputing topic groups from abstract
text alone via embedding clustering recovers only 34–54% purity across
k=600–1200, so the labels would be mostly wrong. Not viable.

Measured on the rebuilt split (2,087 train / 431 test):

- Raw abstract -> topic lookup: 1.0 -> **chance** (no grouping in raw)
- Reviewer's neighbour-recurrence probe: **0.17**
- Was-a-train-true-exemplar probe: **0.00**
- Leak suite: 0 train/test overlap, 0 exemplar==any query, 0 slot
  mismatch, 0 ambiguous rows, 0 consecutive same-topic rows

**Residual count leak, closed.** After the raw fix, a text shown as a
*true* exemplar recurred less across train+test (mean 6.1) than one shown
as a *distractor* (mean 10.1) — a reserved text serves as true only for
its own topic's queries but as a distractor for ~30 neighbour topics'
queries. A count-only probe ("pick the least-recurring candidate") scored
0.43, and fused with the MiniLM reference lifted it 0.539 -> 0.588: real,
worth closing. A least-shown sampler alone made it *worse* (0.33 -> 0.43),
proving the gap is set by the 1:5 true:distractor slot ratio, not by
sampling. What closed it: a per-text exposure cap (`MAX_SHOWS_PER_TEXT`
= 6; a distractor topic whose pool is capped is skipped for the next
neighbour) with **both splits built in one interleaved, seeded pass** so
the cap is spent evenly (test-first alone starved train and reopened the
gap at 0.34). Result: recurrence 5.36 (true) vs 5.62 (distractor); the
count probe 0.43 -> 0.23, and *both directions* of every count probe —
least/most recurring in test, in train+test, as train distractor — sit
within noise of 1/6 (best 0.25). No consistent count signal remains.

**Boundary.** If the platform mounts `prepare.py` into the agent environment, the grouping is readable
and no dataset design can prevent a lookup — `prepare.py` must be able to
do the join, and an agent that can read `prepare.py`'s inputs can do the
same join. That is a platform mount question, not a dataset one.

## Review round 3 — one true exemplar per row

The round-2 build still let two rows of one topic share a true exemplar
(65 test rows, 339 train rows), because the true pick was the least-shown
pool text and distractor shows count toward that. Each topic's pool is as
large as its query set, so `prepare.py` now gives every row a pool text no
other row of its topic has had as true: **0 reused** in either split.

That alone opened a count gap (true texts shown less than distractors,
rarest-candidate probe 0.29 on train+test), because uneven distractor
exposure size-biases the distractor counts. Lowering the cap closed it only
by dropping rows (cap 3: 218 test rows). What closed it: within the
neighbour band, distractor topics are taken **least-exposed first**, so
exposure stays level (2,176 of 2,905 texts shown exactly 6 times; mean 5.85
true vs 5.88 distractor).

Measured on the rebuilt split (2,338 train / 474 test):

- Reviewer's neighbour-recurrence probe: **0.173**
- Count probes, train+test: rarest **0.175**, commonest **0.158**; test only:
  rarest **0.205**, commonest **0.148** (chance ~0.167)
- 0 reused true exemplars, 0 exemplar == any query, 0 consecutive same-topic rows
- Ladder: chance 0.1751, adversary 0.3502, **reference 0.5295**, margin 0.1793
- Rebuilt from the unchanged raw zip in a clean folder: byte-identical, deterministic

## Final verification (all measured, all passing)

- [x] Full-scale fetch (1,574 viable topics after dedup, 7,961 raw rows)
      with the halved-topic, exposure-capped design yields 2,087 train /
      431 test, clearing the >=500-usable-item / >=1000-train-row
      platform minimums with wide margin.
- [x] `prepare.py` runs deterministically: two runs over the same raw/
      produce byte-identical `train.csv`, `test.csv`, and `answers.csv`.
- [x] Full baseline ladder measured on the shipped split; see table above.
      `config.yaml`'s `grade:` block holds the real numbers.
- [x] Margin between reference and keyword-overlap adversary: 0.1755
      (just under the 0.18 guideline; accepted rather than risk
      overshooting into "too hard" with a fourth tuning pass).
- [x] No `topic_id`, `topic_name`, `field_name`, or `domain_name` column
      present in `dataset/public/train.csv` or `test.csv`.
- [x] Zero train/test `work_id` overlap; zero cross-row exemplar leak
      (no test candidate's exemplar text equals another test row's query
      abstract); zero `true_slot` / topic mismatches; zero rows where a
      non-true candidate also matches the query's real topic.
- [x] `grade.py` self-tests (9 hand-built cases: exact match, one error,
      missing row, extra row, duplicated id, out-of-range slot,
      non-numeric slot, empty submission, missing column) all pass.
- [x] Raw+generator zip built FLAT: `works.jsonl` + `generator/` ONLY —
      no topic field anywhere in the archive; grouping lives in
      `prepare.py` itself (`build_raw.py` +
      `build_hard_distractors.py`) only; verified to exclude every
      `public/`/`private/` path; verified to rebuild `dataset/public/` and
      `dataset/private/` byte-identically from a clean extraction.
