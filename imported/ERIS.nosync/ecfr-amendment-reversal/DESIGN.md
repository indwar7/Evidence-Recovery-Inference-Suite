# Design notes — The Vanished Clause

## The task in one line

Given a CFR section as it reads after an amendment, the wording the amendment
removed as loose fragments, and a set of marked places of which one in four
is real, rebuild how the section read before — with the Federal Register's
amendment instruction withheld.

The sections below down to "Revision 2" record the first design (text only,
free generation) and what was measured on it. They are kept because they are
why the task changed; the current numbers are in "Revision 2" and
`config.yaml`.

## Why this shape

Two candidate ideas were gate-checked on the same day. The other one
(Crossref: order citing passages of retracted papers by hedging drift) was
**abandoned at the gates**, and the reasons are worth recording because they
are the kind of thing that only surfaces by probing the API:

- Crossref exposes 186M works and structured `update-to` retraction
  relationships, but **not the text of one paper citing another**. The
  original idea needed exactly that text.
- Retraction "lag" is degenerate: median **1 day**, because Crossref records
  the notice's own publication date, not the retracted paper's.
- 72% of retraction-notice abstracts state no extractable reason. Any reason
  label would have been my own regex heuristic over free text — a Gate 3
  violation (the label must be the archive's own structured value).

eCFR passed every gate instead: the label is the archive's own recorded
`amendment_date`, both text sides are the agency's published wording, the
licence is unambiguous (U.S. Government Work, 17 U.S.C. § 105), and the
source URL is reachable.

## Data

`https://www.ecfr.gov/api/versioner/v1`. Two endpoints: `/versions/title-{n}`
lists every amendment date per section with a `substantive` flag; `/full/
{date}/title-{n}.xml` returns the section text as of a date. Consecutive
substantive amendment dates on the same section give a (before, after) pair.

Two API quirks cost real time and are documented in the fetcher:

- `/full/` **requires** a request header permitting compression and returns
  HTTP 406 without one. This is not prominently documented.
- `/versions/` caps at 1000 results per call and rejects `per_page`.

One content quirk matters more: the eCFR inserts a banner reading *"Link to
an amendment published at 89 FR 47080, May 31, 2024"* into sections whose
amendment is published but not yet incorporated. It names the amendment date
directly. It is stripped from both sides, along with editorial notes that
narrate the change.

## Split

By CFR **part**, never by row. Parts share defined terms, drafting conventions
and cross-references heavily, so a part-mixed split would let those be
memorised rather than generalised. Section disjointness follows and is
asserted anyway. Exact-duplicate `text_after` values are collapsed before
splitting — two sections can carry identical boilerplate, and a duplicated
input spanning the split is a leak even when the sections differ.

## Metric: changed-token F1

The metric changed **three times**, each time because a measurement said it
had to. This is the most important section of this document.

### 1. Whole-string similarity, rejected before building

Measured on the corpus: the median amendment touches **1.0%** of its section
(p25 0.3%, p75 2.9%). Under edit similarity or BLEU against the full section,
copying the input through is ~99% correct and scores near the maximum,
leaving no range to separate real attempts. The identical failure was measured
on a prior challenge in this portfolio, where a rewrite task scored by plain
edit similarity gave copy-through 0.795.

Scoring only the tokens the amendment removed sends copy-through to exactly
**0.000**.

### 2. Recall over changed tokens, measured and rejected

With recall alone, a submission that appends a candidate vocabulary of
plausible regulatory words to the shown text recovers every changed token by
brute force. Measured on a worked example: the shown text plus a 25-word dump
scored **1.000** — identical to the oracle.

Adding a precision term over what the prediction *proposes* closes it. The
same attack, rebuilt from the 60 most frequent train-side additions, scores
**0.008** on the shipped split.

### 3. The oracle did not score 1.0, and that was a real bug

After the precision fix, the oracle — submitting the true prior wording —
scored **0.568**, not 1.000. Root cause: `extra = Counter(prediction) -
Counter(shown)` cancels a changed token that also occurs elsewhere in the
section, so those tokens were unreachable and recall could never saturate.

Fixed by deriving both sides through the *same* alignment: the truth's changed
tokens come from aligning shown→truth, and the prediction's proposed tokens
from aligning shown→prediction. Oracle then scored exactly **1.000**.

This bug was invisible on hand-built unit tests (which used short strings with
no repeated tokens) and only appeared on real section text.

## What the data actually looks like — and what it invalidated

Before building the reference I assumed an amendment rewrites *one* passage.
That assumption was wrong, and measuring it changed the reference design.

| Measurement (inner part-disjoint split) | Value |
|---|---|
| Amended sentences per row, median | **7** |
| Amended sentences per row, mean | 13.3 |
| Amended sentences per row, max | 124 |
| Sentence locator top-1 accuracy | **90.0%** |
| Ceiling: perfectly swap **one** sentence | **0.211** |
| Ceiling: perfectly swap **all** amended sentences | **0.636** |
| Ceiling: submit the true prior wording | 1.000 |

Two conclusions:

- **Finding the amended passage is nearly solved** (90% top-1) and is not
  where the difficulty lives.
- **A single-edit strategy is capped at 0.211** no matter how good the edit.
  My original "one careful edit" reference premise was therefore unsound.

## Reference approaches measured, and why the table-based ones failed

All measured on the inner part-disjoint validation split.

| Approach | Score |
|---|---|
| Copy the input through | 0.000 |
| Phrase substitution table + locator, one edit | 0.000 |
| Phrase + token substitution table, one edit | 0.000 |
| Substitution tables, edit **all** detected sentences (best threshold 0.3) | 0.006 |
| Retrieval: nearest prior-wording sentence from train (best sim ≥ 0.4) | 0.023 |
| *(ceiling: perfect all-sentence swap)* | *0.636* |

Why the tables fail is structural, not a tuning problem. The split is
part-disjoint, so an exact phrase mined from one CFR part almost never occurs
verbatim in another. Token-level swaps transfer better in principle — 67 of
106 training rows share changed content-words with at least three other rows —
but a token table cannot reconstruct a *sentence*, and the metric scores
tokens the amendment removed, not tokens that plausibly belong.

The honest conclusion is that this task requires **generation**, and a
retrieval-or-substitution reference is a floor rather than a solution. That is
a desirable property for the challenge: the gap between 0.023 and the 0.636
all-sentence ceiling is real headroom for a solver, and it cannot be closed
by a lookup table.

## Revision after review (2026-09-27)

Reviewer: test rows carried title, part and section number, so the eCFR
version history returned every `text_before` directly; test had too few rows.
A full re-audit then found three more problems. All measured, all fixed in
`prepare.py` / `grade.py`; the raw archive is unchanged.

| Problem | Measured | Fix |
|---|---|---|
| Section identity published | title, part, section_name in both splits; own number in every text | columns dropped; own number masked as `[SECTION]` on both sides |
| Same-section chains in test | 10/161 test answers = another test row's `text_after` (0.062 vs 0.104 ref) | one pair per section in test; rest of chain dropped |
| Fetcher remnants | `21, 2024.` banner tail in 677 raw `text_before`; OFR source note names the amending FR document; correction banners never stripped | cut at the first FR source note; banner variants and remnants removed; 254 citation-only pairs become identical and drop |
| Position-free metric | 60-token dump scored 0.129 | tokens keyed by gap; dump 0.012 |
| Grader too slow for the platform evaluator | 42 s per grade (oracle ~80 s): evaluator could not score | difflib anchors + LCS between them; ~2 s per grade |
| Test size | 161 rows | 353 rows (410 train), one per section |

The earlier tables in this document (locator 90%, ceilings 0.211/0.636,
table/retrieval scores) were measured before these fixes. The remnant made
amended sentences trivially locatable, which is why the locator fell from
90% to 51% once it was removed. Current numbers are in `config.yaml`.

## Revision 2 (2026-09-30): the task had no range, so the removed wording is given

Three agent runs on the platform scored **0.020, 0.000, 0.000**. That agrees
with everything measured here on the first design: seq2seq reference 0.002,
best lookup baseline 0.009. The earlier conclusion in this document — "the
task requires generation, and the gap to the ceiling is real headroom" — was
wrong. The ceilings (0.749, 0.933) were computed *from the truth*; nothing
that could be built from the public data came within 0.74 of them. Prior
wording is not inferable from amended wording: a rule that says *ten days*
does not record that it said *thirty*.

What *is* inferable is where wording belongs. So:

- `removed_fragments` — every maximal run of prior-wording tokens that the
  grader's own alignment puts in one gap, as a literal substring of the prior
  wording, in an order fixed by a hash of the private record id.
- `[GAP]` marks in `text_after` — every real gap, plus decoys.
- `train_slots.csv` — the mark each training fragment came from.
- `grade.py` is unchanged apart from dropping a `[GAP]` left in a prediction.
  The submission is still the full prior wording as text.

### The decoy count is the difficulty knob, measured before it was fixed

| Decoys per real gap (min per row) | Random dealing | Basic pair scorer |
|---|---|---|
| 0 | 0.398 | — (one-fragment rows solved by construction) |
| 1 (2) | 0.168 | 0.572 |
| 2 (3) | 0.131 | 0.522 |
| **3 (5)** | **0.078** | **0.478** |
| 5 (8) | 0.066 | 0.425 |
| 8 (12) | — | 0.377 |

(Probe numbers, from the basic feature set on an earlier decoy hash; the
shipped ladder is below.) Three per real gap puts a label-free heuristic
near 0.3 and a learned scorer near 0.5, with every content-free probe
under 0.12.

### A decoy placement that was built and rejected

An amendment's edits cluster, so uniform decoys leave "real marks sit
together" as a signal. The first attempt to neutralise it put one decoy
within 12 tokens of every real gap. Measured, it did the opposite: a real gap
and its near decoy form a tight pair, so tightness became a *better* locator.

| | Uniform decoys (shipped) | One near decoy per real gap |
|---|---|---|
| Tightest marks, random fragments | 0.119 | 0.157 |
| Fragment-blind mark classifier | 0.188 | 0.227 |
| Reference (counts over train + test inputs, since replaced) | 0.518 | 0.571 |

Uniform decoys are shipped. The clustering that remains is the clustering
amendments really have, and a solver is entitled to it.

### Why decoy positions are keyed on the private record id

If decoys were the first K non-real positions in an order computable from
public data, a solver could recompute the order: any marked position ranked
after an unmarked one would have to be real. Keyed on the record id — which
embeds the title, section and both dates, none of them published — the order
cannot be replayed.

### The shipped ladder (353 test rows)

| Submission | Score |
|---|---|
| Blank / marks removed / marks left in | 0.000 |
| 60 commonest removed tokens appended | 0.012 |
| Every fragment appended to the end | 0.045 |
| Every fragment at the first mark | 0.080 |
| Random dealing | 0.091 |
| Every fragment at the last mark | 0.098 |
| Listed order into the first marks | 0.102 |
| Every fragment at every mark | 0.112 |
| Tightest marks, random fragments | 0.119 |
| Fragment-blind mark classifier | 0.188 |
| Bigram-count heuristic, no labels | 0.295 |
| Reference: linear pair scorer + assignment | 0.497 |
| Ceiling: every fragment at its true mark | 0.982 |
| Oracle | 1.000 |

Row-wise halves of the answers score the reference 0.513 and 0.482.

The true-mark ceiling is 0.982 and not 1: where a fragment ends in a token the
shown text repeats right after the mark, the alignment keys that token to the
neighbouring gap. The true prior wording itself scores exactly 1.000.

### Every test row is predicted on its own

The first reference for this design counted n-grams over the training text
*and the test inputs* and scored 0.518. Fitting anything on the released test
set is transductive adaptation — a rejection class of its own — so it was
removed: counts now come from the training text and from the row's own shown
text only, the rule is stated in the problem description, and a `[REQUIRED]`
rubric mirrors it. The honest reference is **0.497**.

Doing this exposed a second fault. A training row's own prior wording
contains its fragments in place, so counts that include it make every true
training pair look better than it will at test time. With training features
computed from counts that leave the row's own fold out, the same model went
from 0.464 to 0.497, and validation (0.477) now agrees with test.

### Novelty: what the description now says explicitly

A numbered "What is new here" section and a "How it relates to prior work"
section, the latter conceding the classical primitive (linear assignment) and
differentiating four distant antecedents: the cloze procedure, gapped-text
reading items, differential file comparison, and textual criticism.

### Also changed in this revision

- `dataset_stats.json` no longer reports any statistic of a test answer (it
  carried the median changed-token count and fraction of test). Size
  statistics are now train-side only.
- Rules now ban pretrained models explicitly. The first reference used
  flan-t5-small; with placement as the task, a pretrained language model
  would be measuring its pretraining.
- The sample submission is the input with its marks removed.

## Open items

- The platform's copy of the raw archive has been an older, smaller one
  (5.2 MB `amendments.jsonl`, 47 test rows). The numbers here need the 13.9 MB
  archive (1,404 pairs, sha256 in `dataset/raw/meta.json`).
- The title is a creative name rather than a plain description; untouched,
  because no review has raised it.
- Roughly a quarter of pairs are punctuation- or date-only churn rather than
  substantive language changes. `prepare.py` drops changes under three tokens
  and over 60% of the section; two thirds of training fragments are still a
  single token.
- The reference is linear over hand-built pair features. A small language
  model trained from scratch on the shipped text is the obvious stronger
  solver and has not been measured.
