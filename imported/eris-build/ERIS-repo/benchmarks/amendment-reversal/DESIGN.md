# Design notes — The Vanished Clause

## The task in one line

Given a CFR section as it reads after an amendment, reconstruct how it read
before — with the Federal Register's amendment instruction withheld.

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

## Open items

- Resolved: the generative reference in `solution.ipynb` is executed end to
  end and its score is recorded in `config.yaml`.
- The corpus is still being fetched. The fetcher runs as two workers over
  disjoint CFR titles writing separate shards, and `prepare.py` merges every
  `amendments.jsonl` it finds, de-duplicating by `record_id`, so a partial
  fetch is always usable and an overlapping re-run cannot double-count.
- Roughly a quarter of pairs are punctuation- or date-only churn rather than
  substantive language changes. `prepare.py` drops changes under three tokens
  and over 60% of the section, but a stricter content-word filter is worth
  measuring once the full corpus is in.
