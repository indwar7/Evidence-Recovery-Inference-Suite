# Untangling Mixed Edit Histories

## The problem

A wiki page is never written once. It is written, corrected, argued over,
re-priced, vandalised, restored, and quietly copyedited by strangers over
years. Every one of those moments leaves a complete copy of the page behind.

This challenge hands you those copies with the labels torn off.

Each item is a bag of text snapshots. Every snapshot is one genuine revision of
one real travel-guide article — the wikitext exactly as it stood at a single
moment. Snapshots from several different articles are mixed into the same bag
and shuffled, and everything that would identify or date them has been removed:
no timestamps, no revision numbers, no edit summaries, no usernames, and the
article's own name blanked out wherever it appears in its own text.

Your task is to rebuild the family tree. For each snapshot, name the snapshot
it was edited **from** — its immediate parent. If a snapshot's real parent is
not in the bag, it is the head of its lineage and you answer `ROOT`. Get every
pointer right and you have simultaneously recovered which snapshots belong
together, the order they were written in, and where each lineage starts.

## Why it resists the obvious approaches

**The number of lineages is hidden.** A bag contains anywhere from 3 to 11
lineages, 6 on average, across 10 to 42 snapshots. However many there are, that
is exactly how many `ROOT` answers the bag has. Both kinds of mistake cost you:
fuse two lineages and you assert a parent that never existed; break one apart
and you assert a root that isn't there.

**The same article shows up as several separate lineages.** Two to four articles
feed a bag, but each one typically contributes more than a single lineage,
sampled from different periods of its own history with at least one revision
deliberately withheld between them. Those sibling lineages are the same page a
few days apart: the same listings, the same phrasing, the same slow-burning
dispute. Telling them apart is not a clustering exercise you can win with a
better similarity metric — and because the connecting revision is absent, the
later lineage's head is legitimately a `ROOT`, not a missed link.

**Knowing which snapshots are neighbours does not tell you which came first.**
Pick any snapshot and its nearest match by text similarity: roughly 85% of the
time that really is its parent or its child. The hard half is the arrow. Byte
count will not settle it either — these windows were deliberately taken from
spells when a page was being reworked rather than expanded, so size orders a
real lineage correctly only about 60% of the time.

**So the signal you need is editorial, not statistical.** Revisions are
lopsided in a way that is learnable. Pages accumulate specifics as they age — a
price, an opening time, a source, a wikilink, a hedge — and shed what upkeep
clears away: shuttered venues, dead links, repeated sentences, typos, marketing
language. Text is tightened far more often than it is padded; a plain mention
becomes a formatted listing far more often than a listing collapses into a
mention. Read a diff in one direction and it looks like something a person
would do; read it backwards and it usually does not. That judgement, learned
from the labelled lineages and carried to unlabelled ones, is the challenge.

## What makes this task new

The labels here were not produced by an annotator or a model. They are the
wiki's own revision graph — revision *r* descends from revision *r−1* because
that is the copy an editor actually opened and modified, recorded by MediaWiki
as it happened. There is nothing to disagree with.

### How it differs from existing revision-history work

Edit histories have been mined extensively, but always as raw material to learn
edits *from*. No prior resource asks a model to reconstruct the history itself.

**TokTrack** (Flöck et al., ICWSM 2017) traces provenance for all 13.5 billion
token instances in English Wikipedia: which revision introduced each token and
every revision that removed or restored it. Its revision chains arrive complete,
ordered and identified — the provenance is an annotation laid over a graph
already known. Here the graph is the answer.

**WikiHist.html** (Mitrevski et al., ICWSM 2020) republishes Wikipedia's full
revision history as HTML. Same ordered, identified history in a different
format, and no prediction task attached.

**NewsEdits** (Spangher et al., NAACL 2022) assembles 4.6 million versions of
1.2 million news articles together with a document-level reasoning task. Its
versions come pre-ordered and pre-grouped; the question posed is what a future
revision will change, not which revision spawned which.

**PEER** (Schick et al., 2022) learns from Wikipedia edit histories to draft
text, suggest edits and justify them. It generates edits given a history it can
see; it never has to infer a history it cannot.

**WikiSplit** (Botha et al., EMNLP 2018) harvests a million sentence rewrites
from Wikipedia edits to supervise split-and-rephrase. The direction of each
rewrite is supplied as part of the supervision.

Each of those begins with an ordered, identified sequence of revisions. This
challenge deletes precisely that and asks for it to be reconstructed — with a
second obstacle none of them has, since several lineages are interleaved in one
bag and their number is concealed.

Solving it means doing three things at once, and the metric gives credit for
none of them separately: separating interleaved lineages that may originate
from the same article, deciding the direction of an individual edit, and
folding those local decisions into one globally consistent forest.

### How the data was assembled, and what is held back

Everything originates in the official English Wikivoyage full-history XML dump
(`enwikivoyage-latest-pages-meta-history`, dated 2026-08-05, CC BY-SA 4.0).

For each qualifying article, a single window of 14 consecutive revisions is
selected by a churn score computed **purely from byte sizes** — never from the
text, never from the labels, never from how any solver performs — which
prefers stretches where the page was being reworked over stretches where it
simply grew. That window then yields two or three lineages taken from separate
runs within it, always with at least one revision dropped in the gap, which is
what makes each lineage head a true `ROOT`. Lineages from two to four articles
are pooled, shuffled, and only then given the identifiers `n00, n01, …`.

Stripped before release: timestamps, revision ids, edit summaries, usernames,
the article title (replaced by `___` throughout its own text), and every
template parameter whose value is a bare ISO date. That final rule exists for a
concrete reason — Wikivoyage listings carry a `lastedit` field noting when an
editor last checked them, and on its own it orders a lineage at 0.64 pairwise
accuracy. Editors misspell the field often enough that the rule targets the
date-shaped value rather than any particular spelling. Afterwards no
date-valued parameter remains anywhere in the released text, and the ISO dates
that survive in ordinary prose order lineages at 0.497 — indistinguishable from
chance.

**On holdout.** The revisions themselves are public, as they must be for a
challenge built on an open corpus; they are not withheld from the dump they
came from. What is withheld is everything needed to recover the answer without
inferring it: which snapshots share a bag, the shuffle, the identifier mapping,
and the parent labels, all of which live only in the private answer key. No
file a solver receives carries a timestamp, a revision id, or a title, so the
chronology cannot be looked up from the released material. Solutions run
without network access.

## Scoring

The metric is **immediate-parent accuracy** — the share of test snapshots whose
parent you name correctly, averaged over every snapshot in the test set.
Answering `ROOT` for a genuine lineage head counts as correct. Scores fall in
[0, 1] and larger is better.

Every figure below was measured on this dataset:

- Uniformly random parent — **0.053**
- Every snapshot marked `ROOT` — **0.252**
- Cluster, then sort each cluster by byte size — **0.310**
- Attach to the most similar smaller snapshot — **0.358**
- Spanning arborescence with direction taken from size — **0.390**
- Linear model over edit-diff features — **0.493**
- Reference solution — **0.552**
- Nearest neighbour with an oracle supplying direction — **0.852**
- Oracle — **1.000**

That second-to-last figure is the one to keep in mind. Even with a perfect
similarity search and a genie whispering the direction of every edit, the score
tops out at 0.852. Similarity work alone cannot cross it, and everything above
0.390 has to come from reading the edits.

Scoring in code:

```python
def evaluate(submission, answers):
    """
    submission / answers: DataFrames with columns item_id, snapshot_id, parent_id
    Structural validity of `submission` is checked before this point.
    """
    merged = answers.merge(submission, on=["item_id", "snapshot_id"],
                           suffixes=("_true", "_pred"))
    return float((merged["parent_id_true"] == merged["parent_id_pred"]).mean())
```

## The data

Everything sits under `dataset/public/`:

- `snapshots/<item_id>.json` — one item's snapshots, as
  `{snapshot_id: wikitext}`
- `train.csv` — parent pointers for the 630 labelled items
- `test.csv` — the 6,241 snapshots whose parents you must supply
- `sample_submission.csv` — a valid but deliberately poor entry (all `ROOT`)
- `validate_submission.py` — the grader's structural checks, runnable offline
- `dataset_stats.json` — item and snapshot counts

A snapshots file is shaped like this:

```json
{"item_id": "it00042",
 "snapshots": {"n00": "'''___''' is a city in ...", "n01": "...", "n02": "..."}}
```

Identifiers are handed out **after** shuffling, so `n00` is not the oldest and
neighbouring numbers mean nothing.

Columns in `train.csv`:

- `item_id` (string) — the item this snapshot sits in
- `snapshot_id` (string) — the snapshot's identifier within that item
- `parent_id` (string) — the snapshot it was edited from, or `ROOT`

`test.csv` carries the first two columns only.

There are 900 items altogether, 630 labelled and 270 held out. No item spans
the split, and no article appears in more than one item.

## Your submission

Produce a CSV called `submission.csv` with three columns:

- `item_id` (string) — copied from the test file
- `snapshot_id` (string) — copied from the test file
- `parent_id` (string) — your predicted parent, or the literal `ROOT`

It must satisfy all of the following:

- Exactly **6,241 rows** plus a header, one row per row of the test file.
- Each test `(item_id, snapshot_id)` pair present once — no omissions, no
  repeats, no additions.
- Each `parent_id` is either `ROOT` or a `snapshot_id` from the **same item**.
- No snapshot is listed as its own parent.
- The pointers describe a **forest, with no cycles**. An entry containing
  `n01 → n02 → n01` is thrown out in its entirety, so verify before submitting.
  Every lineage has to terminate at `ROOT`.

You can check an entry offline by running
`python validate_submission.py submission.csv dataset/public/test.csv`, which
applies exactly the checks the grader applies.

A well-formed entry starts like this:

```csv
item_id,snapshot_id,parent_id
it00301,n00,n04
it00301,n01,ROOT
it00301,n02,n00
```
