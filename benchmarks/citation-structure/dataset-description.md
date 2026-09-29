# Multi-Jurisdiction U.S. Case Law and Its Citation Graph

Real opinions from United States federal and state courts, paired with
the real citation edges between them, packaged as a candidate-selection
task: one query opinion, twelve candidates, and the question of which
candidates the query cites.

**Jurisdiction coverage.** The collection deliberately spans multiple
court levels rather than being restricted to one. Scanning the shipped
test split for court-identifying phrases that survive redaction finds
references to courts at every level throughout the corpus. A query and
the candidate it cites therefore need not belong to the same court
system, and citations across court levels are ordinary here.

## Provenance

| | |
|---|---|
| Source | [CourtListener Bulk Data](https://www.courtlistener.com/help/api/bulk-data/), Free Law Project |
| Bulk files | `https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/` |
| Snapshot published by the source | 2022-09-30 |
| Licence | [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) |
| Snapshot downloaded for this build | 2026-09-20 |

The two dates differ, and both are correct. CourtListener publishes
dated bulk-data snapshots that remain available long after they are
cut; this build downloaded the archived **2022-09-30** snapshot on
**2026-09-20**. The opinions therefore reflect the state of the archive
as of September 2022, not of the download date.

This is verifiable from the shipped data: across all 4,355 clusters in
`opinion_clusters.csv`, `date_filed` ranges from 1994-05-25 to
2022-01-28, and **zero** opinions are filed after the 2022-09-30
snapshot date.

The Free Law Project states on the bulk-data page: *"Our bulk data files
are free of known copyright restrictions."* The underlying material is
the text of judicial opinions, which is not subject to copyright under
U.S. law — see *Banks v. Manchester*, 128 U.S. 244 (1888) and *Georgia v.
Public.Resource.Org, Inc.*, 590 U.S. 255 (2020). No licence fee,
registration or permission is required to redistribute it, and the
Public Domain Mark imposes no attribution requirement; the attribution
above is a courtesy to the Free Law Project.

## How the task data was built

1. Opinions were matched to their clusters and to the citation-edge
   table. 44,291 opinions survived; 5,792 citation edges run between two
   opinions both present in that set.
2. The citation subgraph was decomposed into 524 connected components.
3. Components were assigned wholesale to train or test, so **no citation
   edge crosses the split**. 521 components went to train, 3 to test.
4. For each query opinion, its true cited opinions (1 to 4 of them) were
   placed in a pool alongside distractors drawn from the same side of the
   split, to a fixed pool size of twelve. Pool order is randomised per
   query with a fixed seed.
5. All text was redacted (below), whitespace-normalised, and truncated to
   a fixed character budget.

`prepare.py` reproduces this end to end and is deterministic — two runs
produce byte-identical outputs.

## Files

This dataset ships the **raw source material and the generator only**. The
train/test split is not distributed here — `prepare.py` builds it from these
files, deterministically, and two runs produce byte-identical outputs.

### `raw/opinions.csv` — 4,151 rows
The opinion text the task is built from: every query and every candidate.
Columns: opinion id, cluster id, and the redacted opinion text.

44,291 opinions were matched against the citation-edge table during the build
(recorded in `build_meta.json` as `n_opinions_matched`), but only those
reachable as a query or a distractor are shipped, so the archive stays small
enough to inspect.

### `raw/opinion_clusters.csv` — 4,355 rows
Cluster-level metadata for those opinions: cluster id, date filed, and the
docket/court identifiers the join needs.

### `raw/citation_edges.csv` — 5,792 rows
The citation graph: one row per directed edge between two of the 44,291
matched opinions. Some endpoints were not kept in `opinions.csv`, and
`prepare.py` skips any edge whose endpoint is missing. This is the archive's own recorded citation
relation and is the ultimate source of the task's labels.

### `raw/build_meta.json`
Counts recorded during the fetch: opinions scanned, opinions matched, edges
retained, and the snapshot date of the CourtListener bulk export used.

### `raw/meta.json`
Provenance for the shipped archive: source URL, retrieval timestamp, licence,
row counts per file, and the redaction settings applied.

### `raw/ATTRIBUTION.txt`
Source, publisher, licence, and the public-domain notice for U.S. judicial
opinions.

### `generator/build_raw.py`, `generator/join_corpus.py`, `generator/redact.py`
The fetch-and-build pipeline that produced the raw archive: `build_raw.py`
streams the CourtListener bulk tables, `join_corpus.py` joins opinions to
clusters and to the citation-edge table, and `redact.py` removes citation
markers and case names. None of them annotates anything or calls a model.

### What `prepare.py` derives (not shipped here)
From the files above, `prepare.py` builds `public/train.csv` (1,954 rows),
`public/test.csv` (400 rows), `public/sample_submission.csv`,
`public/dataset_stats.json` and the held-out `private/answers.csv`. Those are
produced at build time and documented in the challenge's problem description,
not distributed as part of this dataset.

## Label leakage and what was done about it

The obvious shortcut — find the citation string, read off the answer —
is closed deliberately:

- **Reporter citations removed.** Strings matching U.S. reporter patterns
  (`410 U.S. 113`, `347 F.3d 572`, `123 S. Ct. 45`, and ~20 other
  reporter series) are replaced with `[CITATION]`. The pattern tolerates
  runs of whitespace inside the citation, because the OCR'd source text
  frequently breaks one across a page boundary.
- **Case names removed.** Party-name strings from the clusters table are
  replaced with `[CASE]` wherever they appear in body text, along with a
  generic `X v. Y` pattern that over-redacts slightly — the safe
  direction, since the task is about substance, not names.
- **Verified on the built split.** The files `prepare.py` generates contain
  **zero** surviving reporter citations and **zero** surviving `X v. Y`
  patterns. The scan is re-run on every build and its result is written to
  `public/dataset_stats.json` under `leak_scan`, so the check is reproducible
  from the raw archive shipped here rather than taken on trust.

Two further shortcuts were measured and closed:

- **Candidate position carries no signal.** Candidate order is
  randomised per query. Always selecting positions 0 and 1 scores 0.1253, versus 0.1257 for selecting
  two at random — indistinguishable.
- **Text length carries almost no signal.** Selecting the two longest
  candidates scores 0.1262 and the two shortest 0.0963, both *below* the
  0.1257 random floor.

## Intended use

**Primary use.** Training and evaluating models that infer a relationship
between two documents from their substantive content rather than from
explicit surface markers — here, deciding which of twelve candidate
opinions a query opinion cites, with every citation string and party name
redacted.

**Also suitable for.**

- Benchmarking long-context text encoders. Opinions run to several
  thousand characters, and the passage carrying the signal is often deep
  in the discussion, so representations with short input windows discard
  most of each document.
- Studying selection under an unknown positive count — deciding how many
  items to accept is part of the problem, not a hyperparameter supplied
  with the data.
- Research on legal-document similarity and on how well general-purpose
  text representations transfer to specialised professional prose.
- Teaching chance-corrected evaluation: the dataset ships with degenerate
  baselines that score exactly zero, making the argument for MCC over F1
  concrete rather than theoretical.

**Not suitable for.**

- **Legal research or practice.** The text is redacted and truncated, the
  citation graph is a small fraction of the real one, and a "not cited"
  candidate is only *not known to be cited*. Nothing here should inform a
  view about what any opinion actually holds or cites.
- **Measuring authority, influence or precedent strength.** Citation
  counts in this subset reflect what survived the build, not a decision's
  standing. An opinion may be cited to be distinguished or overruled.
- **Training a general citation recommender.** Candidates are drawn from
  the same side of a component-disjoint split, so the pool is far easier
  than retrieval over a full corpus; a model tuned here will not transfer
  to open-set recommendation.
- **Any claim about U.S. case law as a whole.** Only opinions carrying at
  least one internal citation edge can appear, which skews the collection
  toward heavily-citing and frequently-cited decisions from a single
  2022-09-30 snapshot.

## Limitations

- **Not a complete citation graph.** Only edges where *both* endpoints
  survived the build are present — 5,792 of the 28.9 million edges in
  the source table. A candidate marked "not cited" may be cited by the
  query in reality, with the edge simply absent from this subset. Treat
  negatives as "not known to be cited", not as verified non-citations.
- **Citation ≠ endorsement or relevance.** An opinion may cite another to
  distinguish, criticise or overrule it. Models trained here learn "is
  cited by", which is not the same as "is topically similar to" or "is
  good authority for".
- **Redaction removes real signal.** Stripping citations and case names
  is necessary to make the task non-trivial, but it also removes text a
  practising lawyer would use. Performance here is a lower bound on what
  is achievable with unredacted text, and the task is deliberately harder
  than the real-world version of the same question.
- **Skewed toward well-connected opinions.** Only opinions with at least
  one internal citation edge can be queries, which biases the collection
  toward frequently-cited and heavily-citing decisions. It is not a
  representative sample of U.S. case law.
- **One snapshot, one country.** Everything is U.S. law as of
  2022-09-30. Later citing history is absent, and nothing here
  generalises to other legal systems.
- **OCR noise.** The source text comes from scanned reporters and retains
  scanning artifacts — broken words, stray characters, irregular
  whitespace. Control characters were normalised, but the underlying
  transcription errors were not corrected.
- **Test set drawn from few components.** 400 test queries drawn from 3 connected components
  means scores carry meaningful variance, and the component structure
  makes test queries somewhat topically correlated with one another.
- **Truncation.** Opinion text is capped at a fixed character budget, so
  the longest opinions are cut off and a genuinely relevant passage may
  fall outside the retained portion.
