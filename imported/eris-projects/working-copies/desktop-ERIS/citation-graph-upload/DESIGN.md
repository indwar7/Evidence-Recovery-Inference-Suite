# Design notes — Reconstructing Local Citation Structure in Case Law

## The task in one line

Given one real U.S. court opinion and a pool of twelve candidate
opinions, all with every citation marker and case name redacted, decide
which candidates the query opinion cites.

## Why this shape

A prior challenge on this account ("Recovering OCR Batch Origin from
Character Noise") was a partition-recovery / clustering task. This
challenge is deliberately a different task identity per the platform's
own rejection register (section 2.2): the submitted artifact here is a
**subset of a given candidate pool**, selected per query, with the
subset size unknown and variable. It is not a partition into unlabeled
groups, not a ranked retrieval list against an external corpus (the
candidate pool is handed over in the same CSV row), and not a
classification into a fixed label set (the same candidate text can be
cited by one query and not another).

**This shape replaced an earlier bag/all-pairs design.** The original
build presented a "bag" of opinions and asked for the directed edge set
among all pairs within it. That design was abandoned after a measured
positional leak — see "The positional leak that forced a redesign"
below. The current shape was chosen specifically because it makes the
candidate ordering meaningless by construction, which the earlier one
did not.

## Data

Source: Free Law Project's CourtListener bulk legal data
(https://www.courtlistener.com/help/api/bulk-data/), snapshot date
2022-09-30 — chosen as the earliest snapshot where citation-map,
opinion-clusters, opinions, dockets, and courts are all simultaneously
populated (citation-map's 2022-08-03 and 2022-08-31 snapshots are empty
placeholders, discovered by direct inspection before committing to this
snapshot date).

License: Public Domain Mark, quoted verbatim from the archive's own
bulk-data documentation page: "Our bulk data files are free of known
copyright restrictions." Independently, U.S. judicial opinions are
government-edict material and not copyrightable in the first place
(Banks v. Manchester, 128 U.S. 244 (1888); Georgia v.
Public.Resource.Org, 590 U.S. 255 (2020)).

### Measured: the naive pre-filter strategy failed, the citation-graph
### filter worked

The first build attempt pre-filtered to the 13 federal circuit courts
of appeals via a docket->court join BEFORE looking at the opinions
table, on the theory that a bounded jurisdiction family would be a
natural, dense-citation scope. Measured on a 1GB test read of the
opinions table (752,236 raw lines, 8,698 well-formed rows after
correcting a CSV multi-line-field parsing bug): this pre-filter matched
**1 opinion out of 8,698** (0.003% hit rate) against a set of 12,189
target clusters, because opinions.csv is not ordered by any join key
correlated with court identity, and 4M+ total opinion rows vastly
outnumber the pre-filtered scope even in a random uniform sample.

Filtering instead to opinions whose id appears anywhere in the
citation-map table (either as citing_opinion_id or cited_opinion_id —
3,524,995 distinct ids across 28,900,123 total citation edges in the
full table) matched **657 opinions out of 8,698** in the same 1GB test
read (7.6% hit rate) — over 2,500x denser, because citation-map only
records edges between opinions the archive has already indexed with
extractable text, which correlates far more with "present in this table
at all" than court identity does. The final build uses this
citation-graph-membership filter.

### Measured: a naive line-based streaming parser silently corrupts rows

An early version of the streaming fetcher split decompressed bz2 output
on literal `\n` bytes before CSV-parsing each line. This undercounts
real rows whenever a `plain_text` field contains an embedded newline
(real opinion text routinely does, from paragraph breaks in the source).
Verified directly: `csv.DictReader` over the same decompressed data with
Python's stdlib multi-line-aware parsing produced 8,699 well-formed rows
against the naive line-splitter's corrupted/undercounted output on the
same input. Fixed by decompressing to a real file and parsing it with
`csv.DictReader(f, newline="")` rather than manual line-splitting.

### Measured: parallel byte-range downloads meaningfully sped up the build

A single serial download measured ~0.86 MB/s on this network connection
(50MB in 57.9s). Four concurrent 50MB byte-range requests, concatenated
in order, measured ~2.9 MB/s aggregate (200MB in 68.1s) — a ~3.4x
speedup — verified to decompress correctly as a single continuous bz2
stream (concatenating in-order byte ranges is byte-identical to a serial
download of the same range). The final `build_raw.py` /
`join_corpus.py` fetch pipeline uses 8-worker parallel range downloads
for the opinions prefix and 5-worker for the full citation-map table.

## Measured on the shipped build

### Corpus scale

- 44,291 opinions matched (participate in the citation graph), from a
  bounded ~2GB prefix of the opinions table (the full table is ~30GB;
  reading further would find more, but this scope was sufficient).
- 5,792 internal citation edges (both endpoints among the kept
  opinions), out of 28,900,122 total edges in the full citation-map
  table.
- 4,356 of the 44,291 opinions participate in at least one internal
  edge; 520 connected components (one giant component of 1,778 nodes,
  one of 264, 57 more of 8-54 nodes, the rest smaller).
- 4,355 of those opinions' clusters (date_filed, case_name, docket_id)
  were resolved by streaming the full opinion-clusters table (1 cluster
  unresolved — a negligible data-quality edge case in the source).

### Split

The original design held out entire federal circuit courts of appeals
from train.csv (register 4's "split by group, never by row"). Building
that split required joining through the dockets table (~3.4GB
compressed), which measured far too slow for this build's time budget (a
narrow streaming scan for only the ~4,355 needed docket ids found just 7
matches in the first 1,000,000 rows, so the dockets for these opinions
are not concentrated early in the table the way the clusters were).

The split was therefore made COMPONENT-DISJOINT instead of
court-disjoint: connected components of the citation subgraph are
assigned whole to train or test, so no citation edge crosses the split
and a solver cannot look up a labelled citing pair directly. This is a
weaker generalisation test than a true court-disjoint split — a test
component could still share surface stylistic similarity with a train
component from the same court — and is disclosed rather than silently
substituted.

Final split: 524 components, 521 train / 3 test. 1,954 train queries, 400
test queries, 12 candidates each.

### Query and candidate-pool construction

For each query opinion, its true cited opinions (1 to 4, mean 1.972 on
the shipped test set) are placed in a pool with distractors drawn from
the same side of the split, to a fixed pool size of 12. Pool order is
randomised per query under a fixed seed.

The pool size was swept before being fixed at 12, trading the random
floor against the achievable ceiling:

| pool size | random floor | pre-trained top-2 |
|-----------|--------------|-------------------|
| 8         | 0.1838       | 0.6834            |
| **12**    | **0.1376**   | **0.6600**        |
| 16        | 0.1107       | 0.6112            |
| 20        | 0.0768       | 0.6239            |

12 was chosen as the point where the random floor is comfortably below
the 0.2 band floor while the ceiling stays high.

### The positional leak that forced a redesign

The original bag/all-pairs shape was **abandoned after measurement**, not
on suspicion. Content-free probes on that build scored:

| probe                      | MCC    |
|----------------------------|--------|
| bag position only          | 0.1434 |
| text length only           | 0.0964 |
| random at true density     | 0.0829 |
| reference solution         | 0.0980 |

A position-only probe scoring **above** the reference solution is a
design failure severe enough to abandon the source (playbook Part 1.3).
A control confirmed the signal was genuine rather than noise: a random
half of pairs scored 0.082 against the positional half's 0.143.

The query/candidate shape eliminates this by construction, and the fix
is verified on the shipped split:

- Positive rate across the twelve positions: 0.128 to 0.205, mean 0.165.
- `always_first_two` scores **0.1253** against `random_two`'s **0.1257**
  — indistinguishable, i.e. position buys nothing.

### Redaction leak scan

**0 surviving reporter citations and 0 surviving "X v. Y" patterns**
across every shipped text field, in both train.csv and test.csv.

Two bugs were found and fixed here, both by measurement:

1. **Whitespace-split citations survived.** The reporter pattern used
   `\s?` between the volume, reporter and page, but the OCR'd source
   breaks citations across page boundaries, e.g. `80 L. Ed.        2 d
   674`. Eight such fragments survived into an earlier shipped build.
   Widened to `\s+`/`\s*`; re-scanned to 0.
2. **The leak scan itself reported 7 phantom hits.** It joined every
   text field into one string before scanning, letting the "X v. Y"
   pattern match across the seam between two documents (one text ending
   in `...State v.` followed by the next). Per-field scanning finds zero.
   The scan now runs per field. This changed only the reported count —
   the shipped CSV checksums were unaffected.

### Ablation ladder (mean per-query MCC, all 400 test queries)

| approach                                | MCC    |
|-----------------------------------------|--------|
| select none                             | 0.0000 |
| select all twelve                       | 0.0000 |
| two shortest candidates                 | 0.0963 |
| two longest candidates                  | 0.1262 |
| random three per query (5-run mean)     | 0.1306 |
| always positions 0 and 1                | 0.1253 |
| random two per query (5-run mean)       | 0.1257 |
| pre-trained embedding, fixed top-2      | 0.5146 |
| truncated embedding, per-query z>=1.0   | 0.6461 |
| windowed embedding, fixed top-2         | 0.5517 |
| windowed embedding, per-query z>=1.2    | 0.6547 |
| word-overlap top-2                      | 0.6174 |
| **reference: blend w=0.7, z>=1.0**      | **0.6726** |
| oracle                                  | 1.0000 |

TF-IDF was deliberately not used anywhere in this ladder or in the
reference solution.

### Difficulty band

The target agent-score band is 0.2-0.7. The reference solution lands
inside it, and every content-free baseline sits at or below 0.131 —
comfortably under the 0.2 floor, so the band is earned by reading the
text rather than by exploiting structure.

### Finding: on the enlarged split the pre-trained encoder clears lexical

The first build (180 test queries, MIN_TRUE_CITES=2) found word-overlap
**0.6305** beating every single-signal embedding configuration, with the
blend clearing it only narrowly at 0.6348. **That conclusion did not
survive the enlargement.** Re-measured on the shipped 400-query split:

- word-overlap top-2 — **0.6174** (top-1 0.5935, top-3 0.5611)
- windowed embedding, per-query z>=1.2 — **0.6547**
- blend w=0.7, z>=1.0 — **0.6726** (the reference)

The embedding alone now beats the best lexical rule by ~0.037, and the
blend adds a further ~0.018. The earlier narrow-margin finding was an
artefact of the smaller, higher-cite-count sample and is corrected here
rather than left standing.

**Truncation matters less than the first build concluded.** Re-measured:
`all-MiniLM-L6-v2` truncated to its 256-token window scores **0.6461**
at z>=1.0, against **0.6510** for the same model over ~230-token sliding
windows with max-pooling. That is a real but small gain, not the
0.5949 -> 0.6211 jump the first build reported. The larger lever is the
decision rule: a fixed top-2 cut-off on the same windowed scores falls
to 0.5517.

Two further findings from the same sweep:

- **The decision rule matters more than the representation.** Re-measured
  on the shipped 400-query split: fixed top-2 caps out at 0.5517 windowed
  (top-1 0.5810), while z-thresholding the same scores reaches 0.6547.
  That ~0.10 gap is larger than anything gained from windowing or from
  switching between lexical and embedding features.
- From the first build, not re-measured on this split: absolute cosine
  cut-offs collapsed entirely (0.0057 to 0.1643 across thresholds
  0.35-0.65), and max-over-windows beat mean-over-windows (0.6211 vs
  0.6129). Both are consistent with MiniLM's raw similarities being
  uncalibrated across these texts and with a citation turning on one
  passage rather than a document's average topic.

Word overlap remains a strong baseline (0.6174) because citing opinions
quote the cited text closely, and this is disclosed rather than smoothed
over — the reference's margin over it is ~0.055, not a landslide. A long-context encoder that reads all
1,500 tokens natively, or an encoder fine-tuned on train.csv's 300
labelled queries, are the obvious directions above the reference —
neither was attempted in this build. An attempt to measure `bge-m3`
(8192-token window) was abandoned after repeated download failures of
its 2.2GB weights.

### Known limitations of this build, disclosed

- The test set is 180 queries drawn from only 3 connected components, so
  scores carry meaningful variance and test queries are somewhat
  topically correlated.
- The component-disjoint split is a weaker generalisation test than the
  court-disjoint split originally intended (see Split above).
- Negatives are "not known to be cited", not verified non-citations:
  only 5,792 of the source's 28.9M edges have both endpoints in this
  build, so a distractor may genuinely be cited with the edge absent.
- Opinion text is truncated to a fixed character budget, so a relevant
  passage can fall outside the retained portion.
- No transductive/pooled attack was explicitly scripted against this
  shape (register 2A). The per-query independence of the metric and the
  component-disjoint split make one structurally hard to construct, but
  this reasoning is not backed by an actual attack script.
- Date-ordering was measured on the earlier bag shape (a citing opinion
  post-dates its cited opinion 99.4% of the time) and scored only 0.039
  alone. Dates are deliberately not shipped in train.csv/test.csv.


## What is new here

- The graded artifact is a variable-size SUBSET of a closed candidate
  pool, selected from content-redacted text and evaluated with a
  chance-corrected metric (MCC over the full candidate pool, including
  true negatives) rather than a partition, label, or ranked list — a
  distinct task identity from prior citation- or reference-based NLP
  benchmarks, which are typically framed as citation-string extraction,
  citation-recommendation/retrieval, or document classification.
  Deciding how many candidates to accept is part of the task: both
  degenerate answers score exactly 0.000.
- Ground truth is a real, independently-extracted citation-graph record
  (CourtListener's own citation-parsing output), not a human annotation
  or a model's output, applied to real court opinions rather than a
  constructed or synthetic corpus.
- The redaction step is a deliberate, measured intervention (not an
  assumption): both formal citation strings and every case name in the
  raw corpus are masked, closing the two distinct channels a solver
  could otherwise exploit to solve the task via string-matching instead
  of recognizing legal/topical dependency.

## How this differs from adjacent prior work (distant antecedents only)

- Citation-recommendation / citation-prediction research (a large,
  active area in scholarly-document NLP) is the nearest conceptual
  neighbor, but is typically framed as retrieval against a large corpus
  or as extending a document's OWN bibliography, not as recovering a
  small closed graph's internal edge structure from redacted text with
  no external candidate pool.
- Legal NLP benchmarks built on case-law text (broadly, "LegalBench"-
  style efforts) generally test classification, extraction, or judgment
  prediction over case text; this task's target — which members of a
  closed candidate pool a given opinion cites, with the count unknown —
  is structurally distinct from any single-document label or extraction
  target.
- Knowledge-graph link-prediction work (a large separate ML subfield)
  typically operates over structured triples or embeddings already
  extracted from text, not over raw natural-language documents with the
  explicit graph-indicating tokens deliberately removed.
