# Design notes — The Reference Shuffle

## The task in one line

Given one real Wikipedia paragraph with its citation markers replaced by
a neutral token, and a shuffled pool of that paragraph's own true
reference cards (title, author, venue, year), recover the order in which
the references were actually cited.

## Why this shape, and what it replaces

The brief asked for a RAG-themed (retrieval-augmented generation)
challenge with genuinely structured ground truth. RAG systems have two
distinguishable failure surfaces: (1) retrieving/selecting the right
supporting passages for a claim, and (2) correctly attributing which part
of a generated answer is grounded by which retrieved passage. An earlier
challenge on this account, "Reconstructing Local Citation Structure in
Case Law", already covers failure surface (1) in a strong, accepted
shape: given a query document and a pool of candidates (some true
citations, some not), select the true subset, scored by MCC over the
full confusion matrix.

Repeating that shape with new data (e.g. "given a Wikipedia paragraph and
a pool of references, some cited and some not, select the cited ones")
would be the same engine on different data -- explicitly the failure mode
task-design.md warns about ("a second challenge built on the same engine
... passed only because domain and ground truth were genuinely
different"). This task is deliberately the OTHER RAG failure surface:
every candidate in the pool IS a true citation (there is no membership
decision at all), and the only withheld quantity is ORDER. That makes the
submitted artifact a permutation, not a subset, and the metric a
rank-correlation statistic, not a confusion-matrix statistic -- different
in 2 of {input shape, output shape, metric} from the case-law challenge:
- input shape: candidate pool is still text + metadata cards, similar in
  spirit, so this one is NOT claimed as different.
- output shape: DIFFERENT. A subset (MCC task) vs. a permutation
  (this task).
- metric: DIFFERENT. Confusion-matrix MCC vs. rank-correlation Kendall's
  tau.

## Closest public benchmarks, and how this differs

- **HotpotQA / Natural Questions**: multi-hop or single-hop QA over
  Wikipedia, where the model must retrieve supporting passages from a
  large corpus and answer a question. This task retrieves nothing (the
  complete candidate pool is handed over) and asks no question -- the
  target is a citation ORDER, not an answer string or a passage
  selection.
- **MS MARCO / BEIR**: passage-ranking benchmarks where a query is
  matched against a large corpus and relevant passages are ranked by
  relevance to the query. This task has no query in the retrieval sense
  -- the "query" and the "candidates" are fixed by the source article
  itself, every candidate is already known to be relevant (all are true
  citations), and what's recovered is the internal ORDER of citation
  within one fixed paragraph, not a relevance ranking against an open
  corpus.
- **Citation-recommendation / citation-intent research** (a large
  scholarly-NLP subfield): typically predicts WHETHER a paper should cite
  another paper, or classifies WHY a citation was made (background,
  comparison, extension). This task assumes every reference is a known,
  confirmed citation and asks only WHERE (in what order) it was placed --
  a question that field does not pose.
- **Sentence-ordering / shuffled-paragraph-reconstruction tasks** (a
  known NLP task family, e.g. ROCStories ordering, discourse coherence
  benchmarks): those tasks shuffle TEXT UNITS (sentences/paragraphs) and
  ask for their original order using only the text units' own content.
  This task instead shuffles REFERENCE METADATA CARDS external to the
  text (bibliographic records, not prose) and keeps the paragraph text
  itself in its original, unshuffled order -- the ordering target is an
  alignment between two different modalities (prose vs. citation
  records), not a reordering of the prose itself.

## Data

Source: English Wikipedia, live via the public MediaWiki Action API
(`en.wikipedia.org/w/api.php`, `action=parse`, `prop=wikitext`), sampled
from `Category:Good_articles` and `Category:Featured_articles`.

This is a different data origin from `dumps.wikimedia.org`'s enwiktionary
bulk dump used by an earlier challenge on this account: a different
Wikimedia project (encyclopedia articles vs. a dictionary), a different
host and endpoint (the live Action API vs. a static bulk-dump mirror),
and no content overlap (dictionary headword entries vs. paragraph-level
encyclopedia prose with citation records).

License: CC BY-SA 4.0, verified live from Wikipedia's own siteinfo API
(`?action=query&meta=siteinfo&siprop=rightsinfo&format=json` ->
`{"rightsinfo": {"text": "Creative Commons Attribution-Share Alike 4.0"}}`).

### Why Good/Featured articles, not a random article sample

A pilot on 40 random category-listed Good articles found citation
patterns vary sharply by how an article's editors format references:
many articles use `{{sfn}}`/`{{harv}}` short-form citations pointing to a
separate Bibliography section, which this task's design cannot use
(the marker-to-reference mapping would require resolving short forms
against a bibliography list, and named-ref reuse within a paragraph
breaks the clean 1:1 mapping this task needs). Restricting to paragraphs
where EVERY citation is a fully-inline `{{cite ...}}` template, with no
within-paragraph reuse, measured a 60% pass rate among paragraphs that
already had >=3 citations in a 72-unit, 40-article pilot sample -- workable,
though it means the shipped dataset is not representative of "a random
Wikipedia paragraph," and this is disclosed in dataset-description.md's
Limitations.

## Fetch measurements

The fetch drew titles from `Category:Good_articles` (paginated via the
API's own `cmcontinue` continuation token, not a fixed page cap) and
attempted 1,900 distinct article titles in one worker, sleeping
0.8-1.5s between every request, over roughly 27 minutes wall-clock.
1,196 usable citation-order units were extracted from 510 of those
1,900 articles (a 26.8% article yield -- most Good Articles either have
too few fully-inline citations in a single paragraph, or rely on
`{{sfn}}`/`{{harv}}` shorthand this task's filter excludes). The
`Category:Featured_articles` batch was not needed: the 1,900-title Good
Articles sample alone cleared the target unit count. No request failed
outright (broad `except Exception` retry logic never exhausted its 3
attempts on any article during this run); output was written
incrementally, one line per extracted unit, so an interrupted run would
have lost at most the article in flight.

Final `n_refs` distribution across all 1,196 raw units: 3->492, 4->302,
5->169, 6->115, 7->71, 8->47 -- monotonically decreasing, as expected
(paragraphs with more distinct fully-inline citations are rarer).

## Split

The shipped package fixes every unit at EXACTLY 3 references -- see
"Post-build rework" below for why this changed from the original design.
`prepare.py` filters the 1,196 raw units down to the 492 with `n_refs ==
3`, then splits by ARTICLE (226 train articles / 70 test articles, at
`TEST_ARTICLE_FRACTION = 0.22`), producing **383 train units** and **109
test units**. Verified byte-identical across two separate runs of
`prepare.py` with the same seed (`SEED = 20260925`). 109 test rows
clears the >=60-row floor with a ~1.8x margin, and the platform's own
public/private row-wise leaderboard slice (splitting `answers.csv`
roughly in half) still leaves each slice comfortably above the
>=10-row-per-slice requirement, since every unit is exactly one row in
`answers.csv` (no graded unit can be fragmented by a row-wise cut).

## Ordering-shuffle and reference-card construction

Each unit's 3 reference cards are shuffled under a fixed seed
(`random.Random(SEED)`, `SEED = 20260925`) and labelled with opaque
letters `A, B, C` that carry no positional information themselves.
Cards are rendered in a fixed field order (title | author | venue |
year) so field POSITION within a card cannot leak which citation type it
is. The true order is encoded as three independent integer columns
(`rank_A`, `rank_B`, `rank_C` -- see "Post-build rework" below), each the
0-indexed true reading position of that label's citation.

## Metric: rescaled Kendall's tau, verified chance floor

Rescaled Kendall's tau -- `(tau_b + 1) / 2` -- was chosen over exact-match
and over per-column accuracy on rank_A/B/C (see problem-description.md
"Why this metric" for the full argument). Its chance floor for a
uniformly random permutation of 3 items was verified by EXHAUSTIVE
enumeration (not sampling) over all 6 permutations: mean rescaled tau =
0.500000, exactly. (The original design also verified this for n = 4..8,
before the pool size was fixed at 3 -- see "Post-build rework" -- and the
floor held exactly at every n tested.)

### Degrade rule: 0.0 for a non-attempt, not the 0.5 chance floor

A missing, malformed, or duplicated row is scored 0.0, deliberately BELOW
the 0.5 an honest random guess would earn in expectation -- matching the
standard "missing/malformed -> 0" rule used by every prior challenge on
this account (e.g. citation-graph-upload scores a missing query 0.0 even
though ITS metric's own informed-guess floor is ~0.14, not 0.0). The gap
between 0.0 (non-attempt) and ~0.5 (honest random guess) is the intended
incentive to attempt every row.

## Baseline ladder (measured on the FINAL shipped test split: 109 units, 70 articles)

| approach | rescaled Kendall's tau |
|---|---|
| blank / non-attempt | 0.0000 |
| pool order (labels A=0, B=1, C=2, unchanged) | 0.4771 |
| adversarial: global venue -> position-fraction, fit on train | 0.4924 |
| random permutation (mean over 5 seeds) | 0.4954 |
| reversed pool order | 0.5229 |
| year-sort ascending | 0.5352 |
| title-length sort | 0.5413 |
| **reference: word overlap (title+venue) + author-surname bonus (w=2.0) + year-consistency bonus (w=0.05), Hungarian assignment** | **0.7248** |
| oracle | 1.0000 |

TF-IDF was deliberately not used anywhere in this ladder or in the
reference solution -- word overlap is a raw set-intersection count, no
inverse-document-frequency weighting.

### Reference solution construction, and what was tried

The reference solution scores each (marker position, candidate card)
pair by: (1) raw word overlap between the paragraph text immediately
preceding that `[CITE]` marker (a growing left-context window, capped at
2000 characters) and the card's title+venue words; (2) a bonus of
`author_w` per author-surname word that appears literally anywhere in
the whole paragraph (author surnames are disproportionately likely to
appear in running prose, e.g. "According to Smith..."); (3) a small
year-consistency bonus rewarding assignments consistent with ascending
publication year. The resulting 3x3 score matrix is solved exactly per
unit with `scipy.optimize.linear_sum_assignment` (Hungarian algorithm),
which guarantees each of the 3 cards is assigned to exactly one marker
position -- unlike a greedy per-marker argmax, which can assign the same
card to two markers and silently break the permutation constraint.

Weights (author bonus, year-consistency bonus) were swept over
`{0, 1, 2, 4, 8} x {0, 0.05, 0.1, 0.2, 0.4}` on `train.csv`'s 383 units
(never against the graded test score), landing on `author_w=2.0,
year_w=0.05` at a train score of 0.7276. Applied unchanged to the test
split, this configuration scores 0.7248 -- the in-sample train score and
the held-out test score are close enough (0.7276 vs 0.7248) to indicate
the weights are not overfit to a particular sample.

## Leak probes run

Every probe from `leak-probes.md`'s table was run on the shipped split:

| Probe | Result | Verdict |
|---|---|---|
| Constant / majority answer | pool order: 0.4771; reversed: 0.5229 | at chance, PASS |
| Item length (title chars) | 0.5413 | at chance, PASS |
| Position in shipped row order / slot letters | pool order 0.4771 vs random 0.4954 -- indistinguishable | PASS |
| Every metadata column shipped (venue) | adversarial global venue-position (train-fit): 0.4924 | at chance, PASS |
| Copy input through unchanged | not applicable -- output is a label permutation, not a rewrite of the input text | n/a |
| Adversarial: ignore task structure (pool across units) | same as the venue-position probe above, which explicitly pools ALL units' (card, true-position) pairs and ignores per-unit paragraph content | 0.4924, PASS (margin 0.232 under reference) |

The margin between the reference (0.7248) and the single strongest probe
(title_length_sort, 0.5413) is 0.1835 -- above both the 0.10 minimum and
the 0.18 target margin named in task-design.md. This is a wider margin
than the original 3-8 range design measured (0.1461), a side effect of
fixing the pool at exactly 3 references (see "Post-build rework" below);
re-verified rather than assumed after the rework.

## Difficulty band

Because this metric's own chance floor is 0.5 rather than 0.0 (an
unavoidable property of any rank-correlation-based ordering metric, and
the exact pattern task-design.md names as the accepted recipe for
ordering/ranking challenges), the "target agent runs 0.2-0.6" guidance is
interpreted here as: every content-free probe should sit close to 0.5,
and the reference should clear it by a measured, disclosed margin while
staying comfortably below 1.0 (oracle) -- the same qualitative shape as
the standard 0-floor band, shifted to this metric's actual chance level.
This departure is stated explicitly here rather than silently assumed.

Measured result: every probe sits in [0.4771, 0.5413], the reference
sits at 0.7248, and oracle sits at 1.0000. The reference is 0.1835 above
the strongest probe and 0.2752 below oracle -- inside a defensible band,
not near either extreme.

## Target Recoverability warning: rare 5-token span overlap (investigated, not a leak)

After the rework below, the platform's Target Recoverability scan
flagged (as a WARNING, not a failure -- its own message states "this is
an overlap warning, not measured target recovery"): "public/test.csv
(ref_A_card): 56/109 inspected examples share a rare 5-token span with
training." Investigated directly against the shipped files rather than
dismissed:

- Exact card-string overlap between train and test is essentially zero:
  of 1,135 distinct train cards and 322 distinct test cards (across
  ref_A/B/C_card), exactly 1 card string is shared, and ZERO test
  `ref_A_card` values match any train card exactly.
- Reproducing the platform's own 5-token-span method directly: 21/109
  test `ref_A_card` rows share a "rare" (<=3 train occurrences) 5-token
  span with a train card. Every example inspected is the VENUE field
  recurring verbatim (e.g. "venue: GrandPrix.com", "venue: Liverpool
  Mercury", "venue: United States Army Reserve") -- the same real-world
  publication is legitimately cited by multiple different Wikipedia
  articles, so the same venue string appears as a candidate card in
  both a train unit and an unrelated test unit. Broadening to any-column
  venue-text match (not just the 5-token-span method) finds 42/109 test
  `ref_A_card` rows whose venue string appears somewhere in train.
- This is card CONTENT recurring, not LABEL leakage: knowing that a
  venue string appears elsewhere in train reveals nothing about that
  specific unit's `(rank_A, rank_B, rank_C)` answer, because rank is a
  property of the SPECIFIC PARAGRAPH's citation order, not of the venue
  identity. This is exactly what `adversarial_global_venue_position` (see
  the ladder above) was built to test: a model fit on train that predicts
  rank purely from venue identity, pooled across all units and blind to
  each unit's own paragraph content, scores 0.4924 -- at chance, 0.2324
  below the reference. If venue recurrence were an exploitable target
  shortcut, this probe would have found it; it does not.
- Not fixed by re-splitting, because there is no available split that
  removes it: venues are a real-world, closed set (a handful of local
  newspapers, sports sites, and government pages disproportionately
  covering same-topic Wikipedia articles), and articles were already
  split so no two PARAGRAPHS share an article -- but two different
  articles legitimately citing the same newspaper is not something a
  train/test article split can or should prevent, since it reflects a
  genuine property of the source corpus, not a construction artifact.
  Suppressing all venue recurrence would require dropping the venue
  field entirely (discarding a legitimate, disclosed input signal used
  by the reference solution) or shrinking the corpus further to force
  every venue to appear on only one side of the split, which was not
  done because the adversarial probe already demonstrates the recurrence
  is not exploitable.

## Post-build rework: two platform rejections and their fixes

The dataset described above (3-8 references, `true_order`/
`predicted_order` string columns) was the FIRST shipped version and
failed two platform checks on submission:

1. **Prepared Data Integrity: "At least one column is missing values in
   half or more profiled rows."** The variable-width `ref_A_card` ..
   `ref_H_card` columns were sparse at high indices by construction --
   measured directly: with `n_refs` ranging 3-8, `ref_D_card` was filled
   in only 58.5% of train rows, `ref_E_card` 34.7%, down to `ref_H_card`
   at 4.2%. This is exactly the Part 4.4 packed-slot failure mode
   `leak-probes.md` warns about. Capping the pool at 5 (`ref_E_card`
   still 17.5% filled) or 4 (`ref_D_card` still 38.0% filled) was
   measured and both still failed the 50% threshold; only fixing every
   unit at EXACTLY 3 references removes the packed-slot problem
   entirely (`ref_A_card`/`B`/`C` are 100% filled by construction, and
   there is no `ref_D_card`+ column to be sparse). This was chosen over
   truncating larger units down to 3 references (which would have kept
   `n_refs` variable while discarding the reference count signal in a
   way that doesn't correspond to any real citation), on the grounds
   that filtering to a fixed count is the simpler, more defensible
   design and n_refs==3 is the single largest bucket in the raw data
   (492 of 1,196), so it costs the least scale of any fixed choice.

2. **Target Recoverability: "the submission and answer columns do not
   identify a supported target."** The original `true_order` /
   `predicted_order` columns encoded the answer as one space-separated
   multi-token string (e.g. "C A B"). The platform's automated target-
   type profiler does not recognise a delimited string as a supported
   target form (see `failures.md`, Target Recoverability: "carry the
   answer in the submission's own form under a clearly-named column").
   Fixed by re-encoding the identical information as three independent
   integer columns, `rank_A`/`rank_B`/`rank_C` (each a plain scalar in
   `{0, 1, 2}`), replacing the one-column string with three one-column
   scalars. `grade.py` was rewritten to compute the same pairwise-
   concordance comparison directly from three columns instead of parsing
   them out of a string; `prepare.py`, `dataset-description.md`,
   `problem-description.md`, `rubrics.md`, `README.md`, both copies of
   `validate_submission.py`, and the reference solution notebook were
   all updated to match, and the full baseline ladder was re-measured
   from scratch on the resulting (smaller, fixed-width) split rather
   than assumed to carry over.

Both fixes were applied together in one rebuild pass (not one platform
resubmission per fix) since both required regenerating `train.csv`/
`test.csv`/`answers.csv` from the same `prepare.py` run; verified after
by re-running `run_checklist.py` (69/69 PASS) and by testing `grade.py`
directly against the oracle (1.0), a blank submission (0.0), and a
missing-column submission (raises `InvalidSubmission`).

A genuine cost of fix (1): the shipped test set shrank from 264 units
(128 articles) to 109 units (70 articles), since only n_refs==3 units
are kept. 109 still clears the >=60-row minimum (see "Split" above), but
with less margin than the original design had.

## What is new here

- The graded artifact is a PERMUTATION of a closed, entirely-relevant
  candidate pool (never a subset, never involving a true/false decision),
  evaluated with a distribution-free, exactly-chance-normalized
  rank-correlation metric -- a distinct task identity from citation
  membership-selection, passage retrieval, or sentence-ordering
  benchmarks (see "Closest public benchmarks" above).
- Ground truth is the archive's own recorded structure: the physical
  position of each `<ref>` marker in Wikipedia's own wikitext, written by
  Wikipedia's editors -- not a human annotation or a model's output,
  applied to real encyclopedia prose.
- The `[CITE]` substitution is a deliberate, measured intervention: it
  removes the one channel (the citation template's own text, sitting
  exactly where the marker was) that would make the task trivially
  solvable by string position rather than by content reasoning.

## Killed candidate ideas

- **Support-selection with distractor references** (same shape as
  citation-graph-upload, new data): killed at the design stage for being
  the same engine on new data, not a genuinely different task identity --
  see "Why this shape" above.
- **Recovering paragraph SENTENCE order** (shuffle sentences, not
  references) from a fixed reference list: this is the well-known
  sentence-ordering / discourse-coherence task family (see "Closest
  public benchmarks"), so it would not clear the originality bar; also
  closer to a text-only NLP task than a RAG-grounding task, since RAG's
  distinguishing feature is the *separate* retrieved-evidence channel.
- **Full PMC/PLOS-style academic citation ordering**: PLOS
  (`api.plos.org`) is already used by a prior challenge on this account
  (subsection ordering); PMC's OA web service endpoints
  (`ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi`) returned HTTP 404 on every
  tested PMCID during source research on 2026-09-25 (endpoint appears to
  have moved/been retired), so this candidate was abandoned at the Gate
  0.5 reachability check before any data was fetched.
