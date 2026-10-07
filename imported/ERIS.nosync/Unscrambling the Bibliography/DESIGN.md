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

The shipped package fixes every unit at EXACTLY 4 references (v3; v1 and
v2 used 3) -- see
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


## v2 metric revision after platform review (2026-10-02)

The platform reviewer returned the first upload with three notes: remove
the data-path discovery snippet from the problem description, add a
"what not to use" section, and "the agents' scores are high, please
modify it a little bit."

The high scores are a property of the v1 metric, not of the data.
Rescaled Kendall's tau over three cards has a chance floor of exactly
0.5, so the bottom half of the scale is unreachable by any honest
submission and every real submission is compressed into the top half.
The same thing happened on this account's subsection-ordering challenge,
where the fix that passed review was to replace tau with position
accuracy and leave the data alone. The same fix is applied here.

**v2 metric: mean per-unit position accuracy.** For each unit, the
fraction of its three cards whose submitted position equals the true
position; a non-permutation triple scores 0. Chance is exactly 1/3 (each
card is right with probability 1/3). It is also the more direct target:
the task asks which marker each card belongs to, and this scores exactly
that, card by card, rather than pairwise order relations -- the
"direct, unambiguous target" shape rather than an after-the-fact
alignment.

The data, the split, `prepare.py` and every public file are unchanged;
`sample_submission.csv` is byte-identical. Only `grade.py` and the
documents change.

Every rung re-measured under both metrics on the shipped test split
(109 units, 70 articles). Reference weights were re-tuned on train
under the v2 metric.

| rung | v1 tau | v2 position accuracy |
|---|---|---|
| blank / non-attempt | 0.0000 | 0.0000 |
| author-surname bonus alone, no overlap | 0.4740 | 0.3150 |
| pool order | 0.4771 | 0.3180 |
| random permutation (mean of 5 seeds) | 0.4954 | 0.3223 |
| adversarial: global venue-position, fit on train | 0.4924 | 0.3272 |
| reversed pool order | 0.5229 | 0.3456 |
| title-length sort | 0.5413 | 0.3639 |
| year-sort ascending | 0.5352 | 0.3761 |
| word overlap only, Hungarian | 0.7034 | 0.5994 |
| reference, v2 weights (author 0, year 0.4) | 0.7187 | **0.6024** |
| oracle | 1.0000 | 1.0000 |

Three things the re-measurement showed:

- **The sweep drops the author-surname bonus.** Under v1 the chosen
  weights were (author 2.0, year 0.05). Under position accuracy the
  train sweep lands on (author 0, year 0.4): the author term is at
  chance on its own (0.3150) and adds nothing to overlap. It is reported
  as its own rung rather than silently removed. Under the v1 metric the
  v2 weights score 0.7187 against 0.7248 for the v1 weights, so the two
  configurations are within 0.006 of each other on the old scale.
- **The margin grows in the metric's own units.** Reference minus the
  strongest content-free probe is 0.226 under v2 (0.6024 - 0.3761)
  against 0.184 under v1. Headroom to the oracle is 0.398.
- **Per-unit scores take three values only.** On the 109 test units the
  reference scores 1.0 on 53, 0.333 on 38 and 0.0 on 18. A permutation
  of three admits no other value, so this granularity is intrinsic and
  is disclosed in `config.yaml` rather than smoothed. A row-wise
  half-split of `answers.csv` scores the reference 0.5926 and 0.6121,
  and pool order 0.3333 and 0.3030, so leaderboard slices preserve the
  ordering.

The grader's degrade behaviour was re-tested on 22 damaged submissions
(repeated rank, out of range, 1.5, string junk, blank, None, NaN,
missing unit, duplicated unit both right and one wrong, unknown id,
whitespace in ids, float and string dtypes, extra column, shuffled rows,
all zeros, global swap, cyclic shift, boolean, header only); every one
returned a number in [0, 1], and only a missing column raised
`InvalidSubmission`, a `ValueError`.

The v1 anchors are retained in `config.yaml` under `v1_tau_*` names.

## v3 data revision after agent-score review (2026-10-02)

The v2 metric change (position accuracy, three cards) was uploaded and the
reviewer's agent runs still came back above the intended band: the user
asked for agents to land between 0.4 and 0.7. On three cards the word
overlap reference already scored 0.60 and two correct placements force
the third, so the metric alone could not move the band far enough. The
data was made harder instead, with no change to the task shape, the card
format, the split rule or the grader logic.

**Change.** `N_REFS` in `prepare.py` goes from 3 to 4. Every shipped unit
now has exactly four cards (`ref_A_card`..`ref_D_card`) and four answer
columns (`rank_A`..`rank_D`, a permutation of `{0, 1, 2, 3}`). The raw
file is unchanged; the filter simply keeps the 302 four-reference units
instead of the 492 three-reference units. Same seed, same article-level
split rule, `TEST_ARTICLE_FRACTION` unchanged at 0.22.

**Measured alternatives.** All three pool sizes were built from the same
raw file and scored with the same ladder code before choosing:

| Pool size | Train / test | Chance | Word overlap | Reference | Strongest probe |
|---|---|---|---|---|---|
| 3 (v2) | 383 / 109 | 0.333 | 0.5994 | 0.6024 | 0.3761 (year sort) |
| **4 (v3)** | **234 / 68** | **0.250** | **0.5331** | **0.5404** | **0.2978 (year sort)** |
| 5 | 131 / 38 | 0.200 | 0.4789 | 0.5053 | 0.2474 (year sort) |

Five cards was rejected because 38 test rows is below the 60-row floor.
Four cards keeps 68 test rows (above the floor, below the 200 preferred)
and 234 train rows, and lowers the reference by 0.062 and the chance
floor by 0.083 relative to v2. The per-unit search space grows from 6
orders to 24.

**Grader.** `grade.py` is the same position-accuracy grader generalised
to `RANK_COLS = [rank_A, rank_B, rank_C, rank_D]`; the permutation check
is against `{0, 1, 2, 3}`. Chance floor verified by enumerating all 24
permutations: 1 scores 1.0, 6 score 0.5, 8 score 0.25, 9 score 0.0; mean
exactly 0.25. Exactly three right is impossible (three fixed points force
the fourth).

**Ladder on the shipped v3 split** (68 test units, 50 articles; shipped
`grade.py`):

| Rung | Score |
|---|---|
| blank / non-attempt | 0.0000 |
| pool order | 0.2206 |
| author-surname bonus alone, no overlap | 0.2426 |
| random permutation (mean of 5 seeds) | 0.2463 |
| sample submission (rotates through the 24 permutations) | 0.2500 |
| reversed pool order | 0.2868 |
| title-length sort | 0.2904 |
| adversarial: global venue-position, fit on train | 0.2941 |
| year-sort ascending | 0.2978 |
| word overlap only, Hungarian | 0.5331 |
| reference, weights (author 8, year 0.2) swept on train | **0.5404** |
| oracle | 1.0000 |

- Margin over the strongest content-free probe: 0.5404 - 0.2978 = 0.2426.
- Reference weights re-swept on the 234 train units over author
  {0,1,2,4,8} x year {0,0.05,0.1,0.2,0.4}: best (8, 0.2), train score
  0.5780, test 0.5404. On four cards the author-surname term helps
  slightly (overlap alone 0.5331), unlike on three cards where the sweep
  zeroed it; the notebook keeps the same sweep and reports whatever it
  lands on.
- Per-unit reference scores: 22 at 1.0, 24 at 0.5, 11 at 0.25, 11 at 0.0.
- Row-wise half-split of answers.csv: reference 0.5515 / 0.5294, pool
  order 0.2353 / 0.2059.
- Venue recurrence (Target Recoverability note above, re-measured): 37 of
  68 test `ref_A_card` venues also appear in some train card; the
  venue-only adversary scores 0.2941, within 0.05 of chance.

**Expected agent band.** Agents on the v2 three-card data were reported
above 0.7. The reference drops by 0.062 and the floor by 0.083; an agent
that beat the v2 reference by the same relative margin would land near
0.60 to 0.68 here, inside the requested 0.4 to 0.7 band. This is an
extrapolation from the reference, not a measurement of agent runs.

**Files changed for this revision.** `prepare.py` (N_REFS and docstring),
`grade.py`, `problem-description.md`, `dataset-description.md`,
`rubrics.md`, `config.yaml`, `README.md`, `solution.ipynb`,
`run_checklist.py`, this file. The raw zip is unchanged.

## v4: one recorded field withheld per card, after novelty review (2026-10-02)

The novelty gate scored the four-card revision 4/10 (threshold 5). The
assessment named SCiFi (ACL Findings 2024) as the closest work -- it also
links in-text citation markers to bibliographic sources on real paragraphs
-- and said a higher tier needed "a materially distinct evaluation harness,
such as variable pool sizes, partial field masking, or a multi-step
reasoning objective, that composes primitives not already present in the
citation-grounding literature."

**Lever chosen: partial field masking.** Variable pool sizes were ruled
out by the fixed-width schema: mixing 3- and 4-reference units leaves
`ref_D_card` empty in 62% of rows, and mixing 4 and 5 leaves `ref_E_card`
empty in 64%, both failing the "no column empty in half or more rows"
check that forced the fixed pool in the first place. Masking keeps the
schema, keeps the grader, and adds a second step to every unit (place the
identifiable cards, then deduce the rest).

**Rule.** On every card, exactly one field the editors actually recorded
is replaced by `[withheld]`. The field is drawn per card with the same
seeded RNG as the shuffle, uniformly among recorded fields, independently
of the card's true position. A never-recorded field keeps its placeholder
and is never the withheld one, so no card gains a second gap. The withheld
value is written to no public or private file. Train is withheld the same
way as test.

**Test-card profile** (272 cards): title withheld on 88, year on 74, venue
on 68, author on 42. Visible recorded fields per card: 3 on 131, 2 on 130,
1 on 11. The 11 single-field cards (typically a title-and-year card that
lost its title) are placeable only by elimination; this is disclosed in
`dataset-description.md` as a limitation rather than smoothed away.

**Ladder on the shipped split** (68 test units, 50 articles; shipped
`grade.py`):

| Rung | Score |
|---|---|
| blank / non-attempt | 0.0000 |
| reversed pool order | 0.1691 |
| year-sort ascending (visible years) | 0.2463 |
| random permutation (mean of 5 seeds) | 0.2574 |
| title-withheld cards placed last | 0.2574 |
| sample submission (rotates through the 24 permutations) | 0.2647 |
| author-surname bonus alone, no overlap | 0.2831 |
| pool order | 0.2941 |
| title-withheld cards placed first | 0.3015 |
| adversarial: global venue-position, fit on train | 0.3088 |
| title-length sort (visible titles) | 0.3125 |
| word overlap only (visible title + venue), Hungarian | 0.4706 |
| reference, weights (author 0, year 0.4) swept on train | **0.5184** |
| oracle | 1.0000 |

- Margin over the strongest content-free probe: 0.5184 - 0.3125 = 0.2059.
- Probe spread: with 68 units the standard error of a random
  permutation's mean is 0.25 / sqrt(68) = 0.03, so a seeded shuffle's
  pool-order probes sitting at 0.17 and 0.29 is within three standard
  errors on either side of 0.25 and is reported as such, not trimmed.
- The two withheld-slot probes bracket the floor (0.2574, 0.3015): the
  slot carries no order information, as the construction guarantees.
- Reference weights re-swept on the 234 train units: best (0, 0.4),
  train 0.4594, test 0.5184. Test above train is a 68-unit sample effect;
  both numbers are reported.
- Per-unit reference scores: 21 at 1.0, 20 at 0.5, 17 at 0.25, 10 at 0.0.
- Row-wise half-split of answers.csv: reference 0.5294 / 0.5074, pool
  order 0.3235 / 0.2647.
- Venue recurrence re-measured on visible venues: 22 of 68 test
  `ref_A_card` venues also appear in some train card; the venue-only
  adversary scores 0.3088.
- Effect of masking alone (same split, same grader): word-overlap-only
  0.5404 -> 0.4706, reference 0.5404 -> 0.5184.

**Prior-work section.** `problem-description.md` now names SCiFi
directly, concedes the shared primitive (the marker-to-source link on
real paragraphs) and states what this harness isolates that a
retrieval-and-generation harness does not: alignment of a known,
complete, distractor-free but incomplete reference set to prose, scored
as a permutation against an exact floor. Distant antecedents (sentence
ordering, context-aware citation recommendation, citation-context
classification) are kept.

**Files changed.** `prepare.py` (card renderer, withheld draw, stats),
`problem-description.md`, `dataset-description.md`, `rubrics.md`,
`config.yaml`, `README.md`, `solution.ipynb`, `run_checklist.py`, this
file. `grade.py` unchanged. The raw zip is unchanged.

## v5: larger test set and trimmed description, after review (2026-10-06)

Reviewer (nxank4, 2026-10-05): 68 test units with a 17-unit public board
(sd about 0.063) could not separate agent runs (0.515, 0.515, 0.588) from
a model-free word-overlap rule (0.500). Asked for several hundred test
units with train larger than test, and for the description to drop the
baseline list, the reference recipe, the novelty / prior-work /
adjacent-task sections, the design defence in Evaluation, the source
name in "What not to use", `dataset_stats.json`, the Docker/notebook
rules and the grading-slice line.

Data change: every raw paragraph with 4 to 8 citations is now used (704
units instead of 302). A paragraph with more than 4 keeps one seeded
window of 4 consecutive citations; markers outside the window are
deleted from its text. Pool size, schema, metric and grade.py logic are
unchanged. TEST_ARTICLE_FRACTION 0.22 -> 0.45. Split: 385 train / 319
test units (205 / 163 articles). Public 25% slice: 80 units, sd of the
reference score 0.033.

Ladder (shipped grade.py, ~/eris-build/refshuffle/ladder6.py): every
content-free probe 0.238-0.263; year sort 0.3033 (strongest); word
overlap only 0.4843; reference 0.4914 (author 0, year 0.2; train 0.494);
margin 0.188. Half-split: reference 0.478 / 0.505, pool order 0.258 /
0.245.
