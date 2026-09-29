# Wikipedia Paragraphs Paired With Their Own Shuffled Reference Cards

Real paragraphs from English Wikipedia's "Good article" and "Featured
article" pages (Wikipedia's own quality-review categories), packaged as a
citation-order-recovery task: one paragraph with its inline citation
markers replaced by a neutral token, a shuffled pool of that paragraph's
own real reference cards, and the question of which order they were truly
cited in.

**Quality-reviewed source pool.** Good and Featured articles have passed
a structured Wikipedia community review that, among other things, checks
citation quality — this correlates strongly with fully-templated
`{{cite ...}}` citations rather than ad hoc inline text, which is what
this task's extraction filter requires. The pool is not restricted to any
one topic area; sampled articles span history, science, sport,
meteorology, biography, and more.

## Provenance

| | |
|---|---|
| Source | [English Wikipedia](https://en.wikipedia.org/) article corpus, as mirrored at `dumps.wikimedia.org/enwiki` |
| Declared source URL | `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2` |
| Article pool | `Category:Good_articles` and `Category:Featured_articles` |
| Fetch mechanism | public MediaWiki Action API (`action=parse`, `prop=wikitext` at `https://en.wikipedia.org/w/api.php`), serving the same live article wikitext as the bulk dump |
| Licence | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) |
| Retrieved | 2026-09-25 |

Verified live from Wikipedia's own MediaWiki siteinfo API at build time:

```
https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json

-> {"rightsinfo": {"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
                    "text": "Creative Commons Attribution-Share Alike 4.0"}}
```

CC BY-SA 4.0 requires attribution to Wikipedia and its contributors, and
requires that this package (and any derivative of it) remain under a
compatible share-alike licence — which this package satisfies by also
being released under CC BY-SA 4.0.

## How the task data was built

1. Article titles were pulled from Wikipedia's own `Category:Good_articles`
   and `Category:Featured_articles` listings via `list=categorymembers`,
   paginated with the API's own continuation token.
2. Each article's raw wikitext was fetched via `action=parse`.
3. The wikitext was split into paragraphs. A paragraph was kept only if
   it contained EXACTLY 3 `<ref>` markers where **every** marker is a
   distinct, fully-inline `{{cite journal|book|web|news|magazine|
   encyclopedia}}` template — no `<ref name="x"/>` reuse within the
   paragraph, no `{{sfn}}`/`{{harv}}` shorthand, no bare citation text.
   This is what keeps the marker-to-reference mapping unambiguous: each
   `[CITE]` token in the shipped text corresponds to exactly one of the
   paragraph's own reference cards. Fixing the count at exactly 3 (rather
   than shipping a variable 3-8 range packed into extra columns) is a
   fix applied after an earlier iteration's variable-width columns failed
   the platform's "no column missing values in half or more profiled
   rows" check — see Limitations.
4. Each kept citation template was parsed into structured fields (title,
   author, venue, year) — the article editors' own recorded values.
5. Each `<ref>` marker was replaced with the token `[CITE]`, and the
   surrounding wikitext markup (templates, wikilinks, bold/italic,
   residual HTML) was stripped to plain prose.
6. Units were split by **source article** into train/test, so no two
   paragraphs from the same page appear on both sides.
7. For each unit, its references were shuffled under a fixed seed and
   assigned opaque letters (A, B, C, ...); the letter sequence
   corresponding to the true, original left-to-right citation order was
   recorded as the answer.

8. `prepare.py` then removes any markup residue that survived extraction
   from the shipped text and cards (image-caption fragments, template
   braces, italic quote marks, HTML entities), so nothing in the shipped
   files identifies the source site. Text is capped at 4,000 characters,
   but the cap never cuts before the last `[CITE]` marker, and every shipped
   row is asserted to contain exactly 3 markers.

`prepare.py` reproduces steps 6-8 end to end from the raw fetched units
and is deterministic — two runs produce byte-identical outputs.

## Files

### `dataset/public/train.csv` — 383 rows
`unit_id`, `text` (paragraph with `[CITE]` markers), `ref_A_card`,
`ref_B_card`, `ref_C_card`, and the answer as three integer columns
`rank_A`, `rank_B`, `rank_C`.

Each `rank_X` column is the 0-indexed true reading-order position of
that lettered card's citation. Together the three columns for a row are
always a permutation of `{0, 1, 2}`.

### `dataset/public/test.csv` — 109 rows
Identical minus `rank_A`, `rank_B`, `rank_C`.

### `dataset/public/sample_submission.csv` — 109 rows
`unit_id`, `rank_A`, `rank_B`, `rank_C`. Format example only.

### `dataset/public/dataset_stats.json`
Split counts and article counts.

### `dataset/private/answers.csv` — 109 rows
`unit_id`, `rank_A`, `rank_B`, `rank_C`. The answer key. None of
`rank_A`/`rank_B`/`rank_C` appear in `test.csv` (test.csv carries only
the shuffled cards, never their true rank), so there is no shared-name
value between the two files to keep distinct — `test.csv` and
`answers.csv` share only the `unit_id` join key.

### `dataset/raw/`
The source material the task was built from: `wiki_citation_units.jsonl`
(one JSON object per extracted paragraph unit, before the train/test
split and before card-shuffling), plus `_fetched_titles.json`,
`build_meta.json`, `meta.json`, and `ATTRIBUTION.txt`.

## Label leakage and what was done about it

The obvious shortcut — leave a citation marker's adjacent template text
in place and read it directly — is closed structurally: every `<ref>`
marker is replaced with the identical `[CITE]` token before the paragraph
is shipped, so no positional trace of which template used to sit where
survives in the text itself.

Further shortcuts measured on the shipped test split (109 units) and
found closed (all scored under rescaled Kendall's tau, exact chance =
0.5000):

- **Pool order carries no signal.** Leaving the shuffled cards in their
  given order (`rank_A=0, rank_B=1, rank_C=2`) scores 0.4771; the reverse
  order scores 0.5229 — both indistinguishable from a random permutation
  (0.4954, mean over 5 seeds).
- **Card length carries almost no signal.** Sorting candidates by title
  string length scores 0.5413 — the strongest content-free probe
  measured, still only mildly above chance.
- **No cross-unit venue shortcut.** A model fit on `train.csv` that
  learns, globally across every paragraph, which venues tend to be cited
  earlier or later in a paragraph (ignoring each unit's own content
  entirely) scores 0.4924 on test — at chance, confirming a solver cannot
  substitute a pooled, content-free venue prior for actually reading the
  paragraph.
- **Publication year carries real, but modest, signal — and is
  disclosed, not hidden.** Sorting candidates by ascending year scores
  0.5352. This is a genuine (if weak) pattern in how encyclopedic
  narrative prose is written and cited, not an artifact of how this
  dataset was built. It is not removed, because `year` is a legitimately
  shipped field a solver is meant to be able to use; the reference
  solution incorporates it explicitly (see `problem-description.md`,
  Evaluation) and still clears every probe by at least 0.183.

## Intended use

- Training and evaluating models that align short structured records
  (bibliographic metadata) to positions in a passage of prose based on
  content, not position.
- Benchmarking whether a system can perform the "grounding" half of a
  retrieval-augmented pipeline in reverse: given a passage and its true
  supporting sources, work out which source backs which part of the
  passage — a structurally different probe from ordinary retrieval
  (nothing is retrieved; everything relevant is already given) or
  ordinary grounding/attribution (nothing is selected; every candidate is
  already known to be relevant).
- Studying rank-correlation evaluation under an exact, distribution-free
  chance floor (0.5), independent of how many items are being ordered.
- Research on how citation practices (which reference types are used
  where in a paragraph) vary across topics and article quality tiers.

## Limitations

- **The same real-world venue is sometimes cited by both a train and a
  test article.** The train/test split is by source article, so no two
  paragraphs share an article — but two different articles can
  legitimately cite the same newspaper, sports site, or government page,
  and when they do, the corresponding `venue` field value in a reference
  card recurs across the split. Measured on the shipped files: 42 of 109
  test `ref_A_card` rows have a venue string that also appears somewhere
  in a train card. This is card content recurring, not label leakage —
  `(rank_A, rank_B, rank_C)` is a property of one paragraph's own
  citation order, not of venue identity, and a model fit on train to
  predict rank purely from venue (ignoring each unit's own paragraph)
  scores 0.4924, at chance (see `DESIGN.md`, "Target Recoverability
  warning"). It is disclosed here rather than engineered away because
  removing it would mean either dropping the `venue` field (a legitimate,
  disclosed input the reference solution uses) or shrinking the corpus to
  force single-sided venues, neither of which the measured evidence
  justifies.
- **Not a representative sample of all Wikipedia citations.** Only
  paragraphs where every citation is a fully-inline template survive the
  filter; paragraphs relying on `{{sfn}}`/`{{harv}}` shorthand or a
  separate "Works cited" section are excluded entirely. This biases the
  collection toward articles/paragraphs written in a particular citation
  style.
- **Restricted to Good/Featured articles.** These pages have passed
  community review and are not representative of Wikipedia's average
  article in citation density or prose quality.
- **Citation metadata is exactly what editors typed, unverified.** Titles,
  authors, venues and years are taken verbatim from each `{{cite ...}}`
  template's fields; any typo, inconsistency, or missing field in the
  original wikitext (e.g. `author: (uncredited)` when no author field was
  present) is preserved as-is, not corrected.
- **Only paragraphs with EXACTLY 3 references are included.** Paragraphs
  with 1-2 citations (too little signal to order meaningfully) or with 4
  or more are excluded entirely. An earlier iteration shipped a variable
  3-8 range packed into `ref_A_card`..`ref_H_card`, but that left
  high-index columns empty in most rows and failed the platform's "no
  column missing values in half or more profiled rows" check; fixing the
  count at exactly 3 was the smallest change that closed it while keeping
  the largest single n_refs bucket in the raw data (492 of 1,196 raw
  units). This means the task never varies in pool size and says nothing
  about recovering order among 4+ references — a real scope reduction
  from the original design, not a claim about typical Wikipedia
  paragraphs.
- **English Wikipedia only, one snapshot.** Fetched live on 2026-09-25;
  Wikipedia articles are edited continuously, so re-fetching the same
  titles later would not reproduce byte-identical wikitext. The shipped
  `dataset/raw/` is the frozen snapshot this package is built from.
- **Card field format is fixed and terse.** Title/author/venue/year are
  rendered as a single compact string per reference; richer bibliographic
  detail available in the original templates (page numbers, DOIs, ISBNs)
  is not shipped, so two citations that differ only in a field this
  format omits are indistinguishable from the card alone (though the
  paragraph's own content is still available to break the tie).
- **Ambiguity by construction is excluded, not resolved.** Paragraphs
  where the same reference is cited more than once are dropped rather
  than handled, so this dataset says nothing about recovering order when
  a source is reused within one paragraph.

## Redundant columns

No numeric column derivable from another shipped column is published
(e.g. no separate `text_length` column — solvers can compute
`len(text)` themselves if needed).
