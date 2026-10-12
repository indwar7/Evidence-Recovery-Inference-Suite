# Wikipedia Paragraphs Paired With Their Own Shuffled Reference Cards

Real paragraphs from English Wikipedia's "Good article" and "Featured
article" pages (Wikipedia's own quality-review categories), each stored
with the full set of references it cites, in the order the article's
editors cited them. Every citation marker in the paragraph is replaced
by the neutral token `[CITE]`, and each reference is stored as the
structured fields the editors recorded in its `{{cite ...}}` template
(kind, title, author, venue, year).

**Quality-reviewed source pool.** Good and Featured articles have passed
a structured Wikipedia community review that, among other things, checks
citation quality. This correlates strongly with fully templated
`{{cite ...}}` citations rather than ad hoc inline text, which is what
the extraction filter requires. The pool is not restricted to any one
topic area; sampled articles span history, science, sport, meteorology,
biography, and more.

## Provenance

- **Source:** [English Wikipedia](https://en.wikipedia.org/) article corpus, as mirrored at `dumps.wikimedia.org/enwiki`
- **Declared source URL:** `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2`
- **Article pool:** `Category:Good_articles` and `Category:Featured_articles`
- **Fetch mechanism:** public MediaWiki Action API (`action=parse`, `prop=wikitext` at `https://en.wikipedia.org/w/api.php`), serving the same live article wikitext as the bulk dump
- **Licence:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)
- **Retrieved:** 2026-09-25

Verified live from Wikipedia's own MediaWiki siteinfo API at build time:

```
https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json

-> {"rightsinfo": {"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
                    "text": "Creative Commons Attribution-Share Alike 4.0"}}
```

CC BY-SA 4.0 requires attribution to Wikipedia and its contributors, and
requires that this package (and any derivative of it) remain under a
compatible share-alike licence. This package is released under
CC BY-SA 4.0 on that basis.

## How the raw data was built

The build script is `dataset/generator/build_raw.py`;
`dataset/generator/finalize_raw_meta.py` writes the metadata files.

1. Article titles were pulled from Wikipedia's own `Category:Good_articles`
   and `Category:Featured_articles` listings via `list=categorymembers`,
   paginated with the API's own continuation token. 1,900 articles were
   attempted.
2. Each article's raw wikitext was fetched via `action=parse`.
3. The wikitext was split into paragraphs. A paragraph was kept only if
   it had between 3 and 8 `<ref>` markers and **every** marker is a
   distinct, fully inline `{{cite journal|book|web|news|magazine|
   encyclopedia}}` template: no `<ref name="x"/>` reuse within the
   paragraph, no `{{sfn}}`/`{{harv}}` shorthand, no bare citation text.
   This keeps the marker-to-reference mapping unambiguous: each `[CITE]`
   token corresponds to exactly one entry in the paragraph's `refs` list.
4. Each kept citation template was parsed into structured fields (kind,
   title, author, venue, year), using the article editors' own recorded
   values.
5. Each `<ref>` marker was replaced with the token `[CITE]`, and the
   surrounding wikitext markup (templates, wikilinks, bold/italic,
   residual HTML) was stripped to plain prose.
6. The references were written in the order their markers appear in the
   wikitext, left to right. This order is read from the article itself.
   It is not inferred, annotated, or generated.

## Files

### `dataset/raw/wiki_citation_units.jsonl`: 1,196 lines, 1.7 MB

One JSON object per extracted paragraph:

| field | type | content |
|---|---|---|
| `page` | string | source article title |
| `text` | string | paragraph prose, markup stripped, every citation marker replaced by `[CITE]` |
| `n_refs` | int | number of citations in the paragraph (3 to 8); always equals the number of `[CITE]` tokens in `text` |
| `refs` | list | `n_refs` objects in true citation order, each with `kind`, `title`, `author`, `venue`, `year` |

Example (text truncated):

```json
{"page": "3rd Plurinational Legislative Assembly of Bolivia",
 "text": "At Arce's inauguration, Mesa and the entire CC caucus left the hemicycle ... [CITE] ...",
 "n_refs": 5,
 "refs": [{"kind": "news", "title": "Legisladores de CC presentan recurso de inconstitucionalidad contra anulación de los 2/3",
           "author": "Corz", "venue": "La Razón", "year": "2020"},
          {"kind": "news", "title": "CC presenta recurso ante el TCP para frenar la anulación de los 2/3",
           "author": null, "venue": "Urgente.bo", "year": "2020"},
          ...]}
```

### `dataset/raw/meta.json`
Schema, row count, size, MD5 checksum, licence, and the list of
modifications made to the source material.

### `dataset/raw/build_meta.json`
Counts recorded at build time: articles attempted, articles that yielded
at least one paragraph, paragraphs extracted, and the distribution of
`n_refs`.

### `dataset/raw/_fetched_titles.json`
Bookkeeping: the 1,900 article titles the fetcher attempted, so that a
second run can resume without re-fetching.

### `dataset/raw/ATTRIBUTION.txt`
Source, licence, retrieval date and the list of modifications.

### `dataset/generator/build_raw.py`, `dataset/generator/finalize_raw_meta.py`
The fetch-and-extract script and the metadata writer.

## Summary statistics

All figures below describe the raw file as a whole.

- **Paragraphs:** 1,196, from 510 distinct articles (of 1,900 attempted).
  Most articles contribute one paragraph; the most from one article is 18.
- **Citations per paragraph (`n_refs`):**

  | n_refs | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|
  | paragraphs | 492 | 302 | 169 | 115 | 71 | 47 |

- **Citations:** 5,092 in total. By template kind: web 2,313, news 2,216,
  magazine 229, book 184, journal 150.
- **Field coverage across the 5,092 citations:** `title` and `kind` are
  always present; `venue` is present in 4,953, `year` in 4,920, and
  `author` in 2,643 (`null` when the editor recorded none).
- **Publication years:** 1586 to 2026, median 2009.
- **Paragraph length:** 181 to 2,950 characters, median 689.

## Intended use

- Training and evaluating models that align short structured records
  (bibliographic metadata) to positions in a passage of prose based on
  content.
- Studying citation grounding in reverse: given a passage and its true
  supporting sources, work out which source backs which part of the
  passage.
- Research on how citation practices (which reference types are used
  where in a paragraph) vary across topics.

## Limitations

- **Not a representative sample of all Wikipedia citations.** Only
  paragraphs where every citation is a fully inline template survive the
  filter. Paragraphs that rely on `{{sfn}}`/`{{harv}}` shorthand or a
  separate "Works cited" section are excluded entirely, which biases the
  collection toward articles written in that citation style.
- **Restricted to Good/Featured articles.** These pages have passed
  community review and are not representative of Wikipedia's average
  article in citation density or prose quality.
- **Restricted to 3 to 8 citations per paragraph.** Paragraphs with
  fewer or more citations are not included.
- **Citation metadata is exactly what editors typed, unverified.**
  Titles, authors, venues and years are taken verbatim from each
  `{{cite ...}}` template's fields. Typos, inconsistencies and missing
  fields in the original wikitext are preserved as they are, not
  corrected.
- **Terse card fields.** Only kind, title, author, venue and year are
  kept. Richer detail in the original templates (page numbers, DOIs,
  ISBNs) is not stored, so two citations that differ only in an omitted
  field look identical in `refs`.
- **Repeated sources within a paragraph are excluded, not resolved.**
  Paragraphs that cite the same reference more than once are dropped, so
  the data says nothing about order when a source is reused.
- **The same venue can appear in many articles.** Different articles
  legitimately cite the same newspaper, sports site or government page,
  so venue strings recur across paragraphs.
- **English Wikipedia only, one snapshot.** Fetched on 2026-09-25.
  Articles are edited continuously, so re-fetching the same titles later
  would not reproduce identical wikitext. `dataset/raw/` is the frozen
  snapshot.
