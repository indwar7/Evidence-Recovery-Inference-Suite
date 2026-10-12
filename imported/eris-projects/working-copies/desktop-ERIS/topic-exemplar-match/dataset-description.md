## Overview

This dataset is a sample of scholarly works drawn from OpenAlex, the open
catalog of research publications maintained by OurResearch. Each record
pairs a paper's title and reconstructed abstract with its OpenAlex-assigned
primary research topic, drawn from OpenAlex's own structured topic
classification (domain -> field -> subfield -> topic, ~4,500 topics total).

## Source

- Catalog: OpenAlex (https://openalex.org), operated by OurResearch.
- Source URL: `https://api.openalex.org/topics` — the OpenAlex topic
  taxonomy that defines this dataset's structured label and its
  nearest-neighbor distractor construction. Individual work records are
  retrieved through per-topic filtered queries to `/works`, built on top
  of that topic list.
- License: CC0 1.0 Universal (Public Domain Dedication). OpenAlex states its
  full dataset, including metadata and the `primary_topic` classification
  field, is released under CC0 — see https://openalex.org and the API's
  `/works` documentation. No attribution is legally required; this
  description credits OpenAlex/OurResearch as a courtesy.
- Abstracts are not stored as plain text by OpenAlex; they are served as a
  word -> position inverted index (a documented, deterministic
  representation). This dataset's generator reconstructs each abstract to
  plain text by placing each word at its recorded position, exactly as
  OpenAlex's own documentation describes.

## Fetch method

`generator/build_raw.py`:

- Discovers the full set of OpenAlex Topic IDs with at least 500 associated
  works (paginating the `/topics` endpoint; 1,574 such topics were found
  at build time), then takes a seeded-random sample of them.
- For each selected topic, requests a seeded random sample of works whose
  `primary_topic` is that topic, restricted to `type:article`, publication
  years 2015-2023, and `has_abstract:true`.
- One worker, sequential requests, 0.8s sleep between calls, broad exception
  handling with retry-and-backoff — the fetch is resumable and never
  parallelized against the public API.
- Reconstructs each abstract from its inverted index and discards any work
  whose reconstructed abstract is under 40 words.

`generator/build_hard_distractors.py`:

- Reads `works.jsonl`, deduplicates by `work_id`, and drops any
  work whose abstract text is duplicated across other work_ids (see
  Limitations).
- For every topic with at least 2 surviving works, encodes up to 5 of that
  topic's abstracts with a pretrained sentence-embedding model
  (`all-MiniLM-L6-v2`) and averages them into a topic centroid.
- Ranks every topic's centroid against every other topic's centroid by
  cosine similarity and writes the ranked neighbour list per topic to a
  build-only folder. This is a one-time, offline step; it does not run at
  grading time and its output is not part of the raw archive.

`generator/build_prepare_groups.py`:

- Compresses the build-only topic map and neighbour lists into
  `prepare.py` itself. This is the only place the topic grouping exists.

## File structure

The raw archive is flat: `works.jsonl` and the `generator/` folder sit at its top level. The raw archive carries **no topic field of any kind** — the topic grouping is embedded inside `prepare.py` itself, so nothing in the data mount pairs an abstract with its topic.

`works.jsonl` — one JSON object per line:

- `work_id` — the OpenAlex work identifier (opaque ID, e.g. `W3118615836`).
- `title` — the work's title as recorded by OpenAlex.
- `abstract` — the plain-text abstract, reconstructed from OpenAlex's
  inverted-index representation.
- `publication_year` — integer year.

`generator/build_raw.py`, `generator/build_hard_distractors.py`,
`generator/build_prepare_groups.py` — the generator scripts described above,
run in that order. The second and third write only to a build-only folder
and into `prepare.py`; neither adds anything to the raw archive.

## What `prepare.py` derives (not shipped here)

`prepare.py` reads `works.jsonl` and the grouping embedded in `prepare.py`
and writes a train/test split plus a private answer key. It groups works
by `topic_id`, holds out an entire subset of topics for testing (never
splitting one topic's works across train and test), and for every work
builds six labeled-by-position "candidate groups" — one sharing the work's
true topic, five drawn from that topic's nearest neighbours — each represented only by one other work's abstract
(never by the topic's name). This derived split, its candidate
construction, and the resulting answer key are not part of this raw
dataset upload.

## Intended use

- **Primary use**: training and evaluating a model that judges topical
  similarity between a query abstract and example abstracts from several
  candidate research areas, without ever seeing a descriptive label for any
  of those areas.
- **Also suitable for**: scientific-document representation learning,
  abstract embedding evaluation, zero-shot topic clustering research.
- **Not suitable for**: inferring an author's or institution's identity (no
  such fields are included); citation-based tasks (no citation data is
  included in this dataset); any use that treats `topic_name` as a fixed
  label vocabulary to predict directly, which this dataset's own task design
  deliberately avoids.

## Limitations

- OpenAlex's topic classifier is itself a trained model (not a human
  annotation), so the `primary_topic` field inherits whatever errors that
  classifier makes; a small fraction of works are plausibly mis-topicked at
  the source.
- Restricting to `type:article` and 2015-2023 excludes books, preprints, and
  older literature, so the sample skews toward recent, journal-formatted
  writing.
- The topic pool is restricted to topics with at least 500 works, which
  slightly under-represents very narrow or newly created topics.
- Abstract reconstruction from an inverted index is exact for word
  identities and positions, but reconstructed text can read as slightly less
  fluent than the original (e.g. where the source abstract contained
  formatting OpenAlex does not preserve, such as LaTeX markup inside
  scientific abstracts).
- Domain and field distribution is not uniform: topic discovery at build
  time found more eligible topics in Physical Sciences (602) and Social
  Sciences (452) than in Health Sciences (312) or Life Sciences (208), so
  this sample reflects that skew rather than a balanced split across
  domains.
- No topic, field, or domain name is included anywhere in the raw archive
  or the public split. The grouping lives only inside `prepare.py`, so
  a solver cannot recover a candidate's topic from any shipped data file;
  this is by design, not an oversight.
- A small number of distinct OpenAlex works share verbatim boilerplate
  text (e.g. a journal's own front-matter self-description) as their
  recorded "abstract," rather than a real per-work summary. These carry no
  topical signal and were measured and removed: any abstract string shared
  by more than one `work_id` is dropped by `prepare.py`'s raw loader before
  any split or candidate logic runs.
- A small number of works were returned more than once by the raw fetch,
  because a work can carry several OpenAlex topic tags and the fetch
  originally queried by any-topic-match before being corrected to query by
  `primary_topic` specifically; `prepare.py` deduplicates by `work_id`
  defensively regardless of which fetch method produced `works.jsonl`.
- Distractor topics are selected from each true topic's nearest neighbors
  by a single pretrained sentence encoder's embedding space
  (`all-MiniLM-L6-v2`); a different encoder would rank topic similarity
  somewhat differently, so the specific set of "hard" distractors reflects
  one reasonable notion of topical closeness, not a ground-truth ranking.
