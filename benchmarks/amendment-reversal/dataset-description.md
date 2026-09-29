# Dataset Description

## Overview

Pairs of consecutive published versions of United States federal regulations,
drawn from the Electronic Code of Federal Regulations (eCFR). Each record is
one CFR section captured on two dates the archive itself records as amendment
dates: the wording immediately **before** an amendment, and the wording
immediately **after** it.

Every published CFR section carries a version history in the eCFR's versioner
API, listing each date the section was amended, whether the amendment was
substantive, and whether the section was removed. For a section with two
consecutive substantive amendment dates, the API returns the full section text
as it read on each. That pair is the entire record. The prediction target is
`text_before` — the prior wording — given `text_after`.

Nothing in this dataset is annotated. The amendment boundary is the archive's
own recorded `amendment_date`; both text fields are the agency's own published
wording on those dates. No human annotator and no language model produced any
value in this dataset at any point.

Two things are removed from the text, both because they would hand over the
answer rather than because they are unwanted:

- **The forward-reference banner.** The eCFR inserts a line reading *"Link to
  an amendment published at 89 FR 47080, May 31, 2024"* into a section whose
  amendment is published but not yet incorporated. It names the Federal
  Register citation and the amendment date directly.
- **Editorial and effective-date notes** appended by the Office of the Federal
  Register, which frequently narrate the change in prose.

Structural XML markup is stripped to plain text. No wording is otherwise
altered, normalised or rewritten.

## Source

- **Source:** Electronic Code of Federal Regulations (eCFR), versioner API v1
- **Publisher:** Office of the Federal Register, National Archives and Records
  Administration; in partnership with the U.S. Government Publishing Office
- **Source URL:** `https://www.ecfr.gov/api/versioner/v1/titles.json`
- **API base:** `https://www.ecfr.gov/api/versioner/v1`
- **Endpoints used:**
  - `/versions/title-{n}.json` — the amendment-date record per section
  - `/full/{date}/title-{n}.xml?part={p}&section={s}` — section text on a date

The eCFR is an editorial compilation of CFR material and is **not** the
official legal edition of the Code of Federal Regulations.

## Fetch method

`dataset/generator/build_raw.py`, run as:

```bash
python build_raw.py --out ../raw --since 2022-01-01
```

Deliberate choices, each for a measured reason:

- **One worker, never parallel.** The API throttles under concurrency; a
  single patient worker finishes sooner than several contended ones.
- **1.2 s between every request.**
- **`Accept-Encoding: gzip, deflate` on every call.** The `/full/` endpoint
  *requires* response compression and returns HTTP 406 without it. This is not
  prominently documented and was found only by reading the 406 body.
- **Broad `except Exception` in the retry loop.** `ConnectionResetError` and
  `TimeoutError` are not `HTTPError`/`URLError` subclasses and would otherwise
  end a long run outright. Bounded at 5 retries with linear backoff; HTTP 404
  and 400 are treated as "this section did not exist on that date" and skipped
  without retrying.
- **Each pair is appended and flushed as it is parsed**, so an interrupted run
  loses at most the pair in flight.
- Pairs are discarded at fetch time when the two versions are byte-identical,
  when either side is under 200 characters (too little text to reason over), or
  when either side exceeds 20,000 characters (length outliers).

## File Structure

Generator:

- `generator/build_raw.py` — the fetch script that produces the whole raw
  archive below. It lists each title's version record, groups substantive
  non-removal amendments by section, walks consecutive amendment dates to form
  (before, after) pairs, fetches both sides, strips XML markup and the
  amendment banner, and writes one JSON object per pair. It performs no
  annotation and calls no model.

Raw archive:

- `raw/amendments.jsonl` — one JSON object per (section, amendment) pair
- `raw/meta.json` — provenance: API base, titles requested, date floor,
  retrieval timestamp, per-title pair counts, the filters applied, and a
  sha256 of `amendments.jsonl`
- `raw/ATTRIBUTION.txt` — source, publisher, licence, and the 17 U.S.C. § 105
  notice

These three files plus the generator are the whole dataset. The challenge's
`prepare.py` derives the train/test split from `raw/amendments.jsonl`; the
split's own files are produced there and are not distributed as part of this
dataset.

## Features

### `raw/amendments.jsonl`

| Field | Type | Description |
|---|---|---|
| `record_id` | string | Stable id: title, section, and the two dates |
| `title` | integer | CFR title number |
| `part` | string | CFR part the section belongs to |
| `section` | string | Section identifier, e.g. `1.94` |
| `section_name` | string | The section's own published heading |
| `subpart` | string | Subpart letter, where the section has one |
| `date_before` | string | The earlier amendment date (ISO) |
| `date_after` | string | The next consecutive amendment date (ISO) |
| `text_before` | string | Full section text as published on `date_before` |
| `text_after` | string | Full section text as published on `date_after` |

### `raw/meta.json`

`generated_by`, `source`, `source_url`, `api_base`, `publisher`, `licence`,
`retrieved_utc`, `titles_requested`, `issue_date_gte`, `pairs_per_title`,
`n_pairs`, `amendments_sha256`, and a `filters` object recording every
build-time exclusion.

## Intended Use

**Primary use.** Evaluating whether a model can recover the prior wording of a
regulation from its amended form — inferring what an amendment removed from
how the surviving text reads, when the amendment instruction itself is
withheld.

**Also suitable for.** Studying how legal and administrative language is
revised over time; evaluating long-document conditioning, since the change is
usually confined to one paragraph of a long section and locating it is most of
the problem; research on partial-credit metrics for rewrite tasks, where the
input and target share most of their content.

**Not suitable for.** Any legal, regulatory or compliance purpose. This is a
research benchmark built from an editorial compilation, not the official legal
edition of the CFR, and the text has had editorial notes and amendment banners
removed. It must not be used to determine what any regulation required at any
time. It is also not suitable for measuring real amendment rates or regulatory
activity, since aggressive filtering distorts those frequencies, and not for
non-US regulatory text or for CFR material outside the titles and date range
fetched.

## Limitations

- **Filtering makes the retained set unrepresentative of real amendments.**
  Pairs are dropped when the two versions are byte-identical, when either side
  is under 200 or over 20,000 characters, when the change is under three
  tokens, when it exceeds 60% of the section (a near-total rewrite rather than
  an amendment), and when the shown text duplicates another record's exactly.
  Small technical corrections and wholesale replacements are therefore both
  under-represented relative to how often agencies actually make them.
- **"Substantive" is the archive's judgement, not a verified property.** The
  versioner API flags each amendment as substantive or not, and only
  substantive ones are kept. That flag is the Office of the Federal Register's
  own editorial classification; a change it marks substantive may still be
  minor in effect, and the reverse also occurs.
- **The amendment banner strip is regex-based.** The forward-reference line and
  editorial notes are removed by pattern. Unusual phrasings may survive, and
  the patterns can occasionally remove a sentence that was not a banner. Any
  surviving banner would name the Federal Register citation and amendment
  date, which is why the pattern is deliberately broad rather than narrow.
- **One direction only, and it is the harder one.** Records are built as
  (after → before). Real regulatory drafting runs the other way, and a model
  trained here learns to invert amendments, not to make them. Whether the two
  directions are equally difficult was not measured.
- **Consecutive amendment dates are not always consecutive *changes*.** If a
  section was amended twice on the same date, or if the archive records a date
  for a change that did not alter the text this pipeline can see, the pair may
  span more or less than exactly one amendment.
- **Titles were sampled for size and amendment activity, not for balance.**
  Large, frequently-amended titles dominate. Subject matter is therefore skewed
  toward the regulatory areas that change most often, and a model's performance
  here should not be read as uniform across the CFR.
- **The eCFR is an editorial compilation.** It is current to within a few days
  of the date requested but is not the official legal edition, and the text on
  a given date reflects the archive's incorporation schedule rather than the
  exact moment a rule took legal effect.

## License

**U.S. Government Work — public domain.**

- Under **17 U.S.C. § 105**, works of the United States Government are not
  subject to copyright protection in the United States and are in the public
  domain.
- The Code of Federal Regulations is a work of the U.S. Government, prepared
  by the Office of the Federal Register (NARA) and published with the U.S.
  Government Publishing Office.
- eCFR developer documentation: `https://www.ecfr.gov/api/versioner/v1/titles.json`

The Office of the Federal Register, the National Archives and Records
Administration, and the U.S. Government Publishing Office do not endorse this
dataset or any work derived from it, and nothing here implies affiliation with
or endorsement by them. This dataset is not official government material and
is not the official legal edition of the Code of Federal Regulations. The full
notice is in `raw/ATTRIBUTION.txt`.
