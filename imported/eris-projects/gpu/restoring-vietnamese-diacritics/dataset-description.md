# GeoNames Vietnam Settlements: Folded vs Native Spellings — Dataset

## Overview

This dataset provides pairs of Vietnamese place names in two forms: the native
spelling with tone and vowel diacritics, and the ASCII-folded spelling with the
marks removed, together with the names of each place's province, district and
nearest training-split settlements. It is built from the Vietnam export of the
GeoNames geographical database. Its evaluation set is defined by held-out
syllables — every evaluation name contains at least one syllable that occurs
in no training name — so it supports evaluating whether a model can restore
diacritics on syllables it has never seen.

## Source

- **GeoNames geographical database**, Vietnam country export.
- Source URL: `https://www.geonames.org/search.html?q=&country=VN` (file used: `https://download.geonames.org/export/dump/VN.zip`)
- The specific file used is `VN.zip`, from which `VN.txt` is extracted. `VN.txt`
  is the standard GeoNames "geoname" table: one tab-separated row per feature,
  whose columns include `geonameid`, `name` (native UTF-8 name), `asciiname`
  (ASCII-folded name), `feature class`, `feature code` and `country code`.
- Fields used downstream, for records whose `feature class` is `P` (populated
  places): `name` (the native spelling), `asciiname` (the folded spelling),
  `latitude` and `longitude` (to find each place's nearest training
  settlements), and `admin1 code` / `admin2 code` (to join each place to its
  `ADM1` and `ADM2` records, whose `name` and `alternatenames` supply the
  province and district names). Coordinates and codes are consumed by
  `prepare.py` and are not published.
- The snapshot is pinned to 2026-10-10 with a SHA-256 recorded in `meta.json`,
  because GeoNames republishes a rolling file. The pin checks `VN.txt`'s own
  hash rather than the `VN.zip` container's: the zip's bytes were observed to
  vary between two fetches of identical `VN.txt` content, so only the
  extracted table's hash is a meaningful reproducibility check.

## Fetch method

`dataset/generator/build_raw.py` downloads `VN.zip` with a single HTTP request
and a generous timeout, catching exceptions broadly, writing to a temporary
`.part` file that is renamed only on success so an interruption never leaves a
truncated file. It verifies the download against the pinned SHA-256 (warning if
GeoNames has since republished the file), extracts `VN.txt`, and writes
`meta.json` and `ATTRIBUTION.txt`. Re-running is idempotent.

## File Structure

The dataset upload contains the raw source and the generator only:

- `raw/VN.txt` — the GeoNames Vietnam geoname table (tab-separated), as fetched.
- `raw/meta.json` — source URL, fetched file, snapshot date, SHA-256 of `VN.zip`
  and `VN.txt`, and the licence with its source URL.
- `raw/ATTRIBUTION.txt` — the CC BY 4.0 attribution for GeoNames.
- `generator/build_raw.py` — the fetch-and-pin script described above.

### What `prepare.py` derives (not shipped here)

Running `prepare.py` on `raw/` produces the public/private split that solvers and
the grader use. These files are generated, not part of this upload:

- `public/train.csv` — `id`, `folded`, `province`, `district`, `neighbors`,
  `name`; the training rows.
- `public/test.csv` — `id`, `folded`, `province`, `district`, `neighbors`;
  inputs only.
- `public/sample_submission.csv` — `id`, `name` (copy baseline).
- `public/dataset_stats.json` — summary counts and lengths.
- `public/validate_submission.py` — a well-formedness checker.
- `private/answers.csv` — `id`, `name`, `folded_src`, `province_src`,
  `district_src`, `neighbors_src`; the held-out targets plus copies of
  test.csv's input and context columns, each carried under a different column
  name so the grader can classify changed positions as context-attributable
  or not without re-deriving test.csv from train.csv.
- `private/split_manifest.json` — split method and counts.

`prepare.py` keeps only records whose ASCII folding is a clean, length-preserving
one-to-one mapping of the native name (so input and output align
position-by-position), drops any name that carries no diacritic, deduplicates on
the (folded, native) pair, and splits by folded string so no folded form appears
in both train and test. The split holds out syllables: a deterministic subset of the
corpus's less common syllable types (chosen by a SHA-256 hash, no random seed)
is held out, every name containing one of them is an evaluation row, and no
training row contains any of them. Every test name therefore carries at least
one syllable absent from training, and since all rows sharing a folded string
share its syllables, no folded string appears in both splits. Row ids within each split are then
assigned after an independent shuffle keyed on each row's GeoNames `geonameid`
(never published) rather than on the sort order used for the split, so id order
cannot leak which native spelling a row has. It then attaches context to every
row: `province` and `district` are the `name` of the row's `ADM1` and `ADM2`
records — replaced by the record's own `alternatenames` spelling when that
lists the same name with Vietnamese marks (ignoring English generics such as
"Province"; single-token exonyms such as `Hanoi` stay as recorded) — and
`neighbors` lists the names of the five nearest training-split settlements by
great-circle distance. Any context value whose folded form contains the row's
own folded name as a phrase is withheld, and neighbours are taken from the
training split for both splits, so no row's context contains its own answer
(measured: zero evaluation rows have their answer, or a phrase matching their
input, in any context field).

## Features

- `name` — native Vietnamese spelling of a populated place, with tone and vowel
  diacritics (UTF-8). This is the prediction target.
- `folded` — the source's `asciiname`: the same name with diacritics removed
  (ASCII). This is the input.
- `province`, `district` — the source's own `name` (or same-name marked
  `alternatenames` spelling) for the place's `ADM1` and `ADM2` records, joined
  on the source's admin codes; withheld when it contains the row's own name.
- `neighbors` — the source's own `name` of the five geographically nearest
  training-split places, ordered nearest first, separated by a spaced vertical bar.
- Every field is the source's own recorded value; no field is annotated or
  generated. Selection of neighbours is a deterministic function of the source's
  coordinates.

## Intended Use

- **Primary use:** training and evaluating models that restore Vietnamese
  tone and vowel diacritics to folded place names whose syllables include ones
  never seen in training — a character-level sequence transduction that
  rewards sub-syllabic generalisation rather than recall, optionally
  conditioned on the names of nearby places.
- **Also suitable for:** studying sub-syllabic orthographic ambiguity in
  Vietnamese, and as a proper-name (out-of-vocabulary) stress test for
  diacritic-restoration methods that normally rely on word lists.
- **Not suitable for:** diacritic restoration of running Vietnamese text or
  common vocabulary (these are place names, not sentences); phonetic
  transcription (the target is spelling, not pronunciation); transliteration
  between scripts (both forms are Latin-based); or geographic analysis (no
  coordinates, population or administrative codes are shipped — only the names
  of the province, the district and nearby settlements).

## Limitations

- **Length-preserving subset only.** Names whose ASCII folding is not a strict
  one-to-one, same-length mapping are dropped. In practice this removes names
  containing `đ`/`Đ` (folded to `d`/`D`, which are not combining-mark
  decompositions), so the stroked-D, a very common Vietnamese letter, is absent
  from the targets. The task therefore measures combining-mark (tone and vowel)
  restoration, not the full orthography.
- **Populated places only.** Only `feature class = P` records are kept; rivers,
  mountains, administrative regions and other feature classes are excluded, so
  the vocabulary of morphemes is narrower than Vietnamese toponymy as a whole.
- **Shallow diacritic density.** The average kept name needs about 1.9 diacritics
  and the maximum in the evaluation split is 6; very long multi-syllable names
  are under-represented after the length and feature-class filters.
- **Source inconsistency.** GeoNames `asciiname` is contributor-maintained and is
  occasionally not a clean fold of `name`; those rows fail the one-to-one check
  and are dropped, which slightly biases the kept set toward regular spellings.
- **Single country, single snapshot.** All names are from the Vietnam export at a
  single pinned date; GeoNames coverage of Vietnam is uneven by region, so the
  distribution of name forms reflects that coverage rather than a balanced sample.
- **Context is train-only and name-only.** Neighbours are chosen among training
  places, so test rows in sparsely covered regions may have neighbours tens of
  kilometres away. Ten of the 34 provinces (22.8% of rows) and 1.4% of district
  values are recorded by the source without marks, as exonyms such as `Hanoi`.
- **Real, searchable names.** The names are genuine and appear in public
  gazetteers, so no transform in the data can make an external lookup physically
  impossible; the held-out-syllable split and the task Rules are what close that route,
  not obfuscation of the inputs.

## License

GeoNames data is licensed under the **Creative Commons Attribution 4.0 License
(CC BY 4.0)**, `https://creativecommons.org/licenses/by/4.0/`, as stated in the
GeoNames `readme.txt` at `https://download.geonames.org/export/dump/readme.txt`.
Attribution is provided in `raw/ATTRIBUTION.txt`. Only database fields are used;
no GeoNames images or separately-licensed content are included.
