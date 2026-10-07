# Dataset Description — eCFR Point-in-Time Section Pairs With the Amendments That Produced Them

## Overview

Real regulatory text, paired across the moment a real rule changed it.

Each record is one section of the **Code of Federal Regulations** captured twice —
as it stood the day before a final rule took effect, and as it stood the day it
took effect — together with the **amendatory instructions** the agency published
in the Federal Register to make that change, and whatever replacement text was
printed beneath them.

Nothing here is written, paraphrased, annotated or model-generated. The
instructions are the agency's own words. The before and after texts are the
Office of the Federal Register's own compilation, served by the eCFR
point-in-time API. The ground truth is what the law actually said on those two
mornings, which is about as uncontestable as text data gets.

The task this supports is the compilation itself: given the old text and the
instruction, produce the new text. It is work that is done largely by hand today,
and the record of it having been done correctly is published daily and free.

## Source

The substance of this dataset — both halves of every labelled pair — comes from
the **eCFR point-in-time API**. The Federal Register supplies only the instruction
strings that sit alongside them.

- **eCFR (the before and after text, and the ground truth)** — PRIMARY
  `https://www.ecfr.gov/api/versioner/v1` — `versions/title-N.json?part=P` lists
  every date on which eCFR recorded a change to each section, and
  `full/{date}/title-N.xml?part=P&section=S` serves that section as it stood on
  that date. Every label in this dataset is one of these point-in-time texts.
- **Federal Register (the amendatory instruction text only)** — SECONDARY
  `https://www.govinfo.gov/bulkdata/FR` — the govinfo bulk XML feed. Instructions
  are the `<AMDPAR>` elements of each `<RULE>`; the text printed beneath them is
  the following `<SECTION>` element. Nothing else from the Federal Register is
  used: not the preamble, not the citations, not the regulatory analysis.
- **Rule metadata (publication and effective dates, CFR references)** — SECONDARY
  `https://www.federalregister.gov/api/v1/documents.json` — the Federal Register
  JSON API. Used only to decide which rules to look at and when; none of its text
  enters the dataset.

**Coverage:** final rules published @@DATE_RANGE@@.

### Relationship to other Federal Register datasets

The Federal Register is a large public corpus and many datasets draw on it. What
makes this one distinct is not the Federal Register at all — it is the **pairing
with eCFR point-in-time text**. The unit of data here is a *versioned pair*: one
CFR section captured on two specific dates either side of one specific rule, which
requires reconciling eCFR's recorded change dates against the rule's stated
effective date and discarding every pair the rule cannot account for.

Concretely, this dataset is not:

- a corpus of Federal Register documents, rules or preambles,
- a citation extraction, linking or resolution resource,
- a classification or summarization set over regulatory text.

It is a set of before/after regulation texts with the published instruction that
transformed one into the other. The label is the law as it stood on a given
morning.

**License: public domain.** Both the Federal Register and the Code of Federal
Regulations are works of the United States Government and are not subject to
copyright in the United States (17 U.S.C. § 105). Commercial use and
redistribution are unrestricted. `attribution.csv` records the Federal Register
document number and citation behind every record, so any row can be traced back to
the rule that produced it.

## File Structure

Three files at the top level of the upload:

```
items.jsonl        one line per (rule, section): instructions plus before/after text
attribution.csv    the Federal Register rule behind every record
meta.json          source URLs, date range, selection parameters, counts
```

`prepare.py` consumes `items.jsonl` and derives the public and private splits from
it — the item files solvers read, the training labels, and the hidden answer key.
Those derived files are products of the prepare step and are not part of this raw
upload.

## Features

Every field below is given with its data type. Nested fields are written with a
dotted path so the structure is unambiguous. Types use standard names: `str`,
`int`, `float`, `list[str]`, `list[object]`, `object`. Each entry reads
`field` — type `type` — description.

### `items.jsonl` — JSON Lines; one `object` per line, one line per (rule, section) pair

- `uid` — type `str` — `document_number:cfr_title:section`, unique per record
- `document_number` — type `str` — Federal Register document number of the rule
- `citation` — type `str` — Federal Register citation, e.g. `86 FR 4`
- `publication_date` — type `str` — Date the rule was published, `YYYY-MM-DD`
- `effective_on` — type `str` — Effective date the rule states, `YYYY-MM-DD`
- `change_date` — type `str` — Date eCFR records the section as having changed, `YYYY-MM-DD`
- `cfr_title` — type `int` — CFR title number, e.g. `40`
- `cfr_part` — type `str` — CFR part number, e.g. `63`
- `section` — type `str` — CFR section identifier, e.g. `63.11`
- `agency` — type `str` — Issuing agency name
- `rule_title` — type `str` — Title of the rule as published
- `before` — type `str` — The section on the day before `change_date`, canonicalised
- `after` — type `str` — The section on `change_date`, canonicalised. This is the label.
- `blocks` — type `list[object]` — The amendment as published; one element per run of instructions
- `blocks[].instructions` — type `list[str]` — Consecutive `<AMDPAR>` texts aimed at this section
- `blocks[].printed` — type `str` — Replacement text the agency printed beneath them; empty string when the amendment only deletes
- `n_ops` — type `int` — Count of token-level edit operations between `before` and `after`

`blocks` is a list of objects. Each object has exactly two keys, `instructions`
and `printed`, documented above as `blocks[].instructions` and `blocks[].printed`.

`before`, `after` and `blocks[].printed` all pass through the same
canonicalisation, so they are directly comparable. `after` is the ground truth and
is not released to solvers.

### `attribution.csv` — CSV with a header row

CSV stores every value as text, so all six columns are `str`. Where a column holds
a number, the underlying value is noted.

- `document_number` — type `str` — Federal Register document number
- `citation` — type `str` — Federal Register citation, e.g. `86 FR 4`
- `publication_date` — type `str` — Date the rule was published, `YYYY-MM-DD`
- `effective_on` — type `str` — Effective date the rule states, `YYYY-MM-DD`
- `cfr_title` — type `str` — CFR title touched, e.g. `40`. Integer-valued; the same number appears as `int` in `items.jsonl`.
- `cfr_part` — type `str` — CFR part touched, e.g. `63`

### `meta.json` — a single JSON `object`

- `source_ecfr_primary` — type `str` — eCFR point-in-time API URL; every label comes from here
- `source_fr_secondary` — type `str` — govinfo bulk Federal Register XML URL; source of the instruction strings
- `source_rule_metadata` — type `str` — Federal Register JSON API URL; used only for dates and CFR references
- `license` — type `str` — `public domain (17 U.S.C. 105)`
- `scan_window_requested` — type `list[str]` — Two elements, `[start, end]`, each `YYYY-MM-DD`: the publication-date window the scan was asked for
- `coverage_actual` — type `list[str]` — Two elements, `[earliest, latest]`, each `YYYY-MM-DD`: the rule publication dates actually present
- `n_records` — type `int` — Number of records in `items.jsonl`
- `n_cfr_parts` — type `int` — Number of distinct CFR parts covered
- `n_agencies` — type `int` — Number of distinct issuing agencies
- `selection` — type `object` — Filters applied during the build; its five fields follow
- `selection.well_posed_min_supplied_token_fraction` — type `float` — Minimum share of the tokens an amendment introduces that must appear in what the agency printed
- `selection.min_before_tokens` — type `int` — Lower bound on section length, in whitespace tokens
- `selection.max_before_tokens` — type `int` — Upper bound on section length, in whitespace tokens
- `selection.max_edit_operations` — type `int` — Cap on token-level edits between `before` and `after`
- `selection.change_date_rule` — type `str` — How a rule is paired with an eCFR change date
- `built` — type `str` — How the scan was run

`coverage_actual` is narrower than `scan_window_requested`: the eCFR prefetch runs
in date order and the API sustains about 24 requests a minute, so a scan stopped
early yields a contiguous prefix of the requested window rather than a sample
spread across it.

## Notes

**How a record is paired.** A rule states an effective date, but eCFR applies
changes on the date it records for the section, and the two differ by a few days
often enough that pairing on the stated date alone loses most of the corpus. So the
pairing runs off eCFR's own version record: for each section a rule prints, take
eCFR's recorded change date on the effective date if there is one, otherwise the
nearest recorded change within thirty days.

**Well-posedness filter.** At least 90% of the tokens the amendment introduces
must be present in what the agency printed. Below that, the after text cannot be
produced from the inputs by any method, and in practice the item is one where the
change belongs to a different rule that happened to land in the same window. Those
records are discarded during the build, which is what keeps mis-attribution out of
the corpus. The threshold sits in a near-empty valley of a sharply bimodal
distribution — measured over 191 candidates, 85 at exactly 1.00 and 69 below 0.50 —
so it separates real amendments from mis-attributed ones rather than tuning
difficulty.

**Canonicalisation, and what it removes.** Text is HTML-unescaped, curly quotes and
dashes are mapped to ASCII, whitespace is collapsed to single spaces, and four
things are stripped because each one leaks the answer or the identity of the rule:

- eCFR's editorial pointer, which literally announces the amendment being asked
  about — *"Link to an amendment published at 88 FR 3502, Jan. 19, 2023."*
- bracketed source credits, which gain the new rule's citation in the after text
- every Federal Register citation, replaced by `<FRCITE>`
- every CFR section number, replaced by `<CITE>` — the section number is the
  lookup key for the answer

Inline emphasis markup is removed without inserting a space, because it sits inside
paragraph designators: turning `<E T="03">a</E>` into ` a ` would rewrite `(a)` as
`( a )`.

**Eligibility.** Final rules with a stated effective date, a single CFR title and
at most three parts. Sections of 25 to 1,500 tokens. At most 40 edit operations
between before and after. Sections are fetched individually rather than by part, so
no part is excluded for size.

**Scale.** The scan covers @@N_RAW@@ records across @@N_RAW_PARTS@@ CFR parts and
@@N_RAW_AGENCIES@@ agencies, from final rules published @@DATE_RANGE@@.
`prepare.py` narrows this to the surgical amendments and enforces a part-disjoint
split.
