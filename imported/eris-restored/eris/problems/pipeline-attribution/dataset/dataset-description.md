# Dataset Description

## Overview

Real historical American newspaper pages digitized under the Library of
Congress's National Digital Newspaper Program (NDNP), fetched from the public
bulk-data mirror at chroniclingamerica.loc.gov. Each page was processed by a
real digitization vendor's OCR pipeline, and every batch's own ALTO XML
metadata records which OCR engine and (often) which contracted vendor handled
it. This dataset pairs the raw, uncorrected OCR text of each article/column
region with the real batch identity that produced it.

35 batches, 8,417 text snippets, 34 distinct contributing-institution
prefixes, spanning issue dates roughly 1831-1960.

## File Structure

- `snippets.jsonl` — one JSON object per line, one per OCR'd text region
  (a `TextBlock` in the source ALTO XML, roughly one article or column):
  `snippet_id`, `batch_id`, `lccn` (the newspaper's Library of Congress
  Control Number), `issue_date`, `text` (raw, uncorrected OCR output).
- `batches_meta.csv` — one row per real digitization batch: `batch_id`,
  `institution_prefix`, `n_snippets`, `n_distinct_issue_dates`,
  `software_creator`, `software_name`, `processing_agency`.
- `meta.json` — fetch provenance: source URL, fetch date, fetch method,
  per-batch snippet/issue-date statistics, vendor-metadata coverage.
- `skip_log.csv` — batch IDs that were attempted and skipped (404, too thin,
  or timed out), and why — kept for reproducibility of the fetch process.
- `ATTRIBUTION.txt` — citation and licensing notes.

## Features

| Column | Type | Description |
|--------|------|-------------|
| snippet_id | string | Unique ID: `<batch_id>_<lccn>_<issue_date>_<seq>_<block_idx>` |
| batch_id | string | The real NDNP digitization batch identifier (the label) |
| lccn | string | Library of Congress Control Number for the newspaper title |
| issue_date | string (YYYY-MM-DD) | Real publication date of the source issue |
| text | string | Raw OCR output for one article/column region, uncleaned |
| institution_prefix | string | The contributing institution's batch-ID prefix |
| software_creator | string | Real OCR engine vendor recorded in the batch's ALTO metadata |
| software_name | string | Real OCR engine/product name |
| processing_agency | string | Real contracted digitization vendor, where recorded (present for 17/35 batches — some awardees never logged this field) |

## Notes

- Text is genuinely noisy: character substitutions, junk characters, and
  broken word boundaries are the real output of period-appropriate OCR
  engines on period newspaper print, not synthetic corruption.
- Every batch's `batch_id` and vendor fields are the archive's own recorded
  values — nothing in this dataset was inferred or human-annotated.
- Fetch was rate-limited by the source mirror; `dataset/generator/build_raw.py`
  is resumable and documents the fetch method and known gotchas (see
  `dataset/generator/README.md`) if the corpus needs to be extended later.
- Licensing: NDNP only admits newspaper titles/runs that the contributing
  state institution has independently confirmed are rights-clear for
  digitization and public redistribution — see `ATTRIBUTION.txt` for the
  full, honest caveat (issue dates span past the unconditional pre-1929
  public-domain boundary; per-issue rights were not independently
  re-verified by this fetch).
