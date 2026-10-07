# Dataset Description

## Overview

This dataset is one dated snapshot of the HUGO Gene Nomenclature Committee
(HGNC) complete gene set: every human gene the committee has named, with the
approved **name**, the approved **symbol**, and the dates on which the
committee approved or changed them.

The HGNC is the body that decides what human genes are called. For each gene
it approves a descriptive name (`solute carrier family 2 member 1`) and a
short symbol (`SLC2A1`), following published guidelines that have accumulated
over four decades. Those two fields, as the committee recorded them, are the
input and the target of the task built on this dataset.

Nothing in this dataset is annotated. Every value is the archive's own. The
snapshot is stored byte for byte as the HGNC published it; no row is removed,
no field is rewritten, and no human annotator or language model produced any
value in it at any point.

- 45,019 gene records, 54 columns, tab-separated, UTF-8
- 19,296 of them are approved protein-coding genes, the subset the task uses
- snapshot date 2026-07-07

## Source

- **Source:** HGNC complete gene set, quarterly archive snapshot
- **Publisher:** HUGO Gene Nomenclature Committee, University of Cambridge
- **Source URL:** `https://www.genenames.org/download/archive/`
- **Snapshot file:** `hgnc_complete_set_2026-07-07.txt`
- **Snapshot URL:**
  `https://storage.googleapis.com/public-download-files/hgnc/archive/archive/quarterly/tsv/hgnc_complete_set_2026-07-07.txt`
- **Licence page:** `https://www.genenames.org/about/license/`

The source URL is the HGNC's own archive page, which links the dated
snapshots. It answers a plain automated request with HTTP 200.

## Fetch method

`dataset/generator/build_raw.py`, run as:

```bash
python dataset/generator/build_raw.py --out dataset/raw
```

Deliberate choices, each for a stated reason:

- **A dated archive snapshot, not the rolling current file.** The HGNC
  republishes its current file several times a week. A dated snapshot is fixed,
  so a re-run fetches the same bytes.
- **The sha256 of the snapshot is pinned in the script** and checked after
  every download. A mismatch stops the build.
- **One worker, one request.** A single file is fetched. Nothing runs in
  parallel.
- **Broad `except Exception` in the retry loop.** `ConnectionResetError` and
  `TimeoutError` are not `HTTPError` subclasses and would otherwise end the run.
  Bounded at 5 attempts with linear backoff of at least 1.2 s.
- **The download is streamed to a `.part` file and renamed only when
  complete**, so an interrupted run never leaves a truncated snapshot behind.
- **No filtering at fetch time.** The snapshot is kept whole so that every
  later exclusion is visible in one place and can be re-examined.

## File Structure

Generator:

- `generator/build_raw.py` — the fetch script that produces the whole raw
  archive below. It downloads the snapshot, verifies its sha256, counts its
  rows, and writes the provenance and attribution files. It performs no
  annotation and calls no model.

Raw archive:

- `raw/hgnc_complete_set_2026-07-07.txt` — the HGNC snapshot, one gene per
  row, 54 tab-separated columns, exactly as published
- `raw/meta.json` — provenance: source and snapshot URLs, licence, retrieval
  timestamp, and the snapshot's sha256, byte size, row count, column count and
  count of approved protein-coding genes
- `raw/ATTRIBUTION.txt` — source, publisher, licence, the HGNC's own licence
  statement quoted verbatim, and the citation the HGNC asks for

These three files plus the generator are the whole dataset.

### What `prepare.py` derives (not shipped here)

The train and test tables, the sample submission, the split statistics and the
answer key are produced from the snapshot by the task's `prepare.py`. They are
not distributed as part of this dataset. `prepare.py` reads seven columns of
the snapshot, keeps approved protein-coding genes, removes rows whose name
already prints the answer, and divides what remains by the date each current
symbol came into force.

## Features

### `raw/hgnc_complete_set_2026-07-07.txt`

Columns the task reads:

- `hgnc_id` — string. The HGNC's stable gene identifier, e.g. `HGNC:11005`.
  Used as a sort key only.
- `symbol` — string. The approved gene symbol, e.g. `SLC2A1`. The prediction
  target.
- `name` — string. The approved gene name, e.g. `solute carrier family 2
  member 1`. The input.
- `status` — string. `Approved` for every row in this snapshot.
- `locus_type` — string. The kind of locus; the task keeps `gene with protein
  product`.
- `date_approved_reserved` — ISO date. When the gene first received an
  approved symbol.
- `date_symbol_changed` — ISO date, often empty. When the symbol was last
  replaced by a different one.

Columns present in the snapshot and not read by the task:

- `locus_group`, `location` — broad locus class and chromosomal band
- `alias_symbol`, `alias_name`, `prev_symbol`, `prev_name` — other symbols and
  names the gene has carried, pipe-separated
- `gene_group`, `gene_group_id` — the HGNC's own gene group assignments
- `date_name_changed`, `date_modified` — further record dates
- `entrez_id`, `ensembl_gene_id`, `vega_id`, `ucsc_id`, `ena`,
  `refseq_accession`, `ccds_id`, `uniprot_ids`, `pubmed_id`, `mgd_id`,
  `rgd_id`, `omim_id`, `orphanet`, `agr`, `mane_select`, `gencc` and the
  remaining specialist database columns — cross-references to other resources

### `raw/meta.json`

`generated_by`, `source`, `publisher`, `source_url`, `snapshot_url`,
`licence`, `licence_url`, `retrieved_utc`, a `files` object holding the
snapshot's `sha256`, `bytes`, `n_rows`, `n_columns` and
`n_approved_protein_coding`, and `filters`, which records that none were
applied.

### `raw/ATTRIBUTION.txt`

Plain text. Source, publisher, landing page, snapshot name and URL, licence,
the HGNC's licence statement, the requested citation, and a non-endorsement
notice.

## Intended Use

**Primary use.** Evaluating whether a model can produce the symbol a
nomenclature committee approved for a gene, given the gene's approved name and
the symbols the committee had approved before — a sequence-to-sequence task
over short strings in which part of the answer follows convention and part of
it does not.

**Also suitable for.** Studying how a naming convention drifts over time,
since every record carries its approval dates; studying abbreviation and
acronym formation in a controlled technical vocabulary; evaluating models
under a date-based split where the test period contains both continuations of
known patterns and patterns that did not exist when the training period ended.

**Not suitable for.** Any use as a nomenclature authority. Approved symbols
change, this is a fixed snapshot, and the current approved symbol for a gene
must be taken from genenames.org. It is not suitable for clinical, diagnostic
or reporting purposes, where an out-of-date or wrong gene symbol has
consequences. It is not suitable for predicting anything about a gene's
biology: a symbol records what a gene is called, not what it does. It is not
suitable for non-human nomenclature, which other committees govern under
different rules.

## Limitations

- **Names are current, symbols are dated.** The snapshot holds each gene's
  name as it reads in 2026. 74.7% of the protein-coding records carry a name
  change date, so a gene whose symbol was approved in 1995 may carry a name
  revised in 2016. The task's training rows are therefore old symbols paired
  with present-day names, not a reconstruction of the nomenclature as it stood
  on any past date.
- **Most symbols in the test period are replacements, not first approvals.**
  Of the 2,887 protein-coding symbols that came into force in 2010 or later,
  82.9% replaced an earlier symbol for the same gene, typically a placeholder such as
  `C6orf89` or `FAM123A`. The test period over-represents genes that were hard
  to name the first time.
- **Rows where the name prints the answer are removed by the task.** 3,076 of
  the 19,296 protein-coding records (15.9%) have a name that contains their own
  symbol or its stem as a token, such as `BRCA1 DNA repair associated`. The
  task drops them, which makes the retained set harder than the nomenclature
  as a whole and removes a convention the committee uses often.
- **The symbol stem is a heuristic, not an HGNC field.** Where the task speaks
  of a stem it means the leading letters of the symbol. That is how the
  retained-set statistics are computed; it is not how the HGNC defines a gene
  family, and `H2AC1` and `H1-0` share the one-letter stem `H` without being
  the same family in any useful sense.
- **Approval dates before 1986 are not distinguished.** The earliest recorded
  approval date in the snapshot is in 1986 and 268 protein-coding records share
  that year, which reflects when recording began and not when those genes were
  named.
- **One species, one committee, one snapshot.** Every regularity in this data
  is a regularity of how one committee names human genes. Nothing here
  supports a claim about nomenclature in general.
- **Symbols are short, so a single character is a large fraction of the
  answer.** The median protein-coding symbol is 5 characters. Any per-character
  measure on this data moves in steps of about 0.2 per row.

## License

**CC0 1.0 Universal (Public Domain Dedication).**

The HGNC states on its licence page,
`https://www.genenames.org/about/license/`:

> all data is released under the Creative Commons Public Domain (CC0)
> License. This means that any form of reuse of the content is permitted.

- Licence URL: `https://creativecommons.org/publicdomain/zero/1.0/`
- CC0 does not require attribution. It is given in `raw/ATTRIBUTION.txt` as
  good practice, with the citation the HGNC asks reusers to give.

The HGNC does not endorse this dataset or any work derived from it, and
nothing here implies affiliation with or endorsement by the HGNC.
