# Dataset Description

## Overview

Clinical Pharmacology sections from United States prescription drug labels,
paired with the mechanism-of-action classes the FDA itself assigned to each
drug, with all mechanism vocabulary masked out of the text.

Every drug label submitted to the FDA as Structured Product Labeling carries a
Clinical Pharmacology section — several thousand words on what the drug does
in the body. Separately, the FDA records that drug's pharmacologic class in
the `openfda` block of the same label, drawn from its own NDC pharmacologic
class vocabulary: *Cyclooxygenase Inhibitors [MoA]*, *Proton Pump Inhibitors
[MoA]*, *Cytochrome P450 3A4 Inducers [MoA]*, and so on.

The prediction target is that class assignment. It is the FDA's own structured
field, not an annotation created for this dataset, not inferred from the text
by a heuristic, and not produced by any model.

### The masking, and why it is not optional

Measured on the raw corpus before anything was built: **57% of labels state
their own mechanism verbatim** in the pharmacology prose. A label classed
*Cyclooxygenase Inhibitors* routinely contains the phrase "cyclooxygenase
inhibitor". Left alone, a keyword matcher would score highly while learning
nothing about pharmacology.

Every mechanism word is therefore replaced with `[MASK]`, using a vocabulary
built from **every** class name in the corpus — never just the row's own
class. That distinction matters: if only a row's own class words were masked,
*which* words are masked would itself reveal the answer.

Measured effect on a 1,401-label sample: 1,860 of 2,082 label-class pairs
stopped being keyword-recoverable, and the surviving text still runs to a
median of about 7,300 characters of genuine pharmacology — substrates, binding,
metabolism, half-life, clearance.

Words that appear in class names but carry no mechanism information
(*inhibitors*, *agonists*, *receptor*, *agents*, *activity*, and similar) are
exempt, so the prose stays readable. The full vocabulary and the exempt list
are shipped in `raw/mask_vocabulary.json` so the masking is auditable.

## Source

- **Source:** openFDA Drug Label API (Structured Product Labeling)
- **Publisher:** U.S. Food and Drug Administration
- **Source URL:** `https://api.fda.gov/drug/label.json`
  (the partitioned bulk export of the drug label corpus; the manifest at
  `https://api.fda.gov/download.json` lists 14 partitions covering 262,887
  labels. This dataset uses the 7,979 labels carrying both an FDA
  mechanism-of-action class and a Clinical Pharmacology section.)
- **Query:** labels having both `openfda.pharm_class_moa` and
  `clinical_pharmacology`
- **Fields taken:** `clinical_pharmacology` (masked before storage),
  `openfda.pharm_class_moa`, `openfda.pharm_class_epc`, `openfda.route`,
  `openfda.substance_name`, `openfda.generic_name`,
  `openfda.manufacturer_name`, `effective_time`

## Fetch method

`dataset/generator/build_raw.py`, run as:

```bash
python build_raw.py --out ../raw
```

Deliberate choices, each for a measured reason:

- **One worker, never parallel.** openFDA rate-limits by IP; a single patient
  worker finishes sooner than several contended ones.
- **0.35 s between requests**, with `skip`-based paging.
- **Broad `except Exception` in the retry loop.** `ConnectionResetError` and
  `TimeoutError` are not `HTTPError`/`URLError` subclasses and would otherwise
  end a long run outright. Bounded at 5 retries with escalating backoff;
  HTTP 404 and 400 mean "no more results" or "past the paging cap" and stop
  the run cleanly rather than retrying.
- **openFDA caps `skip` at 25,000**, so the fetch stops there by design.
- Labels are dropped at fetch time when the pharmacology section is under 800
  characters (too little to reason over) or over 60,000 (length outliers), and
  de-duplicated by label id.
- **The unmasked text is never written to disk.** Masking happens in the
  fetcher, before `labels.jsonl` is written, so the raw archive cannot leak it.

## File Structure

Generator:

- `generator/build_raw.py` — the fetch script that produces the whole raw
  archive below. It pages the openFDA label endpoint, keeps labels carrying
  both a pharmacology section and an MoA class, builds the corpus-wide mask
  vocabulary, applies it, and writes one JSON object per label. It performs no
  annotation and calls no model.

Raw archive:

- `raw/labels.jsonl` — one JSON object per drug label
- `raw/mask_vocabulary.json` — the mask terms, the exempt stopwords, and a
  note on why the vocabulary is corpus-wide rather than per row
- `raw/meta.json` — provenance: endpoint, query, retrieval timestamp, label
  and class counts, the filters applied, and a sha256 of `labels.jsonl`
- `raw/ATTRIBUTION.txt` — source, publisher, licence, and the openFDA
  disclaimer

These four files plus the generator are the whole dataset. The challenge's
`prepare.py` derives the train/test split and the per-row candidate pools from
`raw/labels.jsonl`; the split's own files are produced there and are not
distributed as part of this dataset.

## Features

### `raw/labels.jsonl`

| Field | Type | Description |
|---|---|---|
| `record_id` | string | The label's own openFDA id |
| `set_id` | string | SPL set id: stable across versions of one label |
| `effective_time` | string | Label effective date as recorded by FDA |
| `generic_name` | string | Generic drug name (used only for splitting; not published to solvers) |
| `substance_name` | list of str | Active substances (used only for splitting) |
| `route` | list of str | Administration route(s) |
| `manufacturer_name` | string | Labeller |
| `pharm_class_moa` | list of str | **Target.** FDA's mechanism-of-action class(es) |
| `pharm_class_epc` | list of str | FDA's established pharmacologic class(es), for reference |
| `pharmacology_masked` | string | Clinical Pharmacology section with mechanism vocabulary replaced by `[MASK]` |
| `n_masked` | integer | How many tokens were masked in this label |

### `raw/mask_vocabulary.json`

`n_terms`, `built_from`, `note`, `stopwords_excluded`, `terms`.

### `raw/meta.json`

`generated_by`, `source`, `source_url`, `publisher`, `licence`, `licence_url`,
`retrieved_utc`, `api_search`, `n_labels`, `n_distinct_moa_classes`,
`mask_vocabulary_terms`, `median_masks_per_label`, `labels_sha256`, and a
`filters` object recording every build-time exclusion.

## Intended Use

**Primary use.** Evaluating whether a model can infer a drug's mechanism of
action from a description of its pharmacological behaviour, when the mechanism
cannot simply be read off the text.

**Also suitable for.** Set-selection under a chance-corrected metric with
variable, unknown target-set size; long-document conditioning, since the
sentences that identify a mechanism are scattered through several thousand
words; research on masking and de-identification robustness, since the corpus
is a worked example of removing a concept from text that heavily implies it.

**Not suitable for.** Any clinical, prescribing, pharmacovigilance or
regulatory purpose. openFDA states plainly that its data should not be relied
on for decisions about medical care and that results are unvalidated; on top of
that, the text here is deliberately mutilated and the drug's identity is
withheld. It is also not suitable for measuring real drug-class prevalence,
since filtering and de-duplication distort those frequencies, nor for
non-US labelling, nor as a source of the FDA pharmacologic class vocabulary,
since classes with thin support are dropped.

## Limitations

- **Filtering makes the retained set unrepresentative of approved drugs.**
  Labels are dropped when the pharmacology section is missing, under 800
  characters or over 60,000, when no MoA class is assigned, when the masked
  text exactly or near-duplicates another label's, and when the class has
  fewer than 20 supporting labels. Common, heavily-genericised drugs are
  therefore collapsed to one representative, and rare mechanisms disappear
  entirely.
- **Masking is lexical and deliberately over-aggressive.** Every word of every
  class name is masked corpus-wide, regardless of whether that word is
  distinctive. That removes genuine subject matter — a label loses the word
  "adrenergic" even where it describes physiology rather than the drug's own
  class. The bias is deliberate: it costs signal but never leaks the answer.
- **The exempt-word list is a judgement call.** Roughly two dozen structural
  words are exempt from masking so the prose stays readable. A different list
  would produce a different dataset, and a word that is generic in one class
  name can be distinctive in another.
- **FDA class assignment is not exhaustive.** A drug may act through several
  mechanisms while its label records only the one FDA indexed. Absence of a
  class in the target set is therefore weaker evidence than its presence, and
  part of the residual error is irreducible.
- **Many labels describe the same molecule.** Generic drugs are labelled
  separately by each manufacturer with near-identical prose. Near-duplicate
  collapse and the substance-level split handle the leak, but the surviving
  distribution over-represents molecules with a single labeller.
- **The class space is long-tailed and the pool is synthetic.** Only classes
  with at least 20 supporting labels survive, and the twelve-candidate pool is
  constructed by frequency-weighted sampling — it is a task artifact, not a
  clinical differential. Performance here says nothing about distinguishing
  mechanisms that never co-occur in a pool.
- **A single snapshot, and label text is not uniform.** Labels are fetched once
  and vary widely in age, depth and house style; some pharmacology sections are
  thorough and some perfunctory. The masked length distribution is wide, and a
  short section may simply not contain enough to identify a mechanism.

## License

**CC0 1.0 Universal** — public domain dedication.

openFDA states: *"unless otherwise noted, the content, data, documentation,
code, and related materials on openFDA is public domain and made available
with a Creative Commons CC0 1.0 Universal dedication. Under CC0, FDA has
dedicated the work to the public domain by waiving all rights to the work
worldwide under copyright law."*

- Licence statement: `https://open.fda.gov/license/`
- Terms of service: `https://open.fda.gov/terms/`

This dataset uses only the drug label endpoint. It deliberately does **not**
use any device endpoint, because device records carry GMDN nomenclature
content that openFDA notes is separately licensed from The GMDN Agency and is
not covered by the CC0 dedication.

The U.S. Food and Drug Administration does not endorse this dataset or any
work derived from it, and nothing here implies affiliation with or endorsement
by the FDA. openFDA's own disclaimer applies: the data should not be relied
upon to make decisions regarding medical care, and all results should be
assumed unvalidated. The full notice is in `raw/ATTRIBUTION.txt`.
