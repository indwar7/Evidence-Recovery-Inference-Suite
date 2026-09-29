# Provenance and Licensing

Every corpus in this suite is commercially usable. Five carry attribution obligations
and two of those also carry share-alike. This page states what each obligation is, what discharges
it, and what evidence the licence determination rests on.

Harness code — graders, generators, preparation scripts, reference solutions,
verification tooling and documentation — is **MIT** ([`LICENSE`](../LICENSE)). The MIT
grant does not relicense the corpora.

---

## Summary

| Benchmark | Corpus | Licence | Commercial use | Attribution | Share-alike |
|---|---|---|---|---|---|
| accent-transfer | English Wiktionary | **CC BY-SA 4.0** | Yes | Required | **Required** |
| experimental-order | PLOS research articles | **CC BY 4.0** | Yes | Required | No |
| pipeline-attribution | Library of Congress NDNP | **Public Domain Mark** | Yes | Not required (see caveat) | No |
| prompt-edit-attribution | Measured model behaviour (original) | **CC0-1.0** | Yes | Not required | No |
| citation-structure | CourtListener U.S. case law | **Public Domain Mark** | Yes | Not required | No |
| reference-order | English Wikipedia | **CC BY-SA 4.0** | Yes | Required | **Required** |
| mechanism-recovery | openFDA drug labels | **CC0 1.0** | Yes | Not required | No |
| amendment-reversal | Electronic Code of Federal Regulations | **U.S. Government Work** | Yes | Not required | No |
| rag-sufficiency | Hand-written corpus + measured behaviour (original) | **CC BY 4.0** | Yes | Required | No |
| gene-symbol-coinage | HGNC complete gene set | **CC0 1.0** | Yes | Not required | No |
| clause-responsibility | System-Prompt-Library + measured behaviour | **CC BY 4.0** | Yes | Required | No |

Each corpus additionally ships its own machine-readable
`benchmarks/<name>/dataset/raw/ATTRIBUTION.txt`, which is the authoritative record.

---

## accent-transfer — English Wiktionary

**Licence:** Creative Commons Attribution-ShareAlike 4.0 International
(`https://creativecommons.org/licenses/by-sa/4.0/`)

**Source:** the official Wikimedia database dump,
`https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2`,
fetched 2026-09-18.

**Evidence.** Confirmed against the MediaWiki `siteinfo` API
(`action=query&meta=siteinfo&siprop=rightsinfo`), which returns:

```json
{"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
 "text": "Creative Commons Attribution-Share Alike 4.0"}
```

The determination rests on the wiki's own rights endpoint, not on a project page.

**What ships.** IPA transcriptions and their accent tags, taken verbatim from each
article's own structured `{{IPA|en|...|a=...}}` template values. These are contributor-
authored archive content, not annotations created for this dataset and not model output.

**Your obligations.** Attribute Wiktionary contributors, and distribute derivatives of
the corpus under CC BY-SA 4.0 or a compatible licence. **Share-alike is viral over the
data**: it reaches works derived from the corpus. It does not reach the MIT-licensed
harness code, and it does not reach model weights trained on the corpus under most
readings — but that is a question for your counsel, not for this file.

Suggested citation:

> Wiktionary contributors. *Wiktionary, the free dictionary.* https://en.wiktionary.org.
> Retrieved from the Wikimedia database dumps, 2026-09-18.

---

## experimental-order — PLOS research articles

**Licence:** Creative Commons Attribution 4.0 International
(`https://creativecommons.org/licenses/by/4.0/`)

**Source:** the PLOS Search API (`https://api.plos.org/search`) and the article XML
endpoint, fetched 2026-09-12.

**Evidence.** Every PLOS research article carries the CC BY licence in its own
`<license>` block: *"This is an open-access article distributed under the terms of the
Creative Commons Attribution License, which permits unrestricted use, distribution, and
reproduction in any medium, provided the original author and source are credited."* The
determination is per-article and read from the article itself.

**What ships.** Results-section subsections as published prose, in the authors' own
order. Each source article is identified by DOI **in the raw corpus only** — no DOI,
title or author appears in any solver-visible file, because the article identity is
answer-adjacent.

**Your obligations.** Credit the original authors and source. Commercial use permitted.
No share-alike.

Suggested citation:

> PLOS (Public Library of Science). Article content retrieved via the PLOS Search API
> and article XML endpoint, 2026-09-13. Individual articles identified by DOI in
> `dataset/raw/papers.jsonl`.

---

## pipeline-attribution — Library of Congress NDNP

**Licence:** Public Domain Mark.

**Source:** the Chronicling America public bulk-data mirror
(`https://chroniclingamerica.loc.gov/data/batches/`), fetched 2026-08-12.

**Evidence.** The Library of Congress's rights statement for Chronicling America is that
the newspapers in the collection *"are in the public domain or have no known copyright
restrictions."* The NDNP programme admits only titles and runs that the contributing
institution has confirmed rights-clear.

**What ships.** Raw, uncorrected OCR output — ALTO XML `<String CONTENT="...">` values,
concatenated. Nothing cleaned, corrected or paraphrased. Real per-batch vendor metadata
(`softwareCreator`, `softwareName`, `processingAgency`) is read directly from each
batch's own ALTO XML.

> **Caveat, carried from the Library's own advisory.** Issue dates span roughly
> 1831–1960. Material published within the last 95 years is believed to be in the public
> domain but may contain copyrighted third-party content. Snippets carry their
> `issue_date` so a downstream user can filter on it. This caveat is reproduced here
> because the Library states it, not because a problem is known.

**Your obligations.** None imposed by the mark. Crediting the Library of Congress NDNP
is courteous and is what this suite does.

Suggested citation:

> National Digital Newspaper Program (NDNP), Library of Congress. *Chronicling America
> bulk data*, fetched 2026-08-12.

---

## prompt-edit-attribution — measured model behaviour

**Licence:** CC0-1.0 (public domain dedication), for the measured fingerprints, the grid
definition and the derived splits.

**Source:** original measurement. `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0,
ungated) was run greedily over 16 hand-written prompt framings × 12 conditions × 6
hand-written input kinds × 4 hand-written probe questions — 4,608 generations. Each
response was reduced to a 16-dimensional numeric behaviour fingerprint **at generation
time**.

**No response text is stored or released anywhere in this repository.**

**Reasoning.** Apache-2.0 governs the model's weights and code, not the text it
generates; the model card places no claim or restriction on outputs. The recorded
measurements are therefore an original derived work, released here under CC0.

**Your obligations.** None. Crediting the model card is courteous:

> HuggingFaceTB, *SmolLM2-1.7B-Instruct* model card,
> `huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct`. Prompt framings, edit operations,
> probe questions and the fingerprint/ordering code are original work for this challenge.

---

## citation-structure — CourtListener U.S. case law

**Licence:** Public Domain Mark.

**Source:** the Free Law Project's CourtListener bulk data
(`https://www.courtlistener.com/help/api/bulk-data/`), snapshot 2022-09-30.

**Evidence.** The Free Law Project's bulk-data documentation states: *"Our bulk data
files are free of known copyright restrictions"*, beside a Public Domain Mark badge.
Independently, U.S. judicial opinions are government-edict material and are not
copyrightable (*Banks v. Manchester*, 128 U.S. 244 (1888); *Georgia v.
Public.Resource.Org*, 590 U.S. 255 (2020)).

**What ships.** Opinion text with every formal citation string and every party name
removed, and the archive's own recorded citation edges between opinions. The edges are
the labels. Nothing is annotated.

**Your obligations.** None imposed by the mark. Crediting the Free Law Project is
courteous and is what this suite does.

Suggested citation:

> Free Law Project. *CourtListener bulk data*, snapshot 2022-09-30.
> https://www.courtlistener.com/help/api/bulk-data/

---

## reference-order — English Wikipedia

**Licence:** Creative Commons Attribution-ShareAlike 4.0 International
(`https://creativecommons.org/licenses/by-sa/4.0/`)

**Source:** English Wikipedia Good and Featured articles, retrieved 2026-09-25. The
declared source is the bulk dump,
`https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2`.

**Evidence.** Confirmed against Wikipedia's own MediaWiki `siteinfo` API
(`action=query&meta=siteinfo&siprop=rightsinfo`), which returns:

```json
{"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
 "text": "Creative Commons Attribution-Share Alike 4.0"}
```

**What ships.** Paragraphs with their inline citation markers replaced by a neutral
token, and each paragraph's own reference cards. The order of citation is the article
editors' own.

**Your obligations.** Attribute Wikipedia contributors, and distribute derivatives of
the corpus under CC BY-SA 4.0 or a compatible licence. This is the second share-alike
corpus in the suite, after Wiktionary; the same reach applies.

Suggested citation:

> Wikipedia contributors. *English Wikipedia*, Good and Featured articles.
> https://en.wikipedia.org. Retrieved 2026-09-25.

---

## mechanism-recovery — openFDA drug labels

**Licence:** CC0 1.0 Universal (public domain dedication).

**Source:** the openFDA Drug Label API (`https://api.fda.gov/drug/label.json`),
retrieved 2026-09-25.

**Evidence.** openFDA's licence page states: *"unless otherwise noted, the content,
data, documentation, code, and related materials on openFDA is public domain and made
available with a Creative Commons CC0 1.0 Universal dedication."*

> **The carve-out is real, and it shaped the build.** "Unless otherwise noted" covers
> device records, which carry nomenclature licensed separately from the GMDN Agency.
> This benchmark uses the drug label endpoint only and no device endpoint.

**What ships.** Clinical Pharmacology sections with mechanism vocabulary replaced by
`[MASK]`, and the FDA's own mechanism-of-action class assignments. The unmasked text is
never written to disk.

**Your obligations.** None. openFDA's disclaimer travels with the data: it should not
be relied on for decisions about medical care, and results should be assumed
unvalidated. This benchmark is not medical information.

Suggested citation:

> U.S. Food and Drug Administration. *openFDA Drug Label API*.
> https://api.fda.gov/drug/label.json. Retrieved 2026-09-25.

---

## amendment-reversal — Electronic Code of Federal Regulations

**Licence:** U.S. Government Work, public domain under 17 U.S.C. § 105.

**Source:** the eCFR versioner API (`https://www.ecfr.gov/api/versioner/v1/`),
retrieved 2026-09-25.

**Evidence.** Under 17 U.S.C. § 105, works of the United States Government are not
subject to copyright protection in the United States. The Code of Federal Regulations
is prepared by the Office of the Federal Register and published with the U.S.
Government Publishing Office.

**What ships.** Pairs of consecutive published versions of a regulation section, on
dates the archive itself records as amendment dates. The eCFR's forward-reference
banner and editorial notes are removed because they name the amendment.

> **Caveat.** The eCFR is an editorial compilation and is not the official legal
> edition of the CFR. This benchmark must not be used to determine what any regulation
> required at any time.

**Your obligations.** None.

Suggested citation:

> Office of the Federal Register, National Archives and Records Administration, and the
> U.S. Government Publishing Office. *Electronic Code of Federal Regulations*, versioner
> API v1. Retrieved 2026-09.

---

## gene-symbol-coinage — HGNC complete gene set

**Licence:** CC0 1.0 Universal (public domain dedication).

**Source:** the HUGO Gene Nomenclature Committee's complete gene set, quarterly snapshot
2026-07-07 (`https://www.genenames.org/download/archive/`). The generator pins the
snapshot and checks its sha256.

**Evidence.** Stated on the HGNC's own licence page,
`https://www.genenames.org/about/license/`.

**What ships.** Approved gene names and approved gene symbols, both as the committee
recorded them.

**Your obligations.** None. Crediting the HGNC is courteous.

---

## clause-responsibility — System-Prompt-Library and measured behaviour

**Licence:** CC BY 4.0.

**Source:** `danielrosehill/System-Prompt-Library` on Hugging Face
(`https://huggingface.co/datasets/danielrosehill/System-Prompt-Library`).

**Evidence.** The Hugging Face API reports `license: cc-by-4.0` in the dataset's own
metadata, read 2026-09-27. The licence is declared on the Hugging Face dataset card; the
mirrored GitHub repository carries no licence file, so the Hugging Face page is the
citation.

**What ships.** Nothing yet. The benchmark is in build and its generation has not been
run. When it is, the corpus will be the filtered prompts split into sentences, with no
wording rewritten, and numeric measurements of `HuggingFaceTB/SmolLM2-1.7B-Instruct`
(Apache-2.0, ungated) run over them.

**Your obligations.** Attribution to the corpus author.

---

## rag-sufficiency — hand-written corpus and measured behaviour

**Licence:** CC BY 4.0, for the hand-written corpus, the measured retrieval records, the
grid definition and the derived splits.

**Source:** original throughout. 24 hand-written knowledge-base topics, 192 passages and
480 queries, all written for this challenge and all fictional — no topic describes any
real organisation's actual policy. `BAAI/bge-small-en-v1.5` (MIT, ungated) computes real
cosine-similarity retrieval over that corpus; `HuggingFaceTB/SmolLM2-1.7B-Instruct`
(Apache-2.0, ungated) is run greedily over each query's actual retrieved context. Every
retrieval rank and every answer is reduced to numeric measurements at generation time.

**Your obligations.** Attribution. Because CC BY carries an obligation that CC0 does
not, the attribution line below must appear, worded identically, in any redistribution:

> This dataset was produced using BAAI/bge-small-en-v1.5 (MIT licence, ungated) for
> retrieval and HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache-2.0 licence, ungated) for
> generation. Neither model's card places any claim or restriction on outputs; the
> recorded measurements are an original derived work.

---

## A note on models used as instruments

Two open-weights models appear in this suite. Both are **instruments that were
measured**, never teachers whose outputs became labels for anything other than the
task's own derived target:

| Model | Licence | Gated | Role |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Apache-2.0 | No | Generator whose behaviour is measured |
| `BAAI/bge-small-en-v1.5` | MIT | No | Retriever whose ranks are measured |

Neither card asserts ownership of or restrictions on model outputs. Both are ungated, so
the measurements are reproducible by anyone without an access request.

---

## If you are evaluating this suite for acquisition

The licensing position in one paragraph: **all eleven corpora permit commercial use.**
Two (Wiktionary and Wikipedia, CC BY-SA 4.0) impose share-alike on derivatives of the
data, and that is the only term that constrains how you redistribute a derived corpus.
Three (PLOS, rag-sufficiency, clause-responsibility) require attribution only. Three
(NDNP, CourtListener, eCFR) are public domain. Three (prompt-edit, openFDA, HGNC) are
CC0 and impose nothing. Every determination is traceable to a primary source quoted
above, and every corpus carries its evidence in its own `ATTRIBUTION.txt`.

This page is a record of the licensing research done during construction. It is not
legal advice.
