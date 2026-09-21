# Provenance and Licensing

Every corpus in this suite is commercially usable. Three carry attribution obligations
and one carries share-alike. This page states what each obligation is, what discharges
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
| rag-sufficiency | Hand-written corpus + measured behaviour (original) | **CC BY 4.0** | Yes | Required | No |

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

The licensing position in one paragraph: **all five corpora permit commercial use.** One
(Wiktionary, CC BY-SA 4.0) imposes share-alike on derivatives of the data and is the
only term that constrains how you redistribute a derived corpus. Two (PLOS, rag-
sufficiency) require attribution only. One (NDNP) is public domain. One (prompt-edit) is
CC0 and imposes nothing. Every determination is traceable to a primary source quoted
above, and every corpus carries its evidence in its own `ATTRIBUTION.txt`.

This page is a record of the licensing research done during construction. It is not
legal advice.
