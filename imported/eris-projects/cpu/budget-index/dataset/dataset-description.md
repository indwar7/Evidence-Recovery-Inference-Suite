# Budget-Constrained Index Construction Dataset — documents, per-document budgets, a query workload, and a proven score band

## Overview

This dataset supports a benchmark that grades the layout of a retrieval index directly — the set of spans a solver chooses to build is the scored artifact, not a downstream retrieval ranking or a generated answer. It contains a corpus of **228 documents / 7,884 sentences** — real-world facts (real countries and their real attributes) planted at controlled positions inside naturalistic filler prose — each carrying a per-document index budget, and a workload of **1,398** natural-language queries.

For each document a solver builds at most `budget_k` non-overlapping index windows (contiguous sentence spans, each at most 60 tokens); a query is answered only if all of its evidence sentences fall inside a single window without any of its poison sentences. Because the budget forbids indexing everything, the task is a capacity-constrained *selection* — the solver must deliberately leave regions un-indexed — not a partition.

A `split` column marks each query as `train` (599 queries) or `test` (799 queries). The raw `queries.csv` in this dataset carries the full ground truth for every row (both splits). The challenge's `prepare.py` later produces the public release, where the 799 test queries have their `required_sent_ids` and `poison_sent_ids` withheld; in this raw file those columns are populated for all rows.

## The poison twins, and the three shortcuts closed by construction

Every true fact sentence is accompanied by exactly one **poison twin**: the same entity, the same attribute, the same sentence template, the same numeric surface style, a different value. Masked of its number, a twin is character-for-character identical to the fact it shadows. The generator refuses to emit a corpus unless all of the following hold, and each is asserted at build time and recorded in `meta.json`:

1. **Frequency carries no information.** The poison value is a deterministic function of the (entity, attribute) pair, and every fact occurrence anywhere in the corpus is matched by exactly one poison occurrence of that same pair. A corpus-consensus vote between the two candidate values is therefore an exact tie. Measured on the shipped data: occurrence counts tie exactly for every pair, by construction.
2. **Membership carries no information.** Poison values are drawn from a **donor pool of 72 real countries** that are present in `real_facts.json` but never appear as the subject of any sentence in the corpus. Both the true value and the poison value are therefore genuine knowledge-base entries, so "is this number in the KB at all?" cannot separate them, and the poison value cannot be traced back to some second entity mentioned in the corpus. The remaining 142 countries are the entity pool.
3. **Position, magnitude and length carry no information.** Poison is placed before or after its fact with equal probability everywhere — adjacent twins, cluster blocks and off-path poison alike. Donor values are steered by a running balance so the poison is larger about half the time, and are matched to the truth's digit length so the two numbers are the same length on the page. Measured: poison after the fact 49.6%, fact larger 51.0%, digit lengths tie on 98.4% of pairs.

What remains is the intended skill: resolve the entity, normalise the value the sentence states, and compare it against that entity's true value in the shipped knowledge base.

## Numeric surface forms

The same value is written several ways. Most fact sentences state it as plain digits, but a substantial minority — **39.4%**, measured over the 599 training fact sentences — use one of several other surface forms, including thousands separators and rounded phrasings. The remaining 60.6% are plain.

The forms are deliberately not enumerated here. Every one of them is present in the training split, whose 599 queries expose their evidence and poison sentence ids, so the full inventory is recoverable by inspecting the released data. Recovering it is part of the task: a solver that compares values as literal integers will misread every non-plain sentence.

The non-plain forms apply only to the wide-ranged attributes (population, land area); border and language counts are always plain digits.

**Decidability is asserted, not assumed.** A fact and its twin always share the same surface form, so the form itself leaks nothing — only the value does. For every rendered surface the generator checks that the pair remains separable from the released files alone, and refuses to emit the corpus otherwise. The task is hard, but no question in the workload is a coin flip for a solver that reads its surface correctly. Donor values are additionally required to sit at least 25% away from the truth on the wide attributes, which is what leaves room for that guarantee.

## Provenance and license

- The planted facts are **real-world reference data**: every entity is a real country, and every fact value is that country's real attribute — population, land area, number of bordering countries, and number of official languages — together with its real ISO 3166-1 alpha-3 code and real capital city. The snapshot `real_facts.json` ships with the dataset, so generation and solving are fully self-contained (no network access at any point in the challenge).

- **Sources.** The snapshot is built by `build_kb.py` from exactly two public, key-free endpoints, and every field is traceable to one of them:

  **Source 1 — `mledoze/countries`**, the open dataset the REST Countries API is built from, pinned to tag **v4.0.0**. Licence **ODbL-1.0**. Supplies `name`, `cca3`, `capital`, `region`, `area`, `num_borders`, `num_languages`.

      https://raw.githubusercontent.com/mledoze/countries/v4.0.0/countries.json

  **Source 2 — World Bank World Development Indicators**, indicator `SP.POP.TOTL` ("Population, total"), reference year **2024**. Licence **CC BY 4.0**. Supplies `population`.

      https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&date=2024

  The SHA-256 digest of each downloaded input is recorded in `real_facts.provenance.json`, so the snapshot can be audited against the sources:
  `mledoze` `ad0ff314af7aff69…`, World Bank `d6e16192cdf34009…`. Of the 250 country rows upstream, 214 country records carry a capital, a positive land area, at least one official language and a World Bank population, and only those are kept — no value in the shipped knowledge base is imputed, defaulted or missing.

  (REST Countries' own v5 API now requires an API key, which is why the upstream dataset is read directly.)

- Filler prose between facts is synthetic, generated from closed slot vocabularies, and task-irrelevant.
- **Licence: ODbL-1.0** (Open Data Commons Open Database License v1.0) for the derived database, inherited as share-alike from `mledoze/countries`; attribution required. World Bank population values are CC BY 4.0 and are attributed above. Commercial use is permitted under both. The share-alike obligation is real: a redistributed or modified version of this database must carry the same licence.
- Determinism: `numpy.random.default_rng` (PCG64), fixed seed schedule (`20260715`); two runs produce byte-identical files, verified. The auto-tuning loop is itself deterministic.

## File Structure

The archive contains eight files:

- `documents.csv` — the corpus with per-document budgets (7,884 rows).
- `queries.csv` — the full workload with ground truth (1,398 rows).
- `real_facts.json` — the baked real-world country reference snapshot: 214 country records (142 entities + 72 poison donors). Not to be confused with the 228 documents in the corpus.
- `meta.json` — verified score anchors and the proven shortcut invariants.
- `generate.py` — the deterministic generator (reads `real_facts.json`, writes the CSVs and `meta.json`). It ships **inside the archive**, so the dataset is reproducible from the archive alone.
- `build_kb.py` — the script that rebuilds `real_facts.json` from its two upstream sources (the only step that touches the network; not part of the solver pipeline).
- `real_facts.provenance.json` — the exact source URLs, pinned refs, SHA-256 digests of the downloaded inputs, licences and row counts.
- `dataset-description.md` — this document.

## Features: documents.csv

One row per sentence. Each column, its type, and its meaning:

- `doc_id` (string): document identifier, `doc_0001` onward.
- `sent_id` (integer): 0-based position of the sentence within its document.
- `n_tokens` (integer): whitespace token count of the sentence (the 60-token window cap is enforced in these token counts). Mean 10.77, max 18.
- `text` (string): the sentence text.
- `budget_k` (integer): the document's index budget — the maximum number of windows a submission may build for this document. Constant within a document. Range 2–4, mean 2.15.

## Features: queries.csv

One row per query. Every column is populated for every row in this raw file (both splits).

- `query_id` (string): unique query identifier.
- `split` (string): `train` or `test`.
- `doc_id` (string): the document this query is about (matches `documents.csv`).
- `question` (string): the natural-language question.
- `required_sent_ids` (string): pipe-separated integer `sent_id`s that must all fall inside one window for the query to be answerable (e.g. `12|15`).
- `poison_sent_ids` (string): pipe-separated integer `sent_id`s that must NOT be inside that window. Every query in this dataset has at least one.
- `qtype` (string): the question form — one of four:
  - `direct`: the country is named. 1 evidence sentence.
  - `aliased`: the country is given by its ISO 3166 alpha-3 code, resolved by a code-definition sentence. 2 evidence sentences.
  - `relational`: the country is given by its capital, resolved by a capital-to-country sentence. 2 evidence sentences.
  - `chained`: the country is given by its capital city, which resolves to an ISO code, which resolves to the country — two hops. 3 evidence sentences.
- `cls` (string): the construction class — one of exactly five values:
  - `easy`: evidence sentences are adjacent (a small window answers the query).
  - `tight`: evidence gap is just under the 60-token cap (a deliberate window is required).
  - `cluster`: several direct-fact queries whose evidence sentences are packed consecutively, so a single window answers all of them at once — merging them saves budget for elsewhere.
  - `conflict`: two queries whose evidence is interleaved so that no non-overlapping window set can answer both.
  - `impossible`: structurally unanswerable — evidence spans more than the cap, or a poison sentence is pinned between the evidence sentences.

Test-split mix: `direct` 521 / `chained` 99 / `relational` 98 / `aliased` 81; `easy` 407 / `cluster` 194 / `tight` 142 / `conflict` 36 / `impossible` 20.

## Features: real_facts.json

A JSON array of **214 country records** — 142 used as question subjects, 72 reserved as poison-value donors. Not to be confused with the 228 documents in the corpus. Each record has these fields:

- `name` (string): the country's common name (used as the entity in questions and facts).
- `cca3` (string): the real ISO 3166-1 alpha-3 code (e.g. `IDN`); used by `aliased` and `chained` questions.
- `capital` (string): the country's capital city; used by `relational` and `chained` questions.
- `region` (string): the country's continental region, one of `Africa`, `Americas`, `Asia`, `Europe`, `Oceania`. Carried through from the source dataset for completeness; **no question, fact or poison sentence uses it**, so a solver may ignore it.
- `population` (integer): real population — an attribute value.
- `area` (integer): real land area in square kilometers — an attribute value.
- `num_borders` (integer): number of bordering countries — an attribute value.
- `num_languages` (integer): number of official languages — an attribute value.

142 of these rows appear as entities in the corpus; the other 72 are the donor pool that supplies poison values and never appears as a subject.

## Features: meta.json

Build metadata, not solver input — a single JSON object recording how the corpus
was tuned and what was measured on it. No solver needs to read it; it is shipped
so the score band can be audited. Nested keys are written with dotted paths.

**Constraint parameters**

- `max_chunk_tokens` (integer): per-window token cap. Always 60.
- `budget_kfrac` (float): fraction of a document's evaluation-query count that sets its window budget. 0.45.
- `budget_min` (integer): minimum window budget any document receives. 2.

**Accepted auto-tuner configuration**

- `accepted_seed` (integer): RNG seed of the accepted corpus.
- `frac_impossible` (float): requested fraction of structurally unanswerable queries.
- `frac_hard_surface` (float): requested probability of a non-plain numeric surface. Requested, not realised — rounded styles do not apply to the small-range attributes, so the realised share is 39.4%.

**Corpus size**

- `n_docs` (integer): number of documents.
- `n_sentences` (integer): sentence rows in `documents.csv`.
- `n_train_q` (integer): queries in the `train` split.
- `n_test_q` (integer): queries in the `test` split.

**Measured score anchors** — every value a float in [0, 1] on the evaluation split

- `verified_optimum_test` (float): budgeted-DP maximum answerability; the hard ceiling.
- `greedy_alloc_test` (float): greedy max-gain allocator on ground-truth evidence.
- `naive_alloc_test` (float): one-minimal-window-per-query allocator on ground-truth evidence.
- `obvious_pipeline_test` (float): pipeline matching stated values by exact integer equality.
- `no_kb_guess_test` (float): evidence located, no knowledge-base lookup, candidate picked at random.
- `fixed_size_baselines_test` (object): budget-blind fixed-size tilings; string window sizes (`"1"`, `"3"`, `"5"`, `"8"`, `"12"`) to float scores.

**Proven shortcut invariants** — asserted at generation time; the build aborts if any fails

- `shortcut_invariants.pairs` (integer): distinct (entity, attribute) pairs planted.
- `shortcut_invariants.poison_before_frac` (float): poison placed before its fact. Required in [0.40, 0.60].
- `shortcut_invariants.fact_larger_frac` (float): true value is the larger number. Required in [0.40, 0.60].
- `shortcut_invariants.fact_longer_frac` (float): true value has more digits as written.
- `shortcut_invariants.poison_longer_frac` (float): poison has more digits. Required within 0.08 of `fact_longer_frac`.
- `shortcut_invariants.entities` (integer): countries used as question subjects.
- `shortcut_invariants.donors` (integer): countries reserved for poison values only.

## Generation process

1. **Knowledge base split**: the 214 country records are sorted by ISO code and split into 142 entities (question subjects) and 72 donors (poison-value sources only). Only those 142 countries are ever asked about, across all 228 documents.
2. **Carrier pool**: 2,442 unique naturalistic filler sentences (5–18 tokens) composed deterministically from closed slot vocabularies — no external text.
3. **Facts and twins**: for each query an entity and attribute are drawn; the poison value is a deterministic donor value obeying the direction-balance, digit-length and 25%-separation rules. Fact and twin are rendered from the *same* template and the *same* numeric surface style, so only the number differs. Within a document, country names are kept mutually non-substring so text evidence is unambiguous.
4. **Placement classes per query**: `easy`, `tight`, `impossible`, `conflict` (interleaved A1 B1 A2 B2 so no non-overlapping window set satisfies both — verified numerically at build time), and `cluster` (several direct facts packed consecutively so one window answers all, rewarding a merge over per-query windows). Poison side is a coin flip in every class.
5. **Budget**: each document's `budget_k` is `max(2, round(0.45 × #test-queries))`, so the budget binds and a solver must choose which spans to index.
6. **Auto-tuning**: candidate corpora are generated on a fixed seed schedule. For each, the exact optimum is computed by a per-document **budgeted** dynamic program and every solver anchor is measured. Two proportional controllers move the unanswerable fraction (which sets the ceiling) and the hard-surface fraction (which sets where a first-draft pipeline lands). The first candidate satisfying every gate is accepted:
   - verified optimum in [0.62, 0.85] → accepted **0.7021**
   - obvious-pipeline anchor in band → accepted **0.5156**
   - best budget-blind tiling ≤ 0.18 → accepted **0.1389**
   - optimum − naive-per-query allocator ≥ 0.10 → accepted **0.1251**
   - optimum − no-knowledge-base guesser ≥ 0.15 → accepted **0.3316**
7. **Invariant proof**: the accepted corpus is then checked against the three shortcut invariants above; generation aborts if any fails.
8. The split is random over queries. The challenge's private answer file is written in a stratified order so any contiguous row slice (a public/private leaderboard split) mirrors the global difficulty mix.

## Skills this dataset tests

Three separable skills, scored jointly by one smooth answerability number, with the measured worth of each:

1. **Workload inference** — reading each question's evidence sentences from raw text and resolving the entity, which for some question forms is named only indirectly. Budget-blind baselines score 0.1164–0.1389; locating evidence alone lifts that to 0.3517.
2. **Knowledge-base reconciliation** — telling the true statement from its adversarial twin, which is identical bar the number, and whose number may be written in several surface forms that the solver must first characterise from the labelled split. Worth **+0.3179** (0.3517 → 0.6696), the largest single lever.
3. **Capacity-constrained selection** — choosing at most `budget_k` non-overlapping token-bounded windows per document, merging clustered facts and conceding unreachable queries. Worth **+0.1251** (0.5770 → 0.7021).

A greedy allocator ties the exact budgeted DP at 0.7021 here, so exact optimization is explicitly *not* among the graded skills; the DP's role is to certify the ceiling.

Skills 2 and 3 are ablated on **separate axes**, each holding the other fixed, so their figures are not comparable across skills. A solver with perfect identification but a one-window-per-query allocator scores 0.5770, and a solver that generalises across wordings but skips the reference table scores only 0.3517 — neither dominates the other. For the same reason their gains must never be added together: they measure independent capabilities on separate axes.

## Notes

Domain-specific context a solver needs, and the assumptions it should not make.

- **`real_facts.json` is a required input, not background trivia.** A planted false sentence is identical to the true one once its number is masked, so the only way to separate them is to resolve the country and compare the value each sentence states against this table.
- **A question and the sentence answering it word the attribute differently.** 88.6% of the workload shares no attribute wording between the question and its evidence, and 14 of the 40 wordings never appear in the training split at all. Matching on shared words, or on a lookup harvested from training, both leave a large part of the evaluation set unreachable.
- **Values are not always plain digits.** 39.4% of true fact sentences use another surface form. Comparing values as literal integers misreads all of them. The forms are recoverable from the labeled split; they are deliberately not enumerated in the documentation.
- **Some questions cannot be answered by any legal submission.** Evidence may be spread wider than the 60-token cap, a false sentence may be pinned between two evidence sentences, or two questions may be interleaved so that no non-overlapping window set satisfies both. A perfect solver scores 0.7021, not 1.0. Chasing unreachable questions wastes budget.
- **The budget is the binding constraint.** `budget_k` averages 2.15 windows against 3.50 evaluation questions per document. Merging nearby evidence into one window and conceding what cannot be reached is worth more than any allocation refinement: a one-window-per-question allocator gives up 0.1251 even with perfect evidence.
- **Sentence order within a document is meaningful.** `sent_id` is the position, and windows are contiguous ranges over it. The filler prose between facts is synthetic and task-irrelevant; no question tests it.
- **Regeneration is exact.** `generate.py` with its fixed seed schedule reproduces `documents.csv`, `queries.csv`, and `meta.json` byte-for-byte, and aborts rather than emit a corpus that violates any of its built-in checks.
