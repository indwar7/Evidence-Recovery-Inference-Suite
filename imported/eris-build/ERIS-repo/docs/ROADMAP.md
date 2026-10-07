# Roadmap

Status of work in progress. Eight benchmarks are shipped, measured and verifiable; three
are in build. An in-build benchmark publishes no anchors in the suite tables until it is
measured and its reference clears the ladder.

---

## Shipped

| Benchmark | Build | Anchors | Verified |
|---|---|---|---|
| The Accent Translator | v1 | 5 rungs | Yes |
| The Edit That Moved the Answer | v1 | 5 rungs | Yes |
| Recovering OCR Batch Origin | **v4** | 14 rungs | Yes |
| The Order of Discovery | v2 metric | 7 rungs | Yes |
| Reconstructing Local Citation Structure | query-and-pool shape | 16 rungs | Floor only |
| The Reference Shuffle | fixed pool of three | 10 rungs | Floor only |
| What Does It Do In There? | ingredient-group split | 15 rungs | Yes |
| The Vanished Clause | gap-anchored metric | 11 rungs | Yes — **no working baseline** |

The six marked Yes reproduce floor and reference under `python tools/verify.py
--reference`. The two marked "Floor only" have references that need packages outside
`requirements.txt`, so `verify.py` re-scores their floors and not their references.

---

## In build — Predicting Retrieval Sufficiency in RAG Pipelines

**`benchmarks/rag-sufficiency/`** · roughly 40% complete · design locked 2026-09-16

### The task

For each item: a query, the top-*k* passages a **real** dense retriever actually returned
from a hand-written corpus, and a **real** generator's actual answer attempt over that
retrieved context. The submitted artifact is a joint decision trace:

```
diagnosis    in {sufficient, insufficient}
action       in {answer_as_is, reformulate, abstain}
reason_code  in {sufficient, gold_low_rank, gold_absent, query_ambiguous}
```

scored per item across all three fields. The fields must be mutually consistent:
`reason_code=gold_absent` implies `action=abstain`.

### Why the shape is a trace and not a label

Both pipeline steps are real and load-bearing. Retrieval is genuine BGE-small cosine
similarity over a real corpus, and the gold passage's rank is **measured per item, never
asserted**. Generation is a genuine SmolLM2 run over the retrieved context, fingerprinted
rather than stored. The retrieved document ids are an *input feature*, never the
submitted artifact — so this is not retrieval-as-the-task — and the artifact is a
three-field internally-consistent decision rather than a single categorical label.

### Corpus construction

Retrieval is engineered to genuinely succeed sometimes and genuinely fail in three
distinct ways:

| Intent | Construction | Measured outcome |
|---|---|---|
| `sufficient` | Gold passage written with natural query-aligned phrasing | Expect real rank near top-1/2 |
| `gold_low_rank` | Gold passage written in deliberately indirect language | Expect real rank outside top-*k*, still in corpus |
| `gold_absent` | No passage in the topic answers the query | Retrieval structurally cannot find it |
| `query_ambiguous` | Query phrased to plausibly match ≥2 topics | Measured by real cross-topic similarity |

`prepare.py` reads the **measured** rank, not the construction intent. An item built to
be `sufficient` whose gold passage actually measures outside top-*k* is relabelled by
what was measured.

### Done

- Generator written: 24 hand-written topics, 192 passages each carrying a genuine
  numeric fact, 480 hand-written queries. Real BGE-small retrieval ranked within each
  query's topic; cross-topic similarity measured separately for ambiguity detection.
  Item multiplier sweeps `gold_low_rank` queries across *k* ∈ {2..7} and all other
  intents across *k* ∈ {3, 5}, producing **1,536 items** — dry-run verified against the
  merged topic list, exact count confirmed.
- Grader written and tested against perfect, empty, partial, NaN, duplicate and
  malformed submissions. Raises `InvalidSubmissionError(ValueError)` — correcting a real
  bug in an earlier task's grader, which subclassed plain `Exception`, rather than
  copying it.
- Preparation script written, implementing the measured-fact label rule, outcome
  balancing and exact-search by-topic split.
- Licence confirmed: CC BY 4.0, with the two-model attribution line worded and locked.
- Title locked.

### Remaining

1. **Run the generator on GPU.** About 10–15 minutes on an A10G. This blocks everything
   below, because `dataset/raw/retrieval.json` does not exist until it runs.
2. **Measure the pre-registered shortcut attacker first.** A solver reading only released
   retrieval similarity scores and thresholding on them, ignoring the generator signal
   entirely. This is the single most important number in the build, and the reason is
   below.
3. **Build the full ladder** — uniform random, majority combination, the threshold
   attacker, a trained classifier on released features, the intended reference — and
   compute the real joint-metric chance floor rather than assuming one.
4. **Write the solver-facing documentation**, fill every `TBD` anchor in `config.yaml`
   from an actual grader run, and build `solution.ipynb`.

### A risk registered before any data existed

The label rule distinguishes `query_ambiguous` from `gold_low_rank` using
`best_own_topic_score − best_other_topic_score` — and both terms are themselves candidate
*released* features. If released verbatim, a solver could threshold on that gap alone and
shortcut `reason_code` without ever reading the generator's output. **Recomputing the
released numbers would be the label rule.** This is the same failure that broke an
earlier task's first label rule in this suite.

Two mitigations are pre-registered:

- **(a)** do not release the raw score gap, only a coarser derived signal; or
- **(b)** make the `query_ambiguous` / `gold_low_rank` distinction depend on a pattern
  only visible in the generator's actual output — `query_ambiguous` items should make the
  generator answer *confidently but from the wrong topic*, which a retrieval-score
  threshold cannot see.

**Which mitigation is needed will be decided from the measured attacker score, not
guessed in advance.** That ordering is the point: the attack is measured first, and the
design responds to the number.

### Prior-art position

The crowded neighbourhoods were scanned and are deliberately avoided: hallucination-span
detection (RAGTruth), citation and attribution scoring (ALCE), distractor robustness
(RGB, NoisyBench), multi-hop RAG QA (MultiHop-RAG), poisoned-corpus attacks
(PoisonedRAG), and general RAG evaluation frameworks (RAGAS, CRAG, RAGBench/TRACe).

The axis this task targets — *query reformulation necessity as a structured decision
trace* — is unoccupied. RetrievalQA and query-performance-prediction work study
retrieve-versus-not, but collapse it to final-answer accuracy or a routing score, never a
structured `(diagnosis, action, reason)` trace as the graded artifact. The closest
neighbours (RAGRouter-Bench, RetrievalQA, QPP variant selection) are named and
differentiated in the task's own documentation rather than ignored.

---

## In build — Coining Gene Symbols from Gene Names

**`benchmarks/gene-symbol-coinage/`** · roughly 60% complete

### The task

Given the approved name of a human gene, `solute carrier family 2 member 1`, produce the
symbol the nomenclature committee coined for it, `SLC2A1`. Scored by prefix agreement:
shared leading characters over the longer of the two lengths.

### Done

- Corpus fetched from a pinned HGNC snapshot with a sha256 check. CC0 1.0, quoted from
  the HGNC's own licence page.
- Split by the archive's own approval date, cutoff 2010-01-01: 13,863 train, 2,270 test.
- Grader, preparation script, submission validator and dataset description written.
- A 13-rung ladder measured on the shipped split.

### A split that measurement refused

A family-disjoint split was built first. On it, a trained sequence model scored `0.258`
against `0.337` for a one-line initials heuristic: the reference landed below a trivial
baseline, because the split had removed the signal along with the shortcut. It was
replaced by the date split, which is also the honest framing. Train is what the
committee had already decided; test is what it decided next.

### Remaining

1. **Close the adversary margin.** A lookup adversary, nearest training name with the
   designator swapped, scores `0.366` against the reference's `0.447`. That is `+0.081`
   against a target of about `+0.18`. A retrieval-augmented reference is written and
   unmeasured.
2. **Build the ablation ladder** of the real solver, not only baselines.
3. **Write `config.yaml`, the problem description, the rubrics and the notebook.**
4. **Run the full checklist as a script.** The ladder was measured; the schema, leakage
   and grader checks have not been run.

The benchmark's own [`STATUS.md`](../benchmarks/gene-symbol-coinage/STATUS.md) carries
the full ladder and the resume commands.

---

## In build — Attributing Assistant Behavior to System Prompt Clauses

**`benchmarks/clause-responsibility/`** · roughly 25% complete · design locked 2026-09-20

### The task

Given a system prompt's sentences in shuffled order and the behaviour measured under the
full prompt, submit a ranking of the sentences by responsibility and a `necessary` or
`correlated` flag for each. Ground truth is a real ablation: each sentence is removed in
turn and the behaviour re-measured on the same 8 probe questions.

### Done

- Source and licence confirmed: `danielrosehill/System-Prompt-Library`, CC BY 4.0.
- Corpus fetched, 923 of 923 files, and filtered to 145 prompts. 440 are dropped for
  naming their author, 286 for numbered or bulleted structure whose steps depend on each
  other, 52 for sentence count outside 3 to 10.
- Generator, preparation script and grader written.
- Three generator faults fixed before any run: an undefined name that would have
  crashed the run after all generation had finished, no checkpoint, and a corpus cache
  that sat inside the raw directory.

### Remaining

1. **Run the generation on a GPU.** 145 prompts, about 8,000 generations. It was
   attempted on an 8 GB laptop and did not complete.
2. **Fix the metric before measuring anything else.** By arithmetic, about 70% of
   sentences are `correlated`, so answering "all correlated" in any order would score
   about `0.60` while reading nothing. Both halves of the score need correcting for
   chance, so that the sample submission scores near zero.
3. **Standardise the measurements before taking distances.** The distance is currently
   in raw units, so word count swamps the yes/no fields.
4. **Replace the probe subsets.** The ten in use all contain probes 0 and 1 and share 2
   or 3 of their 4 probes. The 14 four-element subsets of 0 to 7 whose values XOR to
   zero share either 0 or 2.
5. **Measure the ladder**, with a sentence-length probe first. Removing a long sentence
   removes more of the prompt, so length alone may predict responsibility.
6. **Write the solver-facing documents** against the final metric and fill every `TBD`.

### A risk registered before any data exists

**The experiment can be re-run.** A solver with the same model and the same 8 questions
could remove each sentence and recompute the answer key; decoding is greedy, so the
result is reproducible. The closure is layered: the probe questions appear in no
solver-visible file, the solver-facing text does not name the model, and re-running the
measurement is made invalid by rule. Pretrained text encoders stay allowed. The rule is
against repeating the experiment, not against pretrained models.

---

## Suite-level

- **Scale the `experimental-order` corpus.** At 259 papers it is the smallest in the
  suite; corpus size was limited by PLOS API rate limiting during construction. The
  generator already supports `--articles` and a `--start` page offset for resumable
  scaling.
- **A GPU reference for `accent-transfer`.** The shipped reference is a CPU rewrite
  table at `0.768`. An encoder-decoder transformer over the source phoneme sequence, with
  both accents as prefix tokens, reads full context and is expected to score materially
  higher. The band should be re-measured against it before being treated as final.
- **A working baseline for `amendment-reversal`.** Every shipped approach scores at or
  below `0.012`, and the reference scores `0.002`. The split is by CFR part, so lookup
  does not transfer. Perfectly restoring one sentence per section is worth `0.749`, so
  the path is a model that locates the amended passage and regenerates it.
- **Offline references for `citation-structure` and `reference-order`.** Both
  references sit outside `tools/verify.py` because of their dependencies. Either the
  dependencies move into an optional requirements file with a verify flag, or a
  dependency-free rung is promoted to the verified reference.
- **A cross-encoder reference for `experimental-order`.** The shipped reference is a
  CPU pairwise TF-IDF logistic model at `0.358`. A fine-tuned cross-encoder reading
  subsection pairs jointly can judge presupposition directly, and the same caveat applies.
