# Roadmap

Status of work in progress. Four benchmarks are shipped, measured and verifiable; one is
in build and publishes no anchors until it is measured.

---

## Shipped

| Benchmark | Build | Anchors | Verified |
|---|---|---|---|
| The Accent Translator | v1 | 5 rungs | Yes |
| The Edit That Moved the Answer | v1 | 5 rungs | Yes |
| Recovering OCR Batch Origin | **v4** | 14 rungs | Yes |
| The Order of Discovery | v2 metric | 7 rungs | Yes |

All four reproduce under `python tools/verify.py --reference`.

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

## Suite-level

- **Scale the `experimental-order` corpus.** At 259 papers it is the smallest in the
  suite; corpus size was limited by PLOS API rate limiting during construction. The
  generator already supports `--articles` and a `--start` page offset for resumable
  scaling.
- **A GPU reference for `accent-transfer`.** The shipped reference is a CPU rewrite
  table at `0.768`. An encoder-decoder transformer over the source phoneme sequence, with
  both accents as prefix tokens, reads full context and is expected to score materially
  higher. The band should be re-measured against it before being treated as final.
- **A cross-encoder reference for `experimental-order`.** The shipped reference is a
  CPU pairwise TF-IDF logistic model at `0.358`. A fine-tuned cross-encoder reading
  subsection pairs jointly can judge presupposition directly, and the same caveat applies.
