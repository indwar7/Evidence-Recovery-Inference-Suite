# Predicting Retrieval Sufficiency in RAG Pipelines

**Domain:** retrieval-augmented generation · **Task type:** structured prediction
**Metric:** joint field accuracy · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC BY 4.0

> **Status: in build (~40%).** The design is locked, the generator, grader and
> preparation script are written and tested, and 1,536 items across 24 topics are
> specified. **The baseline ladder has not been measured, so no anchors are published** —
> `config.yaml` carries `TBD` in every anchor slot rather than an estimate. Full status
> and remaining work: [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md).

## The task

For each item you are given a query, the top-*k* passages a **real** dense retriever
actually returned, and a **real** generator's actual answer attempt over that retrieved
context. You submit a joint decision trace:

```
diagnosis    in {sufficient, insufficient}
action       in {answer_as_is, reformulate, abstain}
reason_code  in {sufficient, gold_low_rank, gold_absent, query_ambiguous}
```

All three fields are scored together per item. They must be mutually consistent:
`reason_code=gold_absent` implies `action=abstain`, never `answer_as_is`. Getting the
action right and the reason wrong is not full credit.

The retrieved document ids are an **input feature**, never the submitted artifact — this
is not retrieval-as-the-task, and it is not a single-label classification.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration; anchor slots marked `TBD` until measured |
| `STATE.md` | The full design log: task shape, prior-art scan, label rule, pre-registered shortcut analysis |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, measurement method, schema |
| `rubrics.md` | The rules a submission must satisfy |
| `prepare.py` | Label derivation from measured facts, balancing, exact-search by-topic split |
| `grade.py` | `grade(submission, answers) -> float`; joint per-item three-field average |
| `dataset/generator/gen_raw.py` | The generator — real BGE retrieval, real SmolLM2 generation |

## Build it

```bash
cd dataset/generator
pip install torch transformers sentence-transformers accelerate numpy
python gen_raw.py --out ../raw --smoke     # quick check, 3 topics
python gen_raw.py --out ../raw             # full build, ~10-15 min on an A10G

cd ../.. && python prepare.py              # raw -> public/private
```

## Data

Original throughout. 24 hand-written knowledge-base topics, 192 passages (each carrying a
genuine numeric fact), 480 hand-written queries — all fictional, describing no real
organisation's actual policy. `BAAI/bge-small-en-v1.5` (MIT, ungated) computes real
cosine-similarity retrieval; `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0, ungated)
runs greedily over each query's actual retrieved context.

Retrieval is engineered to genuinely succeed sometimes and genuinely fail in three
distinct ways — and `prepare.py` reads the **measured** rank, not the construction
intent. An item built to be `sufficient` whose gold passage actually measures outside
top-*k* is relabelled by what was measured.

Split is **by topic**, chosen by exhaustive search over same-size topic combinations for
the test ratio *and* the lowest measured majority-combination shortcut.

**CC BY 4.0.** Attribution to both models is required and is worded identically in every
prose file — see [`../../docs/LICENSES.md`](../../docs/LICENSES.md).

## The risk registered before any data existed

The label rule distinguishes `query_ambiguous` from `gold_low_rank` using
`best_own_topic_score − best_other_topic_score` — and both terms are candidate *released*
features. If released verbatim, a solver could threshold on that gap and shortcut
`reason_code` without ever reading the generator's output. Two mitigations are
pre-registered in `STATE.md`, and **which one is needed will be decided from the measured
attacker score, not guessed.** Measuring that attacker is the first number to be taken
once the generator runs.
