# RAG Retrieval-Sufficiency Decision Trace — state as of 2026-09-16

Domain: retrieval-augmented generation (RAG), GPU. Target novelty >= 8
(platform min 6). Status: **design locked, generator not yet written (~10%).**

## Title — LOCKED 2026-09-16 (user confirmed, register rule 1.16 compliant:
## plain, descriptive, no colon subtitle, no creative project name)

    Predicting Retrieval Sufficiency in RAG Pipelines

## The task

For each item: a query, plus the top-k passages a REAL dense retriever
(BGE-small embeddings, cosine similarity, computed on GPU) actually returned
from a hand-written corpus, plus a REAL generator's (SmolLM2-1.7B-Instruct)
actual answer attempt over that retrieved context. The corpus is
deliberately constructed (see RETRIEVAL-FAILURE CONSTRUCTION below) so
retrieval genuinely succeeds sometimes and genuinely fails other times, in
three distinct, engineered ways.

Submitted artifact = a joint 3-field decision trace, not a final answer and
not a single label:

    (diagnosis, action, reason_code)

  diagnosis   in {sufficient, insufficient}
  action      in {answer_as_is, reformulate, abstain}
  reason_code in {sufficient, gold_low_rank, gold_absent, query_ambiguous}

Scored as an exact joint match across all three fields against a gold trace
(see grade.py plan) — getting action right but reason wrong is not full
credit. This is deliberately a structured multi-field decision, not a
single categorical label, to clear rejection rule 2.2 (a single label reads
as "just classification"; joint-tuple-conditioned-on-a-diagnosis is closer
in shape to prompt-delta's (edit_op, slice) pair, which passed).

## Why this is honestly "rag" (register rule 1.19)

Both steps are real and load-bearing, not decorative:
  1. Retrieval is REAL: BGE-small-en-v1.5 embeddings computed on GPU, real
     cosine similarity, real top-k ranking over a real (if hand-written)
     corpus. The gold passage's rank is MEASURED per item, never asserted.
  2. Generation is REAL: SmolLM2-1.7B-Instruct actually runs over (query +
     retrieved top-k) and produces an actual answer. A released confidence
     signal (hedge/refuse/IDK regex fingerprint, reusing prompt-delta's
     proven fingerprint() design) is computed from that REAL generation
     output, not asserted.

The gold trace is derived from retrieval facts only (rank of the gold
passage, corpus-wide ambiguity check) per the user's explicit 2026-09-16
decision — NOT from the generator's output — because that keeps grading
fully deterministic and avoids a subjective/LLM-judged gold label. But the
generator's output is still shown to the solver as part of the input
features and is the intended route to `reason_code`: a solver that ignores
the generation signal and thresholds on raw retrieval scores alone is the
shortcut to measure and close (see SHORTCUT PLAN below) — same discipline
as promp-delta's fmt_json-recall-vs-slice-spread independence check.

## Why this is not "just retrieval" or "just classification" (rule 2.2)

- Not retrieval-as-the-whole-task: the retrieved doc ids/ranks are an INPUT
  feature, never the submitted artifact.
- Not a single-label classification: the submission is a 3-field trace,
  jointly scored, where `action` and `reason_code` must be mutually
  consistent (e.g. reason_code=gold_absent implies action=abstain, never
  answer_as_is) — an internally-consistent structured decision, not an
  independent label per column.
- Not regression: no scalar output anywhere in the submission.

## Prior art scan (see full agent report, 2026-09-16)

CROWDED, avoid re-treading: hallucination-span detection (RAGTruth),
citation/attribution scoring (ALCE), distractor/noise robustness (RGB,
NoisyBench), multi-hop RAG QA (MultiHop-RAG), poisoned-corpus attacks
(PoisonedRAG), general RAG-eval frameworks (RAGAS, CRAG, RAGBench/TRACe).

UNOCCUPIED axis this task targets: "query reformulation necessity as a
structured decision trace." RetrievalQA and QPP/variant-selection papers
study retrieve-vs-not, but always collapse it to final-answer accuracy or
a routing-accuracy score — never a structured (diagnosis, action, reason)
trace as the graded submitted artifact. RAGRouter-Bench is the closest
neighbour (a routing benchmark) and must be named and differentiated in the
"how this relates to prior work" section, per rule 2.1 (cite close
neighbours explicitly and differentiate, or risk cite-or-cap).

Distant antecedents to cite instead of close neighbours (rule 2.1):
  - classical IR relevance feedback / query reformulation (Rocchio,
    pseudo-relevance feedback) — decades-old antecedent for "should the
    query be reformulated," safely distant from any 2024-2026 RAG paper.
  - triage/decision-trace literature outside NLP (e.g. structured
    diagnostic decision trees in clinical decision support) as an analogue
    for "a structured multi-step decision as the graded artifact" rather
    than a single label.

Close neighbours to name-and-differentiate (do NOT ignore, per rule 2.1):
  - RAGRouter-Bench (routing accuracy only, no reason-code, no joint trace)
  - RetrievalQA (retrieve-vs-not framing collapses to final-answer accuracy)
  - QPP-variant-selection work (predicts query difficulty, not a structured
    action+reason trace)

## RETRIEVAL-FAILURE CONSTRUCTION (measured, not asserted)

Hand-written corpus of TOPICS (like prompt-delta's FAMILIES), each with
hand-written passages and hand-written queries. Per query, the gold passage
id is fixed by construction, but WHERE it lands in real BGE retrieval is
MEASURED after embedding, and that measurement (not the construction
intent) is what the label pipeline reads:

  sufficient:      gold passage written with natural query-aligned phrasing
                    -> expect real rank near top-1/2
  gold_low_rank:    gold passage written in deliberately paraphrased /
                    indirect language (low lexical + semantic overlap with
                    the query) -> expect real rank outside top-k but still
                    in-corpus
  gold_absent:      query's answer has NO passage in this topic's corpus at
                    all -> retrieval structurally cannot find it
  query_ambiguous:  query phrased to plausibly match passages from >=2
                    different topics -> measured via real cross-topic
                    similarity, not asserted

prepare.py will READ THE ACTUAL MEASURED RANK (from the real BGE run) to
assign diagnosis/action/reason_code, falling back to reclassifying any item
whose measured rank doesn't match its construction intent (e.g. a
"sufficient"-intended item whose gold passage actually measured outside
top-k gets relabeled by its true measured facts) — same discipline as
prompt-delta never asserting a fingerprint, always reading it off a real
generation.

## LABEL RULE (draft, to implement in prepare.py once raw data exists)

Per item (query, k), read the MEASURED gold_rank_in_topic from retrieval.json
(never the construction_intent directly):

    if construction_intent == "gold_absent":
        diagnosis=insufficient, action=abstain, reason_code=gold_absent
    elif gold_rank_in_topic is None or gold_rank_in_topic > k:
        diagnosis=insufficient
        if best_other_topic_score is close to best_own_topic_score (measured
        margin, threshold TBD from real data):
            reason_code=query_ambiguous, action=reformulate
        else:
            reason_code=gold_low_rank, action=reformulate
    else:  # gold_rank_in_topic <= k
        diagnosis=sufficient, action=answer_as_is, reason_code=sufficient

ALREADY IDENTIFIED RISK (before any data exists, to attack first): this rule
distinguishes query_ambiguous from gold_low_rank using
best_own_topic_score - best_other_topic_score, which are both themselves
candidate RELEASED features. If released verbatim, a solver could
threshold on that gap alone and shortcut reason_code without reading the
generator's output at all -- the same shape of shortcut that broke
prompt-delta's first label rule (recomputing the exact released numbers
IS the label rule). Planned mitigation once real numbers exist: either (a)
do not release the raw score gap as a feature, only a coarser derived
signal, or (b) make the query_ambiguous / gold_low_rank distinction also
depend on the generator's fingerprint (e.g. query_ambiguous items should
make the generator confidently answer using the WRONG topic's fact, a
pattern invisible to a retrieval-score-only threshold). Measure the
threshold-only attacker FIRST, then decide which mitigation is needed --
do not guess which is necessary before seeing the number.

## SHORTCUT PLAN (rule 2.6 — to measure once raw data exists, not yet run)

Primary shortcut to attack: a solver that reads ONLY the released retrieval
similarity scores (e.g. top-1 cosine score, score gap to rank 2) and
thresholds on them, ignoring the generator's answer/confidence signal
entirely. If this attacker scores uncomfortably close to the intended
reference, close it by:
  - not releasing raw cosine scores directly (release a coarser/derived
    feature instead), and/or
  - making `reason_code` (specifically distinguishing gold_low_rank from
    query_ambiguous) depend on a pattern only visible in the generator's
    actual output behaviour (e.g. query_ambiguous items make the generator
    actually answer confidently but WRONG relative to the true gold topic,
    which a pure retrieval-score threshold cannot see).
Secondary shortcut: transductive leakage (rule 2A) — BGE embeddings and any
similarity statistics must be computed per-item against the FULL corpus,
never fit/normalized using test-split rows' hidden content. Corpus passages
themselves are not split by train/test (they are shared infrastructure);
only ITEMS (queries) are split by topic.

## Split (rule 4)

By TOPIC (group), never by row — same mechanism as prompt-delta's by-family
split with exact-search family-combination selection to land train/test
ratio in 15-25% while keeping the majority-baseline shortcut low on the
resulting test set. Sizing TBD once corpus/query counts are fixed (need
train >= 1000).

## Licence

Corpus, queries, retrieved-rank labels, and generation fingerprints are
100% original: corpus/queries hand-written in this repo, embeddings/
generations computed by running open-weights models over hand-written
input. Following prompt-delta's precedent exactly:
  - BAAI/bge-small-en-v1.5 (embedder) — MIT licence, ungated, verified
    2026-09-16: card explicitly states outputs usable commercially with no
    restriction.
  - HuggingFaceTB/SmolLM2-1.7B-Instruct (generator) — Apache-2.0, ungated,
    verified in prompt-delta's build: no output-ownership/restriction
    clause on the model card.
  DATASET LICENCE TO DECLARE ON PLATFORM: CC BY 4.0 (user confirmed
  2026-09-16, explicitly choosing CC BY 4.0 over prompt-delta's CC0
  precedent after the conflict was flagged).

## LICENCE — LOCKED 2026-09-16

CC BY 4.0 on the platform's licence field. Because CC BY carries an
attribution obligation (unlike CC0), every prose doc (dataset-description.md,
problem-description.md) MUST include an explicit attribution line crediting:
  - BAAI/bge-small-en-v1.5 (MIT licence, ungated) — the retrieval embedder
  - HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache-2.0, ungated) — the generator
as the instruments used to produce the released measurements, worded
identically everywhere it appears, and cross-checked against the platform's
declared licence field before packaging (register section -1's exact
lesson: Palimpsest's one real defect was prose saying CC BY-SA while the
platform field said CC BY — cross-check field vs prose word for word).

## Files (planned, mirroring prompt-delta layout)

    dataset/generator/gen_raw.py   THE GENERATOR — hand-written corpus +
                                    queries, real BGE retrieval, real
                                    SmolLM2 generation, --smoke/--full modes
    dataset/prepare.py             reads raw, measures actual ranks, derives
                                    gold trace from MEASURED facts, balances,
                                    splits by topic (exact search like
                                    prompt-delta)
    dataset/grade.py               joint 3-field exact-match metric,
                                    InvalidSubmissionError(ValueError),
                                    degrade never reject
    problem-description.md / dataset-description.md / rubrics.md /
    config.yaml / solution.ipynb   TODO after generator run

## Next session — TODO (updated 2026-09-16, generator ready to run)

1. DONE - dataset/generator/gen_raw.py written: 24 hand-written topics,
   192 passages (every passage carries a genuine numeric fact, verified),
   480 hand-written queries (6 sufficient / 6 gold_low_rank / 4 gold_absent
   / 4 query_ambiguous per topic). Real BGE-small-en-v1.5 retrieval (ranked
   within each query's own topic; cross-topic similarity measured
   separately for ambiguity detection). Item multiplier: gold_low_rank
   queries are swept across k in {2,3,4,5,6,7}; all other intents get k in
   {3,5} as a stress check only (no duplicate-answer sweep) -- produces
   1,536 items total (dry-run verified against the actual merged TOPICS
   list, exact count confirmed, no off-by-one in the k_eff-collapse dedup).
   Real SmolLM2-1.7B-Instruct generation over each item's actual retrieved
   context, fingerprinted (hedge/no-answer/context-citation/etc), never
   raw response text.
2. DONE - CC BY 4.0 licence confirmed by user 2026-09-16 (see LICENCE
   section above). Title confirmed by user: "Predicting Retrieval
   Sufficiency in RAG Pipelines".
3. DONE - grade.py written and tested against perfect/empty/partial/NaN/
   duplicate/malformed submissions. Joint per-item 3-field average,
   InvalidSubmissionError(ValueError) (fixed prompt-delta's own bug of
   subclassing plain Exception -- did not copy it here).
4. DONE - prepare.py written (label rule implementing the LABEL RULE
   section above, balancing, exact-search by-topic split) -- NOT YET RUN
   against real data (raw/retrieval.json does not exist until gen_raw.py
   runs on GPU).
5. TODO (blocks everything below) - RUN gen_raw.py ON GPU. See RUN
   INSTRUCTIONS below.
6. TODO - once raw data exists: run prepare.py, inspect the trace
   distribution it prints, and MEASURE the retrieval-score-threshold
   shortcut attacker flagged in the LABEL RULE section (this is the single
   most important number to get before anything else -- if it's
   uncomfortably high, the AMBIGUITY_MARGIN-based split needs the
   generator-fingerprint mitigation already planned, not a guess).
7. TODO - build and measure the full ablation ladder (5+ rungs): uniform
   random, always-most-common-combo, retrieval-score-threshold-only
   attacker, a real trained/heuristic solver (logistic regression or
   gradient boosting on released features), the intended reference. Report
   every number per the SCORE-BAND REQUIREMENT (chance floor ~0.25-0.33
   per field depending on internal consistency constraints -- compute the
   real joint-metric chance floor once labels exist, do not assume).
8. TODO - problem-description.md, dataset-description.md (describe the RAW
   files: retrieval.json/grid.json/meta.json AND the derived public/private
   files, with the CC BY 4.0 attribution line for both models worded
   identically to the platform's licence field), rubrics.md (5+ specific),
   config.yaml, solution.ipynb.

## RUN INSTRUCTIONS (for the user, on their GPU)

    cd gpu/rag-retrieval-diagnosis/dataset/generator
    pip install torch transformers sentence-transformers accelerate numpy
    python gen_raw.py --out ../raw --smoke     # quick check, 3 topics
    python gen_raw.py --out ../raw             # full build, ~10-15 min on A10G

Full build writes dataset/raw/{retrieval.json,grid.json,meta.json}. After
that, next session runs `python prepare.py` from dataset/ to produce
public/private splits, then measures the shortcut attacker before anything
is packaged.
