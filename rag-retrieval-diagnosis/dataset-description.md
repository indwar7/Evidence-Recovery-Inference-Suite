# Dataset Description

## Overview

This is a fictional, synthetic dataset: every knowledge-base topic,
passage, and query below was hand-written for this challenge and does not
describe any real company, product, law, or identifiable real-world
policy. Any resemblance to a real organization's actual terms is
coincidental.

The raw data is built by running two open-weights models as measurement
instruments over that hand-written corpus:

1. A real dense embedding model, BAAI/bge-small-en-v1.5, encodes every
   passage and every query, and real cosine similarity retrieval is
   computed - for each query, every passage in that query's own topic is
   ranked by similarity, and the actual measured rank of the query's
   designated answering passage is recorded (or that it does not appear in
   its own topic's corpus at all, for queries built with no answering
   passage).
2. A real instruction-tuned language model, HuggingFaceTB/SmolLM2-1.7B-
   Instruct, is run over each query together with its actual top-ranked
   retrieved passages, and its response is reduced to a small set of
   numeric behaviour measurements - never the response text itself.

Twenty-four knowledge-base topics were written by hand (a leave policy, an
expense policy, a device-setup guide, a returns policy, and so on), each
with several hand-written passages and twenty hand-written queries per
topic. Each query is written with one of four intents: a query whose
answering passage should rank near the top of its topic, a query whose
answering passage is deliberately paraphrased so it should rank lower, a
query with no answering passage anywhere in its topic, and a query worded
generically enough to plausibly match a different topic. The actual
measured retrieval rank - not the intent it was written to trigger - is
what the derived task labels are built from; see prepare.py's "LABEL RULE"
for the exact derivation.

Every hand-written query is additionally shown at more than one retrieval
depth (how many top-ranked passages are retrieved), so the same query can
appear multiple times in the raw generation log at different depths - see
"Task Construction" for why this is not a duplicated answer.

No response text is stored or released anywhere in this dataset. The
topics, passages, queries, and retrieval depths are all hand-written; the
two models are used only as instruments being measured, and the recorded
numbers - similarity scores, ranks, and answer-behaviour measurements - are
the measurement.

Models: BAAI/bge-small-en-v1.5 (embedding), licensed MIT; and
HuggingFaceTB/SmolLM2-1.7B-Instruct (generation), licensed Apache-2.0.
Neither model's card places any claim or restriction on outputs. The
recorded measurements are therefore an original, generated artifact,
released here under CC BY 4.0 - attribution to both models is given above
and in config.yaml's citation field, matching the platform's declared
licence field exactly.

Retrieval and generation are both deterministic given fixed weights,
seed, and hardware (greedy decoding, do_sample=False), so the raw file is
reproducible bit-for-bit on the same hardware.

## File Structure

This dataset upload contains exactly these files at the top level of the
raw/ directory:

- retrieval.json - one JSON object holding every item's retrieval
  measurements and every item's generation behaviour fingerprint.
- grid.json - the twenty-four topics, their passages, and their queries
  (with construction intent), exactly as used to build retrieval.json.
- meta.json - build metadata: model ids, seed, item count, elapsed time,
  device, and a sha256 of retrieval.json for integrity checking.

The generator script that produced these files, generator/gen_raw.py, is
included in this upload. It is self-contained and deterministic: given the
same model weights, seed, and hardware, it reproduces retrieval.json
byte-for-byte.

## Features

### retrieval.json

- schema (int): Format version.
- gen_model_id (str): HuggingFaceTB/SmolLM2-1.7B-Instruct.
- embed_model_id (str): BAAI/bge-small-en-v1.5.
- gen_model_licence, embed_model_licence (str): Licence strings for each
  model, and a note that neither restricts output usage.
- dataset_licence (str): CC-BY-4.0.
- seed (int): Fixed generation seed.
- max_new_tokens (int): Generation cap per response.
- decoding (str): greedy (do_sample=False).
- k_sweep_gold_low_rank (list of int): The retrieval depths swept for
  queries whose answering passage was written to rank low.
- k_variants_other (list of int): The (smaller) set of retrieval depths
  used for every other query intent, as a stress check rather than a full
  sweep - see Task Construction for why these differ.
- fingerprint_fields (list of str): The behaviour measurement names, in
  the order every measurement vector uses.
- topics (list of str): The twenty-four hand-written topic ids.
- n_passages (int): Total passages across all topics.
- n_queries (int): Total hand-written queries across all topics.
- n_items (int): Total items after the retrieval-depth multiplier - see
  Task Construction.
- items (object): Keyed by an internal item key, each value holds the
  item's topic_id, query text, gold_id (the true answering passage id or
  "none"), construction_intent, k (retrieval depth), the retrieved passage
  ids and their real similarity scores, the measured gold_rank_in_topic
  (or null if absent from that ranking), and the measured best-own-topic
  and best-other-topic similarity scores used for ambiguity detection.
  This is the entire measured retrieval dataset; everything in public/ and
  private/ is derived from this object plus the generation fingerprints
  below.
- generation_fingerprints (object): Keyed by the same item keys, each
  value a list of floats/ints in fingerprint_fields order, measuring the
  generator's actual answer over that item's retrieved context.

### grid.json

- topics (list of objects): Each has topic_id, passages (an object mapping
  passage id to its hand-written text), and queries (a list of objects,
  each with text, gold_id, and construction intent).

### meta.json

- generated_by, seed, gen_model_id, embed_model_id, n_queries, n_passages,
  n_items, retrieval_sha256, elapsed_seconds, device.

## Task Construction

The task's items are built by prepare.py from retrieval.json and
grid.json:

- Each hand-written query becomes multiple items by varying k (the
  retrieval depth). This is only a genuine, non-duplicate multiplier for
  queries whose answering passage was written to rank low: sweeping k
  across the measured rank flips the true diagnosis from insufficient to
  sufficient once k reaches that rank, producing a real per-item-different
  label from the same query. For every other query intent, the diagnosis
  does not depend on k (the answering passage is always top-ranked, always
  absent, or the query is always cross-topic-confusable regardless of
  depth), so only a small, fixed set of depths is kept per query as a
  stress check, rather than repeating the identical answer at every depth.
- For each item, the diagnosis, action, and reason_code are derived from
  the MEASURED gold_rank_in_topic versus that item's own k - never from
  the query's construction_intent directly. An item written with one
  intent whose true measured rank contradicts that intent is relabeled by
  its measured facts; see prepare.py's "LABEL RULE" docstring for the
  exact rule and the shortcut this was checked against.
- Within each topic, items are capped per (diagnosis, action, reason_code)
  combination so no single combination dominates every item in a topic.
- See STATE.md's "FINAL LOCKED NUMBERS" (added once the full GPU build
  completes) for the exact surviving item count and train/test split.

## Split Design

The split is by **topic**, not by row. Every item built from one topic
lands entirely in train or entirely in test. Several topics are held out
entirely for testing, chosen by exact search over which topics land the
test/train item-count ratio inside 15-25% while also keeping a
majority-vote-among-combinations shortcut near chance on the resulting test
set - both properties measured, not assumed.

With the topic-disjoint split, a solver has never seen a held-out topic's
own passages, phrasing, or baseline retrieval behaviour during training and
cannot look an answer up; it has to learn how retrieval sufficiency and
generation behaviour relate in a way that transfers across topics.

## Measured Difficulty

See STATE.md's "FINAL LOCKED NUMBERS" section, added once the full GPU
build and shortcut-attacker measurement complete. That section reports the
chance floor, every naive baseline, the retrieval-score-only shortcut
attacker, a real trained solver, and the intended reference solution, all
measured directly on the shipped dataset with the task's own grader.

## Intended Use

This dataset is built for one task: given a query, its actual retrieved
passages and their real similarity scores, and the actual behaviour of a
model's answer over that retrieved context, submit a structured decision
trace diagnosing whether retrieval was sufficient and what the system
should do next (see problem-description.md). The submitted answer is a
jointly-scored three-field trace, not a single-label classification, not a
similarity score, and not a re-ranking of the retrieved passages - the
retrieval has already happened and is an input feature, never the target.
It is not intended as a general-purpose corpus of model responses, a
benchmark of either model's overall capability, or a source of training
data for other tasks; no response text is released, which alone rules out
most other uses.

## Known Limitations

- **Responses were capped at a fixed generation length.** A different cap
  could shift some behaviour measurements and could change which items
  land closest to the diagnosis boundary. The cap was fixed for the entire
  build, so it is consistent within this dataset, but the released numbers
  should not be read as either model's unconstrained behaviour.
- **One embedding model and one generation model only.** Every measurement
  comes from BAAI/bge-small-en-v1.5 and HuggingFaceTB/SmolLM2-1.7B-
  Instruct. Which retrieval failures are hardest to diagnose is not
  claimed to generalize to other embedding models, other generators, or
  other model sizes - it is a property measured on this pairing, on this
  corpus, under greedy decoding.
- **Greedy decoding is deterministic per-hardware, not necessarily
  bit-identical across hardware.** The raw file was generated once, on one
  GPU, with a fixed seed; floating-point non-associativity means a
  regeneration on different hardware could produce slightly different
  token-level outputs and, in turn, slightly different fingerprint values,
  even though the generation procedure itself is fully deterministic given
  fixed hardware. The shipped raw/retrieval.json is the authoritative
  artifact; prepare.py derives train/test/answers from it directly and
  does not regenerate anything.
- **The behaviour fingerprint fields are surface statistics, not a
  semantic measure.** They count things like hedging phrases, explicit
  context citation, and length; they do not verify factual correctness or
  capture meaning. Two answers with identical fingerprints could differ
  substantially in content, and this dataset cannot distinguish that.
- **All twenty-four topics, their passages, and their queries were
  hand-written for this task**, not drawn from a broader corpus of
  real-world knowledge bases or real user queries - so the diversity here
  is representative of the categories chosen, not of retrieval-augmented
  systems in production generally.
