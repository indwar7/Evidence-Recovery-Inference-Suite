# Dataset Description

## Overview

The corpus of system prompts is real, publicly published data: the
danielrosehill/System-Prompt-Library collection on Hugging Face, 923
hand-authored assistant system prompts covering a wide range of everyday
tools (coding helpers, writing assistants, tech-recommendation bots,
research summarizers, and more), released under CC BY 4.0.

That raw collection was filtered down to a clean subset before any model
was run: prompts that personalize to the collection's author by name were
excluded entirely, as were prompts built as numbered, code-fenced, or
bulleted multi-step procedures (their steps are causally dependent on
each other - removing step 3 from a five-step procedure is not a clean
test of one independent instruction, unlike removing a flat clause from a
list of separate rules). What remains are prompts composed of three to
ten sentence-level, largely independent instructions - the raw material
this dataset's ablation experiment needs.

Each surviving prompt was split into its individual clauses (one per
sentence) and assigned to a topic group via deterministic keyword
matching on its name and description (no natural category field exists in
the source data - see Task Construction).

For every surviving prompt, a real instruction-tuned language model,
HuggingFaceTB/SmolLM2-1.7B-Instruct, was run against eight hand-written
probe questions, once with the FULL system prompt in effect, and once
more for EACH clause with that clause individually removed (all others
left intact) - a genuine counterfactual ablation experiment, on GPU, per
clause, per prompt. Every generated answer was reduced to a small set of
numeric behavior measurements - never the response text itself.

No response text is stored or released anywhere in this dataset. The
probe questions are hand-written; the filtering rules and topic-keyword
rules are hand-written; the model is used only as an instrument being
measured, and the recorded numbers - behavior measurements per clause
configuration - are the measurement.

Corpus: danielrosehill/System-Prompt-Library
(huggingface.co/datasets/danielrosehill/System-Prompt-Library), licensed
CC BY 4.0. Model: HuggingFaceTB/SmolLM2-1.7B-Instruct, licensed Apache-2.0;
its card places no claim or restriction on generated outputs. The
recorded measurements (the ablation behavior fingerprints) are therefore
an original derived work, released here under CC BY 4.0 with attribution
to the source corpus above, matching the platform's declared licence
field exactly.

Decoding was greedy (do_sample=False) with a fixed seed, so the raw file
is reproducible bit-for-bit on the same hardware and against the same
downloaded corpus snapshot.

## File Structure

This dataset upload contains exactly these files at the top level of the
raw/ directory:

- ablation.json - one JSON object holding every measured per-probe
  behavior fingerprint.
- grid.json - the filtered prompt corpus (each prompt's clauses, topic
  group, and source metadata) exactly as used to build ablation.json.
- meta.json - build metadata: model id, corpus source, seed, prompt and
  generation counts, elapsed time, device, and a sha256 of ablation.json
  for integrity checking.

The generator script that produced these files, generator/gen_raw.py, is
included in this upload. It downloads the source corpus from Hugging Face
at build time and is deterministic given the same model weights, seed,
and corpus snapshot.

## Features

### ablation.json

- schema (int): Format version.
- gen_model_id (str): HuggingFaceTB/SmolLM2-1.7B-Instruct.
- gen_model_licence (str): Apache-2.0; no restriction on outputs.
- corpus_source (str): danielrosehill/System-Prompt-Library (Hugging Face).
- corpus_licence (str): CC-BY-4.0.
- dataset_licence (str): CC-BY-4.0.
- seed (int): Fixed generation seed.
- max_new_tokens (int): Generation cap per response.
- decoding (str): greedy (do_sample=False).
- fingerprint_fields (list of str): The behavior measurement names, in
  the order every measurement vector uses.
- probe_queries (list of str): The eight hand-written probe questions
  asked against every prompt configuration.
- n_prompts (int): Total surviving prompts after filtering.
- n_generations (int): Total individual model generations run.
- raw_fingerprints (object): Keyed "promptId|ablatedClauseOrNone|
  probeIndex", each value a list of floats/ints in fingerprint_fields
  order - the measurement from ONE generation. Released at this per-probe
  granularity (not pre-averaged) so downstream item construction can
  derive as many probe-subset-averaged items as needed without any
  additional model runs. This is the entire measured dataset; everything
  in public/ and private/ is derived from this object plus grid.json.

### grid.json

- prompts (list of objects): Each has prompt_id, agentname, topic,
  clause_ids (list of str, in the ORIGINAL prompt order - not the
  randomized display order used downstream), and clause_texts (an object
  mapping each clause id to its hand-split sentence text).

### meta.json

- generated_by, seed, gen_model_id, corpus_source, n_prompts,
  n_generations, ablation_sha256, elapsed_seconds, device.

## Task Construction

The task's items are built by prepare.py from ablation.json and
grid.json:

- Each surviving prompt is shown across several items by varying WHICH
  four of the eight probe questions its full-prompt fingerprint (and each
  clause's ablated fingerprint) is averaged over. Ten of the seventy
  possible four-of-eight combinations are used, chosen deterministically
  (not randomly), so the same prompt genuinely produces different
  measured responsibility rankings across its several items - this
  multiplier was necessary because the filtered, clean corpus is smaller
  (145 prompts) than a naturally-occurring one-item-per-prompt design
  would need to clear the platform's minimum item count.
- For each item, every clause's responsibility score is the standardized
  distance between the full-prompt fingerprint and that clause's
  individually-ablated fingerprint, both computed over the item's own
  probe subset. The true ranking sorts clauses by this score; the
  necessary/correlated flag threshold is set at the 70th percentile of
  the score distribution across the whole shipped dataset (see STATE.md
  for the exact measured value).
- Each item's DISPLAYED clause order is independently reshuffled
  (deterministically, keyed by prompt and probe-subset) so that a
  clause's position in the original prompt text carries no information
  about its position in what a solver sees - this closes a measured
  shortcut risk (position in the original prompt correlating with
  responsibility, since persona/role clauses conventionally come first
  and formatting/fallback rules last).

## Split Design

The split is by **topic group**, not by row. Every item built from one
topic group lands entirely in train or entirely in test. Several topic
groups are held out entirely for testing, chosen by exact search over
which groups land the test/train item-count ratio inside 15-25% - see
STATE.md's FINAL LOCKED NUMBERS for the exact groups and ratio.

With the topic-disjoint split, a solver has never seen a held-out topic
group's own prompt phrasing or clause style during training and cannot
memorize a per-topic shortcut; it has to learn how the released behavior
measurement relates to clause responsibility in a way that transfers
across domains.

## Measured Difficulty

See STATE.md's FINAL LOCKED NUMBERS section for the chance floor, every
naive baseline, the position-only shortcut attacker, a real trained
solver, and the intended reference solution, all measured directly on the
shipped dataset with the task's own grader.

## Intended Use

This dataset is built for one task: given a system prompt's clauses and
the model's actual full-prompt behavior measurement, submit a ranking of
clause responsibility plus a per-clause necessary/correlated flag (see
problem-description.md). The submitted answer is a structured, jointly-
scored multi-part decision - not a single label, not a scalar sensitivity
score, and not a rewritten or optimized prompt. It is not intended as a
general-purpose corpus of model responses, a benchmark of either the
source corpus's authors or SmolLM2's overall capability, or a source of
training data for other tasks; no response text is released, which alone
rules out most other uses.

## Known Limitations

- **Responses were capped at a fixed generation length.** A different cap
  could shift some behavior measurements and could change which clauses
  land closest to the necessary/correlated boundary. The cap was fixed
  for the entire build, so it is consistent within this dataset, but the
  released numbers should not be read as the model's unconstrained
  behavior.
- **One model only.** Every measurement comes from a single 1.7B-parameter
  instruction-tuned model, HuggingFaceTB/SmolLM2-1.7B-Instruct. Which
  clause is most responsible for a given behavior is not claimed to
  generalize to other models, other model sizes, or other instruction-
  tuning recipes - it is a property measured on this model, on this
  corpus, under greedy decoding.
- **The filtered corpus favors flatter, shorter prompts.** Excluding
  numbered/bulleted/code-fenced procedures and personalized prompts means
  this dataset's clause-attribution problem is easier to pose cleanly
  than it would be on the full, unfiltered range of real-world system
  prompts, many of which are longer, more structured, or more
  personalized than what is released here.
- **Greedy decoding is deterministic per-hardware, not necessarily
  bit-identical across hardware.** The raw file was generated once, on
  one GPU, with a fixed seed; floating-point non-associativity means a
  regeneration on different hardware could produce slightly different
  token-level outputs and, in turn, slightly different fingerprint
  values, even though the generation procedure itself is fully
  deterministic given fixed hardware. The shipped raw/ablation.json is
  the authoritative artifact; prepare.py derives train/test/answers from
  it directly and does not regenerate anything.
- **The behavior fingerprint fields are surface statistics, not a
  semantic measure.** They count things like hedging phrases, refusal
  language, and length; they do not verify factual correctness or
  capture meaning. Two answers with identical fingerprints could differ
  substantially in content, and this dataset cannot distinguish that.
- **The source corpus is one collection, not a random sample of
  real-world system prompts.** danielrosehill/System-Prompt-Library
  reflects one author's personal tool-building habits; the diversity of
  domains here is representative of that collection's own range, not of
  system-prompt engineering practice in general.
