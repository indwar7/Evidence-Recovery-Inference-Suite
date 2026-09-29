# System-Prompt Clause Responsibility Assignment — state as of 2026-09-20

Domain: prompt engineering, GPU. Target novelty >= 8 (platform min 6).
Status: **design locked, generator not yet written.**

REPLACES the earlier RAG design (gpu/rag-retrieval-diagnosis/): abandoned
because that corpus was hand-written with no citable public source_url,
and the platform requires a real, citable dataset source (see that
project's STATE.md for the full pivot history — kept for reference, not
being built further).

## Title — LOCKED 2026-09-20 (user confirmed, register rule 1.16 compliant)

    Attributing Assistant Behavior to System Prompt Clauses

## The task (storytelling framing for problem-description.md)

An assistant is behaving oddly on some queries — too hedgy, too verbose,
refusing things it shouldn't, or the reverse. Its system prompt has several
independent instructions stacked together (a persona line, a scope
restriction, a tone rule, a fallback rule, and so on). Which instruction is
actually CAUSING the odd behavior, and which ones are just sitting there
uninvolved?

Given a system prompt split into its constituent clauses, and the model's
REAL behavior fingerprint (numeric measurements on held-out probe queries
run WITH the full prompt), submit:
  1. an ORDERED ranking of the clauses by responsibility for that behavior
  2. a necessary/merely-correlated binary flag per clause

Ground truth comes from REAL counterfactual ablation: the full prompt is
run, then EACH clause is individually removed and the prompt re-run, on
GPU, and the fingerprint shift is measured per clause. This is the answer
key; it is never shown to the solver, only the full-prompt fingerprint is
shown (see LABEL RULE).

Submitted artifact = (ranking, necessary/correlated flags) — a structured
multi-part decision, not a scalar, not a single label, not a rewritten
prompt. Clears rule 2.2 the same way prompt-delta's ranking did, on a
different underlying question (composite multi-clause responsibility
attribution, not single-hidden-edit effect ranking).

## Why this is not a duplicate of "Prompt Edit Effect Attribution"

prompt-delta: ONE hidden edit from a KNOWN vocabulary of ~12 ops; task is
which of 4 shown INPUT CATEGORIES it moved most. The edit identity is a
hidden label to recover.

This task: the system prompt's clauses are ALL VISIBLE (no hidden edit
vocabulary); the task is which of the prompt's OWN clauses is
CAUSALLY RESPONSIBLE for an observed behavior, verified by REAL ablation,
not by category-selection. Different object (composite multi-clause
prompt vs single before/after edit pair), different mechanism (ablation
causality vs categorical edit-op identification), different submitted
shape (ranking + per-clause flags vs a 4-way ordering).

## Prior art scan (see full agent report, 2026-09-20)

OCCUPIED, avoid: PromptSET (scalar prompt-sensitivity regression),
LLM-inversion (PTP/PROVE/output2prompt — prompt reconstruction from
output), prompt-delta's own edit-attribution axis (see above), prompt
compression (LLMLingua — generation-shaped), meta-prompting/APO (APE,
OPRO, PromptBreeder — rewritten-prompt generation), jailbreak-susceptibility
benchmarks (JailbreakBench, SG-Bench — label + scalar attack-success-rate),
persona/system-prompt scalar-effect papers (IAMs regression framework —
per-component SCALAR coefficient, the closest near-neighbour to avoid
being confused with).

UNOCCUPIED axis confirmed: "which specific component of a structured
prompt object is causally responsible for a failure, packaged as a
structured multi-field diagnosis" is not the headline SCORED artifact of
any found 2023-2026 benchmark. Adjacent-but-different: DETAIL (NeurIPS
2024, per-few-shot-example influence SCORE, continuous not ranked-flagged),
ReasonDiag (2603.21286, interactive step-error viz tool, not a scored
benchmark), TELBench/DRIFT (2606.02060, agent-TRAJECTORY error
localization, not static prompt engineering).

Close neighbour to name-and-differentiate (rule 2.1, cite + differentiate
or risk cite-or-cap): IAMs (2603.26830) decomposes system-prompt
components via regression into per-component SCALAR coefficients — this
task instead requires an ORDERED RANKING plus a discrete necessary/
correlated FLAG per clause, verified by ablation rather than fit by
regression, and is scored on ranking/flag correctness rather than
coefficient-magnitude accuracy.

Distant antecedents to cite (rule 2.1, prefer over close neighbours):
  - classical software engineering fault localization (spectrum-based
    fault localization, Tarantula/Ochiai — "which code component is
    responsible for the failing test") — same causal-ablation logic,
    decades-old and safely distant from any 2024-2026 prompt-engineering
    paper.
  - Shapley-value / counterfactual credit assignment in general ML
    interpretability (LIME/SHAP-style ablation-based attribution) as the
    general apparatus, applied here to prompt CLAUSES instead of model
    FEATURES.

## Dataset source (LOCKED 2026-09-20)

Primary corpus: danielrosehill/System-Prompt-Library, Hugging Face.
  URL: https://huggingface.co/datasets/danielrosehill/System-Prompt-Library
  Licence: CC BY 4.0, verified directly on the HF dataset card (independently
  re-verified via WebFetch, 2026-09-20 — 923 items, JSON fields agentname/
  description/systemprompt/creation_date/etc confirmed).
  FLAG: licence lives on the HF card, not a LICENSE file in the mirrored
  GitHub repo (github.com/danielrosehill/System-Prompt-Library returns no
  LICENSE file) — cite the HF page as source_url, not GitHub, so the
  citation and the licence claim point at the same place (register
  section -1 discipline: field and prose must agree, extended here to
  "citation and licence source must agree").
  No natural single categorical grouping field exists in the raw schema
  (is-agent etc are mostly booleans) — topic groups for the by-group split
  will be DERIVED by clustering agentname/description into domains
  (support/coding/writing/research/etc), not taken verbatim from a field.
  This derivation must be done deterministically (e.g. keyword rules or a
  fixed clustering seed), not by hand-labeling each of 923 items -- TODO
  next session, needs a concrete method chosen and written into prepare.py
  or a pre-processing step in gen_raw.py.

Backup/supplement if 200 subsampled prompts is not enough for train>=1000
after later balancing losses: fka/prompts.chat (CC0 1.0, 2,169 items,
huggingface.co/datasets/fka/prompts.chat) — noted but not needed unless
the primary corpus falls short; do not mix licences into one declared
dataset licence field without re-checking rule -1's field-vs-prose
discipline if this becomes necessary.

## CORPUS FILTERING (measured 2026-09-20, real sample of 231/923 files)

Real inspection found the raw corpus is NOT uniformly "5-7 flat independent
clauses" as assumed: ~48% of prompts personalize to the author by name
("Daniel"/"Daniel Rosehill"), and structure ranges from single run-on
sentences to 59-line numbered multi-step procedures with DEPENDENT steps
(Step 2 only makes sense after Step 1) -- ablating one numbered step from
such a procedure is not a clean independent-clause causal experiment.

DECISION (user confirmed 2026-09-20): EXCLUDE personalized prompts
entirely rather than genericizing the name. Filter rule:
  1. Exclude any systemprompt matching /\bdaniel\b/i (case-insensitive).
  2. Exclude any systemprompt containing code fences, markdown headers, or
     numbered-step markers (proxy for dependent multi-step procedures):
     /```|###|Step \d|\n\s*\d+\.\s/
  3. Keep only prompts whose remaining sentence count (split on
     .!? boundaries, discarding fragments under 10 chars) falls in
     [3, 10] -- enough for a real, non-trivial clause set, not so many
     that clauses stop being independent.

MEASURED YIELD on a 231-file real sample (evenly sampled across the
full 923, not cherry-picked): 39/231 pass with 3-8 sentences (16.9%,
projects to ~156/923); widening to 3-10 sentences did not meaningfully
change (~21%, ~196/923). REJECTED the alternative of genericizing the
name (which would have raised yield to ~44.6%, ~412/923) per explicit
user decision to keep exclusion simple and avoid altering source text
beyond clause-splitting.

CONSEQUENCE: usable prompt base is ~150-200 (measure exactly on the full
923 once gen_raw.py downloads it, this sample projection is not the final
count). This is smaller than prompt-delta's or the RAG design's item
bases, so the probe-query-subset item multiplier must be higher (~6-10x)
to clear train>=1000 -- HEIGHTENED RISK of the "repeated views share one
answer" trap (rule 4 spirit, prompt-delta's item-definition lesson): with
fewer base prompts and a higher multiplier, it is EASIER for a solver to
memorize per-prompt-family responsibility patterns instead of reasoning
per item. MUST verify, once real ablation data exists, that varying the
probe-query subset shown per item genuinely changes which clause is
top-ranked for a meaningful fraction of items -- if it does not (i.e. the
same clause wins regardless of which probes are shown), the multiplier is
producing duplicate answers and needs a different design (e.g. also vary
which SUBSET of clauses is shown/ranked per item, similar to the RAG
design's k-sweep, rather than only varying probes).

## LABEL RULE (draft — real counterfactual ablation, not asserted)

Per system prompt with clauses c_1..c_n (n typically 5-7, derived by
splitting systemprompt on sentence boundaries — confirmed natural on this
corpus via manual inspection of sample items):

  1. Run the FULL prompt over P probe queries (P=4 planned), get the
     released fingerprint (reuse prompt-delta's proven fingerprint()
     design: length, hedging, refusal, bullet use, etc — never response
     text).
  2. For each clause c_i, run the prompt WITH c_i REMOVED (all other
     clauses intact) over the SAME P probe queries, get its own
     fingerprint.
  3. responsibility_score(c_i) = measured distance between the full-prompt
     fingerprint and the c_i-ablated fingerprint, standardized across the
     item's own clause set (so score is relative WITHIN one item, not a
     raw magnitude that could vary prompt-to-prompt).
  4. TRUE ranking = clauses sorted by responsibility_score, descending.
  5. necessary flag: responsibility_score(c_i) exceeds a threshold
     (TBD, measured from real data distribution once generation exists —
     do not guess this number, same discipline as prompt-delta's
     AMBIGUITY_MARGIN-style constants).
  correlated flag: below that threshold but non-zero (clause exists and
  is not the top-ranked null-effect case).

RISK TO MEASURE (rule 2.6, before shipping): a solver could shortcut by
always predicting clause ORDER-OF-APPEARANCE in the prompt text (e.g.
first clause = persona = often necessary, last clause = fallback rule =
often least responsible) without reading the fingerprint at all, if
clause position correlates with responsibility across this corpus by
construction (system prompts conventionally put persona/role first).
MUST measure a position-only baseline against real data before shipping;
if it scores uncomfortably close to the reference, shuffle each item's
displayed clause order independently of true prompt order (release
clauses in randomized order per item, breaking the position-based
shortcut) — same category of fix as prompt-delta's per-probe-variance
mitigation.

## Split (rule 4)

By DERIVED TOPIC GROUP (clustered from agentname/description — see
Dataset source above), never by row. Exact sizing TBD once the topic
clustering method is chosen and item counts are known.

## GPU budget (measured via arithmetic, not yet run)

200 subsampled prompts x (1 full-prompt run + ~6 avg clause-ablation runs)
x 4 probe queries = ~5,600 generations. At prompt-delta's measured
~0.3s/generation, ~28 minutes of raw generation time, well inside the
1.5h budget with room for embedding/setup overhead and a higher probe
count if needed for fingerprint stability.

## Files (planned, mirroring prompt-delta/rag-retrieval-diagnosis layout)

    dataset/generator/gen_raw.py   THE GENERATOR — subsample+cluster the
                                    HF corpus, split each prompt into
                                    clauses, run full + per-clause-ablated
                                    generations on GPU, release fingerprints
                                    per (prompt, clause-ablation) cell
    dataset/prepare.py             derive responsibility scores + ranking +
                                    necessary/correlated flags from MEASURED
                                    ablation deltas, balance, split by
                                    derived topic group
    dataset/grade.py               ranking + flag joint metric (TBD exact
                                    formula — needs design), InvalidSubmission
                                    Error(ValueError), degrade never reject
    problem-description.md / dataset-description.md / rubrics.md /
    config.yaml / solution.ipynb   TODO after generator run

## FINAL LOCKED NUMBERS (2026-09-20, verified against the REAL full 923-file
## corpus, before any GPU generation — pipeline dry-run tested end-to-end
## with synthetic fingerprints standing in for real GPU output)

Real corpus download: 917/923 files fetched successfully (6 failed, 0.6%
loss, acceptable). After CORPUS FILTERING (exclude-by-name, exclude-by-
structure including bulleted lists — an earlier pass missed bullets and
had to be fixed — sentence count in [3,10]):

    kept: 145 prompts (out of 917 checked)
    excluded_name: 437  (personalization to the author)
    excluded_structure: 285  (numbered/code-fenced/bulleted procedures)
    excluded_sentence_count: 50  (too flat or too sprawling)
    clause count distribution: min 3, max 10, avg 5.9
    topic groups: 13 (writing_editing 26, general_other 26,
        tech_recommendation 22, ai_meta_tooling 18, coding_dev 16,
        research_analysis 12, career_professional 6,
        format_lookup_utility 5, home_lifestyle 4,
        productivity_planning 3, finance_shopping 3, health_wellness 2,
        creative_roleplay 2)

ARCHITECTURAL FIX during design (before any GPU cost was spent): the
original per-subset generation design would have cost ~40,000 generations
(~3.3h, over budget) because it re-ran ablations once per probe-subset
combination. Fixed by generating ONCE per (prompt, ablation-or-full,
single probe) cell (8 probes total, not per-subset), and deriving as many
subset-averaged ITEMS as needed at ZERO extra GPU cost by re-averaging
the same per-probe fingerprints in prepare.py. This dropped the real
generation count to ~7,976 (145 prompts x (1 full + 5.9 avg ablations) x
8 probes), ~40 minutes estimated, comfortably inside budget.

ITEM MULTIPLIER: 10 of the 70 possible 4-of-8 probe-query subset
combinations per prompt (deterministic, not random — first 10 of
itertools.combinations(range(8), 4)) -> 1,450 total items.

SPLIT (by topic, exact-search, verified in dry run): 10 topics train / 3
topics test (ai_meta_tooling, creative_roleplay, format_lookup_utility
held out), ratio 20.8% (inside the 15-25% band) -> 1,200 train / 250 test.

PIPELINE DRY-RUN VERIFICATION (synthetic fingerprints with an injected
signal, real filtering + real prepare.py logic, no GPU): label-recovery
check confirmed 1,200/1,200 items correctly identified the
artificially-boosted clause as top_clause_id (the full ablation ->
standardized-distance -> ranking chain is wired correctly). The
position-only shortcut (guessing the first clause in DISPLAYED order)
scored 20.6% against a 17.0% chance floor (avg 5.88 clauses/item) even
under this artificially-skewed synthetic setup — confirming the per-item
SEEDED SHUFFLE of displayed clause order successfully breaks the
position-based shortcut flagged in STATE.md before any data existed.

STILL TO MEASURE ONCE REAL GPU DATA EXISTS (do not skip): the
NECESSARY_THRESHOLD (70th percentile of real standardized responsibility
scores — the dry run's -0.2099 is a synthetic-data artifact, not a real
number), the full ablation ladder (naive baselines, a real trained
solver, the reference), and the reason_code/flag distribution for
balance. Report all per the SCORE-BAND REQUIREMENT before declaring done.

## Next session — TODO

1. Choose and lock a deterministic method for deriving topic groups from
   agentname/description (keyword rules vs a fixed embedding-clustering
   seed) — needed before prepare.py's split logic can be written.
2. Design grade.py's exact metric: how ranking-correctness and per-clause
   necessary/correlated-flag-correctness combine into one score (needs the
   same care prompt-delta's positional-credit and the RAG joint-field
   average got — decide before writing, don't improvise at grade time).
3. Confirm title with user (register rule 1.16).
4. Write gen_raw.py: download/subsample the HF corpus, split prompts into
   clauses (verify sentence-boundary splitting is clean across a larger
   sample than the 2 manually inspected so far), run the ablation grid on
   GPU, release fingerprints.
5. Run gen_raw.py on GPU (user's A10G) — MUST push to GitHub release or
   similar BEFORE declaring done this time, since this is the second
   attempt after the RAG pivot's source_url gap; do not skip the source_url
   step again.
6. prepare.py: derive labels from measured ablation deltas, measure the
   position-only shortcut attacker FIRST (see LABEL RULE risk above),
   close it if needed, balance, split.
7. Build and measure the 5+ rung ablation ladder per the SCORE-BAND
   REQUIREMENT: chance floor, naive baselines, the position-only shortcut,
   a real trained/heuristic solver, the reference — report all numbers.
8. problem-description.md (storytelling-first framing, per user's explicit
   request 2026-09-20 for the narrative to read well), dataset-description.md
   (describes the RAW files, rule 1.7), rubrics.md, config.yaml (with a
   real source_url this time — the HF dataset page), solution.ipynb.
