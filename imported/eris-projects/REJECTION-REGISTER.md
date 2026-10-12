# Eris / Shipd — Rejection Register

Every failure mode actually observed across the portfolio, with the challenge that
hit it. This is the pre-flight checklist for any new build. Nothing here is
theoretical — each line cost a revision round.

Sources: `ERIS-GUIDE.md`, `claude/ai&novelty.md`, and the per-challenge memories.

---

## -1 · What a genuinely passing review rewards (the positive control)

Palimpsest ("Untangling Interleaved Revisions") got the first fully-praising
review in the portfolio's history. Its specific praise points are as load-bearing
as any rejection above — they are the confirmed target, not a guess.

- **A dense, independently-verifiable ladder is what gets noticed first.** The
  reviewer named all **nine** measured points by value (0.053 random parent ->
  0.252 all-root -> 0.358 similarity -> 0.390 arborescence -> 0.493 linear ->
  0.552 reference -> 0.852 oracle ceiling) and then **re-ran the platform sample
  check himself** (got 0.2470 against the published 0.252) — the ladder is only
  trusted once it reproduces outside your own numbers.
- **"Immediate parent exact match avoids the matched assignment trap that
  quietly hands out credit on clustering tasks."** A NEW metric-design lesson:
  scoring via a best-matching/Hungarian-style assignment between predicted and
  true groups can silently grant credit the model never earned, on any task
  shaped like clustering. Prefer a metric that scores each element's DIRECT,
  unambiguous target (a parent pointer, a specific label) over one that finds
  the best alignment after the fact.
- **"The leak work is measured rather than asserted, which is rarer than it
  should be."** Not "we removed the leak" but the actual before/after number:
  a template-date field ordered lineages at 0.64 pairwise; after the fix,
  surviving prose dates ordered at 0.497 = chance. Quantify the leak's
  magnitude AND the fix's effect, both as numbers.
- **Target the SHAPE of a leak, not its spelling.** The fix stripped every
  template parameter whose *value* matched an ISO-date pattern, rather than
  matching on the field's name/spelling — because editors misspell field names
  but the date's shape is what actually leaks order. Pattern-class removal
  beats keyword removal.
- **The raw-vs-public split can resolve a real licence-vs-leakage tension.**
  `attribution.csv` (titles + revision IDs, required for CC BY-SA compliance)
  stayed in the RAW archive for licence/provenance purposes but was
  deliberately **excluded from the public split** because titles/IDs are
  exactly the lookup key that would leak the answer. Real attribution data and
  leak-safe public data can be the same file, scoped differently — not a
  contradiction to resolve by dropping one requirement.
- **Agents beating the reference is fine — IF three conditions hold together.**
  All three agents scored above the reference here (0.60/0.64/0.64 vs 0.552)
  and the reviewer treated this as healthy, not a red flag, BECAUSE: (a) real
  headroom remained beneath the oracle ceiling (0.852, so agents were nowhere
  near maxed), (b) the **ordering was identical on public and private**
  halves, and the **spread matched almost exactly** (0.0396 vs 0.0413), and
  (c) the public-to-private gain was small and consistent across all three
  agents. This refines section 3 above: "reference below agents" is only a
  problem when it *also* shows saturation, inverted ordering, or an
  inconsistent public/private split — check all four together, not the score
  alone.
- **The one real defect was a metadata field, not the design.** Every prose
  document correctly said CC BY-SA 4.0 (matching Wikivoyage's actual licence
  and the attribution file's own label), but the platform's **declared licence
  field** said CC BY 4.0 — silently dropping the ShareAlike obligation, which
  has real legal weight (BY-SA binds downstream derivatives to the same
  terms; BY does not). Lesson: cross-check the literal metadata FIELD against
  the prose everywhere it's stated — they can drift independently, and only
  the field is machine-read.

---

## 0 · Hard platform constraints (violating any = auto-reject)

- **GPU-only.** `ERIS-GUIDE.md` line 13/17. Solver gets **1x NVIDIA A10G + 10 CPU
  cores + 62 GB RAM**. The task must genuinely *leverage* the GPU, not merely
  tolerate it.
- **Runtime** max **1.5 h**, target ~1 h.
- **Novelty >= 6.** Below 6 blocks submission outright.
- **No tabular / regression challenge.** Shipping one bars you from creating
  another.
- Deterministic evaluation AND deterministic seeded dataset generation.
- No internet, no external datasets at solve time. Self-contained.
- CSV submission only, exact filename / columns / dtypes.

---

## 1 · Upload & validator mechanics (fail before a human ever reads it)

| # | Rule | Learned from |
|---|---|---|
| 1.1 | Dataset zip must be **FLAT** — the validator cannot see inside a top-level folder | pipeline-forensics, babel-gauge |
| 1.2 | The dataset zip must **already contain `public/` and `private/`**. `grade_entrypoint.py` reads `dataset/public/sample_submission.csv` **directly** and may never run `prepare.py` | repeat-harvest round 3 |
| 1.3 | Prepared splits live at **`dataset/public/`** and **`dataset/private/`** — not at the challenge root, not at a sibling `prepared/` | repeat-harvest round 2 |
| 1.4 | **No markdown tables anywhere** in problem-description / dataset-description. The renderer eats pipe characters. Use bullet lists | palimpsest, repeat-harvest, babel-gauge |
| 1.5 | **No tilde code fences** (`~~~`) — breaks the parser. Only triple backticks | palimpsest |
| 1.6 | Copy from the RAW file, never a rendered preview — rendered copies drop pipes and closing backticks | recompile |
| 1.7 | The dataset-description field describes the **RAW upload's files**, never the prepared split. Pasting problem-description there = "fundamental mismatch" | palimpsest, recompile |
| 1.8 | `answers.csv` **first column must equal the submission's ID column**, or you get "Submission missing required ID column" | cut-the-corpus |
| 1.9 | The platform **slices `answers.csv` row-wise** into public/private leaderboards. One row per submission ID, and the grader must **ignore IDs absent from the slice and tolerate extra rows** | pipeline-forensics, palimpsest |
| 1.10 | Slices must be **representative** — order rows by a stratifying key then golden-ratio rekey + pairwise balance, and verify every contiguous slice scores near the full set | cut-the-corpus, repeat-harvest |
| 1.11 | `prepare.py` must be **self-healing**. Never `assert col in df.columns` for a value that is deterministically reconstructable — the platform may feed a raw file missing derived columns | cut-the-corpus |
| 1.12 | **Solvers discover the data path**, never assume it. The agent env mounts `/data/` with cwd `/workspace`; docs saying `dataset/public/` produced no score | recompile |
| 1.13 | Grader must **degrade, never reject**. Missing rows / blanks / dupes / NaN score 0 for that item. `InvalidSubmission` must subclass **`ValueError`**, not `Exception` | recompile, babel-gauge |
| 1.14 | **Data secrecy**: don't name the assembly / track / region / coordinates that let a participant download the exact data. Cite the *method*, not the pointer | repeat-harvest |
| 1.15 | **Source URL triggers duplicate detection on URL/domain, not task.** Gutenberg #1342 and a bare CC0 deed URL both collided with unrelated datasets | cut-the-corpus, babel-gauge |
| 1.16 | Title: plain and descriptive. Rejected for being **too long/paper-like**, **too catchy**, and for a **colon subtitle / creative project name** | cut-the-corpus, palimpsest |
| 1.17 | Description must open with a **clean declarative sentence**, then short bold-labelled sections and bullets. Dense mid-sentence openings get rejected | cut-the-corpus |
| 1.18 | The **generator must ship** inside the archive | budget-index |
| 1.19 | Tag the **domain honestly** (`rag` on a task that retrieves nothing was flagged) | budget-index |
| 1.20 | `ast.parse` every notebook cell **before** writing the `.ipynb`; a literal `\n` in a patch silently broke a cell and burned a 40-min run | palimpsest |
| 1.21 | Re-uploading an **accepted** dataset forces you to withdraw it. Ship revisions through challenge-side files only | repeat-harvest v2 |
| 1.22 | Before packaging, **replay the platform's own entrypoint paths from the extracted zip**. Local runs pass while the harness fails purely on output location | repeat-harvest |

---

## 2 · Review-committee rejections (`claude/ai&novelty.md`)

The committee scores: originality, task formulation, dataset, metric, difficulty,
scientific value. **Any reviewer rejecting means redesign from scratch.**

- **2.1 Novelty is stochastic.** The same task scored 6, 4 and 3 on re-runs.
  Threshold is 6. Levers that measurably moved it:
  - a dedicated **"What is new here"** numbered section;
  - a **"How it relates to prior work"** section citing **distant antecedents
    only** — naming a close recent neighbour triggers cite-or-cap and *drops*
    the score (naming "HiCBench 2026" cost points; naming TokTrack/WikiHist and
    differentiating them *raised* palimpsest 6 -> accepted);
  - **real data +1**;
  - changing the **task identity**, not the wording. Reframing alone took
    budget-index 4 -> 7; the remaining point needed conceding the classical
    primitive openly and naming the new capability.
- **2.2 Task formulation.** Must not read as a variation of classification,
  regression, ranking, retrieval, OCR, segmentation, captioning, sequence
  prediction, or information extraction. The **submitted artifact** is what makes
  this true — a forest, a partition, a permutation, an index geometry, a program,
  an ordered op pair. A numeric prediction reads as "just regression" no matter
  how it is framed.
- **2.3 The task must RANK solutions.** budget-index's real defect was
  *"otherwise this is an optimization puzzle that does not rank solutions."*
  A task where **every stage has an exact algorithm converges** — all three
  agents landed on 0.4 exactly. Fix: at least one stage must have **no exhaustive
  pattern** (held-out wordings, unlearnable boundary noise, a genuine perceptual
  judgment).
- **2.4 Description numbers must match the shipped data exactly.** A reviewer
  re-counted rows and found the description and the upload came from different
  generations.
- **2.5 Truth set must be complete.** 21.6% of repeat-harvest windows contained
  real annotations in the flanks that were *not* in the answer key — a solver
  could be right and score 0.
- **2.6 Shortcut opportunities must be measured, not argued.** Attack your own
  shipped data with concrete solvers. Corpus-consensus voting scored 0.389
  against a 0.425 ceiling and collapsed the advertised skill gap to +0.036 —
  found only by writing the attacker.

---

## 2A · Transductive / cross-row leakage — a distinct rejection class

A challenge was rejected for **"violating independent single-sample inference and
using transductive test-set adaptation."** Concretely: the reference/agent
solution **fit global TF-IDF / N-gram models on the test data itself**, and
**grouped separate test rows together to intersect their candidate banks** —
eliminating distractors by exploiting a **dataset construction artifact** across
rows, instead of making an independent, generalizable per-row prediction.

This is a different failure from leakage-in-prepare (section 5) — the columns
are clean, the split is clean, but the **task as posed lets a solver treat the
test set as a corpus** rather than score each item independently.

**What to check for any new design:**
- Can a solver's prediction for row/item *i* be computed **without looking at
  any other test row's hidden content**? If the intended solution legitimately
  needs cross-item statistics (e.g. corpus term frequencies), those statistics
  must come from **train only**, never fit on the released test set.
- If items are naturally grouped (bags, candidate sets, shared keys), verify no
  **cross-item shortcut**: e.g. a public field/key that is unique-mapped to its
  label across the **whole corpus** (train+test combined) lets a solver build a
  global lookup table and skip the intended per-item reasoning entirely — the
  same failure shape as budget-index's consensus-voting shortcut (section
  2.6/1.6), generalized beyond RAG to any bagged or keyed task.
- Reject any design where "intersect what's left across several test rows" is a
  viable strategy — verify by explicitly attempting that attack with a solver
  before shipping, the same way corpus-consensus and lookup attacks were
  measured on budget-index.
- Directly relevant to **any task whose unit of submission is a bag/group**
  (e.g. a partition-recovery task): the intended cross-item signal is *within*
  one bag by design, but a solver must not be able to shortcut using structure
  **across different bags/rows in the released test set** — e.g. a key that
  deterministically identifies its group everywhere it appears in the corpus.

## 3 · The agent-difficulty gate — the single biggest killer

Every rejection below is the same gate failing in a different direction.

- **3.1 Agents at 1.0** — pipeline-forensics (twice), babel-gauge (3/3 at 1.0000).
  Cause: every fact is exactly recoverable from the input, so a competent agent
  reaches the ceiling. No amount of format-starving or corruption fixes it;
  difficulty must be **irreducible loss built into the data**.
- **3.2 Agents at ~0.07** — repeat-harvest at IoU 0.40. Cause: the threshold was
  tuned against the *upper* bound (stop guesses) and never stress-tested against
  a detector that **finds the target but localises loosely** — which is exactly
  what real agents do.
- **3.3 Agents saturating the ceiling** — repeat-harvest scored 0.39 against a
  0.3965 ceiling = **98.4% of max**, and the compressed sub-0.5 scale *hid* it.
  The reviewer rejected it and said do **not** rescale the ceiling; make the task
  harder. **Prefer a metric with an attainable 1.0** so saturation is visible.
- **3.4 All agents at the same score** — budget-index, three agents at exactly
  0.4. A deterministic task with a verifiable train split converges exactly.
- **3.5 Reference below a naive baseline** — repeat-harvest round 3: reference
  0.2743 vs best fixed tile 0.2778. The claimed ordering was **inverted**. Always
  measure the naive rungs, not just the reference.
- **3.6 An ablation overtaking the reference** — recompile, when the test split
  was steered past ~0.70 hard-fraction every solver collapsed to the same score
  and one ablation beat the full reference. **Sweep the difficulty knob and
  measure the whole ladder at each point.**
- **3.7 Bimodal per-item scoring** — recompile's exact set-intersection gave 13
  items at exactly 1.000 and 8 at exactly 0.000 of 30. Different solvers collide.
  Score **overlap + content**, aligned softly.
- **3.8 Split composition silently sets the band** — recompile's random split drew
  45% hard items from a 58%-hard corpus and inflated every score (agent hit 0.86
  vs a 0.650 reference). Stratify the test split *and* the train split.

**Target shape:** a ladder of **5+ distinct, strictly ordered** scores from
faithful partial implementations (the real solver with one capability ablated) —
not just baselines. Those ablations are the shapes agents actually land on.

---

## 4 · Split geometry

- Split must be **disjoint by group**, never by row — by CFR part, by galaxy, by
  article, by element. A row-level split leaks.
- **train > test.** Reviewer explicitly rejected train 101 / test 179.
- **train >= 1000** (platform minimum, flagged on repeat-harvest).
- **test/train ratio 15–25%.** 66.6% was called too high; 133% is still open on
  budget-index.
- Drop items that are **unwinnable by construction** (unreachable targets,
  misclassified-when-clean galaxies) so every graded row is winnable.

---

## 5 · Pre-flight checklist

Before packaging, all of these must be *measured*, not assumed:

- [ ] Grader returns a **number** for ~20 deliberately damaged submissions
- [ ] Reference per-item score distribution has **many distinct values**, not mostly 0s and 1s
- [ ] Ladder of **5+ ablations**, distinct and strictly ordered, reference on top
- [ ] Naive/blind baselines measured and **below** the weakest real solver
- [ ] Answer-knowing adversary cannot exceed the stated ceiling
- [ ] **No transductive shortcut**: a solver cannot fit statistics on the released
      test set itself, or intersect/cross-reference multiple test rows or bags to
      eliminate distractors, to solve items without independent per-item reasoning
- [ ] `prepare.py` byte-identical across two runs
- [ ] No answer leak in `public/`; no train/test group overlap
- [ ] Every contiguous leaderboard slice scores near the full set
- [ ] Description counts equal the shipped data counts
- [ ] Platform entrypoint replayed **from the extracted zip**
- [ ] Notebook executed end-to-end, cells `ast.parse`-clean
- [ ] Reference runtime well inside 1.5 h on A10G

---

## 6 · The irreducible risk

The agent-difficulty gate can only be run on Shipd. Everything above is
verifiable locally; the band is not. The best available proxy is the ablation
ladder plus a ratio extrapolation once any single real agent score exists
(`agent/reference x new reference`). Run **one** agent, read the number, tune,
then run the rest — do not spend three credits on an unverified band.
