# Palimpsest — Design Record

How this challenge was arrived at, what was measured before the spec was
frozen, and which shortcuts were found and killed. Numbers marked *(probe)*
were measured on the ~130-article API-collected pool during design; the final
anchors in `config.yaml` are measured on the shipped dataset.

## The capability being measured

Given a shuffled bag of document snapshots drawn from several unidentified
documents' revision histories, reconstruct the **edit genealogy**: which
snapshots belong to the same document, in what order they were written, and
where each lineage begins. The submission is a parent pointer per snapshot;
the metric is immediate-parent accuracy.

Three sub-skills have to work together, and the metric rewards none of them
in isolation:

1. **Grouping** — how many lineages are in the bag (never disclosed, 3–11) and
   which snapshots belong together. An article usually contributes several
   lineages from different stretches of its own history, so two near-identical
   lineages in one bag are common and similarity alone cannot separate them;
2. **Edit directionality** — for two adjacent revisions, which one came first.
   This is the core learned skill: a parent→child diff is not the reverse of a
   child→parent diff, and the difference (what edits *add* vs what they
   *remove*) is a property of how humans revise documents;
3. **Global structure** — pairwise beliefs must be decoded into a consistent
   forest of chains (each snapshot at most one child, no cycles).

## Why this is not a known task

Sentence/paragraph reordering (ROCStories, SIND, arXiv abstracts) orders parts
of ONE document by discourse coherence; here the units are whole revisions of
SEVERAL documents, coherence between snapshots is near-total by construction,
and the output is a forest, not a permutation. Temporal ordering benchmarks
(LongBench Pro timeline reconstruction, HistoryBankQA) order *events described
in text*; here nothing describes anything — the signal is the writing process
itself. Wikipedia revision corpora (WikiAtomicEdits, WikiRevisions) supply
edits as *input* for tasks like grammatical error correction or edit intent
classification; none asks a model to *recover* the revision graph. Stemmatology
in digital humanities reconstructs manuscript descent trees, but from copying
errors between scribal copies, with tiny hand-curated corpora and no ML
formulation. The occupancy scan (2026-08-05) found no benchmark, competition,
or dataset with this formulation.

## Why this is an NLP task, not graph reconstruction with text attached

A reviewer could reasonably ask whether the language matters, given the output
is a forest and the reference decodes it with a matching algorithm. The
measurements answer this directly.

Every non-linguistic signal available was measured, and each one plateaus early:

| signal used | parent_acc |
| --------------------------------------------------------- | ---------- |
| document statistics only (byte size, cluster membership) | 0.310 |
| symmetric text similarity + size for direction | 0.390 |
| **everything above is available without reading the edit** | **≤ 0.390** |
| add the content of what each edit adds/removes | 0.493 → 0.552 |

Both independent solver agents confirmed this from the other side: each
reported that its binding constraint was **direction**, not structure — per-pair
direction accuracy plateaued at 0.74 and chain orientation, not clustering or
decoding, is where their remaining error lived.

The entire 0.17 above the structural ceiling comes from the *content* of the
diff — which tokens a revision introduced and which it deleted. The structure
half of the problem (grouping, acyclicity, the matching decode) is worth about
0.03 on top of good pairwise scores; the language half is worth 0.17. The task
is graph-shaped in its *output format* and linguistic in its *difficulty*.

What the model has to learn is an asymmetry in editing behaviour: revisions
accrete specifics (prices, hours, citations, wikilinks, qualifiers) and shed
maintenance debris (closed listings, dead URLs, duplicated sentences, typos,
promotional phrasing). This is a property of how people write, not of the
graph. It is also why byte size fails — a copyedit that tightens prose and a
vandalism revert both shrink the page, and only the language distinguishes them.

The reference uses hashed token bags over the diff rather than a sequence
model, which is a statement about the compute budget (CPU, 1.5 h), not about
the ceiling. The gap between the reference (0.552) and the similarity-only
bound (0.852) is where richer language modelling — contextual encoders,
attention over the aligned diff, modelling *where* in the document a change
landed — has room to pay off.

## Why Wikivoyage

- **Ground truth is the wiki's own revision graph** — recorded by MediaWiki at
  edit time, not annotated, not model-derived, not disputable.
- **CC BY-SA 4.0** — commercial use and redistribution permitted (attribution
  ships in `raw/attribution.csv`).
- **Less mined than Wikipedia** — Wikipedia revision histories underlie several
  known corpora; Wikivoyage's essentially none.
- **Maintenance-heavy editing** — travel guides get re-priced, reorganised and
  pruned, so windows exist where the page churns without growing. This is what
  makes the byte-size shortcut weak (measured below).

## The shortcut hunt (run BEFORE the spec was frozen)

Two prior challenges in this repo died on the constant-answer gate
(`pipeline-forensics` 1.0 twice; `storm-genealogy` 0.91), so the cheap-shortcut
measurements were run before any spec prose was written.

### Shortcut 1 — byte length. WEAKENED BY WINDOW SELECTION.

On arbitrary revision windows, sorting by size recovers pairwise order at
**0.76** *(probe over Wikivoyage and Wikipedia both)* — fatal. Fix: windows are
selected by a churn criterion, `churn / (1 + |net drift|)`, computed **only
from byte sizes in the dump** — never from text, labels, or any baseline's
behaviour. Selected windows order by size at **~0.60** — barely above chance,
weak enough that size-based direction caps out at the low size-chain baseline.

### Shortcut 2 — `lastedit=` template dates. FOUND AND STRIPPED.

Wikivoyage `{{listing}}` templates record when an editor last verified a
listing. Measured: present in **45%** of revisions; ordering lineages at
**0.64** pairwise; 93,107 occurrences in the shipped pool. Editors also
misspell it — `lastedited`, `lastdit`, `lasted it`, `lastedot`, `lastedir` —
and citation `accessdate` carries the same kind of signal, so rather than
chase spellings `prepare.py` drops **any template parameter whose value is a
bare ISO date**. That rule is exhaustive by construction and cannot be
defeated by a typo. After stripping, no date-valued parameter survives
anywhere in the released text; the 1.85% of snapshots still containing an ISO
date carry it in prose, and those order lineages at **0.497 — chance**. The
raw data keeps the parameter so the measurement is reproducible.

### Shortcut 3 — article titles in the prose. MASKED.

The page name appears throughout its own wikitext, which would make grouping a
string-match exercise. All title tokens longer than 3 characters are replaced
with `___`.

### Shortcut 5 — grouping by similarity. BROKEN BY SAME-ARTICLE LINEAGES.

In the v1 build, grouping was near-free (tf-idf clustering purity ~0.92) and
one agent reported 99.7% candidate recall. An article now contributes two or
three lineages taken from different stretches of its own history, separated by
a dropped revision, so a bag routinely holds near-identical lineages that no
similarity measure can separate — the shipped-build agent's undirected edge
recall fell to 0.84, and it attributed the loss precisely to "two disjoint
time-windows of the same article". The dropped revision is the second
lineage's true parent, so its head genuinely is a `ROOT`: the item stays
well-posed, it is just no longer solvable by clustering.

### Shortcut 4 — snapshot ids and file ordering. RANDOMISED.

Ids are assigned AFTER shuffling (`n00…` in shuffled order), and JSON keys are
sorted, so neither carries chronology.

### What similarity alone can do — measured, and published to solvers

The most-similar snapshot is a true neighbour (parent or child) **~0.80** of
the time *(probe)* — but similarity is symmetric and cannot orient the edge.
Published in the problem statement as the "genie" bound so solvers know
similarity engineering alone cannot win.

## Measured difficulty gradient — FINAL, on the shipped dataset

900 items (630 train / 270 test, 6,241 test snapshots), scored through
`grade.py` against `private/answers.csv` — the exact solver pipeline.

| anchor | parent_acc |
| ------------------------------------------------- | ---------- |
| random parent | 0.053 |
| all ROOT | 0.252 |
| cluster + size order | 0.310 |
| most similar smaller snapshot | 0.358 |
| arborescence, direction by size | 0.390 |
| linear (hashed diff + LR) | 0.493 |
| **reference: diff EmbeddingBag + per-item softmax, matching decode** | **0.552** |
| **frontier coding agent, public data only, 75-90 min CPU** | **0.667 / 0.685** |
| similarity-only ceiling (direction genie) | 0.852 |
| oracle | 1.000 |

### The agent-difficulty gate, and the two designs it rejected

The gate was run with real coding agents given nothing but `dataset/public/`
and the problem statement, in an isolated directory, CPU-only. Two earlier
builds were **measured and rejected** before this one:

| build | chain length | lineages/item | agent score | verdict |
| ----- | ------------ | ------------- | ----------- | ------- |
| v1 | 8 | 3.0 | 0.750, 0.798 | rejected — above band |
| v2 | 5 | 4.7 | 0.720 | rejected — above band |
| **v3 (shipped)** | **4** | **5.8** | **0.667, 0.685** | **accepted** |

Both v1 agents independently found the same amplifier, and one of them
measured it precisely: per-pair direction accuracy was only 0.705, but because
a lineage's orientation is ONE global decision, summing pair logits along an
8-revision chain lifted chain-level orientation to 0.89 — and they reported
0.95 at length 10 versus 0.86 at length 6. Long chains average per-pair noise
away. The fix follows directly from that measurement: shorten lineages so the
amplifier is weak, and put several near-identical lineages from the *same
article* in one bag so grouping can no longer be solved by similarity alone.

The shipped build's agents confirm the design measures what it claims: both
reported direction — not clustering, not decoding — as their binding
constraint, one plateauing at 0.744 per-pair direction accuracy and noting
that the remaining signal "is genuinely semantic".

*Caveat, stated for the record:* the two agent scores were measured on a build
identical to the shipped one except that 26 misspelled `lastedit` variants
(`lastdit`, `lasted it`, …) across ~35,000 revisions had not yet been
scrubbed. The shipped build strips every template parameter whose value is a
bare ISO date, which is exhaustive by construction; residual content dates
order lineages at 0.497, i.e. chance. The effect on the agent numbers is
below measurement noise but was not re-measured.

## Why the metric is parent accuracy, not an ordering correlation

Kendall-tau-style metrics presuppose the grouping is known and grade only the
permutation. Parent accuracy grades all three sub-skills in one number, is
trivially in [0,1], is interpretable (fraction of edges recovered), and is
insensitive to lineage length imbalance in a way ancestor-pair F1 is not
(long chains dominate F1 quadratically; each snapshot contributes exactly one
edge to parent accuracy).

Submissions must be structurally valid forests — cycles are rejected, not
scored 0, and `validate_submission.py` ships in public/ so the check costs a
solver nothing.

## Well-posedness filter

Exact reverts restore a byte-identical page. When two snapshots in a bag have
identical text, the parent of the following revision is undecidable **by any
method** — and the answer key becomes arbitrary among the twins. Measured on
the probe pool: 7.4% of snapshots had an identical twin. `prepare.py` therefore
samples each article's run only from contiguous revision stretches whose
scrubbed texts are all distinct (falling back to shorter runs, minimum 4).
After the filter: 0 duplicates. *Near*-reverts — 99% similar but not identical
— are untouched; they remain the hard, decidable cases the challenge is about.
This mirrors the reachable-targets filter in blind-spot: unwinnable rows add
noise, not difficulty.

## Leakage analysis

- Items never share an article; articles never cross the train/test split.
- Timestamps, revids, comments, usernames, titles: never released.
- The hidden answers are the parent pointers alone; nothing in public/ encodes
  them (ids are shuffled-order, JSON keys sorted, CSV rows in id order).
- Wikivoyage is public, so a determined human could in principle look test
  snapshots up. This is outside the agent threat model (no internet in the
  solver environment) and is true of every real-data challenge in the repo.

## Known limitations

- One wiki, one language, one editing culture. The challenge measures lineage
  recovery on collaboratively edited travel guides; transfer to code, contracts
  or fiction is plausible but unmeasured.
- Windows are 14 consecutive revisions; very long-range structure (years of
  drift) is not represented.
- ~6% of steps are revert-like (r_t closer to r_{t-2} than to r_{t-1}) and ~7%
  land near-identical to the grandparent *(measured on the probe pool)*, which
  makes some parents genuinely ambiguous. This is treated as irreducible noise
  in the ceiling, not filtered out — reverts are what makes nearest-neighbour
  chaining fail, and removing them would reopen that shortcut.
