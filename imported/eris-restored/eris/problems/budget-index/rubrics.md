# Rubrics — Budget-Constrained Index Construction

Task-specific criteria for good engineering on this challenge. Evaluated
alongside the answerability score. Every score quoted below was measured by
running the described implementation against the private answers on this exact
dataset.

## Rubric 1 — REQUIRED

**Locates each evaluation query's evidence from the question text.** The
solution parses all four question forms and resolves the entity each asks
about: directly by name, through an ISO-code definition sentence, through a
capital-to-country link sentence, and through the two-hop `chained` form where a
capital resolves to an ISO code which resolves to a country. Solutions that
select windows blindly score 0.1164–0.1389; handling only the single-hop forms
costs 0.0325 against a solution that handles all four.

## Rubric 2 — REQUIRED

**Reconciles the value each sentence states against the knowledge base.** Every
fact sits beside a poison twin that is character-for-character identical once
the number is masked, states a real attribute value of a real country, and is
placed on either side with equal probability. Frequency, membership and
positional shortcuts are all closed by construction, so the twins can only be
separated by resolving the entity in `real_facts.json` and comparing the value.

The stated value is **not always plain digits**: 39.4% of true fact sentences
use one of several other surface forms. A strong solution characterises that
inventory from the labelled training split rather than assuming it, then
normalises accordingly — comparing values as literal integers silently misreads
better than a third of the corpus, and the failure is silent because such a
solver still returns a confident answer. Measured with the allocator held fixed
at the budgeted DP, so these are directly comparable: no knowledge-base lookup at
all scores **0.3517**; matching by shared words **0.4706**; a lookup harvested from the
training split **0.4105**; meaning-based matching with value reconciliation
**0.6696**. This
stage is worth **0.3179** of the score and is the single largest lever in the
challenge.

## Rubric 3 — REQUIRED

**Treats the window budget as a binding constraint.** `budget_k` averages 2.15
windows against 3.50 evaluation questions per document, so the solution must
merge — one window over a cluster of consecutive facts answers several questions
at once — and must concede questions that no legal allocation can reach rather
than spending windows on them. Measured with identification held fixed at perfect, so these two are directly
comparable: emitting one minimal window per query until the budget runs out
scores **0.5770** against **0.7021** for a budget-aware allocator, so this is
worth **0.1251**. Note this row is not comparable to the identification figures
in Rubric 2 — it degrades a different stage.

Note what is *not* required: a greedy max-gain allocator ties the exact budgeted
dynamic program at 0.7021 on this dataset. Either is fine. Exact optimization is
not the graded skill — budget-awareness is.

## Rubric 4 — REQUIRED

**Produces a valid submission by construction.** For every document the windows
are in range, mutually non-overlapping, at most `budget_k` in number, and each
verified to be at most 60 tokens before writing `submission.csv` — the grader
zeroes all questions of any document whose windows break a rule.

## Rubric 5 — RECOMMENDED

**Validates end to end on the supervised split.** The 599 training queries
reveal their evidence and poison ids. A strong solution measures its evidence
locator's span accuracy per question form and — using the revealed poison ids —
verifies its value-reconciliation against `real_facts.json` before touching the
evaluation set. This is where a solver discovers that exact integer matching
fails on a third of the facts; the training split is what makes that failure
visible rather than silent.

## Rubric 6 — RECOMMENDED

**Handles unanswerable and conflicting queries deliberately.** The workload is
deliberately over-constrained: some queries are structurally unanswerable
(evidence wider than the token cap, or poison pinned between evidence
sentences), and `conflict` pairs are interleaved so that no non-overlapping
window set can satisfy both. A strong solution does not burn scarce windows
chasing them, and resolves conflicts by expected gain rather than arbitrarily.
A truth-optimal allocation scores 0.7021, not 1.0.

## Rubric 7 — OPTIONAL

**Reports diagnostics** — per-question-form evidence-location accuracy, per
numeric-surface-style discrimination accuracy, score by construction class,
budget utilization, and the gap to the verified optimum (0.7021) — demonstrating
the solution's behavior is understood rather than accidental.

**Level summary:** 4 REQUIRED + 2 RECOMMENDED + 1 OPTIONAL (majority
REQUIRED/RECOMMENDED, as the spec requires).
