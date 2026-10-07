# Attributing Assistant Behavior to System Prompt Clauses — Rubrics

Criteria are graded alongside the leaderboard score. They describe what
good ML engineering looks like *for this task specifically*: recovering
which clause of a real system prompt caused an observed behavior, from
real ablation-derived measurements alone.

---

### 1. Validates on a topic-level split — REQUIRED · TRAINING

Every item's behavior measurements come from one of the training topic
groups; the held-out test groups never appear in train.csv at all.
Validation must hold out **whole topic groups**, mirroring the real
split, not individual rows.

*Fails if:* a row-level random split is used for validation, letting
items from the same topic group sit on both sides.

### 2. Does not reduce to clause position — REQUIRED · MODELING

The clause_ids column's display order has been deliberately reshuffled
per item so it carries no information about a clause's position in the
original prompt. A solution that predicts responsibility purely from
where a clause appears in the shown clause_ids/clause_texts order, without
using the fp__ behavior columns, is answering a question this dataset was
specifically built to make unanswerable.

*Fails if:* the model's only input is the clause's index in clause_ids,
with the fp__ behavior columns unused.

### 3. Predicts a per-prompt ranking, not a per-clause independent label — REQUIRED · MODELING

The target is a relative ORDERING of a prompt's own clauses, not an
absolute label applied to each clause in isolation. A solution that
scores every clause independently (e.g. a fixed threshold on some
learned per-clause score) without ever comparing clauses WITHIN the same
item can produce internally inconsistent or invalid rankings.

*Fails if:* the submitted ranking for any row is not an exact permutation
of that row's own clause_ids.

### 4. Necessary/correlated flags are not just "top-1 gets necessary" — RECOMMENDED · MODELING

The necessary/correlated flag is a threshold judgment on the underlying
responsibility score, not simply "whichever clause ranks first." In many
items, the top-ranked clause still falls under the necessary threshold
(no single clause produced a large individual shift), and in some items
more than one clause clears it. A solution that always flags exactly the
top-ranked clause as necessary and everything else as correlated has not
engaged with the actual threshold structure of the labels.

*Fails if:* the flags column is a deterministic function of ranking
position alone (e.g. always "necessary" for rank 1, always "correlated"
otherwise) rather than a genuinely separate prediction.

### 5. Beats the trained baseline reported in STATE.md — RECOMMENDED · TRAINING

A trained classifier (logistic regression or gradient boosting) on the
released behavior measurements is measured directly on the shipped
dataset; see STATE.md's FINAL LOCKED NUMBERS for the exact score. A
solution that does not clear this has not found signal beyond what a
standard baseline model already finds in the released features.

*Fails if:* the final held-out combined score is at or below the
trained-baseline number recorded in STATE.md.

### 6. Handles a variable clause count correctly — REQUIRED · DATA_HANDLING

Each item's prompt has its own clause count (3 to 10), and the submitted
ranking and flags must cover exactly that item's own clause_ids - never a
fixed-size vocabulary shared across rows. A solution that assumes a
constant number of clauses per item will emit invalid or incomplete
submissions on many rows.

*Fails if:* any row's ranking or flags omits a clause_id present in that
row's own clause_ids, or includes one that is not.

### 7. Emits a structurally valid submission — REQUIRED · CODE_QUALITY

Exactly one row per item_id in test.csv with ranking a permutation of
that row's own clause_ids and flags covering every one of them.
sample_submission.csv in public/ shows the exact expected format.

*Fails if:* the grader cannot match a row's item_id, or either of the two
required prediction columns is missing entirely.

### 8. Completes within the compute budget — REQUIRED · CODE_QUALITY

The full pipeline - loading train.csv, fitting whatever model is chosen,
and writing submission.csv for the test rows - must finish well inside
the 1.5-hour GPU-time limit. The released feature set is small and
tabular; there is no reason a competent solution should approach the
limit.

*Fails if:* the notebook does not complete top-to-bottom inside the
budget.

### 9. Reports which topics it generalizes to — RECOMMENDED · COMMUNICATION

Because the test set holds out only a few topic groups, per-topic scores
on a held-out validation group (not just an aggregate number) are
informative about whether a solution is learning a genuine
behavior-to-clause attribution signal, or overfitting to quirks of the
training topics' specific prompt style.

*Fails if:* no per-topic breakdown is reported anywhere in the notebook,
only a single aggregate score.
