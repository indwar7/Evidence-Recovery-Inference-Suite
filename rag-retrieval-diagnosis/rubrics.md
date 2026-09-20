# Predicting Retrieval Sufficiency in RAG Pipelines — Rubrics

Criteria are graded alongside the leaderboard score. They describe what good
ML engineering looks like *for this task specifically*: diagnosing whether a
real retrieval step gave a model enough to answer, and what it should do
next, from retrieval scores and answer behaviour alone.

---

### 1. Validates on a topic-level split — REQUIRED · TRAINING

Every item's retrieval and behaviour measurements come from one of the
training topics; the held-out test topics never appear in train.csv at
all. Validation must hold out **whole topics**, mirroring the real split,
not individual rows.

*Fails if:* a row-level random split is used for validation, letting items
from the same topic sit on both sides.

### 2. Does not reduce to thresholding retrieval scores alone — REQUIRED · MODELING

The released retrieved_score_1..k columns alone can approximate the
diagnosis/insufficient split for many items, but reliably telling
gold_low_rank apart from query_ambiguous - two different insufficient
reasons - requires reading the released generation behaviour measurements
(fp__ columns), not just the similarity scores. A solution that ignores the
fp__ columns entirely is answering an easier, narrower question than the
one this task asks.

*Fails if:* the model's only input is retrieved_score_1..k and k, with the
fp__ behaviour columns unused.

### 3. Predicts a consistent joint trace, not three independent labels — REQUIRED · MODELING

diagnosis, action, and reason_code are not independent: certain
combinations are structurally invalid (for example, reason_code=gold_absent
paired with action=answer_as_is never occurs in the true labels). A
solution that predicts each of the three fields with a separate,
uncoordinated model can emit internally inconsistent rows even when each
field's marginal accuracy looks reasonable.

*Fails if:* the submitted rows contain combinations of diagnosis, action,
and reason_code that never appear together anywhere in train.csv's target
columns.

### 4. Does not simply recompute a retrieval-score gap — RECOMMENDED · FEATURE_ENGINEERING

Thresholding on (best own-topic score) minus (best cross-topic score) 
alone - both derivable from the released retrieved_score_1..k columns
directly - was the shortcut this task was explicitly checked against
before shipping (see STATE.md's SHORTCUT PLAN and FINAL LOCKED NUMBERS). A
stronger solution engineers features that combine retrieval scores with
the generation behaviour measurements (e.g. whether the answer hedges,
whether it explicitly says the context lacks the answer) rather than
treating the score gap as the whole feature set.

*Fails if:* the only engineered feature is a single retrieval-score gap or
threshold, with no use of the fp__ behaviour columns.

### 5. Beats the trained baseline reported in STATE.md — RECOMMENDED · TRAINING

A trained classifier (logistic regression or gradient boosting) on the
full released feature set - both retrieval scores and behaviour
measurements - is measured directly on the shipped dataset; see STATE.md's
FINAL LOCKED NUMBERS for the exact score. A solution that does not clear
this has not found signal beyond what a standard baseline model already
finds in the released features.

*Fails if:* the final held-out joint-accuracy score is at or below the
trained-baseline number recorded in STATE.md.

### 6. Emits only valid field values — REQUIRED · DATA_HANDLING

diagnosis must be one of {sufficient, insufficient}; action must be one of
{answer_as_is, reformulate, abstain}; reason_code must be one of
{sufficient, gold_low_rank, gold_absent, query_ambiguous}. A solution that
emits values outside these fixed vocabularies scores 0 on that field for
that row rather than being rejected, but has not engaged correctly with
the task's output space.

*Fails if:* any submitted diagnosis, action, or reason_code value falls
outside its fixed vocabulary above.

### 7. Emits a structurally valid submission — REQUIRED · CODE_QUALITY

Exactly one row per item_id in test.csv with all three of diagnosis,
action, reason_code present. sample_submission.csv in public/ shows the
exact expected format.

*Fails if:* the grader cannot match a row's item_id, or any of the three
required columns is missing entirely.

### 8. Completes within the compute budget — REQUIRED · CODE_QUALITY

The full pipeline - loading train.csv, fitting whatever model is chosen,
and writing submission.csv for the test rows - must finish well inside the
1.5-hour GPU-time limit. The released feature set is small and tabular;
there is no reason a competent solution should approach the limit.

*Fails if:* the notebook does not complete top-to-bottom inside the
budget.

### 9. Reports which topics it generalizes to — RECOMMENDED · COMMUNICATION

Because the test set holds out only a handful of topics, per-topic scores
on a held-out validation topic (not just an aggregate number) are
informative about whether a solution is learning a retrieval-sufficiency
signal that transfers, or overfitting to quirks of the training topics'
specific phrasing.

*Fails if:* no per-topic breakdown is reported anywhere in the notebook,
only a single aggregate score.
