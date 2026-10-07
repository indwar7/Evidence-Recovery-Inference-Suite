# Palimpsest — Rubrics

Criteria are graded alongside the leaderboard score. They describe what good ML
engineering looks like *for this task specifically*: recovering a forest of
edit lineages from a shuffled bag of document snapshots.

---

### 1. Infers the number of lineages rather than assuming it — REQUIRED · DATA_HANDLING

The number of source documents per item **varies between 2 and 4 and is never
disclosed**; the number of `ROOT` snapshots equals it. The solution must derive
the grouping from the data (clustering, threshold on similarity, model-scored
attachment) instead of hard-coding a constant number of chains or a constant
number of roots per item.

*Fails if:* a fixed `n_clusters=3` (or any constant) is used for every item, or
the root count is fixed.

### 2. Validates on an item-level split — REQUIRED · TRAINING

Snapshots inside one item are highly interdependent: a pair label is only
meaningful relative to the other snapshots in that bag. Validation must hold
out **whole items**. Splitting individual snapshots or individual pairs across
train and validation puts near-duplicate revisions of the same article on both
sides and produces a validation score that will not survive the private test.

*Fails if:* a row-level or pair-level random split is used for validation.

### 3. Models edit direction, not just similarity — RECOMMENDED · MODELING

Text similarity identifies which snapshots are *adjacent* in a lineage — the
most-similar snapshot is a true neighbour about 80% of the time — but it is
symmetric and cannot say which of the two came first. A strong solution scores
**ordered** pairs asymmetrically, using what a revision *adds* versus what it
*removes* (a parent→child diff is not the reverse of the child→parent diff).

*Fails if:* the parent is chosen by a symmetric similarity score with no
directional component beyond byte size.

### 4. Decodes a globally consistent forest — RECOMMENDED · MODELING

Independently taking each snapshot's best-scoring parent produces cycles and
duplicate parents, and it throws away the constraint that each lineage is a
chain. Solutions that decode the pairwise scores into a structure globally —
maximum spanning arborescence (Chu-Liu/Edmonds), Hungarian assignment,
cycle-breaking, or beam search over orderings — should outperform per-node
argmax on the same scores.

*Fails if:* per-node argmax is used and cycles are patched arbitrarily with no
attempt at global consistency.

### 5. Does not lean on byte length as the ordering signal — RECOMMENDED · FEATURE_ENGINEERING

Windows were selected for churn rather than growth, so within a true lineage
byte size orders revisions at only ~0.60 pairwise accuracy — barely above the
0.50 coin flip. Size is a legitimate weak feature, but a solution that uses it
as the primary direction signal will plateau near the 0.32 size-chain baseline.

*Fails if:* direction is decided by size comparison alone.

### 6. Emits a structurally valid submission — REQUIRED · CODE_QUALITY

Exactly one row per `(item_id, snapshot_id)` in `test.csv`; every `parent_id`
is either `ROOT` or a snapshot **of the same item**; no snapshot is its own
parent; no cycles. `validate_submission.py` ships in `public/` and applies the
identical checks the grader applies.

*Fails if:* the grader rejects the submission for any format or forest violation.

### 7. Beats the similarity-nearest-neighbour baseline — RECOMMENDED · TRAINING

Attaching each snapshot to its most similar smaller snapshot scores about
**0.385**. A solution that does not clear this has not learned anything the
metric rewards, whatever its internal validation says.

### 8. Probes for shortcut signals instead of assuming clean text — RECOMMENDED · AGENT_BEHAVIOR

The snapshots are raw wikitext. A careful solution inspects it for accidental
ordering giveaways (embedded dates, revision counters, template metadata) and
reports what it finds — rather than either assuming the text is sanitised or
silently depending on an artifact. The released data has had `lastedit=`
template parameters and article titles removed for exactly this reason; a
solver that verifies this is doing the right thing.

### 9. Completes within the compute budget — REQUIRED · CODE_QUALITY

The full pipeline — featurisation of all ordered pairs, training, decoding, and
writing `submission.csv` — must finish inside the 1.5-hour CPU-only limit. Pair featurisation is quadratic in item size (up to 40 snapshots means
1,560 ordered pairs per item); a solution that ignores this will not finish.
