# Rubrics — Rust Toolchain Local Variable Names

Each rubric is specific to this task and approach-neutral. Every rule in
`problem-description.md` has a matching `[REQUIRED]` rubric here.

**[REQUIRED] Uses only the provided data directory; no lookup of the real
source.** The Rust project's source is real and public, and the exact
function behind any masked row is trivially searchable on GitHub by its
unmasked tokens, comments, or structure. A submission that recovers names
this way — or depends on any file, network call, or cached copy of the
`rust-lang/rust` repository outside the shipped data directory — is invalid
regardless of its score. This matters because the corpus cannot be made
unsearchable (it is real, published code); the environment's lack of network
access is what actually closes this path, and a solution assuming network
access will not run at all in the solving environment.

**[REQUIRED] No pretrained code model and no static rename/refactoring
tool.** Predictions must come from parameters trained on `train.csv` alone.
This matters because a pretrained source-code model's training corpus
plausibly already contains the Rust standard library verbatim, which would
let it recall rather than infer the masked names — and because IDE-style
"suggest a name from this type" refactoring tools are hand-coded heuristics,
not trained parameters, and are not a valid solution path under this rubric
even though some externally-written ones might score well.

**[REQUIRED] Predicts a full, correctly-ordered, correctly-counted name
sequence per row.** `num_vars` varies from 2 to 17 across the test set (mean
3.0). A submission must emit exactly that many `|`-joined names in `VAR0,
VAR1, ...` order for every row; a submission that always emits a fixed
count, or emits names in an order not matching the placeholder indices, is
leaving points on the table by construction and will score 0 on every row
where the count is wrong.

**[REQUIRED] Learns real per-function usage context, not just positional or
type-based priors.** The positional most-common-name baseline (predict the
training set's most frequent name at slot index *i*, ignoring the code
entirely) scores 0.021; the much stronger type-signature heuristic (predict
the most frequent name seen for a parameter's declared type) scores 0.256,
because compiler and IDE code binds some names tightly to their types. Both
are measured, non-trained adversaries. A valid submission is expected to
exceed both by actually encoding the function's surrounding code, not only
replicating one of these two shortcuts under a different name.

**[REQUIRED] No cross-row lookup of identical masked functions, and no
cross-row lookup via a shared source file.** The dataset is deduplicated by
exact masked-text skeleton AND split so that every function from one source
file sits entirely on one side; an exact train-skeleton lookup baseline
measures 0.010 and a per-file most-common-name baseline measures 0.013
(identical to the file-blind global baseline) on the shipped split —
indistinguishable from the floor in both cases, confirming no graded test
row has an identical counterpart, or a same-file sibling, in `train.csv`. A
submission that performs either kind of lookup is not solving the task and
will not score above these floors regardless.

**[RECOMMENDED] Exact match is the standard, not near-miss credit.** Per-slot
scoring is exact string match (case-sensitive) with no credit for an
edit-distance-close wrong guess. A submission tuned to produce
plausible-but-generic names (e.g. always guessing a short common noun) rather
than the specific name this function's author chose will not benefit from
partial string similarity it does not actually have.

**[RECOMMENDED] Train/val split should respect the shipped split exactly.**
`train.csv` and `test.csv` are already split by a stable hash of each
function's source file and masked text; a submission that re-merges and
re-splits the two files (for example, to get more training data by including
masked test rows) is working against a split that was deliberately
deduplicated and hashed to prevent exactly this kind of leakage back in.
