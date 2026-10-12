# Rust Toolchain Local Variable Names

A real function from the Rust project's source code, with every local
variable and parameter name replaced by a positional placeholder, looks like
this:

```rust
fn now() -> SystemTime {
    let VAR0 = unsafe {
        let mut VAR1 = MaybeUninit::zeroed();
        expect_success(abi::SOLID_RTC_ReadTime(VAR1.as_mut_ptr()), &"SOLID_RTC_ReadTime");
        VAR1.assume_init()
    };
    let VAR2 = unsafe {
        libc::mktime(&mut libc::tm { tm_sec: VAR0.tm_sec, tm_min: VAR0.tm_min })
    };
    VAR2
}
```

The original names, in declaration order, were `rtc`, `out`, `t`. This task
asks you to recover that sequence — the exact names a human developer chose —
from the masked code alone.

## Task

Each row of the test set gives one masked Rust function in a column named
`masked`, and the number of placeholders it contains in `num_vars`. Your job
is to output, for each row, the sequence of original identifier names that
belong at `VAR0`, `VAR1`, ..., `VAR{num_vars-1}`, in that order, joined by `|`.

Every placeholder in a row refers to exactly one real local variable or
function parameter, consistently substituted at every place it is used in
that function (not only at its declaration site). A function is only
included if none of its masked names appears anywhere else in it — not as a
field or method name, path segment, struct label, string literal or
(for names of 3+ characters) comment — so the answer is never visible in
the text. No two placeholders in the same row share a name by construction
(the dataset drops any function where a name is shadowed/re-bound, to avoid
an ambiguous answer).

## What you are actually being asked

This is not a code-completion or code-generation task: the code's logic,
control flow, and every non-identifier token are already given in full and
unchanged. It is also not classification into a fixed label set: the space of
possible names is open-ended (6,937 distinct names appear in the shipped
training data alone, and about one test slot in six carries a name that never
occurs in training), and the correct output is a *sequence* of names whose
length varies per row (from 2 to 17 in the test set), not a single label.

The closest adjacent line of research is variable-name prediction in the
code2vec / code2seq family, which predicts names from a learned graph or path
embedding of a function's AST, typically evaluated per-variable in isolation.
This task differs in two ways: the unit of prediction is the full ordered
sequence of every masked name in one function scored together (not one
variable judged alone), and the corpus is Rust systems and compiler code
(the standard library, the rustc compiler and rust-analyzer) rather than the
Java/C# corpora those benchmarks typically use — Rust's ownership and
borrow-checker idioms (`&mut`, `unsafe` blocks, `MaybeUninit`) produce a
different, less-studied naming distribution than garbage-collected-language
corpora.

## Dataset

Functions come from three codebases in the `rust-lang/rust` repository at one
pinned commit: the standard library, the rustc compiler, and rust-analyzer.
Discover the data under the mounted data directory at solve time (its exact
path is provided by the environment; do not hard-code a path from this
document). You will find:

- `train.csv` — columns `id`, `masked`, `num_vars`, `names`. `names` is the
  true `|`-joined answer sequence for that row.
- `test.csv` — columns `id`, `masked`, `num_vars`. No `names` column.
- `sample_submission.csv` — the exact submission format, with a placeholder
  name for every slot.
- `dataset_stats.json` — row counts and the distribution of `num_vars`.

Train and test are split by source file, not by function: no function in
`test.csv` has a sibling function from the same original file sitting in
`train.csv`.

## Rules

**Use only the data directory.** The Rust project's source is real, public,
and trivially searchable on GitHub — looking up the exact
function by its remaining (unmasked) code, comments, or structure and copying
its real identifier names from the live repository is not solving the task.
A submission built this way is invalid regardless of its score, and so is any
submission that depends on data not shipped in the data directory. The solver
environment is expected to have no network access; your solution must run to
completion from the data directory alone.

**No pretrained code-language model and no static-analysis identifier-naming
tool.** Specifically, no pretrained source-code model (Copilot-style
completion models, pretrained code2vec/code2seq checkpoints, or any model
whose pretraining corpus plausibly included the Rust standard library itself)
and no third-party rename/refactoring tool that infers "idiomatic" names from
a type signature. Predictions must come from parameters trained on the
provided `train.csv` alone.

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

For each test row, score is the fraction of its `VARi` slots whose predicted
name exactly matches the true name (case-sensitive, exact string match, no
partial credit for a close-but-different spelling). The final score is this
per-row fraction averaged across all rows:

```python
def row_score(pred_names: list[str], gold_names: list[str]) -> float:
    n = len(gold_names)
    correct = sum(p == g for p, g in zip(pred_names, gold_names))
    return correct / n
```

**Why this metric and not whole-row exact match.** Requiring every slot in a
row to match before awarding any credit collapses most of the score range to
zero for rows with many variables (mean 3.0 masked names per test row, up to
17):
a model correct on 4 of 5 names in a row would score identically to one
correct on 0 of 5. Per-slot accuracy rewards partial correctness, which is
where most of the real signal in this task lives.

**Why not a similarity-based string metric** (edit distance, embedding
cosine similarity between predicted and true names). Identifier names are
short, and a plausible-looking wrong guess is often *more* superficially
similar to the true name than a correct-but-differently-styled one (`buf`
vs. `buffer` are closer under edit distance than `buf` vs. the equally valid
`data`) — a soft metric would reward superficial string overlap rather than
actually recovering the author's choice. Exact match avoids rewarding
near-misses that are not actually correct.

**Measured baselines**, on the shipped split (train and test are disjoint by
source file, not just by function — see Dataset above):

- Blank submission: 0.000
- `sample_submission.csv` (constant placeholder "x" at every slot): 0.010
- Train-skeleton exact lookup (predict by matching this row's masked text
  against identical masked text in `train.csv`): 0.010 — confirms the
  cross-row boilerplate shortcut this dataset was built to close is in fact
  closed.
- Per-file most-common-name (predict the most frequent training name seen
  elsewhere in this row's own source file): 0.013 — identical to the
  file-blind version below, confirming the file-level split closes the
  same-file-sibling shortcut.
- Global most-common-name (predict the single most frequent training name at
  every slot): 0.013
- Positional most-common-name (predict, for slot index *i*, the most frequent
  training name seen at that index — ignores the code entirely): 0.021
- Type-signature heuristic (predict the most frequent training name observed
  for a parameter's declared type, falling back to the positional baseline):
  0.256 — the strongest non-trained adversary measured. It is strong on this
  corpus because compiler and IDE code binds some names tightly to types
  (`tcx: TyCtxt`, `span: Span`, `db: &dyn HirDatabase`); a trained model has
  to learn that signal *and* the usage context the heuristic ignores.
- Reference solution (trained character-level encoder/decoder, see
  `solution.ipynb`): {{REF}}, a margin of {{MARGIN}} over the strongest
  adversary.
- Oracle (true answers): 1.000

## Submission

Submit a CSV named `submission.csv` with exactly two columns: `id`, `names`.
One row per test id. `names` is the predicted sequence, `|`-joined, in
`VAR0, VAR1, ...` order, with exactly as many entries as that row's
`num_vars`.

A missing id, a duplicated id, a predicted sequence of the wrong length, or a
blank entry scores 0 for that row only — the rest of your submission is
still scored. Only a submission missing the required `id` or `names` column
is rejected outright.

## Compute

Budget: 1x NVIDIA A10G, 10 CPU cores, 62 GB RAM, under 1.5 hours. The intended
approach is a small sequence encoder (character- or subword-level) over each
function's masked source, pooling the hidden states at each placeholder's
occurrence positions, followed by a short autoregressive decoder that emits
each name character-by-character — the shipped reference uses exactly this
architecture and fits comfortably within the budget on a single GPU.
