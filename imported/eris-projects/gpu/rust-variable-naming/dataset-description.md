# Rust Toolchain Local Variable Names — Dataset

## Overview

This dataset pairs real Rust functions from the Rust project's own codebases
— the standard library, the rustc compiler and rust-analyzer — with the
original local variable and parameter names their authors chose, versus a
version of the same function where those names are replaced by positional
placeholders (`VAR0`, `VAR1`, ...). It supports training and evaluating
models that recover the identifier names a human developer would actually
choose, from code structure and usage context alone.

## Source

- **The Rust project's source code**, from the official `rust-lang/rust`
  repository, pinned at commit `76c90957b7e422c4b9c45192b0197214d7de5a54`
  (2026-10-09).
- Source URL: `https://github.com/rust-lang/rust`
- Subtrees used, each copied to the same repo-relative path:
  - `library/core`, `library/alloc`, `library/std` — the standard library;
  - `library/coretests`, `library/alloctests` — its hand-written test crates;
  - `compiler/` — the rustc compiler (all `rustc_*` crates);
  - `src/tools/rust-analyzer/crates/` — rust-analyzer.
- Fields used: the raw `.rs` source text of each file. No commit metadata,
  issue data, or other repository content is used.
- License: dual MIT / Apache-2.0, the project's own licensing terms (see
  `dataset/raw/LICENSE-MIT` and `dataset/raw/LICENSE-APACHE`, and
  `dataset/raw/ATTRIBUTION.txt`).

## Fetch method

- A single `git clone --filter=blob:none --sparse` against the pinned commit,
  followed by `git sparse-checkout set` of the seven subtrees above — this
  downloads only the ~70 MB actually used, not the full ~1 GB repository.
- No pagination, rate limits, or retries are needed: this is a single local
  clone operation, not a paginated API walk.
- Two deterministic post-processing steps are applied to the checkout:
  - every directory named `test_data` is removed — rust-analyzer's parser and
    syntax fixture corpora (542 tiny synthetic `.rs` snippets) are not
    production code;
  - later byte-identical copies of a file are dropped (first in sorted-path
    order kept), so the snapshot contains no duplicate contents. Upstream has
    a few such pairs (`os/rtems/mod.rs` == `os/nuttx/mod.rs`,
    `panic/location/file_b.rs` == `file_a.rs`); none yields a usable
    function, so the derived split is unaffected.
- `dataset/generator/build_raw.py` reproduces this fetch and post-processing
  exactly, pinned to the commit above.

## File Structure

What this dataset ships (the raw archive, 79 MB, 4,439 `.rs` files):

- `dataset/raw/library/{core,alloc,std,coretests,alloctests}/**` — standard
  library source.
- `dataset/raw/compiler/**` — rustc source.
- `dataset/raw/src/tools/rust-analyzer/crates/**` — rust-analyzer source.
- `dataset/raw/LICENSE-MIT`, `dataset/raw/LICENSE-APACHE` — the project's own
  dual license texts.
- `dataset/raw/ATTRIBUTION.txt` — attribution statement and provenance.
- `dataset/raw/meta.json` — pinned commit, subtrees included, post-processing
  applied, and the rationale for what is excluded (see Limitations).
- `dataset/generator/build_raw.py` — reproduces this raw snapshot from the
  pinned commit.
- `dataset/generator/validate_submission.py` — a standalone submission
  format checker.

The full masking and split logic lives in the self-contained `prepare.py`,
which locates the snapshot root under whatever directory it is given.

### What `prepare.py` derives (not shipped here)

Running `prepare.py` against the raw archive produces, deterministically and
without any random seed:

- a masked version of every usable function, with every local variable and
  parameter name that can be unambiguously identified and safely masked
  replaced by `VAR0`, `VAR1`, ... in declaration order;
- a train/test split (`train.csv`, `test.csv`) by a stable hash of each
  function's **source file** — every function from one `.rs` file is placed
  entirely on one side, so no test function has a same-file sibling in
  training data;
- `sample_submission.csv` and `dataset_stats.json`;
- the held-out `answers.csv` naming the true identifier sequence for each
  test row.

## Features

Each usable unit is one Rust `fn` item with a body, located by a
standard-library Rust tokenizer in `prepare.py` (comments, string, char and
raw-string literals, lifetimes, numbers, identifiers and punctuation), taken
only outside macro bodies and never nested inside another `fn`, for which:

- its brackets balance and it is between 60 and 4,000 characters long;
- it has at least two maskable bindings: simple identifier parameters
  (`name: T`, `mut name: T`) and the lower-case identifiers bound by `let`
  patterns (`let x`, `let (a, b)`, `let Some(x)`, `let Foo { a, b: c }`);
  constructors, paths, field labels and upper-case names are never bindings;
- no masked name is bound twice (shadowed) — such a function is dropped
  entirely rather than guessed at;
- after masking, **no masked name survives anywhere in the function in any
  role** — not as a method or field after `.`, a path segment, a struct
  field label, a macro name, or a binding introduced by `for`/`match`/a
  closure outside the masked binding's scope. A function with such a
  survivor is dropped, so the text never shows an answer, even as context;
- no masked name appears, as a whole word, in any string or char literal
  of the function (Rust 2021 `format!("{name}")` captures a variable by name
  inside the string) or, for names of 3+ characters, in any of its comments
  — this closes a measured leak where authors' comments state a parameter's
  name in prose (for example, "exclusive reference ... through `orig`");
- its masked text is not a duplicate, after whitespace normalization, of any
  other kept function's masked text — this closes a measured cross-row lookup
  shortcut from repeated boilerplate (for example, trait-impl one-liners that
  recur with identical structure and identical names);
- it is placed on only one side of the train/test split, together with every
  other function from the same source file — this closes a measured
  same-file shortcut: on an earlier per-function split, predicting the most
  frequent name seen elsewhere in a test function's own file scored 0.178
  against a file-blind baseline's 0.034.

## Intended Use

- **Primary use**: training and evaluating sequence models that predict a
  developer's original local identifier names from a function's masked
  source and usage context.
- **Also suitable for**: studying naming-convention regularities across three
  large, real, permissively-licensed Rust codebases; probing how much
  identifier choice is recoverable from static type and usage signals alone
  versus requiring learned, corpus-wide naming conventions.
- **Not suitable for**: evaluating code *generation*, type inference, or any
  task where the input/output is a full program rather than a name sequence;
  evaluating on languages other than Rust; claiming the measured scores
  generalize to application code, since this corpus is systems, compiler and
  IDE-tooling code written to unusually consistent internal style guides.

## Limitations

1. **Three codebases, one project.** All functions come from the Rust
   project's contributors, who follow strong internal conventions. Some
   names are extremely conventional and type-bound (`tcx: TyCtxt`,
   `span: Span`, `db: &dyn HirDatabase`, `cx: &LateContext`), which is why
   a declared-type -> most-frequent-name heuristic is a strong adversary on
   this corpus (see `config.yaml`); naming patterns measured here will not
   transfer uniformly to application code written by a broader population.
2. **`library/stdarch` and the other `src/tools/*` crates deliberately
   excluded.** `stdarch` is dominated by macro-generated SIMD intrinsics
   whose local variables are almost exclusively single letters (`r`, `a`,
   `b`, `e`, `c`) — including it collapsed naming diversity and made the task
   close to a template-position lookup (with it included, the most-common-name
   content-free baseline measured 0.165 on the original standard-library-only
   corpus; with it excluded, 0.033). The remaining `src/tools/*` crates
   (clippy, rustfmt, miri, ...) are left out so the snapshot is three
   well-known codebases rather than a grab-bag.
3. **Shadowing functions are dropped, not resolved.** Real Rust frequently
   re-binds a name to a transformed value in the same function (`let x = f(x)`
   pattern). Rather than guess which occurrence belongs to which binding, any
   function containing this pattern (5,057 of the 69,581 extracted `fn`
   items) is dropped entirely — the dataset is therefore biased toward
   functions with simpler, non-shadowing local scopes.
4. **Leak filtering is name-based, not semantic, and strict.** A function is
   dropped if a masked name survives anywhere in its text in any role
   (4,263 functions: typically `let len = self.len()` or `Self { name, .. }`
   style code where a field or method shares the variable's name), appears
   in a string or char literal (1,907), or appears as a whole word in its
   own comments (1,850). These are conservative fixes for real, measured
   leaks — but they also drop functions where the coincidence is harmless,
   and they bias the corpus away from the common "name the variable after
   the accessor" idiom, trading recall for a hard no-answer-in-text
   guarantee.
5. **Most functions are too small to be units.** 38,930 of the 69,581
   extracted `fn` items have fewer than two maskable local names and are
   dropped; the kept 13,331 skew toward functions with real local state.
   Bindings introduced by `for`, `match` arms, `if let` and closure
   parameters are not masked (they are not answers); when such a binding
   reuses a masked name the function is dropped by the rule in point 4.
6. **Context length is truncated for the reference model only.** The shipped
   `masked` column is never truncated — full function text (up to 4,000
   characters) is always kept — but the reference solution caps the context
   it feeds its encoder at 600 characters per function; the dataset itself
   has no such limit, and a different solution could use the full text.
7. **Long-tailed, open answer space.** 8,041 unique identifier names appear
   across the corpus (6,937 in the training split), 4,941 of them exactly
   once; the top 20 names account for 20.0% of training slots. 17.7% of test
   slots carry a name that never occurs in `train.csv`, so no model trained
   on the provided data alone can exceed 0.823 — a bound the shipped
   reference is far below, leaving the remaining headroom genuine.

## License

Dual-licensed MIT OR Apache-2.0, per the Rust project's own licensing terms.
Both license texts are shipped verbatim at `dataset/raw/LICENSE-MIT` and
`dataset/raw/LICENSE-APACHE`; attribution is recorded in
`dataset/raw/ATTRIBUTION.txt`. No modifications were made to the licensed
source files themselves — only read access to produce the derived,
deterministic train/test split.
