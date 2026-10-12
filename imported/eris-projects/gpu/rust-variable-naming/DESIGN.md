# DESIGN — Rust Toolchain Local Variable Names

The measured design record. Every number here was produced by a script run
against the exact files shipped in this package, except where a number is
explicitly labelled as coming from an earlier build.

## Task in one line

Given a Rust function with every local variable/parameter name replaced by a
positional placeholder, recover the original name sequence — an open-vocabulary,
variable-length, per-function sequence-to-sequence task whose target is the
source's own identifier names, never an annotation added for this dataset.

## Why this shape (and what was rejected first)

The hard requirement, per this project's own rejection register, was a coding-
domain seq2seq task where no single deterministic tool computes the answer
from the input alone — otherwise an agent can trivially "solve" the task by
calling that tool, which is a near-certain saturation shortcut (score 1.0,
Rejection Register section 3.1).

Candidates considered and rejected before this one, each killed by measurement
or by this exact shortcut risk:

- **GBIF taxonomic synonym -> accepted name** (biological nomenclature):
  rejected before fetching any scale of data — inspection of real synonym
  pairs (`Panthera gombaszoegensis -> Leo gombaszoegensis`) showed the mapping
  is a discrete historical reclassification event, not a learnable
  orthographic transform; closer to a lookup/retrieval task over genus names
  than to sequence transduction.
- **Rosetta Code polyglot programs** (same algorithm, many languages):
  rejected at the license gate — Rosetta Code is GFDL-1.2 only (confirmed by
  fetching its own Copyrights page), which is not an accepted platform
  license.
- **SQL dialect translation via sqlglot's own test fixtures**: the richest
  candidate found (MIT-licensed, tens of MB of real `(sql, dialect) ->
  equivalent_sql` triples recorded as the library's own test assertions,
  genuinely seq2seq, real cross-dialect syntax differences). Rejected
  specifically because `sqlglot` itself is `pip install`-able and *is* the
  exact function that generated every label — an agent need only
  `import sqlglot; sqlglot.transpile(...)` to score 1.0. Unlike an external
  gazetteer (which at least requires network access the solver environment
  lacks), this is a silent, undetectable shortcut: importing a popular SQL
  library looks like ordinary code, not an obvious violation of a Rules
  section.

**Why variable-naming restoration survives this filter**: no tool, given
arbitrary source code, can compute "what name would this specific human
developer have chosen." It is a historical fact about one person's naming
choice, not a deterministic function of the code — the same code structure
is consistent with many valid names. This is the same property that makes
the Vietnamese tone-mark and accent-translation challenges in this portfolio
resistant to a single-tool shortcut, applied to a new domain (code) with a
new source and a new learnable signal (local usage/type context rather than
phonetic/orthographic context).

**Relationship to code2vec/code2seq**: that line of research established
variable-naming prediction as learnable from code structure, typically scored
per-variable in isolation on Java/C#/JavaScript corpora via a path- or
graph-based embedding. This task differs in input shape (plain masked source
text, not an AST path corpus), output shape (the full ordered sequence of
every masked name in one function, scored together, variable length), and
corpus (Rust systems, compiler and IDE-tooling code, whose ownership and
borrow-checker idioms produce a different naming distribution than
garbage-collected-language corpora) — differing in at least 2 of {input
shape, output shape, metric} per this project's own originality bar, while
deliberately not claiming to be the first work on variable-naming prediction
as a general idea.

## The data and why this source

`rust-lang/rust`, pinned at commit `76c90957b7e422c4b9c45192b0197214d7de5a54`
(2026-10-09). Dual MIT/Apache-2.0, confirmed from the repository's own
`LICENSE-MIT`/`LICENSE-APACHE` files and the GitHub API's `license.spdx_id`
field. Seven subtrees, copied to their repo-relative paths: the standard
library (`library/{core,alloc,std}`) and its test crates
(`library/{coretests,alloctests}`), the rustc compiler (`compiler/`) and
rust-analyzer (`src/tools/rust-analyzer/crates/`) — 79 MB, 4,439 `.rs` files
after post-processing. Fetched via a sparse, blobless, depth-1 git clone
restricted to those subtrees, not the full ~1 GB repository — a single local
operation, no API pagination or rate limiting involved.

The corpus started as the standard library alone and was widened to rustc and
rust-analyzer only after measurement showed data volume, not architecture,
was the lever limiting the reference's margin (see "Reference architecture"
below). The three codebases are hand-written, idiomatic Rust maintained by
the same project under the same license and commit, which keeps provenance
to one source URL and one license check.

Two things are deliberately excluded, with reasons:

- `library/stdarch`, after measurement, not by assumption: an early
  corpus-wide probe including it found the most-common-name baseline at
  0.165 (single-letter names `r`, `a`, `b`, `e`, `c` dominate, from
  macro-generated SIMD intrinsics) versus 0.033 with it excluded, on the
  original standard-library-only corpus — a large, measured collapse in
  naming diversity that would have made the task close to template-position
  lookup rather than genuine naming recovery.
- rust-analyzer's `test_data` directories (542 tiny synthetic parser/syntax
  fixture files), removed by `build_raw.py` because they are not production
  code. The other `src/tools/*` crates (clippy, rustfmt, miri) were sized
  with the earlier parser (roughly 2,600 / 600 / 570 further usable
  functions) and left out so the snapshot stays three well-known codebases
  rather than a grab-bag.

`build_raw.py` also drops later byte-identical copies of a file (two such
pairs exist upstream; neither yields a usable function), so the snapshot
contains no duplicate contents.

## Masking correctness (measured, not assumed)

A first implementation used plain regex substitution of `let NAME = ...`
binding sites only. Measured on three hand-inspected examples, this leaked
the answer in two independent ways:

1. **Usage sites, not just declaration sites, leaked the real name.** Masking
   only the `let` binding left every later reference (`attr.permissions()`,
   `p.set_readonly(true)`, `out.as_mut_ptr()`) showing the real name in plain
   text. Fixed by masking every occurrence of a bound name throughout its
   scope, consistently.
2. **Shadowing produced an ambiguous answer.** A name rebound in the same
   function (`let attr = ...; ...; let attr = ...;`) has two different true
   values depending on which occurrence is asked about, with no way to
   express "which one" in a flat per-slot answer. Any function containing
   this pattern is dropped entirely.

A second, independently measured leak: **comments state the masked name in
prose.** Example: a function's own `SAFETY` comment reads "exclusive
reference for the duration of its call through `orig`" — the code's `orig`
is correctly masked to `VAR1`, but the comment literally spells out the
answer. Measured hit rate on the original standard-library build: 0.150 of
otherwise-usable functions leaked at least one name (length >= 3 characters)
this way. Fixed by dropping any function where a masked name appears as a
whole word inside its own comments.

A third leak, found by explicitly checking for it rather than assuming the
first two fixes were sufficient: **repeated boilerplate produces duplicate
masked skeletons** (on the standard-library build, 385 of 3,371 functions,
~11%, shared an exact masked-text skeleton with at least one other function —
most commonly trait-impl one-liners like `fn copy<R,W>(VAR0: &mut R, VAR1:
&mut W) -> Result<CopyState> { SpecCopy::copy(Copier { read, write }) }`,
repeated 12 times with identical names). Because the *answers* for duplicate
skeletons are also identical by construction, this is exactly the cross-row
lookup shortcut this project's rejection register warns about (section
2.6/2A). Fixed by keeping only the first-encountered function per normalized
masked skeleton.

**From an AST parser to a standard-library tokenizer.** Builds two to four
used `tree-sitter-rust` for masking. That dependency is not available in the
platform's `prepare.py` runtime (every accepted challenge's `prepare.py` in
this portfolio is standard library or pandas only, and the first upload
failed at import), so the shipped pipeline is a self-contained Rust
tokenizer: comments (nested block comments included), string / char /
raw-string literals, lifetimes, numbers, identifiers and punctuation, with
bracket matching on the token stream. `fn` items are taken only with a body,
only outside macro bodies, and never nested inside another `fn` — so code
quoted inside string literals, doc comments or `macro_rules!` bodies can
never become a unit (rust-analyzer's crates are full of Rust snippets quoted
inside test strings; the original balanced-brace regex had extracted ~700 of
them as items with toy names).

The rewrite also fixed a bug found while auditing the tree-sitter output:
tuple-struct and enum constructors inside `let` patterns had been collected
as bindings, so `let Some(x) = ...` was shipped as `let VAR10(VAR11) = ...`
with the answer `Some` — 398 rows (0.6%) of the previous build. The
tokenizer collects only lower-case identifiers in binding positions
(constructors, paths, field labels and upper-case unit-struct patterns such
as rustc's `LocalCrate: LocalCrate` are never bindings).

Finally, the masking rule was made **strict**: after masking, if any masked
name still appears anywhere in the function as an identifier — a method or
field after `.`, a path segment, a struct field label, a macro name, or a
binding introduced by `for`/`match`/`if let`/a closure outside the masked
binding's scope — the function is dropped. The tree-sitter builds had
tolerated the first four (`let VAR0 = self.len()` shipped with `.len()`
visible), which a reviewer would rightly call the answer sitting in the
text. The same whole-word rule is applied to string and char literals (Rust
2021 `format!("{name}")` captures a variable by name inside the string, a
leak the earlier builds missed). A self-audit over every shipped row
confirms zero answer identifiers survive anywhere in the masked text.

Measured breakdown of the shipped pipeline, over all 4,439 files: 69,581
`fn` items with a body; 1,672 dropped as longer than 4,000 characters and
2,219 as shorter than 60; 1 with unbalanced brackets; 38,930 (56%) have
fewer than two maskable names — the dominant cause, most Rust functions are
short enough to have 0 or 1 local binding; 5,057 are dropped for shadowing;
4,263 because a masked name survives in another role; 1,907 for a name in a
string literal; 1,850 for a name in a comment; 351 as duplicate skeletons;
**13,331 kept** (library 3,343: core 725, alloc 496, std 1,147, coretests
488, alloctests 487; compiler 7,046; rust-analyzer 2,942).

## The split

13,331 usable functions, split **by source file**, not by function: every
function extracted from a given `.rs` file goes entirely to train or entirely
to test, ranked by a stable SHA-256 hash of the repo-relative file path —
deterministic and seedless (no RNG; re-running `prepare.py` from the same raw
archive reproduces byte-identical `train.csv`/`test.csv`/`answers.csv`,
verified by running it twice and diffing, and from three different
`raw_root` layouts). **10,659 train rows across 2,105 files, 2,672 test rows
(20.0%) across 439 files, zero overlap** — well above the 200-row minimum
and the 2,000+/200+ preferred floors from this project's own build guidance.

**A fourth leak, found by an independent audit rather than by this build's
own measurement pass.** The first version of this split hashed by
`(source_file, masked_text)` — disjoint on exact function text, but NOT
disjoint by file: on the standard-library build, 248 of 574 source files
(43%) had functions on both sides, and 93.2% of test functions had at least
one same-file sibling sitting in train. Measured directly: a "most frequent
name observed elsewhere in this same file" baseline (train-only, file-scoped)
scored **0.178** against the file-blind global most-common-name baseline's
**0.034** — a 0.144 margin earned purely by exploiting same-file naming
idiolect, no code understanding at all. Fixed by re-splitting at the file
level; on the shipped split the probe collapses to exactly the global
baseline (0.0131 = 0.0131) because no test file has a sibling in train.

## The metric and what it replaced after measurement

Per-row, per-slot exact-match accuracy, averaged across rows:
`sum(pred_i == gold_i for i in slots) / len(slots)`, then mean over all test
rows. Considered and rejected:

- **Whole-row exact match** (every slot must match to score anything):
  rejected without needing to ship it — `num_vars` ranges 2-17 per test row
  (mean 3.0; up to 38 in train), so a model correct on all-but-one slot of a
  row would score identically to one correct on none, collapsing most of the
  achievable score range to zero and discarding the partial-credit signal
  that is the real point of measuring naming quality.
- **Edit-distance / embedding similarity between predicted and true name**:
  rejected because short identifier names make superficial string similarity
  actively misleading (`buf` is closer to `buffer` under edit distance than
  to the equally valid `data`), which would reward near-miss guesses that are
  not the author's actual choice over correct-but-differently-styled ones.

Chance-normalization (e.g. ARI-style correction) was not needed: there is no
natural "everyone in one group" degenerate answer analogous to a partition
task, since the label space per slot is open-vocabulary and a constant-guess
strategy is already measured at the metric's floor (see ladder below).

A hard ceiling worth stating: 17.7% of test slots carry a name that never
occurs in `train.csv`, so no model trained on the provided data alone can
score above 0.823. The reference sits far below that, so the remaining
headroom is genuine.

## The full measured ladder (on the shipped split, via the shipped `grade.py`)

| Anchor | Score | What it tests |
|---|---|---|
| Blank submission | 0.000 | floor |
| Constant placeholder ("x" at every slot) | 0.010 | floor, content-free |
| Train-skeleton exact lookup | 0.010 | the cross-row shortcut above, confirmed closed |
| Per-file most-common-name | 0.013 | the file-level shortcut above, confirmed closed (collapses to the global baseline below) |
| Global most-common-name | 0.013 | content-free, ignores code entirely |
| Positional most-common-name | 0.021 | content-free, ignores code entirely |
| Type-signature heuristic | 0.256 | strongest non-trained adversary: declared-type -> most frequent training name for that type |
| **Reference** (trained char-level encoder/decoder) | **{{REF}}** | trained parameters, attention-pooled occurrence context (see below) |
| Oracle (true answers) | 1.000 | ceiling |

All numbers in this table are measured on the shipped, file-disjoint split
via the real shipped `grade.py` against the real shipped `answers.csv` — not
estimates. `note_band` in `config.yaml` explains each anchor.

**Why the type heuristic is strong here.** Compiler and IDE code binds some
names almost deterministically to their types — `tcx: TyCtxt`, `span: Span`,
`db: &dyn HirDatabase`, `cx: &LateContext`, `def_id: DefId` are the most
frequent names in the corpus — so "most frequent training name for this
declared type" rises from 0.126 on the standard-library-only build to 0.256
here. That is real signal, not a leak: a trained model has to learn it too,
and then add what the heuristic ignores (every `let` binding, which has no
declared type, and the usage context of every placeholder).

## Reference architecture (measured iteration — five attempts)

Reaching a defensible margin over the type-signature heuristic took five
measured iterations, each one a genuine attempt to close the gap, not a
cosmetic rerun. The full history is kept here because it is itself evidence
that the margin was earned, not assumed. Attempts 1-4 are on the
standard-library-only corpus.

1. **Mean-pooling, per-function split (pre-fix).** Plain mean-pooling across
   all occurrence positions of each masked variable, trained 6 epochs on the
   *original* per-function split (before the same-file leak was found):
   scored 0.236 against that split's 0.234 type-signature heuristic — no real
   margin. Diagnosed as architecture-limited (mean pooling washes out the
   declaration-site and distinguishing usage-site signal by averaging it with
   every other occurrence) rather than undertrained.
2. **First/last/mean pooling, file-disjoint split, `min_vars=2`.** After
   fixing the file-level leak (see "The split"), the stronger pooling
   architecture was retrained on the corrected, harder split: 0.1789, 0.1698,
   and 0.1822 across three re-runs of essentially the same architecture (plus
   a dropout variant) — a genuine, reproducible ceiling, not noise, but only
   a ~0.05 margin over that split's 0.1296 type-signature heuristic.
3. **Raising `min_vars` to 3 (tried and reverted).** Hypothesis: requiring
   more maskable variables per function gives the model more context per item
   relative to the type-heuristic. Measured result: the dataset shrank from
   3,143 to 1,530 items, the type-heuristic *did* drop to 0.0856, but the
   trained reference collapsed to 0.0858 — a margin of 0.0002, worse than
   before. The smaller training set hurt the trained model far more than it
   hurt the heuristic (which needs very little data to be strong — it is
   just a frequency count). Reverted to `min_vars=2`.
4. **Growing the raw corpus within the standard library.** Added
   `library/coretests` and `library/alloctests` — the real, hand-written
   test-code counterparts of `core` and `alloc` — growing the usable item
   count from 3,143 to 4,374 (+39%). Retrained an attention-pooling
   architecture (learned softmax attention over all occurrence positions,
   replacing the fixed first/last/mean split) for 15 epochs: 0.2135 against
   that split's 0.1257 type-signature heuristic, a margin of 0.088. Real and
   reproducible, but with the reference sitting at the very bottom of the
   0.2-0.6 target band for agent runs.
5. **Growing the raw corpus to the compiler and rust-analyzer (shipped).**
   Attempt 4 showed +39% data was worth +0.04 of margin; the same repository
   held far more hand-written Rust at the same commit. Adding `compiler/`
   and `src/tools/rust-analyzer/crates/` (with the stricter tokenizer
   pipeline described under "Masking correctness") grew the usable item
   count from 4,374 to 13,331 (3.0x). The type-signature heuristic rose to
   0.256 on this corpus (see above); the reference, retrained for 20 epochs on
   the new split, scores **{{REF}}** — a margin of **{{MARGIN}}** over the
   strongest adversary, the number shipped in `config.yaml`. One
   implementation change came with the corpus growth: the encoder is a
   dilated character CNN (five residual 1-D convolution layers, receptive
   field ~125 characters around every position) instead of the 2-layer
   bi-GRU of attempts 2-4, because a 600-step recurrent encoder could not
   train on 3x the data in reasonable time on the build machine. The
   learned attention over every occurrence position of a placeholder and
   the character-level GRU decoder are unchanged. A CNN sees less
   long-range context than the recurrent encoder, so if anything this
   under-states what the earlier architecture would reach on this data.

**What this history establishes:** three independent architectures (mean
pool, first/last/mean pool, attention pool) converge to within a few points
of each other on any *given* split, which rules out "wrong architecture" as
the explanation for a thin margin — the lever that moved the number, twice,
was training-data volume. The shipped margin is reported with the full
attempt history so a reviewer can see the ceiling was investigated, not
assumed.

## Leakage and shortcut closure (measured)

- Target names absent from `test.csv`; see Masking correctness above for the
  three independently measured and fixed leaks (usage-site leakage,
  comment-name leakage, cross-row skeleton duplication), the extraction
  change that keeps quoted fixture snippets out, and "The split" above for
  the fourth leak (same-file sibling overlap).
- Train/test disjoint on exact masked-text skeleton after deduplication AND
  on source file (every function from one `.rs` file sits entirely on one
  side); the train-skeleton lookup baseline measures 0.010 and the per-file
  most-common-name baseline 0.013 on the real shipped `test.csv`/`answers.csv`
  via the real `grade.py` (not an estimate) — the per-file baseline is
  identical to the file-blind global baseline, confirming no test file has a
  sibling function in `train.csv` to exploit.
- Each row is predicted from its own masked function text alone; no cross-row
  statistic is required by the intended solution, and the dataset's own
  deduplication and file-level split together remove both cross-row
  shortcuts that were found and measured (repeated boilerplate skeletons,
  and same-file naming idiolect).
- What this closure cannot do, stated plainly: the source is real, public
  code. A solver with network access could look up any masked function's
  real names directly from GitHub by its remaining (unmasked) tokens. The
  Rules section states this is invalid regardless of score; the solving
  environment's lack of network access is what actually prevents it
  mechanically, the same residual-risk pattern this project's own Vietnamese
  tone-mark-recovery and eCFR amendment-inversion challenges already carry
  for their own real, public corpora.
