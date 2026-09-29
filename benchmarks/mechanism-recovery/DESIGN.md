# Design notes — What Does It Do In There?

## The task in one line

Given a drug label's Clinical Pharmacology section with all mechanism
vocabulary masked, plus a pool of twelve candidate mechanism-of-action
classes, select the ones the FDA actually assigned.

## Why this source, and what was rejected first

Three sources were gate-checked and dropped before openFDA, each for a reason
that only surfaced by probing the API rather than reading the docs:

- **Crossref** (order citing passages of retracted papers by hedging drift).
  186M works and structured `update-to` retraction relationships, but **no
  citing-passage text** — the idea needed exactly that. Retraction "lag" is
  also degenerate: median 1 day, because Crossref records the notice's own
  date, not the retracted paper's. And 72% of retraction-notice abstracts
  state no extractable reason, so any reason label would have been my own
  regex over free text — a Gate 3 violation.
- **NYC 311 service requests.** Real complaint text with structured agency
  routing, but NYC Open Data publishes a Terms of Use, not a named licence.
  Gate 1 reject.
- **GBIF occurrence records.** Explicit per-record CC0/CC BY, but the text
  fields are sparse and tiny — `habitat` median 85 characters. It is a
  photo-and-coordinate archive, not a text corpus.
- **FDA recall enforcement** (recover the states a recall shipped to). Real
  set-selection shape, but content-free probes hit **0.33–0.35** just by
  naming PA/CA/IL/NY/OH. Too high a floor to be worth building on.

openFDA drug labels passed everything: CC0 1.0 quoted verbatim from
open.fda.gov/license, a reachable endpoint, FDA's own structured class
assignment as the label, and 8,767-character pharmacology sections.

### A licence subtlety worth recording

openFDA's CC0 dedication says *"unless otherwise noted"*, and the carve-out is
real: device records carry GMDN Agency nomenclature under a separate licence.
The build therefore uses **only** `drug/label.json` and no device endpoint.
This is documented in `config.yaml`'s `license_note` rather than left implicit.

## The leak, measured before anything was built

**57% of labels state their own mechanism verbatim** in the pharmacology
prose. A label classed *Cyclooxygenase Inhibitors [MoA]* routinely contains
the phrase "cyclooxygenase inhibitor". Left alone, a keyword matcher scores
well while learning nothing.

The fix is masking, with one design decision that matters more than it looks:
**the mask vocabulary is built corpus-wide, from every class name, never from
the row's own class.** If only a row's own class words were masked, *which*
words are masked would itself reveal the answer — the masking pattern becomes
a perfect leak.

Measured on a 1,401-label sample:

| | value |
|---|---|
| label-class pairs keyword-recoverable before masking | 100% of the 57% |
| pairs blocked after masking | **1,860 of 2,082** |
| pairs still recoverable | 222 |
| surviving text, median | ~7,300 characters |

The unmasked text is never written to disk — masking happens inside the
fetcher, so the raw archive cannot leak it.

## Metric: chance-corrected set F1

Raw set F1 was rejected on measurement. The MoA class distribution is heavily
skewed, and on a 1,001-label sample **always naming the single most common
class scored 0.182 raw F1** while reading nothing at all. That is a large part
of the range spent on knowing which classes are common.

The shipped metric subtracts the expected F1 of a same-size random draw from
the row's own pool and renormalises. Two consequences, both verified on
hand-built cases before any data was fetched:

- Frequency strategies collapse toward 0.
- **Set size becomes an honest decision.** Naming more raises recall and the
  chance correction together, so padding buys nothing — submitting all twelve
  candidates scores exactly **0.000**, where under raw F1 it would score well.

This is the same failure mode that broke the first metric on a sibling
challenge in this portfolio, where a vocabulary dump scored 1.000 against the
oracle's 1.000. Building the correction in from the start was a direct
consequence of that.

## Pool construction, built to defeat two shortcuts

- **Frequency.** Distractors are sampled with probability proportional to
  train-side true-class frequency, so a common class is a common distractor.
  Without this, "pick the most frequent candidate" is a real strategy.
- **Position.** Pool order is shuffled with a per-row seed derived from the
  row id. `dataset_stats.json` ships the first-true-class position histogram
  so the claim is checkable.
- **Size.** Fixed at 12 so pool size cannot signal how many are true.

## Split: by active substance

Many labels are the same molecule from different manufacturers, with
near-identical prose and identical classes. A row-wise split would put a
generic and its brand-name twin on opposite sides and leak the answer
outright. Every label sharing a substance signature lands wholly in one split.

Exact-duplicate masked texts are collapsed first, then near-duplicates by a
word-5-gram shingle signature. Of 3,091 fetched labels, **946 were exact
duplicates and 468 near-duplicates** — nearly half the corpus. That is the
single largest filter in the build and it exists entirely because of generic
re-labelling.

### A ratio bug the re-balancing exists to fix

The class filter — a class must appear in *both* splits, or a test row could
require a class no training example demonstrates — runs after the split and
removes rows unevenly. Measured: a 24.2% split-time ratio became **33.7%**
final, outside the 15–25% band.

Fixed by releasing whole test substances back to train, largest first, until
the ratio is in band. The move is one-directional (test → train only), so
substance-disjointness is preserved by construction. Final ratio: **22.2%**.

## Measured ladder

On the shipped split (744 train / 165 test, 29 classes, pool size 12):

| Baseline | Score |
|---|---|
| Blank submission | 0.000 |
| Select all twelve candidates | **0.000** |
| Random 1 from pool | 0.078 |
| Position 1 only | 0.079 |
| Random 2 from pool | 0.092 |
| Most-frequent-in-pool, k=3 | 0.107 |
| Most-frequent-in-pool, k=1 | 0.112 |
| Sample submission (first candidate) | 0.113 |
| Position 0 only | 0.113 |
| Text-length-only heuristic | 0.122 |
| Most-frequent-in-pool, k=2 | 0.122 |
| Random, but with the correct set size | 0.126 |
| ~~Pair scorer (rejected reference)~~ | ~~0.145~~ |
| **Reference** — one-vs-rest + pool intersection | **0.393** |
| *Diagnostic: name one guaranteed-correct class* | *0.674* |
| *Diagnostic: name two guaranteed-correct classes* | *0.815* |
| Oracle | 1.000 |

Every content-free probe sits in a tight 0.078–0.122 band, and the two
degenerate submissions score exactly 0 — the chance correction is doing its
job.

### The reference was wrong twice, and the ladder caught it both times

**First attempt — pair scorer, threshold only: 0.143.** One shared model
scoring each (text, candidate) pair, with a probability threshold. It selected
3.24 classes per row against a median truth of 1, wrecking precision. Against
a 0.122 content-free probe that is a 0.02 margin — not skill.

**Second attempt — pair scorer, threshold + cap: 0.145.** Tuning a cap on how
many may be named alongside the threshold fixed the over-selection (2.35 per
row) and moved the score by 0.002. That was the signal that the architecture,
not the tuning, was the problem.

**Third — one-vs-rest over the class vocabulary, intersected with the row's
pool: 0.393.** A 2.7× improvement and a 0.27 margin over the strongest probe.

The reason is capacity. A shared pair model has to learn one global notion of
"does this text support this class". One-vs-rest gives each of the 29 classes
its own decision boundary over the vocabulary, and with a few hundred examples
each those are learnable where the shared one is not.

Confirmation that the task, not the data, was fine: naming a single
guaranteed-correct class scores **0.674**. The headroom was always there.

### A second, subtler leak in the reference's own tuning

The reference first tuned `k` on a row-wise slice of train. That is the same
mistake the shipped split exists to prevent: generic drugs appear many times
with near-identical prose, so a row-wise slice puts the same molecule on both
sides. It reported **val 0.868** against a true test score of 0.393, and that
inflated gap picked k=1 when k=2 is better on the real split.

Switching the inner split to group by the drug's own text signature narrowed
the reported gap to 0.766. It is still optimistic — the 400-character
signature does not group tightly enough to fully separate reformulations — and
that residual is recorded here rather than papered over. The shipped reference
uses the grouped split.

## Open items

- Reference score on the shipped split is being measured; `config.yaml`
  carries `TBD` until it lands.
- 222 of 2,082 sampled label-class pairs remain keyword-recoverable after
  masking. Worth a look at which class names survive and whether the exempt
  stopword list is letting a distinctive word through.
- The corpus is capped at 3,091 fetched labels against ~11,000 available.
  More data would raise per-class support and let `MIN_CLASS_SUPPORT` rise
  above 20, which would likely sharpen the reference.

## Revision after review (2026-09-27)

Reviewer findings, each measured and fixed:

| Finding | Measured | Fix |
|---|---|---|
| Solver-facing text names the data source | `problem-description.md`, `rubrics.md` and the notebook named openFDA/FDA and a `/kaggle/input/fda-moa` path | source names and the path removed from every solver-facing file |
| Split grouped exact ingredient combinations | 30 of 165 test rows shared an ingredient with train | ingredients reduced to moieties, union-find over shared ingredients, whole groups held out; asserted in `prepare.py` |
| Test-label statistics in public files | `dataset_stats.json` shipped true-classes-per-test-row and the test position histogram | both removed; the stats file now holds split counts and build drops only |
| Compute mismatch | challenge runs on CPU, description promised an A10G | description and config say CPU; the GPU transformer section (which also used a pretrained model the rules forbid) is removed from the notebook |
| "Chance correction sends random guessing to ~0" | random single guess 0.085, random pair 0.113 (clipping at 0) | claim corrected everywhere, with the measured numbers |

New split: 505 train / 125 test, 22 classes. Reference 0.462 (CPU, ~15 s);
strongest content-free probe 0.193. The tables above this section were
measured on the earlier substance-combination split and are kept as history;
current numbers are in `config.yaml`.

