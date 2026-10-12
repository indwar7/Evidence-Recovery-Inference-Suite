# Repeat-Pattern Span Detection in Genomic Windows

## Overview

Genetic sequences are dotted with short patterns that repeat back-to-back — a
motif copied several times in a row, forming a compact "stutter" inside
otherwise ordinary DNA. These repeated-pattern stretches matter: they drive
genome variability, mark unstable regions, and flag sites tied to real
biology. The catch is that such a stretch hides inside a much longer window of
non-repeating sequence, and it looks superficially like other locally busy
stretches that are not truly periodic.

This benchmark asks a solution to pinpoint the one repeated-pattern stretch
inside a window of real human DNA and report exactly where it starts and ends.

Each input is a DNA window from the GRCh38 (hg38) human reference assembly.
Somewhere inside it sits a single stretch where a short motif repeats
head-to-tail. The stretch takes up only a small slice of the window — the rest
is genuine surrounding sequence at an unpredictable offset — so guessing a
fixed spot never works. A solution has to actually recognise the periodic
"beat" of the motif and lock onto its edges.

The task is a span-detection problem:

```
Input DNA window
        ↓
Spot the repeating "beat" (the motif copied in a row)
        ↓
Read off the exact stretch of bases
        ↓
Output that stretch as a DNA subsequence
```

Inputs use the alphabet `A`, `C`, `G`, `T`. The training split shows each
stretch; the evaluation split hides it.

## Dataset

The dataset holds 1,700 DNA windows sampled from 60 regions across every
autosome and chrX of GRCh38. Each window is self-contained: it carries exactly
one repeated-pattern stretch set in real surrounding sequence, with no stray
stretch from a neighbouring site leaking in.

| Split | Number of windows | Purpose |
|---|---|---|
| Training | 1,400 | Learn what makes a repeated-pattern stretch stand out from its surroundings |
| Test | 300 | Score edge-finding on unseen windows |

The build produces these public files:

| File | Purpose | Stretch shown |
|---|---|---|
| `sequences_train.csv` | Training examples | Yes |
| `sequences_test.csv` | Evaluation inputs | No |
| `sample_submission.csv` | Submission template | No |

The underlying source file is `sequences.csv`, with these fields:

| Column | Type | Description |
|---|---|---|
| `seq_id` | string | Unique window identifier |
| `split` | string | `train` or `test` assignment |
| `length` | integer | Length of the window in bases |
| `sequence` | string | Real hg38 DNA window over `A`, `C`, `G`, `T` |
| `true_intervals` | string | Pipe-separated spans `start-end`, half-open `[start, end)` |
| `n_true` | integer | Number of annotated spans at the site |

A single stretch is often labelled by several overlapping spans — the same run
of copies framed at different motif lengths (a 2-base beat versus a 4-base
beat, eight copies versus twelve). These overlapping frames describe one
stretch, not several.

**The public tables carry no numeric column.** `sequences_train.csv` and
`sequences_test.csv` ship `seq_id` (an opaque string key), `sequence`, and —
on train only — `target`. The raw corpus's `length` field is not republished:
it is exactly `len(sequence)`, so it tells a solver nothing it cannot compute
in one call, while its natural long tail (most windows in a narrow band, a
minority much longer) would put more than 5% of rows outside a 1.5×-IQR fence
and read as an extreme-outlier-heavy column. Compute window length from
`sequence` directly if you need it.

## Sequence Representation

Each window is DNA over four bases.

| Symbol | Meaning |
|---|---|
| `A` | Adenine |
| `C` | Cytosine |
| `G` | Guanine |
| `T` | Thymine |

Positions are zero-based; for a window of length `L`, valid positions run
`0 … L-1`. Spans are half-open `[start, end)`; for example `138-177` covers
positions 138 through 176.

## The Target Stretch

The target is the dominant repeated-pattern stretch in the window. It is hard
to bound because:

- the surrounding sequence can share the same base mix;
- other locally busy or low-variety stretches mimic a real "beat";
- the motif and how many times it repeats are not given at test time; and
- the edges are exact — a span twice the true width already misses badly under
  the score below.

Participants do not reproduce any internal label format. The stretch is handed
to training participants directly, as a span.

### Which Span Counts (Canonical Interval)

Because one stretch carries several overlapping frames, one canonical span is
chosen per window, deterministically:

| Priority | Rule |
|---|---|
| 1 | Take the widest annotated span (the fullest frame of the run) |
| 2 | If tied, take the earliest start |
| 3 | If still tied, take the earliest end |

The training answer is that canonical `[start, end)`. For test windows it
stays private.

## Training Data — `sequences_train.csv`

| Column | Type | Description |
|---|---|---|
| `seq_id` | string | Unique window identifier |
| `sequence` | string | Complete hg38 DNA window |
| `target` | string | The dominant stretch, given as a substring |

Training pairs map directly: a full DNA window to the DNA subsequence of its
dominant repeated-pattern stretch. The `target` column is exactly that
subsequence — the same kind of output a solution submits for test windows (see
Output Format below).

## Test Data — `sequences_test.csv`

| Column | Type | Description |
|---|---|---|
| `seq_id` | string | Unique window identifier |
| `sequence` | string | Complete hg38 DNA window |

The stretch is hidden for every test window; a solution must work out which
slice of the window is the repeated-pattern run.

## Task

For every test window, output the DNA subsequence covering the dominant
repeated-pattern stretch. Each prediction must:

- contain only `A`, `C`, `G`, and `T`;
- occur as a contiguous substring of the corresponding window;
- preserve the original base order; and
- cover one stretch.

The output is a DNA subsequence — not a motif, a repeat count, or a coordinate
pair.

## Output Format

Submit a file named `submission.csv` with exactly two columns.

| Column | Type | Description |
|---|---|---|
| `seq_id` | string | Test window identifier |
| `prediction` | string | The extracted stretch, as a DNA subsequence |

Example:

```
seq_id,prediction
seq_0001,AAAGAAAGAAAGAAAG
seq_0002,ACACACACACAC
seq_0003,GTTGTTGTTGTT
```

Each test window should appear exactly once.

| Requirement | Rule |
|---|---|
| Alphabet | Only `A`, `C`, `G`, and `T` |
| Contiguity | Prediction must occur as a substring of the window |
| Base order | Original order must be preserved |
| Empty prediction | Allowed, but scores zero |
| Duplicate `seq_id` | Not permitted |
| Unknown `seq_id` | Ignored |

A prediction that does not occur as a contiguous substring of its window
scores zero for that window.

## Evaluation

The held-out answer for each test window is `target`: the DNA substring of the
canonical stretch — the same object, in the same form, as the `prediction` a
solution submits. Submitting exactly that substring scores 1.0.

The submitted subsequence is located inside its window. If it occurs more than
once, every occurrence is considered and the one that best matches the hidden
stretch is used — so a correct answer is never penalised just because an
identical short motif recurs elsewhere in the same window.

### Edge Agreement (Overlap Ratio)

The best occurrence of the prediction and the hidden stretch are compared by
overlap ratio (intersection over union):

```
score = overlap length / combined length
```

The overlap is the count of positions the two spans share; the combined length
is the count of positions either one covers. The score runs continuously from
0 to 1.

| Score | Meaning |
|---|---|
| 0.00 | No overlap |
| 0.25 | Roughly in the right place, edges poor |
| 0.50 | Located, edges loose |
| 0.75 | Tight edges |
| 1.00 | Exact match |

### Why overlap ratio, and why not period-correctness

The thing being measured is **where the repeat run begins and ends**, so the
metric scores exactly that and nothing else. Three alternatives were
considered and rejected against this task's own data:

**Exact string match (1/0).** A prediction that finds the right run but stops
one copy short is, biologically, almost the whole answer; under exact match it
is worth the same as a prediction that lands in unrelated sequence. That
collapses every partial result into a single bucket and throws away the signal
that separates a solver which detects the beat from one which does not. The
overlap ratio keeps them apart: same-place-loose-edges scores ~0.5,
wrong-place scores 0.

**Requiring the motif period to be correct.** This is the natural-sounding
alternative, and it is wrong for this task for two reasons, both visible in
the corpus. First, **the period is not uniquely defined.** The same run of
copies is genuinely describable at several periods at once — an `ACACAC…`
stretch is a 2-base beat repeated six times and equally a 4-base beat repeated
three times, and the raw annotation records exactly these competing frames as
multiple overlapping spans per site (see *Which Span Counts*). Scoring period
correctness would force one arbitrary frame to be "the" answer and mark the
others wrong, when both describe the identical stretch of DNA. Second,
**period correctness does not imply the run is bounded correctly.** A solution
can name the right motif and still return a span twice the true width, or one
that starts fifty bases early — the period is right and the answer is still
useless for the question being asked, which is where the run sits. Conversely
a solution can bound the run perfectly while internally modelling it at a
different-but-equivalent period, and it would be penalised for being right.
Period is an intermediate quantity here, not the deliverable; the deliverable
is the extent.

**Edges, and why credit is proportional to edge tightness.** Once the target
is an extent, the useful question is how much of it a prediction actually
covers and how much slop it drags in. Intersection over union is the standard
answer because it is the one measure that penalises both errors at once and
symmetrically: it falls when the prediction misses part of the true run
(intersection shrinks) and equally when it overreaches into surrounding
sequence (union grows). A metric built only on overlap would be trivially
gamed by submitting the entire window — every true stretch would be fully
covered — which is why the union term is not optional. Because it is
continuous, it also gives the precise gradient this task needs: tightening one
edge by a few bases raises the score by a few points, so a solver is rewarded
in proportion to how well it localises, not handed a cliff at some arbitrary
tolerance threshold. That is what "partial credit proportional to edge
tightness" buys — a score that moves smoothly with the actual quality of the
boundary call.

### Window Score and Final Score

For each test window, window score = overlap ratio between the best occurrence
of the prediction and the hidden stretch. The metric is smooth — tighter start
and end edges score higher.

The final score is the mean window score over all test windows:

```
Final Score = sum of window scores / number of test windows
```

An exact hit scores 1.0; the scale is a true `[0, 1]` with no cap below 1.0.

### Example

A window `TTGCAAAGAAAGAAAGAAAGCCTTAC` with the hidden stretch
`AAAGAAAGAAAGAAAG`:

- Output `AAAGAAAGAAAGAAAG` recovers the full stretch and scores 1.0.
- Output `AAAGAAAG` covers half the stretch and scores a lower overlap ratio.
- Output `AAAAAAA` does not occur as a contiguous substring and scores 0.

## Baseline

A fixed-spot, fixed-width guess (the same relative slice of every window)
gives an answer-blind reference. Because stretches sit at unpredictable
offsets and cover only a small slice of each window, this guess scores low —
proof that a real solution must detect the repeating "beat", not guess a
location.

## Constraints

| Constraint | Requirement |
|---|---|
| Input alphabet | `A`, `C`, `G`, `T` |
| Output type | Contiguous DNA subsequence |
| Predictions | One subsequence per test window |
| External data | Not required |
| Evaluation | Hidden stretches |
| Reproducibility | Solutions should be deterministic |

The benchmark runs with standard CPU-based sequence-processing and
machine-learning techniques.

## Objective

Learn a mapping from a full hg38 DNA window to the subsequence of its dominant
repeated-pattern stretch: catch the periodic beat, separate it from
busy-but-random surroundings, lock down tight edges, and generalise to unseen
windows. The final score reflects how closely the extracted subsequence
matches the hidden stretch.
