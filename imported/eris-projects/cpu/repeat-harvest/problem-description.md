# Per-Base Annotation Depth Prediction from DNA

## What you build

A function from one string to another string of equal length. Feed it DNA over `A, C, G, T`; it returns digits. The digit standing at each position is your estimate of **how many times a reference software program flagged that position**.

Usually that count is zero. Sometimes it is four.

## Why the count varies

The program in question — Benson, *Nucleic Acids Research* 27(2):573–580, 1999, the standard published tool for this job — is chronically indecisive. Point it at a stretch of DNA where a short unit is copied back to back, and it does not emit one verdict. It emits a pile of them, each a different reading of the same physical stretch. One reading takes the copied unit to be two letters long; another takes it to be the four-letter unit built from two of those. One counts eight copies before quitting; another pushes to twelve. One clips to the pristine middle; another tolerates the ragged tail.

Lay all those verdicts on top of one another and count them position by position. The result climbs and falls: zero across ordinary DNA, one where only the boldest verdict stretches, two as verdicts begin agreeing, three or four across the span every verdict claims. That rising-and-falling integer curve is the object you predict.

Read it as a legibility map. Where the count runs high, the program saw something unambiguous. Where it tapers to one, the program was reaching. The number is the program's own hesitation, written down.

Prior work on copied-unit DNA collapses that pile into a single verdict and throws the disagreement away. This benchmark keeps the disagreement and asks you to forecast its exact shape from letters alone.

## Why a perfect mask is not enough

The nearest familiar shape to this task is marking, for every position, whether it is inside an object — and that shape is not sufficient here, by a measured margin. A flawless binary mask, covering exactly the right span with a `1` at every position inside it and nothing outside, scores **0.6355** — *below* the 0.6574 of the worked reference, which places its span less accurately but predicts the height. Solving the marking problem perfectly does not solve this one.

Two things cause that. The value at a position is a **count of independent verdicts**, not a membership flag: over a third of what is on offer sits at counts of 3 or more, and a mask cannot express it. And there is no single true boundary to recover in the first place — the program never committed to one, and the spread between its verdicts is the signal rather than noise to be cleaned up. Any method that resolves the fragment into one tidy span has thrown away the quantity being scored before it starts.

## The output

Given a fragment of length `L`, emit `L` digits:

```
DNA      ...ACGTTGCA AAAGAAAGAAAGAAAGAAAG TCCAGT...
digits   ...00000000 11223333333333322111 000000...
```

No boundary to nominate, no region to draw, no probability to threshold. One integer per letter. Every belief you hold is visible in the output; nothing hides behind a formatting choice.

## How it is judged

Call the program's own curve `t_true` and yours `t_pred`. The judgement is their soft agreement, accumulated across the whole graded collection:

```
AGREEMENT = 2 * Σ_s Σ_i min( t_true[i], t_pred[i] )
            / Σ_s ( Σ_i t_true[i] + Σ_i t_pred[i] )
```

One continuous number between 0 and 1; bigger is better. Four consequences fall straight out of the algebra:

- **Nothing is all-or-nothing.** No overlap test, no cutoff, no pass mark. Draw a curve six letters too wide and you forfeit six letters' worth of credit — not the position. An improvement of any size shows up in the number.
- **The pressure is symmetric and exact.** Pushing a digit up while it sits under the truth always gains. Pushing it up anywhere else always costs. Inflation punishes itself with no special rule: digit `1` everywhere earns 0.2182, digit `2` everywhere 0.2162, digit `9` everywhere 0.0658.
- **The height carries most of the value.** Of everything on offer, 11.7% lies at count 1, **39.6% at count 2, 35.3% at count 3 and 12.7% at count 4**. Find every flagged stretch flawlessly, draw it flat at 1, and most of the prize goes unclaimed.
- **Perfection exists.** 1.0 means every curve exactly reproduced. There is no cap below 1 and no rescaling to apply — whatever you earn is already the fraction of what was available.

## The material

Authentic human chromosomal DNA, carrying authentic output from an actual run of the program. None of it is fabricated, and the letters betray their origin: mobile-element copies, single-letter runs, real base-composition skew, and `CG` pairs turning up at **0.29** of the rate the letter frequencies alone predict — a hallmark of vertebrate DNA, where invented `ACGT` scores 1.00. Whatever succeeds here succeeds because it reads DNA, not because it reverse-engineered a simulator.

Four properties of how the fragments were cut, each ruling something out:

- **A pile is always present, never a lone verdict.** The shallowest pile is 2 deep, the deepest 11. Not one fragment has a flat curve of 1 as its answer.
- **The flagged stretch is a sliver of its fragment.** It spans at most a fifth of the letters (typically a seventh), buried under surrounding DNA 4 to 8 times its own length, and it starts anywhere between 5% and 82% of the way along. Guessing a position cannot replace searching for one.
- **Nothing flaggable is left unlabelled.** A fragment was published only when every verdict falling inside it belonged to that fragment's own stretch. Spot a genuine copied-unit run here and it is in the key — being right is never penalised.
- **The letter mix is authentic, and the choice of fragments shows in it.** `G`+`C` sits near 0.45 where the chromosomes overall sit near 0.41. That lift comes from choosing what to cut, not from manufacturing: fragments are cut around short-unit copy runs, the two- and three-letter units dominating the program's output carry `G` and `C`, so choosing flagged material chooses slightly `GC`-heavier neighbourhoods.

What is released:

- `sequences_train.csv` — 1,400 worked examples: `seq_id`, `length`, `sequence`, `true_intervals` (pipe-separated `start-end`, end-exclusive — the program's entire pile of verdicts, which you stack to obtain the true curve), `n_true`.
- `sequences_test.csv` — 300 graded fragments: `seq_id`, `length`, `sequence`.
- `sample_submission.csv` — a legal, feeble baseline in the compact form described below.

The worked examples outnumber the graded fragments 4.7 to 1 and withhold nothing, so the curve's behaviour is available to be studied rather than assumed.

A note on the packaged data card: it documents these same fragments under an earlier judging rule that graded one nominated region per fragment. The task and the judgement that apply here are the ones stated above and implemented in `grade.py`.

## File you submit

`submission.csv`, one line per fragment in `sequences_test.csv`:

```
seq_id,depth_track
seq_0002,00000112223333222111000
seq_0004,00000000000000000000000
```

- `seq_id` — required, never duplicated.
- `depth_track` — exactly `length` digits, one per letter. Counts saturate at 9.

**Compact form, equally accepted.** A column called `intervals`, holding pipe-separated half-open `start-end` ranges, is read as the curve you get by *piling them up* — where three ranges cover a letter, the curve reads 3 there. It expresses the identical object whenever your curve is flat in steps, and it is what the released `sample_submission.csv` uses. Both forms run through the same code and produce the same number; pick whichever your method emits naturally.

Handling of imperfect files — judging never aborts:

- Too few digits are padded with zeros; too many are cut off.
- An unreadable line, a broken or off-the-end range, or an absent `seq_id` earns nothing for that fragment, while that fragment's share of the denominator still counts against you.
- The compact form reads at most 16 ranges per line — a safety stop, not an allowance, since piling on already loses.
- Unrecognised `seq_id`s are ignored. Duplicated `seq_id`s, and a file offering neither `depth_track` nor `intervals`, are refused.

## Where the difficulty lives

Each figure below was obtained on this graded collection by running the released judging code:

- nothing submitted, or all zeros — **0.0000**
- the released sample line — **0.0964**
- digit `1` across every letter of every fragment — **0.2182**
- the strongest label-free constant block, searched over every start, end and height on a 1% grid — **0.2715**
- a crisp unsupervised search drawing a flat `1` over the stretch it finds — **0.4644**
- a flawless binary mask — the exact span, every position inside it `1` — **0.6355**
- that same search drawing a flat `2` — **0.6424**
- the **worked reference**, which learns the curve's profile from the training examples — **0.6574**
- knowing the exact stretch, but drawing it flat at `2` — **0.8410**
- the program's own curve — **1.0000**

The staircase says where the work is. Climbing 0.27 to 0.46 is **finding the thing**. Climbing 0.46 to 0.64 is merely noticing that heights above 1 exist. From 0.64 to 0.84 is **edge accuracy** — the reference is held back by where its curve rises and falls, not by its concept, and the program's own edges are ragged enough that catching them takes calibration against the program rather than a keener search for copied units. The last 0.16 is **contour**: exact placement drawn flat at `2` still forfeits it, because over a third of the value sits at height 3 or above and the curve has to climb and descend in the right places.

The reference closes neither of the last two, and no change of output format reaches them. Both demand a model of what the program itself does.

## Conditions

- Processor only: 10 cores, 62 GB memory, 1.5 hours from start to finish. The worked reference finishes in roughly four seconds on numpy alone.
- Packages limited to those in the Kaggle Python Docker image (numpy, pandas, scipy, scikit-learn and the like).
- Network unavailable; nothing beyond the released files.
- No language-model-authored outputs.
- No identifier-to-answer tables, and no attempt to rebuild withheld truth from anything except the released public files.
- Your notebook must execute start to finish — ingest the fragments, predict a curve for each, emit `submission.csv` — with fixed seeds and reproducible results.
