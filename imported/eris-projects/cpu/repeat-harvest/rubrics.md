# Rubrics — Per-Base Annotation Depth Prediction from DNA

Task-specific criteria for good engineering on this challenge. Weighed alongside
the AGREEMENT score.

## Rubric 1 — REQUIRED

**Locates the stretch by searching, not by guessing a position.** The solution
finds the flagged stretch by looking for a short unit copied back to back —
scoring candidate unit lengths and finding the run where copies hold — rather
than writing a fixed block. The strongest label-free constant block, searched
over every start, end and height, scores 0.2715 against the reference's 0.6574.
Because the surrounding DNA is authentic, the solution must tell the true
stretch apart from mobile-element copies and single-letter runs.

## Rubric 2 — REQUIRED

**Predicts height, not merely presence.** The solution treats its output as an
integer curve and writes values above 1 where the program's verdicts pile up.
Over a third of the available value sits at height 3 or above, so a flat `1` over
a perfectly placed stretch reaches only 0.4644 and a flat `2` only 0.6424. Going
further means predicting how many readings coincide at each letter, learned from
the worked examples rather than assumed.

## Rubric 3 — REQUIRED

**Puts the rise and fall in the right places.** Judging charges letter by letter
with no cutoff, so each edge is paid for on its own. The solution sets its
transitions where the program's verdicts actually begin and end — running on
while copies hold, stopping where they break — instead of padding a generous
block around a hit. This is worth 0.184: knowing the exact span scores 0.8410
with the very same flat `2` shape that scores 0.6424 when the span is merely
searched for.

## Rubric 4 — REQUIRED

**Keeps its predicted mass where it belongs.** Because credit at each letter is
capped at the true height, every unit written outside the stretch is a straight
loss. The solution does not inflate its curve as a hedge: a constant `1` scores
0.2182, a constant `2` 0.2162 and a constant `9` 0.0658, all far below one
well-placed stretch.

## Rubric 5 — REQUIRED

**Produces a valid file by construction.** Every line carries exactly `length`
digits (or a well-formed compact form), one line per graded `seq_id`, none
duplicated. The judging code pads, truncates and skips forgivingly, but a
solution should not lean on that forgiveness.

## Rubric 6 — RECOMMENDED

**Runs deterministically and cheaply.** The pipeline is seeded and reproducible,
uses only the processor and standard scientific Python (numpy suffices), and
finishes well inside the time budget — the worked reference takes about four
seconds. No identifier-to-answer tables and no attempt to rebuild withheld truth
from anything except the released public files.
