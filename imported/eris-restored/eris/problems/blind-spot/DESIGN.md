# Blind Spot — Design Record

How the challenge was arrived at, including the two designs that were measured
and rejected. Kept because the rejections are the evidence that the final
difficulty is real rather than asserted.

## The capability being measured

Given a black-box image classifier and a catalog of ordinary, mild image
operations, predict **which operations land in a given image's blind spot** —
and specifically, which ones push the model toward a *nominated* wrong class.

This is not adversarial robustness evaluation (ImageNet-C, RobustBench), which
applies a fixed corruption set and measures the model. Here the corruption
choice is the submission and the model is hidden. It is not gradient-based
attack synthesis either: there is no gradient access, no pixel optimisation, and
only 400 legal moves per row. What has to be learned is a surrogate of one
specific model's decision geometry, accurate enough to rank 400 discrete
interventions per image.

## The failure mode this design was built against

Two prior challenges in this repo died on the same gate: a trivial heuristic
scored near the oracle (`pipeline-forensics` hit 1.0 twice; `storm-genealogy`'s
naive tracker hit 0.91–1.00). So the constant-answer gate was run **before any
specification prose was written**, on real recorded behaviour.

### Attempt 1 — untargeted damage, unaugmented hidden model — REJECTED

Task: pick the pair that most reduces `p_H(true class)`.

```
damage range per image   mean 0.998   d_max mean 1.000   d_min mean 0.002
best constant pair       0.8823   (equalize -> chanswap)
best class-conditional   1.0000   <-- fatal
```

Two defects. The transform catalog was far too strong — `d_max` was 1.000 on
essentially every image, so the damage surface was saturated and carried no
image-specific structure. And a per-class lookup table of ten entries scored a
**perfect 1.0**: predict the land-cover class (easy, ~95%) and read off the
answer.

Root cause: `H` was trained without augmentation, making it *blanket-brittle*.
Blanket brittleness is a property of the class, not of the image.

### Attempt 2 — robust hidden model, strength-calibrated catalog — REJECTED

Fixes applied: retrain `H` with strong augmentation (95.85% clean accuracy);
retune every operation so that, applied alone, it flips `H` on ~8% of a held-out
calibration split (measured spread 0.071–0.087, so no operation dominates).

```
                        untargeted, calibrated
random pair                  0.2216
best constant (train-fit)    0.4892
class-conditional (train)    0.9073   <-- still fatal
kNN on generic features      0.7225
oracle                       1.0000
```

Better — the best constant fell from 0.88 to 0.49 — but the class lookup still
scored 0.907, leaving only 0.09 of headroom, and a generic-feature kNN could not
beat it. Damage magnitude is simply more class-determined than image-determined.

A per-op strength probe confirmed the mechanism: restricting to the 15 operations
whose solo break rates were already tightly clustered moved the best constant
from 0.72 to 0.49, which is why calibration (rather than dropping operations)
became the fix.

### Attempt 3 — targeted steering — ACCEPTED

The change: a row is `(image, target_class)`, and the objective is
`p_H(target)`, not `1 - p_H(true)`. The best pair now depends on where *this
image* sits relative to *that specific boundary* — a genuinely image-level
quantity.

```
                          untargeted    targeted
random pair                 0.2216       0.0774
best constant (train-fit)   0.4892       0.1672
target-only lookup            —          0.4794
(class,target) lookup       0.9073       0.6890
oracle                      1.0000       1.0000

distinct winning pairs        84/400      221/400
top single pair covers        13.0%         7.7%
within-group argmax agreement   —          36% mean / 30% median
```

The strongest cheap heuristic drops to 0.689 with 0.31 of headroom, the winning
pair spreads across 221 of 400 options, and within a `(class, target)` group the
modal answer is right only about a third of the time. The answer is image-specific.

**No metric contortion was used.** The scoring formula is identical across all
three attempts — per-row min-max normalisation of the response. What changed was
the hidden model's robustness, the operation strengths, and the artifact the
solver submits. Rescuing a dead design by reshaping the metric reads as
artificial difficulty to reviewers; rebuilding the data until the difficulty is
real does not.

## Why the class label is given away

Since the `(class, target)` lookup is available to any solver willing to spend
five minutes predicting the class (~95% accurate), withholding `true_class` would
only add a rote step. Publishing it turns that heuristic from a hidden exploit
into the stated baseline everyone starts from, and focuses the challenge on the
part that is actually hard.

## Well-posedness filters

Applied in `prepare.py`, both principled rather than difficulty-tuning:

- **Clean-correct only.** Images `H` already misclassifies have no intact
  prediction to steer away from (~4% of images dropped).
- **Reachable targets only.** A `(image, target)` row is kept only if some pair
  drives `p_H(target) >= 0.30`. Unreachable rows are unwinnable and would add
  noise, not difficulty. This leaves ~3.1 of the 9 possible targets per image.

## Leakage

- `H`'s training pool (45% of EuroSAT) is disjoint from every released image.
- Operation strengths were calibrated on `H`'s own pool, never on released data.
- The train/test split is **by image**: an image contributes up to 9 rows, so a
  row-level split would place the same image on both sides. Rubric 2 holds
  solvers to the same standard for their internal validation.
- The hidden model's weights are never released, and the test responses live
  only in `private/answers.csv`.

## Known limitations

- The hidden model is a specific 4-block CNN. Findings about *its* blind spots
  need not transfer to other architectures — but the challenge only claims to
  measure whether a solver can model **a** black box from its recorded behaviour.
- Baseline anchors were measured on a 700-image probe; the shipped dataset is
  6,000 images, so leaderboard values will shift slightly.
- EuroSAT is a moderately well-known dataset. The novelty here is the recorded
  behaviour layer and the task formulation, not the imagery.
