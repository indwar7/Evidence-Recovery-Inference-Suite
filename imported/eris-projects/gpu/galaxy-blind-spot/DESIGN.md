# Observing Conditions — design record

## The task in one line

Given a real SDSS galaxy and a target morphology, name the ordered pair of
observing conditions that drives a hidden classifier's probability for that
morphology as high as possible.

## Why this shape

An earlier occupancy scan across candidate GPU challenge spaces killed eight
ideas because a published benchmark already occupied them. The one space that
came back empty was **grading against a hidden system's measured behaviour** —
the ground truth is neither a human annotation nor another algorithm's output,
it is what a specific model actually did, recorded once and replayed at grading
time. That makes the ground truth exact, the grading deterministic and pure-CSV,
and the task un-scrapeable, because the responses exist nowhere else.

This challenge instantiates that engine in observational astronomy.

## Why observing conditions rather than generic corruptions

A predecessor built on EuroSAT used a catalog of ordinary image operations
(blur, JPEG, brightness). It worked, but the answer it demanded — "which image
codec setting breaks this model" — has little standing outside ML robustness
folklore.

Replacing that catalog with **real survey systematics** changes what the task
is about. Seeing, guiding trail, airmass reddening, cirrus, sky background,
scattered light, read and shot noise, saturation, detector nonlinearity, cosmic
rays, bleed columns, a blended foreground star, a deblending residual, pixel
binning, cutout scale error, coadd misregistration and delivery compression are
the actual error budget of an imaging survey. Asking which *pair of observing
conditions* would tip a specific galaxy into a specific wrong morphology is a
question survey teams genuinely have: automated morphology catalogues from SDSS,
DES and Rubin/LSST are known to be biased by observing conditions, and nobody
can currently answer it per-object.

Two of these conditions do work no generic catalog can:

* **`trail_h` vs `trail_v`.** Which one damages a galaxy depends on its position
  angle on the sky, which is essentially random and is *not* a property of its
  morphology class. An edge-on disk trailed along its major axis survives;
  trailed across it, it stops looking edge-on.
* **Seeded realisations.** Where the cosmic rays land, where the blended star
  sits, which way the scattered-light gradient runs — all seeded from the public
  `image_id`. Fixed per galaxy, reproducible by the solver, and orthogonal to
  class.

Both push the answer away from being a per-class constant, which is the failure
mode that has killed more of these designs than anything else.

## Design decisions that came from previous failures

**Targeted, not untargeted.** An earlier iteration asked "which pair damages the
model most". A per-class lookup table scored 0.907 against a reference of ~0.99
— the answer was a property of the class, not the image. Adding a named *target*
class makes the answer depend on which decision boundary this particular galaxy
sits near. The metric was never touched; only the artifact changed.

**Strength calibration on break rate.** Every condition is bisected to the same
solo break rate (~8%) against `H`. Without this, the best pair degenerates to
"name the strongest condition", which is identical for every row. Calibrating on
*break rate* rather than mean pixel damage matters: mean damage has a floor at
the model's own baseline uncertainty, so equalising it silently leaves the
conditions at very unequal real strengths. Break rate has a floor at zero.

**A robust hidden model.** `H` is trained with augmentation spanning the same
families as the catalog. A model trained without augmentation is *blanket*
brittle, and blanket brittleness is a class property. Robustness to any single
condition is what forces the winning pair to be about the individual galaxy.

**Disjoint pools.** `H` trains on 45% of the capped Galaxy10 pool; the challenge
ships the other 55%. `H` has never seen a released galaxy, so its responses are
not memorisation.

**Public stimulus, hidden response.** `catalog.py` ships with the data and
reproduces the exact degraded image for any pair. Solvers can render what `H`
saw; they cannot know what `H` did about it. This keeps the difficulty in
modelling the system rather than in guessing at the rendering, and it gives the
GPU budget something real to do — rendering 400 conditions per galaxy and
training a surface model is the intended solution shape.

**Split by galaxy, never by row.** One galaxy yields up to seven rows sharing an
image. A row-level split leaks; the rubric makes this an explicit criterion
because it is the single easiest way for a solver to fool itself.

## Metric

Mean row-normalised target response. For each row the 400 measured responses are
min-max normalised at build time, so submitting the best available pair scores
1.0 and the worst scores 0.0, regardless of how attainable that row's target
was. Normalising at build time keeps the grader a pure lookup — it never runs
the model, so two grading runs cannot disagree.

Rows are filtered twice for fairness: galaxies `H` misclassifies when clean are
dropped, and rows whose target never reaches p = 0.30 under any pair are
dropped, so every graded row is winnable.

## Gate results

GATE_RESULTS_PLACEHOLDER

## Honest novelty assessment

Scored as a reviewer would, against what exists:

* **Adversarial / corruption robustness benchmarks** (ImageNet-C, and the
  adversarial-attack literature) measure *aggregate degradation* under a fixed
  corruption suite, or search a continuous perturbation with white-box gradients.
  Neither asks a model to *predict, per image, which discrete intervention pair
  steers another model to a named class* without query access. The prediction
  target here is an intervention, not a label or a score.
* **Galaxy morphology benchmarks** (Galaxy Zoo challenges, astroNN) are
  classification of morphology. This is not classification: morphology labels are
  given as *input*.
* **Survey-systematics work** exists but is population-level — how does seeing
  bias the catalogue in aggregate — not per-object and not adversarially targeted.
* **Black-box attack literature** is the closest relative, but it assumes query
  access to the victim. Here there is none: the entire signal is a static table
  of past behaviour, so the task is transfer/surrogate modelling of a system from
  observation alone.

The mechanism (grade against a hidden system's recorded behaviour) is shared
with a sibling challenge built on EuroSAT; the *task* here — physically-grounded
observing conditions, an orientation-dependent answer, and a survey-auditing
framing — is distinct in what capability it measures. Deducting for that shared
mechanism, this lands at **8**, not 9 or 10.

## Residual risks

RESIDUAL_RISKS_PLACEHOLDER
