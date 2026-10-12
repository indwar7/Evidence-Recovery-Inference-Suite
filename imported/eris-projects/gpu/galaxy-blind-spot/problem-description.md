# Observing Conditions: Which Night Would Fool the Morphology Classifier?

## Overview

Automated morphology catalogues are built by running a classifier over millions
of survey cutouts. The classifier never sees a galaxy under ideal conditions —
it sees whatever the atmosphere, the telescope, the detector and the reduction
pipeline delivered that night. Survey teams know this biases their catalogues.
What they do not have is a way to say, for an *individual* galaxy, which
combination of conditions would tip it into a *particular* wrong morphology.

That is this task.

You are given real SDSS galaxy cutouts, each labelled with its Galaxy Zoo
morphology, and a catalog of **20 observing conditions** — seeing, guiding
trail, airmass reddening, cirrus, sky background, scattered light, read noise,
shot noise, saturation, detector nonlinearity, cosmic rays, a bleed column, a
blended foreground star, a deblending residual, pixel binning, cutout scale
error, coadd misregistration, and delivery compression.

Behind the task sits a **hidden morphology classifier**, `H`. You cannot query
it, you never see its weights, and it is not included in the data. What you are
given is a large table of its *measured behaviour*: for thousands of galaxies
and all 400 ordered pairs of conditions, the full probability vector `H`
returned.

Each row you must answer is a **(galaxy, target morphology)** pair. Your job:

> Name the ordered pair of observing conditions that drives `H`'s probability
> for the **target** morphology as high as possible for **that** galaxy.

The first condition is applied to the clean cutout; the second is applied to the
result. Order matters physically — clipping a saturated core and then convolving
with a seeing kernel is not the same as blurring first and clipping after.

This is not classification, segmentation, or regression on a label. The thing
you predict is an *intervention*: a pair of indices into a catalog of physical
conditions, chosen per galaxy and per target. Getting it right means having
learned where a specific model's decision boundary sits around a specific
galaxy — a robustness-auditing capability, not a recognition one.

## Evaluation

Submissions are scored by the **mean row-normalised target response**.

For each test row `i`, let `v_i` be the length-400 vector of hidden-model
probabilities assigned to that row's target morphology, one entry per ordered
condition pair, min-max normalised across the 400 entries. If you submit
conditions `op1` and `op2` for that row, you receive `v_i[op1*20 + op2]`.

~~~python
def evaluate(submission, answers):
    """
    submission: DataFrame with columns row_id, op1, op2
    answers:    DataFrame with row_id and r0..r399, where column r{k} holds the
                normalised response of the hidden model to pair k = op1*20 + op2
    """
    import numpy as np
    V = answers[[f"r{k}" for k in range(400)]].to_numpy(float)
    sub = submission.set_index("row_id").loc[answers["row_id"]]
    idx = sub["op1"].to_numpy() * 20 + sub["op2"].to_numpy()
    return float(np.mean(V[np.arange(len(V)), idx]))
~~~

Per-row normalisation puts every row on the same footing regardless of how
attainable its target is: **1.0** means you picked the single best pair
available for that row, **0.0** the single worst. The reported score is the mean
across all test rows, so it always lies in [0, 1] — higher is better.

Two filters applied during dataset preparation keep every graded row fair:
galaxies that `H` misclassifies when clean are dropped entirely, and rows whose
target is unreachable — no pair drives its probability to at least 0.30 — are
dropped as well.

For orientation, measured on the shipped data:

| approach | score |
|---|---|
| sample submission (`identity, identity`) | SAMPLE_SUB |
| a uniformly random pair | RANDOM |
| one fixed pair for every row | CONSTANT |
| best pair per target morphology | TARGET_LOOKUP |
| best pair per (true morphology, target morphology) | CLASS_TARGET_LOOKUP |
| reference solution | REFERENCE |
| oracle | 1.000 |

## Dataset

Everything lives under `dataset/public/`.

| file | contents |
|---|---|
| `images/IMAGE_ID.png` | the clean 69x69 RGB SDSS cutout |
| `train.csv` | `row_id, image_id, true_class, target_class` |
| `train_response.csv` | `row_id, r0 … r399` — the full measured surface |
| `test.csv` | `row_id, image_id, true_class, target_class` to answer |
| `ops.csv` | `op_index, op_name, family, strength_param, description` |
| `classes.csv` | `class_index -> Galaxy Zoo morphology name` |
| `catalog.py` | the condition implementations, verbatim |
| `sample_submission.csv` | correctly-shaped submission scoring SAMPLE_SUB |

`catalog.py` is the same code used to build the dataset. Given a cutout and an
`image_id`, it reproduces **exactly** the degraded image any pair produces, at
the calibrated strengths listed in `ops.csv`:

~~~python
import catalog, json
from PIL import Image
ops = catalog.build({r.op_name: r.strength_param for r in ops_df.itertuples()})
im  = Image.open(f"images/{image_id}.png")
degraded = ops[op2][1](ops[op1][1](im, image_id), image_id)   # op1 then op2
~~~

Conditions with a random realisation — where the cosmic rays landed, where the
blended star sits, which way the scattered-light gradient runs — are seeded from
the `image_id`, so they are fixed per galaxy and reproducible by you. What you
cannot reproduce is `H`'s reaction to them.

### How the catalog is calibrated

Every condition was individually tuned, by bisection, so that on its own it
flips `H` on the same fraction of galaxies (about 8%). Without that step the
answer collapses to "name the strongest condition", which is a property of the
catalog rather than of the galaxy. With it, every condition is equally dangerous
alone, and which *pair* wins has to come from how the two interact on this
particular galaxy — its size, its inclination, its position angle, how much of
its light sits in the core.

### Source data

Galaxy10 SDSS (astroNN): 21,785 real SDSS galaxy cutouts, 69x69 RGB, with
morphology labels from Galaxy Zoo volunteers.
https://zenodo.org/records/10844811, CC-BY-4.0. Two of the ten classes are
dropped as too rare for any model to learn (349 and 17 examples), leaving eight
morphologies; the remainder are capped so `H` is not dominated by the three
most common. The galaxies `H` trained on and the galaxies shipped here are
disjoint pools.

## Submission

Write `./working/submission.csv` with exactly one row per `row_id` in
`test.csv`:

~~~
row_id,op1,op2
0,7,13
1,2,2
...
~~~

`op1` and `op2` are integers in `[0, 19]` indexing `ops.csv`. `op1` is applied
first. Repeating a condition is allowed, and `0` is `identity` — a legal way to
say "one condition is enough".
