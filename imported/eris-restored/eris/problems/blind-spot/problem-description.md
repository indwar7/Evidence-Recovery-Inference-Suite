# Blind Spot: Steering a Hidden Classifier With Ordinary Image Operations

## Overview

Somewhere behind this challenge sits a trained image classifier. You will never
see its weights, its architecture, or its predictions on the test set. It takes
a 64×64 satellite image and returns a probability over ten land-cover classes,
and on clean imagery it is right about 96% of the time.

Your job is to find its blind spots — and to aim them.

You are given a catalog of **20 ordinary image operations**: JPEG compression,
Gaussian blur, resampling, brightness and contrast shifts, desaturation,
sharpening, motion blur, and so on. Each is mild enough to leave the image
plainly recognisable, and each has been strength-calibrated so that, applied on
its own, it flips the hidden model on roughly 8% of images. No operation is a
brute-force wrecking ball, and none dominates the others.

Each test row hands you an image, its true land-cover class, and a **target
class**. You choose an ordered pair of operations from the catalog — apply the
first, then the second. Your score is how strongly that choice pushes the hidden
model's probability toward the target class you were given.

This is not adversarial perturbation in the usual sense: there is no gradient to
follow, no pixel-level optimisation, and only 400 legal moves per row. It is a
question about a *specific model's* idiosyncratic geometry. Two images of the
same class, steered toward the same target, frequently want different
operations — the winning pair for a given row agrees with its
(class, target) group's most common answer only about a third of the time. What
generalises is not a rule of thumb about which operation is strongest, but a
learned model of where this particular classifier's decision boundaries sit
around this particular image.

The public training split gives you the hidden model's **complete measured
response** — all 400 pairs — for every training row. That is your only window
onto its behaviour. Whether you use it to fit a surrogate, to distil the
model's boundary structure, or something else entirely is up to you.

## Evaluation

Submissions are scored by the **mean row-normalised target response**.

For each test row `i`, let `v_i` be the length-400 vector of hidden-model
probabilities assigned to that row's target class, one entry per ordered
operation pair. If you submit operations `op1` and `op2` for that row:

~~~python
def evaluate(submission, answers):
    """
    submission: DataFrame with columns row_id, op1, op2
    answers:    DataFrame with row_id and r0..r399, where column
                r{k} holds p_hidden(target_class | pair k) and k = op1*20 + op2
    """
    import numpy as np
    V = answers[[f"r{k}" for k in range(400)]].to_numpy(float)
    sub = submission.set_index("row_id").loc[answers["row_id"]]
    idx = sub["op1"].to_numpy() * 20 + sub["op2"].to_numpy()

    chosen = V[np.arange(len(V)), idx]
    lo, hi = V.min(axis=1), V.max(axis=1)
    return float(np.mean((chosen - lo) / np.maximum(hi - lo, 1e-9)))
~~~

Per-row normalisation puts every row on the same footing regardless of how
attainable its target is: **1.0** means you picked the single best pair
available for that row, **0.0** the single worst. The reported score is the mean
across all test rows, so it always lies in [0, 1] — higher is better.

Rows where the target class is unreachable (no pair drives its probability to at
least 0.30) are excluded during dataset preparation, so every graded row is
winnable.

For orientation: always submitting one fixed pair scores about 0.18. Looking up
the best pair for each (true class, target class) combination from the training
data — a strong, cheap baseline — scores about 0.70. The oracle is 1.0.

## Dataset

Everything lives under `dataset/public/`.

```
public/
├── images/<image_id>.png   64×64 RGB satellite patches
├── train.csv               training rows
├── train_response.csv      the hidden model's full response for each training row
├── test.csv                test rows — predict for these
├── ops.csv                 the 20-operation catalog
└── sample_submission.csv   a valid but deliberately weak submission
```

**`train.csv` / `test.csv`**

| Column | Type | Description |
|---|---|---|
| `row_id` | str | Unique row identifier (`tr_000123`, `te_000123`) |
| `image_id` | str | Image for this row; file is `images/<image_id>.png` |
| `true_class` | int | The image's actual land-cover class, 0–9 |
| `target_class` | int | The class you must steer the hidden model toward; never equals `true_class` |

An image may appear in several rows with different targets. **Images do not
cross the split** — no test image appears in training under any target.

**`train_response.csv`** — one row per training row.

| Column | Type | Description |
|---|---|---|
| `row_id` | str | Matches `train.csv` |
| `r0` … `r399` | float | `p_hidden(target_class)` after applying pair `k`, where `k = op1*20 + op2` |

**`ops.csv`**

| Column | Type | Description |
|---|---|---|
| `op_index` | int | 0–19; the value you put in `op1` / `op2` |
| `op_name` | str | e.g. `identity`, `jpeg`, `motion_h`, `desat` |
| `strength_param` | float | The calibrated strength that operation was applied at |

Operation index **0 is `identity`**, so a pair like `(0, 7)` means "apply
operation 7 alone".

Class codes: 0 `AnnualCrop`, 1 `Forest`, 2 `HerbaceousVegetation`, 3 `Highway`,
4 `Industrial`, 5 `Pasture`, 6 `PermanentCrop`, 7 `Residential`, 8 `River`,
9 `SeaLake`.

## Submission

Submit a CSV named `submission.csv`:

| Column | Type | Description |
|---|---|---|
| `row_id` | str | Row identifier from `test.csv` |
| `op1` | int | Index of the operation applied **first**, 0–19 |
| `op2` | int | Index of the operation applied **second**, 0–19 |

**Requirements**

- Exactly **7,045 rows**, one per row in `test.csv`, plus a header row.
- Every `row_id` from `test.csv` present exactly once; no extras, no duplicates.
- `op1` and `op2` are integers in `[0, 19]`. They may be equal — applying the
  same operation twice is a legal move.
- Order matters: `(3, 11)` and `(11, 3)` are different submissions and generally
  score differently.

Example:

```csv
row_id,op1,op2
te_000000,0,0
te_000001,4,17
te_000002,11,11
```
