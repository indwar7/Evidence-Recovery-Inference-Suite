# Observing Conditions — Initial Rubrics

Criteria for judging *how* a solution was built, alongside its leaderboard
score. Each is stated so it can be checked against the agent's code and output
without knowing which modelling approach it chose.

---

### 1. Respects the ordered-pair semantics
**Type:** DATA_HANDLING · **Importance:** REQUIRED

`train_response.csv` column `r{k}` corresponds to `k = op1*20 + op2`, with `op1`
applied to the clean image first and `op2` applied to the result. The solution
must index it that way and must not symmetrise the 400 columns into 210
unordered combinations, nor transpose the two axes.

Order is physical, not bookkeeping: saturation clipping a core and *then*
convolving with a seeing kernel spreads a flat-topped core, while blurring first
and clipping after removes the peak entirely. A solution that conflates the two
is reading its own supervision wrong.

**Fails if:** the response vector is reshaped to (20,20) with the axes swapped,
or `(a,b)` and `(b,a)` are averaged together.

---

### 2. Validates on a galaxy-disjoint split
**Type:** TRAINING · **Importance:** REQUIRED

One galaxy contributes up to seven training rows, one per target morphology. A
random row-level train/validation split therefore places the *same galaxy* on
both sides and reports a validation score well above what the leaderboard will
give. The split used for model selection must be grouped by `image_id`.

**Fails if:** validation rows are drawn by shuffling `train.csv` without
grouping, or no validation split exists at all.

---

### 3. Emits a structurally valid submission
**Type:** CODE_QUALITY · **Importance:** REQUIRED

`submission.csv` contains exactly one row per `test.csv` `row_id` — no
duplicates, none missing, no extras — with `op1` and `op2` as integers in
`[0, 19]`. Written to `./working/submission.csv`.

**Fails if:** operation indices are emitted as floats, strings, or op *names*;
or the row count does not match `test.csv`.

---

### 4. Conditions the prediction on the target morphology
**Type:** MODELLING · **Importance:** REQUIRED

The artifact is "which conditions push *this* galaxy toward *that* morphology".
A model that ingests only the image and ignores `target_class` is answering a
different, easier question — which conditions damage the galaxy at all — and
will score near the untargeted baseline no matter how well it is trained.

**Fails if:** `target_class` never enters the model's input.

---

### 5. Learns the surface, not only its argmax
**Type:** MODELLING · **Importance:** RECOMMENDED

Each training row ships the full 400-value response vector, not just the index
of its best pair. Supervising on the whole surface gives roughly 400x the
gradient signal per row and is markedly more sample-efficient than treating the
row as a 400-way classification with one hot label.

**Met by:** a regression, ranking, or soft-target objective over all 400 values.

---

### 6. Beats the per-(morphology, target) lookup
**Type:** MODELLING · **Importance:** REQUIRED

Picking the pair that works best on average for each `(true_class,
target_class)` combination is a few lines of pandas and needs no model. A
submission must beat that baseline on the leaderboard; matching it means the
solution has learned the class-level prior and nothing about the individual
galaxy.

**Fails if:** the final score does not exceed the documented
`class_target_lookup` anchor.

---

### 7. Handles orientation-dependent conditions correctly
**Type:** TRAINING · **Importance:** RECOMMENDED

Horizontal and vertical flips map every op in the catalog to itself, so they are
safe augmentations. Rotations by 90 degrees are **not**: they exchange `trail_h`
with `trail_v`, and any augmentation that rotates the image without permuting
the corresponding response entries silently corrupts the labels for those ops.

**Fails if:** 90-degree rotation or transpose augmentation is applied to inputs
while the 400-vector target is left unpermuted.

---

### 8. Uses the released catalog rather than guessing at it
**Type:** DATA_HANDLING · **Importance:** RECOMMENDED

`catalog.py` ships with the data and reproduces exactly the degraded image any
`(op1, op2)` pair produces, at the calibrated strengths in `ops.csv`. Rendering
those images and feeding them to the model is the intended use of the GPU
budget, and is what separates a solution that reasons about the *stimulus* from
one that treats the 400 columns as anonymous features.

**Met by:** calling `catalog.build(...)` (or an equivalent reimplementation) to
render degraded views during training or inference.

---

### 9. Reproducible within the compute budget
**Type:** CODE_QUALITY · **Importance:** REQUIRED

The notebook runs end to end on the provided hardware inside the time limit,
with seeds set, and produces the submitted file. Rendering all 400 conditions
for every galaxy is affordable but not free; a solution that would need many
hours does not satisfy this even if its logic is sound.

**Fails if:** the run exceeds the time limit, or the reported score cannot be
reproduced from the submitted code.

---

### 10. Reports an honest holdout estimate
**Type:** CODE_QUALITY · **Importance:** OPTIONAL

The write-up states a galaxy-grouped holdout score before submission and
comments on the gap to the leaderboard. Claiming a number without a disjoint
split, or reporting only the training-set fit, does not count.
