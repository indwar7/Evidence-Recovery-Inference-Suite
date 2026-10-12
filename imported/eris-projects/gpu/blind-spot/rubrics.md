# Blind Spot — Initial Rubrics

Criteria for judging *how* a solution was built, alongside its leaderboard
score. Each is stated so it can be checked against the agent's code and output
without knowing which modelling approach it chose.

---

### 1. Respects the ordered-pair semantics
**Type:** DATA_HANDLING · **Importance:** REQUIRED

`train_response.csv` column `r{k}` corresponds to `k = op1*20 + op2`, applying
`op1` first. The solution must index it that way and must not symmetrise the
400 columns into 210 unordered combinations, nor transpose the two axes.
`jpeg → noise` and `noise → jpeg` have materially different effects; a solution
that conflates them is reading its own supervision wrong.

**Fails if:** the response vector is reshaped to (20,20) with the axes swapped,
or duplicate/unordered pairs are averaged together.

---

### 2. Validates on an image-disjoint split
**Type:** TRAINING · **Importance:** REQUIRED

A single image contributes up to nine training rows, one per target class. A
random row-level train/validation split therefore places the *same image* on
both sides and reports a validation score well above what the leaderboard will
give. The split used for model selection must be grouped by `image_id`.

**Fails if:** validation rows are drawn by shuffling `train.csv` rows without
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

### 4. Extracts image-specific signal, not just a lookup table
**Type:** MODELING · **Importance:** REQUIRED

Choosing the best pair for each `(true_class, target_class)` combination from
the training data is a cheap baseline worth roughly 0.69. Beating it requires
using the image itself, since the winning pair agrees with its group's modal
answer only about a third of the time. A solution scoring at or below that
baseline has not engaged with the actual task.

**Fails if:** the predicted pair is a deterministic function of
`(true_class, target_class)` alone.

---

### 5. Conditions the prediction on the target class
**Type:** MODELING · **Importance:** REQUIRED

The response surface is target-specific: the pair that best steers an image
toward `Highway` is usually not the one that best steers it toward `SeaLake`.
A model that ignores `target_class` and predicts a single "most damaging" pair
per image is solving a different, easier problem and is capped near 0.48.

**Fails if:** `target_class` is absent from the model's inputs.

---

### 6. Predicts the response surface rather than classifying the argmax
**Type:** MODELING · **Importance:** RECOMMENDED

Training a 400-way classifier on the argmax label throws away most of the
signal: many rows have several near-tied pairs, and the margin between rank 1
and rank 5 is often negligible. Modelling the response values across pairs —
and then taking the argmax at inference — uses the graded supervision the
training split actually provides.

**Rewards:** any approach that learns from the full 400-length vector, by
regression, ranking, or distributional loss.

---

### 7. Optimises the quantity that is actually scored
**Type:** TRAINING · **Importance:** RECOMMENDED

The metric is row-normalised: what matters is the chosen pair's position
between the best and worst pair *for that row*, not the raw probability. Rows
whose targets are easy to reach contribute no more than hard ones. A solution
that normalises per row during training or model selection is aligned with the
leaderboard; one that fits raw probabilities will over-serve easily-steered
rows.

**Rewards:** per-row normalisation, rank-based losses, or explicit
top-1-accuracy-against-oracle validation.

---

### 8. Uses the pixel data
**Type:** FEATURE_ENGINEERING · **Importance:** REQUIRED

The images in `images/` are the only source of image-specific information; the
CSVs carry nothing but ids and class codes. A solution must load and derive
representations from the pixels — learned or hand-built, at whatever
resolution it likes.

**Fails if:** `images/` is never read.

---

### 9. Fits the compute budget and uses the GPU
**Type:** CODE_QUALITY · **Importance:** RECOMMENDED

End-to-end runtime under 1.5 hours on 1×A10G, with the model actually placed on
the GPU. The training split is roughly 10⁴ rows × 400 targets over 3.5k images —
enough that a GPU-trained network is the natural fit and a CPU-bound approach
will either time out or underfit.

**Rewards:** batched GPU training/inference; explicit runtime awareness.

---

### 10. Establishes a baseline before modelling
**Type:** AGENT_BEHAVIOR · **Importance:** RECOMMENDED

Computes at least one reference point — constant pair, target-only lookup, or
(class, target) lookup — and reports its validation score before building
anything more elaborate, so that later work can be shown to improve on it.

**Rewards:** a stated progression with numbers attached.

---

### 11. Reports validation score and reasoning
**Type:** COMMUNICATION · **Importance:** RECOMMENDED

States the validation score achieved, how the validation split was constructed,
and why the modelling choices were made. Enough that the leaderboard number is
reproducible in principle from the write-up.

**Fails if:** no score is reported, or the reported number comes from a split
that violates rubric 2.
