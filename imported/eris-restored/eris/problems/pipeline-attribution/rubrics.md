# Rubrics — Digitization Pipeline Attribution

- **[REQUIRED] Generalizes to unseen batches, does not memorize train batch
  identities.** Train and test batches are disjoint by construction. A
  correct approach learns what makes one OCR pipeline's noise pattern differ
  from another's in general, then applies that notion to snippets from
  batches never seen during training. Any approach that only works because it
  secretly assumes test items belong to one of the 19 known train batches
  (e.g. a closed-set classifier applied to open-set data without adaptation)
  is not solving the stated task.

- **[REQUIRED] Produces a valid partition per bag without assuming a fixed
  group count.** The number of true groups varies from bag to bag and is
  never given. A submission that hardcodes a single cluster count across all
  bags, or that silently fails on bags smaller or larger than an assumed
  size, does not meet the task as specified.

- **[REQUIRED] Does not exploit cross-bag or whole-corpus structure as a
  substitute for per-item modeling.** A real adversarial attack was measured
  during this task's construction: ignoring bag boundaries and running a
  single unsupervised clustering pass over the entire test corpus at once,
  using no labeled training data, is a documented shortcut that partially
  works because so few real batches exist in this dataset (see DESIGN.md). A
  submission built this way — rather than by learning from `train.csv` and
  applying that learned notion independently within each bag — should be
  penalized even where it happens to score reasonably, since it does not
  demonstrate the intended generalizable capability.

- **[RECOMMENDED] Moves beyond a single cheap heuristic scalar.** Thresholding
  or clustering on one hand-picked statistic (e.g. non-ASCII character rate
  alone) was measured to plateau well below a properly trained multi-feature
  or learned-embedding approach during this task's construction. Solutions
  that combine multiple noise signals, or learn a representation end-to-end,
  are expected to outperform single-feature heuristics.

- **[RECOMMENDED] Uses the available GPU for a learned representation, not
  only hand-engineered features.** A character-level or subword neural
  encoder trained with a similarity-style objective on the labeled train
  batches (so it generalizes to unseen test batches by similarity rather than
  lookup) is the intended way to use the compute budget. Hand-crafted noise
  statistics are a reasonable baseline but were measured to plateau below a
  properly trained embedding approach at this data scale.

- **[RECOMMENDED] Validates using held-out train batches, not test signal.**
  Since test batches are never labeled, any hyperparameter or threshold
  tuning (e.g. choosing a similarity threshold or an expected group-count
  heuristic) should be validated on a held-out slice of the labeled train
  batches, not on the test set itself.

- **[UNIVERSAL] Documents the group-count estimation strategy used.** Since
  the true number of groups per bag is never given, a clear, brief note on
  how the submission estimates or avoids needing that count (e.g. a distance
  threshold plus connected components, silhouette-based selection, or a
  learned stopping criterion) helps interpret the results.
