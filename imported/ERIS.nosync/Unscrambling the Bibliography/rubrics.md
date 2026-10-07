# Rubrics — The Reference Shuffle

- **[REQUIRED] Returns a full permutation per unit, not a partial or
  tied guess.** For each `unit_id` in `test.csv`, the submitted
  `(rank_A, rank_B, rank_C, rank_D)` tuple must be a permutation of
  exactly `{0, 1, 2, 3}`. A solution that emits ties (e.g.
  `rank_A=rank_B=0`) or leaves a value out of range has not attempted the
  task as specified — the metric scores every card's position against
  the full true order, so every position must be committed to.

- **[REQUIRED] Uses only that unit's own reference pool — no cross-unit
  lookup.** Each paragraph is scored independently against its own
  shuffled cards. Using `train.csv` in any way the solver likes
  (supervised training, feature engineering, calibration) is the intended
  path. Using a *different test row's* reference pool or paragraph to
  help resolve the current one is not, and would not generalise to a
  single paragraph presented in isolation.

- **[REQUIRED] Does not depend on looking up the source paragraph
  online.** The paragraphs are real published prose (citation markers
  replaced with `[CITE]` and leftover markup removed) and are, in
  principle, findable by searching a distinctive phrase from the text. A
  submission whose approach depends on recovering the article and reading
  its real citation order, rather than reasoning about which reference
  card's title/author/venue/year content matches which sentence, is not
  solving the intended task, whatever it scores. The solver environment
  is expected to have no network access.

- **[REQUIRED] Uses none of the tools the problem statement rules out.**
  No language model (hosted or local), no model trained on citation or
  bibliographic data, no external data at solve time, no statistic fitted
  on `test.csv` or on train and test pooled, and no hand-labelled test
  units. Any one of these makes the submission invalid regardless of its
  score.

- **[REQUIRED] Treats a withheld field as withheld.** Every card shows
  `[withheld]` in exactly one recorded slot. Recovering that value from
  anywhere outside the data directory, or from the raw source, is not
  solving the task; a card missing its title is meant to be placed from
  its remaining fields or by elimination against its pool-mates.

- **[REQUIRED] Treats the chance floor as 0.25, not 0.0, when judging
  its own progress.** Position accuracy has an exact expected value of
  1/4 for a uniformly random permutation of four cards (verified by
  exhaustive enumeration over all 24 permutations; see DESIGN.md). A
  submission that scores close to 0.25 has not extracted usable signal
  from the reference metadata, even though 0.25 is well above the
  grader's 0.0 floor for a non-attempt. Calibrate expectations against
  0.25, not against 0.0.

- **[RECOMMENDED] Uses the reference cards' content (title, author,
  venue, year), not just their count or letter.** The cards carry real
  bibliographic metadata specific to each citation. A solution that
  ignores card content and instead exploits some structural regularity in
  how cards are laid out is capped well below the reference score —
  content-free position, length and year probes sit within 0.06 of the
  0.25 chance floor.

- **[RECOMMENDED] Calibrates on `train.csv`'s held-out articles, not
  against the graded score.** Test units come from articles entirely
  disjoint from the training articles, so any model or heuristic should
  be validated against a held-out slice of `train.csv` rather than
  iterated against the leaderboard.

- **[RECOMMENDED] Does not rely on `tf-idf` weighting or any external
  pretrained language/embedding model.** The measured reference solution
  uses raw word-overlap counts (no `tf-idf`) and no pretrained embeddings,
  and still clears every content-free/adversarial probe by 0.188. A
  submission is free to use stronger methods, but the reference
  demonstrates that neither is required to clear the difficulty band.
  Whatever statistics a submission does fit must come from `train.csv`
  only — never from `test.csv` or from train and test pooled.
