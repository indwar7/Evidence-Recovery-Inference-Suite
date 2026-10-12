# Rubrics — Reconstructing Local Citation Structure in Case Law

- **[REQUIRED] Returns a candidate subset per query, not a ranking or a
  single pick.** For each `query_id` in `test.csv` the submitted artifact
  is a set of candidate positions drawn from that query's own twelve
  candidates. A solution that emits only a best-matching candidate, or a
  ranked list without committing to a cut-off, has not attempted the task
  as specified: the metric scores the selected set against the true set,
  so the decision of *how many* to select is part of the problem.

- **[REQUIRED] Uses only that query's own candidate pool — no cross-query
  lookup.** Each query is scored independently against its own twelve
  candidates. Using `train.csv` in any way the solver likes (supervised
  classifier, threshold tuning, feature engineering, a pre-trained
  encoder) is the intended path. Using a *different test query's* row to
  help resolve the current one is not, and would not generalise to a
  single query presented in isolation.

- **[REQUIRED] Does not rely on surviving citation strings or case
  names.** Both are redacted from every shipped opinion and the shipped
  split is verified to contain zero of either (see
  `dataset/public/dataset_stats.json`, field `leak_scan`). A submission
  whose approach depends on recovering leftover citation syntax rather
  than on the substance of the legal discussion is not solving the
  intended task, whatever it scores.

- **[REQUIRED] Selects a variable number of candidates per query.** The
  number of true citations varies from 1 to 4 across queries. Both degenerate strategies score exactly
  0.000 under the chance-corrected metric — selecting nothing and
  selecting all twelve — and a fixed-size selection is measurably worse
  than one that adapts per query. A submission that always emits the
  same count is leaving points on the table by construction.

- **[RECOMMENDED] Reads the whole opinion, not just its opening.**
  Opinions here run to roughly 1,500 tokens, and the passage that signals
  a dependency is often deep in the discussion rather than in the first
  paragraph. Measured on the shipped split: `all-MiniLM-L6-v2` truncating
  at 256 tokens scores 0.6481, while the same model over sliding windows —
  scoring each pair by its best-matching window pair — scores 0.6494 under
  an otherwise identical decision rule. The gain from reading the whole
  document is real but small; the larger lever is the decision rule (a
  fixed top-2 cut-off on those same scores falls to 0.5517).

- **[RECOMMENDED] Calibrates the selection threshold on `train.csv`, not
  against the graded score.** Test queries come from connected components
  of the citation graph entirely disjoint from the training components,
  so any threshold or hyperparameter should be tuned against a held-out
  slice of `train.csv` rather than iterated against the leaderboard. A
  threshold fitted to the test set will not reflect genuine skill.

- **[RECOMMENDED] Normalises similarity within each query's pool.**
  Absolute similarity scores are not comparable across queries — opinion
  length, subject matter and drafting style all shift the scale. Measured
  on the shipped split, a fixed absolute cut-off performs far worse than
  the same model's scores standardised within each query's own twelve
  candidates before thresholding.
