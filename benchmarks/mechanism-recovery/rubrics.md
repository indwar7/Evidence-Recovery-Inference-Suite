# Rubrics — What Does It Do In There?

- **[REQUIRED] Uses only the data directory — no drug-identity lookup.** These
  labels are real, public documents. Recovering an answer by identifying the
  drug and looking up its mechanism class — in a drug-label archive, a drug
  database, a pharmacologic class directory, a biomedical ontology, or any
  mirror of them — is not solving the task. A submission
  built that way is invalid regardless of its score, as is any submission
  depending on data not shipped in the data directory. The solver environment
  is expected to have no network access.

- **[REQUIRED] Does not attempt to defeat the masking.** Mechanism vocabulary
  is masked using a vocabulary built from every class in the corpus, and the
  drug's brand and generic names are withheld. Reconstructing a masked term
  from residual context in order to string-match it against a candidate name
  substitutes unmasking for the pharmacological inference being measured, and
  is treated the same as an identity lookup.

- **[REQUIRED] Selects from the row's own candidate pool.** Each row carries
  its own twelve candidates and the same class is a correct answer on one row
  and a distractor on another. A solution that predicts over a single global
  label set, ignoring the pool it was given, has not attempted the task as
  specified.

- **[REQUIRED] Decides how many classes to name, per row.** The number of true
  classes varies and is never given. The metric subtracts the expected score
  of a same-size random draw, so naming all twelve scores exactly 0.000.
  Always naming exactly one class caps a solver at 0.806 even when that class
  is always right, because about a third of training rows (34%) have more than
  one true class. A submission that always names the same number is leaving points on
  the table by construction.

- **[RECOMMENDED] Reads the whole pharmacology section.** Sections run to
  several thousand words, and the sentences that identify a mechanism —
  substrate, binding target, metabolic pathway — are frequently deep in the
  text rather than in the opening paragraph. A representation that truncates
  early discards the region the answer usually lives in.

- **[RECOMMENDED] Scores each candidate against the text, rather than ranking
  candidates against each other.** The pool is a fixed size but the number of
  correct answers is not, so the useful question is "does this text support
  this class?" one candidate at a time. A model that only ranks within the
  pool still has to choose a cut-off, and scoring pairs independently makes
  that threshold something you can tune on train.

- **[RECOMMENDED] Treats `route` as legitimate signal.** Administration route
  is published in both splits and genuinely constrains mechanism — a topical
  antiseptic and an oral enzyme inhibitor do not share classes. Using it is
  intended, not a shortcut.

- **[UNIVERSAL] Reproducible end to end.** The notebook runs top to bottom,
  loads only the public data, seeds every source of randomness, and writes
  `submission.csv` with exactly `row_id` and `moa_classes`, pipe-separated.
