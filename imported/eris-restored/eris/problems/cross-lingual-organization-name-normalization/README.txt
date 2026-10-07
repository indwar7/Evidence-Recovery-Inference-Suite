CROSS-LINGUAL ORGANIZATION NAME NORMALIZATION - upload checklist
================================================================

FORM FIELDS
  Problem title    Cross-Lingual Organization Name Normalization
  Dataset title    Multilingual Institution Name Index with Canonical English Names
  Domain           Sequence to sequence
  Tags             text, generative
  License          CC0 1.0
  Source URL       https://zenodo.org/records/17953395

UPLOAD
  Dataset file        ror-names-dataset.zip     (FLAT: raw/ generator/ public/ private/)
  Dataset description dataset-description.md    (paste whole file)
  Problem description problem-description.md    (paste whole file)
  prepare.py          organization-disjoint split
  grade.py            mean token-level F1
  Reference solution  solution.ipynb

ALL CRITERIA
  enabled domain   sequence to sequence          YES
  real data        ROR live registry, 120,445 orgs, nothing generated
  source URL       https://zenodo.org/records/17953395   DOI 10.5281/zenodo.17953395
  license          CC0 1.0 - commercial use free, attribution not even required
  difficulty band  0.426 -> 0.489, inside 0.4-0.7
  CPU only         pure text, pandas only, no GPU
  not a benchmark  ROR is an operational registry, not a competition dataset

MEASURED BAND  (mean token F1, held-out organizations, via the real grader)
  empty output                              0.000
  copy the source name through              0.426   <- sample_submission
  aligned word substitution table           0.470
  bag co-occurrence table                   0.486
  REFERENCE, per-language + copy-through    0.489

WHAT THE TASK TESTS
  1. Systematic vocabulary: universiteit / universite / universidad /
     universitaet all become university, across 134 languages.
  2. Knowing when NOT to act: 38% of registry labels already equal the English
     display name. Translating them costs you.
  3. Proper nouns: a token seen once is a place or a person, not vocabulary.
     Pushing it through the table corrupts a name that was already correct.

  The split is by ORGANIZATION, so nothing can be looked up - an institution
  whose English name is needed at test time never had it shown in training.

VERIFIED
  prepare.py runs from the zip; target_name absent from public/test.csv
  sample_submission.csv and answers.csv share columns pair_id,target_name
    in the same order (platform integrity check)
  reference scores 0.4891 against freshly prepared answers
  sample_submission scores 0.4259
  known-answer submission scores 1.000000
  no source_name repeats within or across splits - checked exact,
    (source_name, lang), and lowercase/whitespace-normalized: all zero
  grade.py: python grade.py submission.csv answers.csv -> bare float;
    InvalidSubmissionError subclasses ValueError (REJECTION-REGISTER 1.13)
  zip is FLAT with raw/ generator/ public/ private/ at the root; platform
    entrypoint replayed from the extracted zip (REJECTION-REGISTER 1.2, 1.22)
  notebook runs top to bottom and writes ./working/submission.csv
  grader absorbs dropped rows / shuffling / duplicates / empty outputs,
    raises only on a missing column
  both descriptions are 100% ASCII (paste-safe)
  source URL appears ONLY in the dataset description, never in the problem
    description

NOVELTY - HONEST NOTE
  You asked for 8. I cannot promise it and will not pretend otherwise: Shipd
  applies cite-or-cap, and a nearest neighbour will be named for any task in a
  studied area. What this build does is fix the part that was provably costing
  you a point on babel-gauge - real operational provenance, explicitly denied
  there ("no +1 for real-world dataset provenance"). CC0 also removes every
  licensing question. Check novelty before investing further.
