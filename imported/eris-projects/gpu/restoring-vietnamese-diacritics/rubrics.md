# Rubrics — Restoring Tone Marks on Vietnamese Place Names Built from Unseen Syllables

Each rubric is specific to this task and approach-neutral (no particular
architecture is mandated beyond the stated requirement that predictions come
from trained parameters). Every rule in `problem-description.md` has a matching
`[REQUIRED]` rubric here.

**[REQUIRED] Predictions come from trained neural parameters.** Hand-coded
rules, character n-gram or context tables, most-frequent per-letter or
per-syllable substitution (including tables conditioned on the province or
district), and dictionary or nearest-neighbour lookup, including copying
marks from the `province`, `district` or `neighbors` columns onto matching
syllables, are not valid solutions. On this split every such table scores
below a trained character model, but a submission that is really a table
dressed up — or a trained model used only as a fallback behind one — is not
demonstrating the sub-syllabic generalisation the task measures. Heuristic
features may be fed to a trained model as input.

**[REQUIRED] No external data or lookup of the native spelling.** No external
gazetteer, place-name or geographic database, map or geocoding service, word
list, pretrained model or embedding, or network access may be used. The names
are real and public, so the native spelling is externally discoverable; a
submission that recovers it from outside the data directory is invalid
regardless of score.

**[REQUIRED] Train-only vocabulary; no test-set statistics.** Any character
vocabulary must be built from the training data (or a fixed character set).
Each test row must be predicted from its own input plus the training data;
statistics and vocabularies must not be fitted on the test inputs as a corpus.

**[REQUIRED] No LLM-generated content anywhere in the pipeline.** Neither the
solution outputs nor any intermediate artefact may be produced by a large
language model.

**[REQUIRED] Output preserves length and order.** The task is a one-to-one,
length-preserving substitution: the predicted `name` must have the same number
of characters as the `folded` input, changing only diacritics. A prediction
whose length differs from the input cannot align to the target and scores zero
for that row.

**[REQUIRED] Reproducible, self-contained run.** The notebook must run top to
bottom from the data directory alone, with all randomness seeded, and must
discover the data directory at run time rather than hardcoding a single path.

**[RECOMMENDED] Generalises to unseen syllables.** Every test name contains
at least one syllable that occurs in no training name, and no test name occurs
in training. A competitive solution must restore marks on syllables it has
never seen, which requires modelling how vowels, finals and tones combine
below the syllable rather than memorising syllables or names.

**[RECOMMENDED] Restores multiple marks per name.** Most names need more than
one mark. A method that reliably fixes only the easiest single position leaves
most of the score on the table; the metric gives graded credit for each
correctly restored position.

**[RECOMMENDED] Uses whole-name context, not just the bare letter.** A folded
vowel is ambiguous in isolation; the correct restoration depends on the
syllable and the surrounding name. Solutions that condition each position on
the full folded string outperform position-local substitution.

**[RECOMMENDED] Conditions on the geographic context.** The grader judges the
marks a row's context could supply separately from those it could not, so a
solution that ignores `province`, `district` and `neighbors` forfeits
headroom on the context-supplied marks. Feeding the context names to the
trained model (for example as a second sequence or as per-syllable features),
and showing on a validation split that it helps over the bare folded name, is
the direct way to use them; a pure copy from these columns is both invalid and
weak.
