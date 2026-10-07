# Rubrics — The Accent Translator

- **[REQUIRED] Works from spelling and accent labels only.** The source
  transcription is hidden. A solution must not try to recover it from other
  files; the test set has one row per word, so no sibling row exists.

- **[REQUIRED] Uses the target accent.** Different target accents demand different
  outputs for the same spelling. A model that ignores `tgt_accent` cannot score
  well on the rewrite segments.

- **[REQUIRED] Knows when NOT to change.** 12% of rows have identical source
  and target. Rewriting them scores 0 for that row. A model must learn that
  the rewrite is conditional, not that "something always changes".

- **[REQUIRED] Validates on held-out words, not rows.** The test set is
  word-disjoint. A random row split leaves other directions of the same word
  in the training fold, which leaks the target through its siblings.

- **[RECOMMENDED] Models the rewrite as context-sensitive.** The changes are
  governed by phonological environment (following consonant, syllable
  position, neighbouring vowel). A per-segment lookup without context was
  measured as the CPU reference; a model that reads the whole sequence should
  clearly exceed it.

- **[RECOMMENDED] Uses the GPU for a sequence model over phonemes.** An
  encoder–decoder with both accents as prefix tokens, or a tagging model that
  predicts an edit per source segment, are the intended uses of the budget.

- **[RECOMMENDED] Reports accuracy by direction.** RP→GA and GA→RP dominate;
  NZ and AU directions are sparse and harder. A per-direction breakdown shows
  whether the model learned the rules or just the common pair.

- **[RECOMMENDED] Inspects errors by phonological class.** Rhoticity, the
  TRAP–BATH set, KIT centralisation, GOAT fronting — the errors cluster. A
  solution that names where it fails demonstrates understanding beyond the
  number.

- **[UNIVERSAL] Documents where each accent token enters the model and how
  the no-change case is handled.**
