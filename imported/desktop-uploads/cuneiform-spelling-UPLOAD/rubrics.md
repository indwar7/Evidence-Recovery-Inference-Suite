# Rubrics — Cuneiform Spelling of Unseen Akkadian Words

**[REQUIRED] Learns from the provided data only.** No pretrained models, no LLMs, no external Akkadian dictionaries, sign lists (ePSD, ORACC gloss files, CDA, CAD), word lists, or corpora of any kind are used. The solver environment has no network access; a solution that depends on anything outside the data directory is invalid regardless of its score.

**[REQUIRED] Does not look the answer up.** The source editions are public and searchable. Recovering a spelling by retrieving the edition of the text behind a row (by its normalized line, its metadata, or any other key) is not solving the task and invalidates the submission.

**[REQUIRED] Scores each row independently.** The prediction for a row is a function of that row's own columns and the training data. Fitting statistics on the released evaluation inputs, or combining several evaluation rows to resolve one, is not permitted.

**[REQUIRED] Produces one spelling per row in ORACC transliteration form.** Signs joined by hyphens, determinatives in curly braces, sign indices as subscript digits, exactly as the training targets are written. A row whose prediction is blank, missing, or duplicated scores 0 for that row; only a missing column rejects the file.

**[REQUIRED] Notebook runs end to end on the stated budget.** One A10G, 1.5 hours wall clock, everything seeded and reproducible from the data directory alone.

**[RECOMMENDED] Generalises beyond a lookup.** Every evaluation row's normalized form is absent from training by construction, so a norm-to-spelling table has zero coverage. A strong solution uses the lemma's attested spellings, the gloss, the part of speech, the surrounding line and the text metadata to decide whether the word is written logographically or syllabically and which signs carry which syllables.

**[RECOMMENDED] Treats the metric's partial credit as sign choice, not string overlap.** The score is a sign-level edit similarity: one wrong homophone sign (`u` for `u₂`) or one missing plene vowel costs one sign out of the word's length. Solutions should therefore decode whole sign sequences rather than copy the normalized string.

**[RECOMMENDED] Measures itself against the published baselines.** The naive syllabifier and the stem-plus-suffix heuristic are documented with their scores; a solution that does not clear both on a held-out slice of the training texts is not ready to submit.

**[RECOMMENDED] Reports a held-out score built the same way as the evaluation set.** Hold out whole training texts and keep only words whose normalized form is absent from the remaining texts; a random row-level split overstates the score by a wide margin because the lookup table then covers most rows.
