# Rubrics — Spelling Unseen Akkadian Words in Cuneiform Signs

**[REQUIRED] Learns from the provided data only.** No pretrained models, no LLMs, and no external Akkadian or Sumerian resources of any kind are used: no sign lists, syllabaries, dictionaries, glossaries, lemmatisers, normalisers or corpora (ePSD, the Oracc gloss files, CDA, CAD, or language models trained on Akkadian). The solver environment has no network access. A solution that depends on anything outside the data directory is invalid regardless of its score.

**[REQUIRED] Does not look the answer up.** The source editions are public and searchable. Recovering a spelling by retrieving the edition of the text behind a row, through its normalized line, its metadata or any other key, is not solving the task and invalidates the submission.

**[REQUIRED] Scores each row independently.** The prediction for a row is a function of that row's own columns and the training data. Fitting statistics on the evaluation inputs, or combining several evaluation rows to resolve one, is not permitted.

**[REQUIRED] Produces one spelling per row in the training targets' convention.** Signs are joined by hyphens, logograms are upper case, sign indices are subscript digits, and determinatives and phonetic complements sit in braces, exactly as the training targets are written. A row whose prediction is blank, missing or duplicated scores 0 for that row; only a missing column rejects the file.

**[REQUIRED] Notebook runs end to end on the stated budget.** One A10G, 1.5 hours wall clock, top to bottom, everything seeded and reproducible from the data directory alone.

**[REQUIRED] Uses only libraries available in the Kaggle Python Docker image.**

**[REQUIRED] Uses no LLM-generated outputs anywhere.** No labels, spellings, rules, code or predictions produced by an LLM are part of the solution.

**[RECOMMENDED] Generalises beyond a lookup.** Every evaluation row's normalized form occurs on no word of any training text, so an exact lookup has zero coverage. The measured rule systems reach 0.32 to 0.36; the strongest, a vowel-length-blind lookup within the lemma backed by stem-plus-suffix rules with each syllable written by its most frequent sign, scores 0.3641. A strong solution uses the lemma's attested spellings, the gloss, the part of speech and the text metadata to decide whether the word is written logographically or syllabically, which homophone signs carry which syllables, and how the ending is written.

**[RECOMMENDED] Aims for the scribe's whole spelling, not only overlapping signs.** Half of each row's score is given only when the predicted sign sequence equals the scribe's exactly. One wrong homophone sign in a six-sign word costs more than half the row (0.417 instead of 1). Solutions should decode whole sign sequences and prefer spellings that are internally consistent with the lemma's attested forms.

**[RECOMMENDED] Validates the way the evaluation set is built.** Hold out whole training texts using `train_docs.csv`, and keep only rows whose normalized form occurs nowhere else in training. A random row-level split overstates the score by a wide margin, because most normalized forms then also appear in the remaining training rows.

**[RECOMMENDED] Measures itself against the published baselines.** The naive syllabifier (0.3182), the stem-plus-suffix heuristic (0.3305), the strongest hand-written rule system (0.3641) and a classical joint-sequence transducer with a learned reranker (CLASSICAL_SCORE, CPU only) are documented with their scores. A trained model that does not clear the classical transducer on a held-out slice of the training texts is not yet using what a GPU-trained model can learn here.

**[RECOMMENDED] Gives the model the evidence a scribe's convention leaves in the training texts.** The reference gains most from retrieving, for each row, spellings of the lemma's closest training forms and of other words with the same ending (0.5200 without them, REFERENCE_SCORE with them, together with more epochs and beam search). Any such retrieval must use training rows only, and must not let a training row see its own form or its own text.
