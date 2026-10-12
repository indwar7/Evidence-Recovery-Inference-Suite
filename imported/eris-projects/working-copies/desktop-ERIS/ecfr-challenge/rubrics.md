# Rubrics — The Vanished Clause

- **[REQUIRED] Uses only the data directory — no CFR or Federal Register
  lookup.** These are real regulations and the eCFR publishes every historical
  version of every section, addressable by date. Recovering an answer from the
  eCFR, the Federal Register, govinfo, any mirror or bulk copy of them, or any
  service exposing CFR version history is not solving the task. A submission
  built that way is invalid regardless of its score, as is any submission
  depending on data not shipped in the data directory. The solver environment
  is expected to have no network access.

- **[REQUIRED] Does not reconstruct or use the amendment instruction.** Every
  change here was made by a Federal Register notice stating in plain language
  what to remove and where. Those instructions are deliberately withheld: with
  one in hand the task is executing an edit script rather than inferring where
  wording belongs. Recovering the notice from an FR citation, a docket number,
  or a published amendment index substitutes lookup for the inference being
  measured.

- **[REQUIRED] Learns from the provided data only — no pretrained model.** No
  pretrained language model, encoder, embedding or tokenizer, no LLM, and no
  LLM-generated output anywhere in the solution. The 410 training sections
  (both versions of each) and `train_slots.csv` are the whole of what a
  solution may learn from. A model that arrives already knowing how
  English regulatory prose reads is measuring its pretraining, not this task.

- **[REQUIRED] Predicts each test row on its own.** Every prediction is made
  from the training files and that row's own `text_after` and
  `removed_fragments`. Nothing is fitted on `test.csv` as a corpus — no counts,
  vocabulary, language model or embedding — and no test row is settled by
  pooling or comparing it with other test rows. A solution that adapts to the
  test set as a whole is invalid regardless of its score.

- **[REQUIRED] Writes the prior wording as text, with the marks removed.**
  `text_before` is the section text with fragments put back in place. The
  grader never sees which mark was chosen — it aligns the submitted text to
  the shown text and scores the tokens that were added, each in the gap where
  it sits. A submission of mark indices, or of the fragments alone, is not the
  object being graded and scores 0.

- **[REQUIRED] Commits each fragment to a place rather than hedging across
  marks.** The metric balances recall of the removed tokens against precision
  over what the prediction adds. Putting every fragment at every mark was
  measured: it scores 0.112 against the oracle's 1.000, because every copy in
  a wrong gap costs precision. Appending the fragments to the end of the
  section scores 0.045. Breadth is not a strategy here.

- **[RECOMMENDED] Decides which marks are real and which fragment goes where
  jointly.** About one mark in four is real. A classifier that finds
  likely-amended marks without reading the fragments scores 0.188; matching
  fragments to marks one-to-one over a learned pair score reaches 0.497. A
  solution that treats the two questions separately leaves most of that
  difference unused.

- **[RECOMMENDED] Reads the text after the mark as possibly new.** Where the
  amendment replaced wording, the replacement is still in the shown text right
  after the real mark. A fragment therefore need not read on smoothly into the
  next token — it may read on into the token after the replacement — and the
  replacement often resembles the fragment it displaced. A scorer that only
  asks "does the fragment fit between these two tokens" misses both signals.

- **[RECOMMENDED] Weights effort by tokens, not by fragments.** Two thirds of
  training fragments are a single token, but a row's score is an F1 over
  tokens, so one long fragment can outweigh every short one in its row.
  Getting the long fragments right first is worth more than treating all
  fragments alike.

- **[UNIVERSAL] Reproducible end to end.** The notebook runs top to bottom,
  loads only the public data, seeds every source of randomness, and writes
  `submission.csv` with exactly `row_id` and `text_before`.
