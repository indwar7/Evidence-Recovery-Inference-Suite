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
  what to remove and what to insert. Those instructions are deliberately
  withheld: with one in hand the task is executing an edit script rather than
  inferring a change. Recovering the notice from an FR citation, a docket
  number, or a published amendment index substitutes lookup for the inference
  being measured.

- **[REQUIRED] Produces the prior wording, not a paraphrase of the current
  wording.** The target is a specific text that actually existed. Only the
  tokens where the true prior wording departs from the shown text are scored,
  so fluent regulatory prose that does not recover the removed language earns
  nothing, however well it reads. A solution that treats this as style transfer
  has not attempted the task as specified.

- **[REQUIRED] Commits to a specific reconstruction rather than hedging over a
  vocabulary.** The metric balances recall of the removed tokens against
  precision over what the prediction adds. Appending a bank of plausible
  regulatory words to the shown text was measured: it scores 0.012 against the
  oracle's 1.00, because a proposed token only counts in the gap where the
  removed token actually sat, and every extra token costs precision. Breadth
  is not a strategy here; a submission that proposes many candidates instead
  of one answer is penalised by construction.

- **[RECOMMENDED] Locates the amended passage before rewriting it.** The median
  amendment touches a small fraction of a long section. A model that rewrites
  the whole provision disturbs text that never changed, which costs precision
  without improving recall. A sentence locator ranks an amended sentence first
  only about half the time, so locating is a real part of the problem, and the
  metric rewards leaving the rest alone.

- **[RECOMMENDED] Reads the section's subject from its own text.** No title,
  part or section number is published and the section's own number is masked
  as `[SECTION]`, but the heading and surviving cross-references still say
  what a section governs. Different regulatory areas amend different things —
  a deadline in an import-procedure section, an emission threshold in an
  environmental one — and using that context is legitimate. Trying to
  un-mask or identify the section is not (see the lookup rule above).

- **[RECOMMENDED] Handles sections longer than a single model window.**
  Sections here run long and the change is frequently deep in the provision
  rather than in its opening. A representation that truncates early silently
  discards the region the answer usually lives in.

- **[UNIVERSAL] Reproducible end to end.** The notebook runs top to bottom,
  loads only the public data, seeds every source of randomness, and writes
  `submission.csv` with exactly `row_id` and `text_before`.
