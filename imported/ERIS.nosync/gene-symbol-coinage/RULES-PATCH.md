# Rules-section patch for problem-description.md (platform)

Paste this in place of the current "Rules" section. Changes from what's there
now:

1. Made "trained GPU parameters" a `[REQUIRED]`-style hard gate with a named
   check, not just prose, and gave the agent a concrete number to fail against
   (0.481 CPU rule vs 0.531 reference — the CPU rule must not win).
2. Made the vocabulary-isolation rule explicit about *when* it is checked
   (tokenizer construction time) and named the two common ways agents violate
   it (reading test.csv before building the vocab; fitting any statistic,
   including a "coverage check," on the test names).
3. Kept the named lookup resources and the 43.5/56.5 prose exactly as they
   were — those are honest, task-level disclosures, not the leak the reviewer
   flagged. The leak was `dataset_stats.json` publishing the gold seen/new
   split at the PER-ROW level; that's fixed in prepare.py, not here.

---

## Rules

Use only the data directory. The names in this task are real, public gene
names, so the answer for any test row can in principle be looked up.
Recovering a symbol by searching for the name in any gene nomenclature
resource or gene database (NCBI Gene, Ensembl, UniProt, OMIM and the like), or
through any gene-annotation package (mygene, biomaRt, pyensembl, org.Hs.eg.db,
gseapy gene-set libraries, pybiomart or equivalent), is not solving the task.
A submission built that way is invalid regardless of its score, and so is any
submission that depends on data not shipped in the data directory. Nothing in
the data can make a lookup impossible, so this rule is what closes that
route. The solver environment is expected to have no network access, and your
solution must run to completion from the data directory alone.

Do not use any model or resource that already knows human gene symbols.
Pretrained language models and biomedical embeddings (BioBERT, PubMedBERT,
BioWordVec, SciBERT and the like) have read the literature in which every
approved symbol appears next to its name. Using one turns coinage back into
recall.

**[REQUIRED] Trained GPU parameters must materially drive every prediction.**
A hand-coded rule table (nearest training family if one is close, otherwise
initials) scores 0.481 on this split; the reference, a trained character
transformer, scores 0.531. A valid submission's score must come from
parameters fit by gradient descent on the GPU against the training pairs —
not from a rule table, a nearest-neighbour lookup, or a trained model used
only as a fallback behind one of those. Concretely: (a) a solution with no
`torch`/`jax`/equivalent training loop that updates weights is invalid
regardless of score; (b) a solution whose predictions are unchanged when its
trained model's output is replaced with random noise, for rows where the
rule table would have fired, is invalid; (c) reporting both a rule-table
score and a trained-model score and submitting whichever is higher is
invalid — submit the trained model's own output.

**[REQUIRED] The character vocabulary (or tokenizer) is built from the
training names and symbols only, before test.csv is read for anything other
than producing the submission.** Two concrete violations to avoid: reading
`test.csv` to compute a character set or vocabulary size "for padding" before
training starts; and fitting any frequency statistic — a smoothing table, an
embedding initialisation, a coverage check — on the test names as a corpus.
If your tokenizer would raise on an unseen character, use a fixed,
sufficient character set (upper- and lower-case Latin letters, digits,
space, hyphen) rather than reading test.csv to discover one; every character
in this task's test names already occurs in train.

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Predict each test row from its own name plus the training data. Do not fit
  statistics on the test names as a corpus.
- Your notebook must run end to end, top to bottom: load the data, train,
  then write `submission.csv`.
- Seed everything; your run should be reproducible.

What not to use: pretrained models or embeddings of any kind, task-specific
or gene-annotation libraries, external word lists, gene lists or nomenclature
tables, any network access, any lookup of the answer outside the data
directory, and statistics or vocabularies fitted on the test names.
