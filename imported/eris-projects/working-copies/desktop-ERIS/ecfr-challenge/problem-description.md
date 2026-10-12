# The Vanished Clause

A rule in the Code of Federal Regulations is amended. The old wording does not
get struck through or footnoted — it is simply gone, replaced in place by the
new text. Anyone reading the regulation afterwards sees only what it says now.

Here is part of an offshore-leasing rule as it reads after an amendment. Every
`[GAP]` is a place where wording *may* have been taken out:

> *"…(c) A statement that a well has been drilled on the lease and determined
> to be producible according to `[GAP]` 30 CFR 550`[GAP]` .115 `[GAP]` or
> 30 CFR 550.116;…"*

And here is what the amendment removed, as two loose fragments in no
particular order:

```
","        "§ 250.1603 (SOP only),"
```

Before the amendment the passage read *"…producible according to **§ 250.1603
(SOP only),** 30 CFR 550.115**,** or 30 CFR 550.116;"*. The first and third
marks were real. The second is a decoy — nothing was ever removed there — and
so are five more marks elsewhere in the same section.

That is this task. You are given a regulation section as it reads *after* an
amendment, the wording the amendment took out, and a set of marked places.
Reconstruct how the section read *before*.

## What is new here

1. **The answer is a recorded historical state, not an annotation.** Both
   wordings are the agency's own published text on two consecutive amendment
   dates in the eCFR's version record. Nobody chose what to delete in order to
   make a test; an agency deleted it, for regulatory reasons, and the archive
   kept both versions.
2. **Restoration is a partial matching, with the decoys on the side of the
   places.** Only about one marked place in four ever held removed wording.
   The solver decides which places are real *and* which fragment belongs to
   each, jointly. No fill-in convention applies: most marks must stay empty.
3. **The replacement is still on the page.** Where an amendment substituted
   wording, the new wording sits immediately after the true gap. A fragment's
   right-hand context is therefore often text that did not exist when the
   fragment did — and that replacement is itself evidence of what it
   displaced.
4. **The fragments are the amendment's own units.** They are not fixed-width
   masks drawn by a masking policy. In train they run from a single comma to a
   455-token paragraph, and two thirds of them are one token.
5. **It is scored as rebuilt text, token by token and place by place.** The
   grader never sees a choice of mark. It sees the prior wording you wrote,
   finds what you added and where, and compares that with what the amendment
   removed and where. Leaving a fragment out, doubling it up, or putting it
   one token off are all priced by the same rule.
6. **Transfer across regulatory areas is built into the split.** Test sections
   come from CFR parts that appear nowhere in train, so the defined terms and
   cross-references of a test section have never been seen.

The capability this measures is knowing where wording belongs in a legal text
— the inverse of drafting an amendment.

## How it relates to prior work

The assignment step is classical. Once every (mark, fragment) pair has a
score, choosing a one-to-one matching is the linear assignment problem (Kuhn,
1955), and the reference solution uses it off the shelf. What is new is what
has to be scored, and on what evidence.

- **The cloze procedure (Taylor, 1953)** deletes words from a passage by rule
  and asks a reader to supply them. Here the deletions were made by an agency
  amending a rule, not by a test-maker; they run from one token to a
  paragraph; and the removed wording is supplied. The question is where it
  goes, among places that mostly lost nothing.
- **Gapped-text reading items**, the examination format in which sentences
  are taken out of a passage and must be put back, carry one spare sentence as
  a distractor. Here the spares are *places* — three for every real one — and
  the passage has been rewritten around some of the gaps.
- **Differential file comparison (Hunt and McIlroy, 1976)** records what was
  removed and *where*, which makes undoing a change mechanical. This task
  hands over the removed side of such a comparison with the addresses erased
  and the order shuffled, and asks for the addresses.
- **Textual criticism** reconstructs an earlier state of a text from later
  witnesses. Here there is a single later witness and a set of loose earlier
  readings, and the reconstruction is checked against the earlier state as it
  was actually published rather than argued for.

## Task

Each row is one CFR section as it stands after a recorded amendment:

- `text_after` — the full section text as it read after the amendment,
  including its heading, with `[GAP]` marks inserted
- `removed_fragments` — a JSON list of strings: every run of wording the
  amendment removed, exactly as it was printed

Produce `text_before`: the full section text as it read immediately before
that amendment, with no `[GAP]` marks in it.

How the two inputs relate:

- Every fragment came out of **exactly one** mark, and no mark lost more than
  one fragment. Put each fragment back at its own mark and you have the prior
  wording.
- **Most marks are decoys.** A row carries three decoy marks for every real
  gap, and never fewer than five, so about one mark in four is real. Decoys
  are spread over the whole section.
- The fragments are listed in an order that carries no information. The
  number of fragments tells you how many marks are real, not which.
- Where the amendment *replaced* wording, the replacement is still in
  `text_after`, immediately after the real mark. You do not have to delete it:
  only the wording you put back is scored (see Evaluation).

The section is not identified. No CFR title, part or section number is given,
and the section's own number is replaced by `[SECTION]` wherever it appears in
the text (`§ [SECTION] Inventory allowance…`, `paragraph (a) of § [SECTION]`).
The same mask is applied to the prior wording, so you never have to predict
it. References to *other* sections are left exactly as published — they are
part of the regulation's wording, and often what an amendment changed.

The amendment boundary is not something this dataset invented. Every CFR
section carries the dates it was amended in the eCFR's own version record, and
`text_after` / `text_before` are the agency's published wording on two
consecutive such dates.

## What you are actually being asked

**A placement, delivered as text.** What you decide is a partial one-to-one
matching between fragments and marks; what you submit is the section rebuilt
according to it.

**Which marks are real, and which fragment belongs in which.** A fragment such
as `"§ 250.1603 (SOP only),"` reads naturally in one place in a section and
nowhere else; a fragment such as `","` fits in a great many. Deciding between
them is a question about how regulatory prose is drafted — how a list of
cross-references runs on, where a paragraph designation can sit, what a
replaced phrase looked like before it was replaced.

**This is not diff application.** The Federal Register publishes an
instruction alongside every amendment — *"in paragraph (b)(2), remove the
words 'thirty days' and add in their place 'ten days'"*. Those instructions
are deliberately withheld. With one in hand the task is a string-editing
exercise.

**This is not free generation.** The removed wording is handed to you because
it cannot be inferred: nothing in a rule that now says *ten days* records that
it used to say *thirty*. What can be inferred is where wording belongs, and
that is what is scored.

**Most of the section did not change.** A typical amendment touches about 2%
of a section's tokens. Removing the marks and submitting the rest unchanged is
therefore *almost* right as a string — and scores exactly zero here, by design.

## Dataset

```python
import json
from pathlib import Path
import pandas as pd
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/ecfr-amendment-reversal/public"))
            if (p / "test.csv").exists())
train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")
slots = pd.read_csv(DATA / "train_slots.csv")

row       = test.iloc[0]
pieces    = row.text_after.split("[GAP]")      # n_marks + 1 pieces of text
fragments = json.loads(row.removed_fragments)  # list of strings
# mark k sits between pieces[k] and pieces[k + 1]
```

- `train.csv` — `row_id, text_after, removed_fragments, text_before`
  (410 rows)
- `test.csv` — `row_id, text_after, removed_fragments` (353 rows)
- `train_slots.csv` — `row_id, gap_index, fragment_index`: for every fragment
  of every **train** row, the mark it came out of. `gap_index` is the 0-based
  position of the mark among the `[GAP]` marks of that row's `text_after`, in
  reading order; `fragment_index` is the 0-based position of the fragment in
  that row's `removed_fragments` list. A mark that appears in no row of this
  file is a decoy. This is the training label in its direct form — the same
  thing `text_before` tells you, without having to align two texts. There is
  no such file for test: that is the answer.
- `sample_submission.csv` — `row_id, text_before`: the input with its marks
  removed and nothing put back
- `dataset_stats.json` — split counts, the decoy settings, train-side size
  statistics, and what was dropped at build time

What the training rows look like: a median of **4** fragments and **16** marks
per row (the largest row has 147 fragments); 58 of the 410 rows have a single
fragment. Two thirds of all fragments are a single token — a comma, a
paragraph letter, one changed word — and one in ten runs to 7 tokens or more,
up to whole paragraphs. A row's score is weighted by tokens, so long fragments
matter most within a row.

The split is **by CFR part**: every section of a given part lands entirely in
train or entirely in test. Parts share defined terms, drafting conventions and
cross-references heavily, so a part-mixed split would let those be memorised
instead of generalised.

**Each test row is a different section.** A section amended several times
gives a chain of pairs, and one pair's amended text is the next pair's prior
wording. Test therefore holds exactly one pair per section; train keeps every
pair, so consecutive train rows are often successive versions of the same
section (`train.csv` is in source order). Test rows are in an order that
carries no information, and `row_id` is opaque.

No numeric column is published. A character count is derivable from the text
and adds nothing.

## Rules

**Use only the data directory.** These are real regulations and the eCFR
publishes every historical version of every section, searchable by date.
Recovering an answer by looking the section up — in the eCFR, in the Federal
Register, in govinfo, in any mirror or bulk copy of them, or through any
service that exposes CFR version history — is not solving the task; a
submission built that way is invalid regardless of its score, and so is any
submission that depends on data not shipped in the data directory. The solver
environment is expected to have no network access; your solution must run to
completion from the data directory alone.

**The amendment instruction is withheld on purpose.** The Federal Register
notice that produced each change states in plain language what to remove and
where. Recovering that notice — from its FR citation, its docket, or any
published amendment index — substitutes an explicit edit script for the
inference the task is measuring.

**Each test row stands alone.** Predict every test row from the training
files and that row's own fields. Do not fit anything — counts, vocabularies,
language models, embeddings — on `test.csv` as a corpus, and do not pool or
compare test rows to settle any one of them. A solution that adapts to the
test set as a whole is invalid regardless of its score.

- Learn from the provided data only. No pretrained models of any kind
  (language models, encoders, embeddings, tokenizers trained elsewhere), no
  LLMs, no external corpus of regulatory text, no CFR snapshot, no Federal
  Register archive.
- No LLM-generated outputs may be used anywhere in your solution.
- Use only libraries available in the Kaggle Python Docker image.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

**Gap-anchored changed-token F1.** Your submission is scored as text — the
grader never sees which mark you chose, only the `text_before` you wrote. For
each row, the true prior wording is aligned to the shown wording (`text_after`
without its marks), and the tokens where they differ are identified — those
are what the amendment removed. Each is keyed by *where* it sat: the gap
between two shown tokens it belongs in. Your prediction is aligned to the
shown wording the same way.

```python
changed  = {(gap, token)} for tokens of the true `text_before` not matched to the shown text
proposed = {(gap, token)} for tokens of your prediction not matched to the shown text
hits     = |changed ∩ proposed|          # multiset intersection

recall    = hits / |changed|
precision = hits / |proposed|
row_score = 2·precision·recall / (precision + recall)
```

A token counts only if it is the right token in the right place. Mean over all
test rows. Range `[0, 1]`, higher is better. Tokens are `\w+|[^\w\s]`, so
spacing never matters. A `[GAP]` mark left in a prediction is dropped before
scoring.

What this means for a fragment: put it back at its real mark and every one of
its tokens is a hit; put it at a decoy and every one of its tokens is a miss
*and* a precision error. A fragment left out costs recall only. Putting the
same fragment at several marks is allowed, and is paid for in precision.

### Why this metric and not the obvious alternatives

**Not whole-string similarity.** Consecutive versions of a regulation share
almost all their text: the median amendment in train changes **2.2%** of its
section's tokens (p25 1.0%, p75 5.4%). Under edit distance or BLEU against the
full section, submitting the input with its marks removed is ~98% correct and
scores near the maximum, leaving no range to separate real attempts. Scoring
only the changed positions sends it to exactly 0.

**Not exact string match.** A row has a median of four fragments. A
submission that places three of them has done most of the work;
all-or-nothing throws that away.

**Not accuracy over (mark, fragment) pairs.** Fragments differ enormously in
weight — one is a comma, the next a whole paragraph. Counting pairs would pay
the same for either. Scoring tokens makes the paragraph worth what it is.

**Not recall over changed tokens alone.** With recall only, putting every
fragment at every mark recovers every removed token and scores like the
oracle. With the precision term it scores **0.112**.

**Not position-free token matching.** Without the gap in the key, appending
all the fragments to the end of the section — reading nothing — would be a
perfect answer. Keyed by gap it scores **0.045**.

Baselines measured on the shipped split:

- blank submission — **0.000**
- marks removed, nothing put back — **0.000** (the sample submission)
- the input submitted with its marks still in — **0.000**
- the 60 commonest removed tokens from train, appended — **0.012**
- every fragment appended to the end of the section — **0.045**
- every fragment at the first mark — **0.080**
- fragments dealt to marks at random — **0.091**
- every fragment at the last mark — **0.098**
- fragments in listed order into the first marks — **0.102**
- every fragment at every mark — **0.112**
- the most tightly clustered marks, fragments dealt at random — **0.119**
- a classifier that picks the likeliest real marks from each mark's own
  surroundings, never reading a fragment, fragments dealt at random among
  them — **0.188**
- bigram-count heuristic, no training labels, counts from the training text
  and the row's own text: each fragment goes where the word before the mark
  is most often followed by the fragment's first word — **0.295**
- learned (mark, fragment) pair scorer with a one-to-one assignment
  (`solution.ipynb`) — **0.497**
- *(diagnostic ceiling)* every fragment at its true mark — **0.982**
- submit the true prior wording — **1.000**

### What the numbers say about the task

**Position and order carry nothing.** The first mark, the last mark, the
listed order and random dealing all sit between 0.08 and 0.10.

**Finding the real marks is only part of it.** Knowing which marks look
amended, without reading the fragments, is worth 0.188. The rest comes from
matching each fragment to its own place.

**The counts alone get under a third of the way; learning gets to half.** A
heuristic with no training labels reaches 0.295. A plain linear scorer over
pair features, trained on `train_slots.csv`, reaches 0.497. Both predict each
test row on its own. The gap from there
to 0.982 is single-token fragments and near-identical candidate places, where
word counts stop discriminating and a model has to actually read the sentence.

The true-mark ceiling is 0.982 rather than 1: where a fragment ends in a token
that the shown text repeats right after the mark, the alignment can key that
one token to the neighbouring gap.

## Submission

`submission.csv`, exactly two columns:

- `row_id` — one row per `row_id` in `test.csv`
- `text_before` — your reconstruction of the prior wording: the text of
  `text_after` with the fragments put back and the marks removed

Degradation behaviour:

- A blank or entirely wrong prediction scores **0.0** for that row.
- A `row_id` missing from your submission scores **0.0** for that row rather
  than failing the run.
- `row_id` values not in the test set are ignored.
- A duplicated `row_id` is a conflict the grader will not resolve: every row
  for it is discarded and that row scores 0.
- A `[GAP]` mark left in `text_before` is ignored, not penalised.
- Only a **missing required column** is rejected outright, with
  `InvalidSubmission`.

## Compute

1× NVIDIA A10G (16 GB), 10 CPU cores, 62 GB RAM, under 1.5 hours.

The natural approach scores every (mark, fragment) pair of a section for how
well the fragment reads in that place, then assigns fragments to marks
one-to-one. The scorer is where the work is: `train_slots.csv` labels 4,631
fragments, each with its real mark among all the marks of its section, and
the 410 training sections — both versions of each — are enough text to train
a small language model of this register from scratch.
Sections run long, but only the few tokens either side of a mark and the two
ends of a fragment decide a pair, so the budget is far more than the task
needs; spend it on the model, not on the data.
