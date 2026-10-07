# The Vanished Clause

A rule in the Code of Federal Regulations is amended. The old wording does not
get struck through or footnoted — it is simply gone, replaced in place by the
new text. Anyone reading the regulation afterwards sees only what it says now.

Here is a food-import rule as it reads today:

> *"…the division director shall give the owner or consignee written or
> electronic notice of the opportunity for a hearing, to be held **within ten
> days** of the notice…"*

Before the amendment, that same sentence read **thirty days**. Nothing in the
current text says so. You would have to know how this kind of provision is
normally drafted, what the surrounding paragraphs still assume, and which way
a deadline like this tends to move, to work out what was taken out.

That is this task. You are given a regulation section as it reads *after* an
amendment. Reconstruct how it read *before*.

## Task

Each row gives you one CFR section as it stands after a recorded amendment:

- `text_after` — the full section text as it read after the amendment,
  including its heading

The section is not identified. No CFR title, part or section number is given,
and the section's own number is replaced by `[SECTION]` wherever it appears in
the text (`§ [SECTION] Inventory allowance…`, `paragraph (a) of § [SECTION]`).
The same mask is applied to the prior wording, so you never have to predict
it. References to *other* sections are left exactly as published — they are
part of the regulation's wording, and often what an amendment changed.

Produce `text_before`: the full section text as it read immediately before
that amendment.

The amendment boundary is not something this dataset invented. Every CFR
section carries the dates it was amended in the eCFR's own version record, and
`text_after` / `text_before` are the agency's published wording on two
consecutive such dates.

## What you are actually being asked

**This is not diff application.** The Federal Register publishes an
instruction alongside every amendment — *"in paragraph (b)(2), remove the
words 'thirty days' and add in their place 'ten days'"*. Those instructions
are deliberately withheld. With one in hand the task is a string-editing
exercise; without it you have to infer what changed from how the regulation
now reads.

**This is not summarisation or paraphrase.** The answer is a specific prior
wording that actually existed, not a plausible rewrite. Only the tokens the
amendment removed are scored (see Evaluation), so fluent prose that does not
recover the old language earns nothing.

**Most of the section did not change.** A typical amendment touches a few
sentences of a long provision. Copying the input through unchanged is
therefore *almost* right as a string — and scores approximately zero here, by
design.

## Dataset

```python
from pathlib import Path
import pandas as pd
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/ecfr-amendment-reversal/public"))
            if (p / "test.csv").exists())
train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")
```

- `train.csv` — `row_id, text_after, text_before`
- `test.csv` — `row_id, text_after`
- `sample_submission.csv` — `row_id, text_before`, the input copied through
- `dataset_stats.json` — split counts, the change-size distribution, and what
  was dropped at build time

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
what to insert. Recovering that notice — from its FR citation, its docket, or
any published amendment index — substitutes an explicit edit script for the
inference the task is measuring.

- Learn from the provided data only. No external corpus of regulatory text,
  no CFR snapshot, no Federal Register archive.
- No LLM-generated outputs may be used anywhere in your solution.
- Use only libraries available in the Kaggle Python Docker image.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

**Gap-anchored changed-token F1.** For each row, the true prior wording is
aligned to the wording you were shown, and the tokens where they differ are
identified — those are what the amendment removed. Each is keyed by *where* it
sat: the gap between two shown tokens it belongs in. Your prediction is aligned
to the shown wording the same way.

```python
changed  = {(gap, token)} for tokens of the true `text_before` not matched to `text_after`
proposed = {(gap, token)} for tokens of your prediction not matched to `text_after`
hits     = |changed ∩ proposed|          # multiset intersection

recall    = hits / |changed|
precision = hits / |proposed|
row_score = 2·precision·recall / (precision + recall)
```

A token counts only if it is the right token in the right place. Mean over all
test rows. Range `[0, 1]`, higher is better. Copying the input through scores
0; submitting the true prior wording scores 1.

### Why this metric and not the obvious alternatives

**Not whole-string similarity.** Consecutive versions of a regulation share
almost all their text: the median amendment here changes **2.9%** of its
section's tokens (p25 1.1%, p75 8.3%). Under edit distance or BLEU against the
full section, copying the input through is ~97% correct and scores near the
maximum, leaving no range to separate real attempts. Scoring only the changed
positions sends copy-through to exactly 0.

**Not exact string match.** An amendment can change a deadline, a defined
term and a cross-reference in one pass. A submission that recovers two of the
three has done most of the work; all-or-nothing throws that away.

**Not recall over changed tokens alone.** With recall only, appending a large
vocabulary to the shown text recovers the changed tokens by brute force and
scores like the oracle. The precision term over what the prediction adds
closes that.

**Not position-free token matching.** This was the previous version of the
metric, and it was measured and rejected. Most changed tokens are common ones
— `,` `.` `the` `(` `of` — so appending the 60 most frequent removed tokens
from train to the end of every section, reading nothing, scored **0.129**
position-free: far above every honest lookup baseline. Keyed by gap, the same
dump scores **0.012**, and a smarter dump placed right after the sentence a
locator flags as most likely amended scores **0.002**.

Baselines measured on the shipped split:

- blank submission — **0.000**
- copy the input through unchanged — **0.000** (the sample submission)
- append the 60 commonest removed tokens — **0.012** (adversarial, banned by rule)
- the commonest 8 tokens inserted after the likeliest amended sentence — **0.002**
- retrieve the nearest prior-wording sentence seen in train — **0.000**
- a substitution table mined from train, one confident edit — **0.003**
- the same table applied to every flagged sentence — **0.009**
- seq2seq reference (`solution.ipynb`) — **0.002**
- *(diagnostic ceiling)* replace only the single most-changed sentence with its
  true prior text — **0.749**
- *(diagnostic ceiling)* replace every amended sentence with its true prior
  text — **0.933**
- submit the true prior wording — **1.000**

### What the numbers say about the task

**Finding the amended passage is half the problem.** A character n-gram
logistic regression over sentences puts an amended sentence first only
**51%** of the time. The median amendment touches **4** sentences (mean 6.8).

**One right edit is worth most of the score.** Perfectly reverting the single
most-changed sentence already reaches 0.749; reverting all of them 0.933. A
solver that finds the right sentence and regenerates it well has most of the
available score in hand.

Every lookup-style baseline sits near zero: the split is part-disjoint, so
phrases mined from one CFR part rarely recur verbatim in another, and a table
cannot reconstruct a sentence it has never seen. The headroom belongs to
locating well and generating the prior wording.

## Submission

`submission.csv`, exactly two columns:

- `row_id` — one row per `row_id` in `test.csv`
- `text_before` — your reconstruction of the prior wording

Degradation behaviour:

- A blank or entirely wrong prediction scores **0.0** for that row.
- A `row_id` missing from your submission scores **0.0** for that row rather
  than failing the run.
- `row_id` values not in the test set are ignored.
- A duplicated `row_id` is a conflict the grader will not resolve: every row
  for it is discarded and that row scores 0.
- Only a **missing required column** is rejected outright, with
  `InvalidSubmission`.

## Compute

1× NVIDIA A10G (16 GB), 10 CPU cores, 62 GB RAM, under 1.5 hours.

The natural approach is a sequence-to-sequence model conditioned on the
amended section — the metric rewards recovering specific removed wording, so a
model that learns how this register drafts deadlines, defined terms and
cross-references will beat one that paraphrases fluently. Sections run long,
so how you window or attend over the input matters more than raw model size;
the change is usually confined to one paragraph, and finding which one is most
of the problem.
