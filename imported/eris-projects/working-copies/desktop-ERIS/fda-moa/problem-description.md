# What Does It Do In There?

Every prescription drug label carries a Clinical Pharmacology section: several
thousand words on what the drug does once it is inside a person. What it binds
to. What it blocks. Which enzyme takes it apart, and how fast.

Separately, the drug regulator assigns that same drug a **mechanism-of-action
class** from its own controlled vocabulary — *Cyclooxygenase Inhibitors*, *Proton Pump
Inhibitors*, *Cytochrome P450 3A4 Inducers*. One short phrase naming the thing
the pharmacology section spent three thousand words describing.

Here is a label with every mechanism word masked:

> *"…produces its effects through reversible inhibition of the [MASK] enzyme,
> reducing conversion of arachidonic acid to [MASK] precursors. Peak plasma
> concentrations occur within 2 to 4 hours. Plasma protein binding exceeds 99%.
> Elimination is primarily hepatic, with less than 1% excreted unchanged…"*

The class is *Cyclooxygenase Inhibitors [MoA]*. The word "cyclooxygenase" is
not in the text — it was masked. You would have to know that arachidonic acid,
prostaglandin precursors and reversible enzyme inhibition point there.

That is this task. Given masked pharmacology and a pool of twelve candidate
mechanism classes, pick the ones the regulator actually assigned.

## Task

Each row gives you:

- `route` — how the drug is administered (`ORAL`, `INTRAVENOUS`, …)
- `pharmacology` — the label's Clinical Pharmacology section, with every
  mechanism-vocabulary word replaced by `[MASK]`
- `candidates` — twelve mechanism-of-action class names, in shuffled order

Return `moa_classes`: the subset of those twelve that the regulator assigned to this
drug, pipe-separated.

**How many are correct varies per row** and is not given to you. Most drugs
have one; some have several. Deciding how many to name is part of the problem.

## What you are actually being asked

**This is not classification.** The candidate pool is handed to you in your own
row and differs from row to row. A class that is a correct answer here is a
distractor there. There is no fixed label set to learn.

**This is not keyword matching.** Measured on the raw corpus, 57% of labels
state their own mechanism verbatim in the pharmacology prose. Every one of
those words is masked — using a vocabulary built from *every* class in the
corpus, not just the row's own, so the masking pattern itself gives nothing
away. What survives is the pharmacology: substrates, binding, metabolism,
half-life, the pathway the drug acts on.

**Naming everything does not work.** The metric subtracts what a random
selection of the same size would have scored, so submitting all twelve
candidates scores exactly 0.000.

## Dataset

```python
from pathlib import Path
import pandas as pd
DATA = next(p for p in (Path("/data"), Path("dataset/public"))
            if (p / "test.csv").exists())
train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")
```

- `train.csv` — `row_id, route, pharmacology, candidates, moa_classes`
- `test.csv` — the same without `moa_classes`
- `sample_submission.csv` — `row_id, moa_classes`, naming the first candidate
- `dataset_stats.json` — split counts and what was dropped at build time;
  nothing in it is derived from test labels

Sets are pipe-separated: `Proton Pump Inhibitors [MoA]|Cytochrome P450 2C19 Inhibitors [MoA]`.

The split is **by shared active ingredient**. Many labels are the same drug
from different manufacturers, with near-identical prose and identical classes,
and combination products share ingredients with single-ingredient ones. So
any two labels sharing even one active ingredient (salt forms counted as the
same ingredient) land on the same side: no test drug shares an ingredient
with any training drug.
Exact- and near-duplicate texts are collapsed before splitting. `row_id` is
opaque.

No numeric column is published — text length is derivable from the text.

## Rules

**Use only the data directory.** These labels are real, public documents.
Recovering an answer by looking the drug up — in any drug-label archive, drug
database, pharmacologic class directory, or any mirror of them — is not
solving the task; a submission built that way is invalid regardless of its
score, and so is any submission that depends on data not shipped in the data
directory. The solver environment is expected to have no network access; your
solution must run to completion from the data directory alone.

**Do not try to unmask.** The drug's identity is withheld deliberately: brand
and generic names are not published, and mechanism vocabulary is masked.
Recovering the drug's identity from residual detail and then looking up its
class substitutes identification for the pharmacological inference being
measured.

- Learn from the provided data only. No pretrained pharmacology or biomedical
  knowledge base, no external drug corpus, no ontology lookup.
- No LLM-generated outputs may be used anywhere in your solution.
- Use only libraries available in the Kaggle Python Docker image.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.

## Evaluation

**Chance-corrected set F1.** For each row, set F1 between what you named and
what is true, minus what a random guess of the same size would have earned:

```python
F1        = 2·|P ∩ T| / (|P| + |T|)

# expected F1 of a uniformly random size-|P| subset of the row's n-candidate pool
E[F1]     = 2·(|P|·|T|/n) / (|P| + |T|)

row_score = max(0, (F1 − E[F1]) / (1 − E[F1]))
```

Mean over test rows. Range `[0, 1]`, higher is better. A class outside the
row's pool is dropped before scoring rather than penalised.

### Why chance correction and not raw F1

**Raw F1 pays for distributional luck.** The class distribution is heavily
skewed. Measured on a sample before the split was built, always naming the
single most common class scored **0.182** raw F1 while reading nothing.
Guessing the pool's most frequent few does better still. That is a large part
of the range spent on knowing which classes are common rather than on
pharmacology.

**Subtracting the expected random score removes most of that — but not all
of it.** Measured on the shipped split, a uniformly random single guess scores
**0.085** and a random pair **0.113**, not 0: the per-row score is clipped at
zero, so a lucky row counts and an unlucky one cannot go negative. Always
naming the pool's most frequent class scores **0.193**. Under raw F1 these
strategies would score far higher; the correction shrinks them to a narrow
low band, and the reference clears the strongest of them by 0.27.

**It also makes set size an honest decision.** Naming more classes raises raw
recall but raises `E[F1]` in step, so padding buys nothing — submitting all
twelve candidates scores exactly **0.000**. Under raw F1 it would score well.

**Not exact set match.** A drug with three mechanism classes where you found
two has genuinely done most of the work. All-or-nothing discards that, and
with variable set sizes it would collapse the range.

Baselines measured on the shipped split:

- blank submission — **0.000**
- all twelve candidates — **0.000**
- a random single candidate — **0.085**; a random pair — **0.113**
- a random guess of the correct size — **0.106**
- first candidate in pool order (the sample submission) — **0.070**
- most frequent class in the pool (adversarial frequency attack): k=1 **0.193**,
  k=2 **0.180**, k=3 **0.137**
- text-length-only probe — **0.116**
- reference: one-vs-rest over the masked text, intersected with the pool —
  **0.462**
- *(diagnostic)* name one guaranteed-correct class — **0.806**; two — **0.929**
- submit the true set — **1.000**

## Submission

`submission.csv`, exactly two columns:

- `row_id` — one row per `row_id` in `test.csv`
- `moa_classes` — pipe-separated class names from that row's `candidates`

Degradation behaviour:

- A blank or entirely wrong selection scores **0.0** for that row.
- A `row_id` missing from your submission scores **0.0** for that row rather
  than failing the run.
- `row_id` values not in the test set are ignored.
- A class name outside that row's pool is dropped before scoring.
- A duplicated `row_id` is a conflict the grader will not resolve: every row
  for it is discarded and that row scores 0.
- Only a **missing required column** is rejected outright, with
  `InvalidSubmission`.

## Compute

**CPU only — no GPU.** Under 1.5 hours.

Pharmacology sections run to several thousand words and the sentences that
identify a mechanism are scattered through them, so how you window or attend
over the input matters. The reference is a bag-of-words model over the masked
text — one classifier per class, scored against the row's own pool — and runs
end to end on CPU in well under a minute. There is room above it: how many
classes to name per row is a decision worth modelling rather than fixing, and
class co-occurrence is not exploited by the reference.
