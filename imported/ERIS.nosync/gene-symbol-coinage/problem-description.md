# Coining Gene Symbols from Gene Names

## Overview

Every human gene has an approved descriptive name and a short approved
symbol, both decided by the human gene nomenclature committee.

```
solute carrier family 2 member 1          ->  SLC2A1
adhesion G protein-coupled receptor E5    ->  ADGRE5
tubulin tyrosine ligase like 13           ->  TTLL13
```

The training set holds the symbols the committee had approved before 1
January 2010. The test set holds the symbols it approved from that day on.
Your model is in the committee's position: it knows every earlier decision
and has to make the next one.

## Task

Each test row gives one field, `name`, the gene's approved name. Produce
`symbol`, the gene's approved symbol, as a string.

Some test genes join a family the committee had already named. `ADGRE5`
extends a stem that is in train, and the job is to work out which designator
the new member gets. Others found a family that did not exist before the
cutoff, and the stem itself has to be coined. 43.5% of test rows have a stem
that occurs in train and 56.5% do not. Nothing in a row says which kind it
is, and this split-level figure is not published per row anywhere in the
data directory — see Rules for what that means for a valid solution.

## What makes this hard

The answer is never printed in the input. This is not acronym detection,
which finds a short form written next to its long form. Every row whose name
already contains its own symbol or stem (`BRCA1 DNA repair associated`) was
removed, and so was every test row whose symbol or stem appears as a word in
any other shipped name.

It is not a lookup within the data either. No name and no symbol is shared
between train and test. The nearest training name points to a gene's
probable family, but not to the member's designator, and it says nothing for
the 56.5% of rows whose family is new. On a later-in-time test set the
nearest training name is often from the wrong family.

Numbers in the name usually reappear in the symbol (`solute carrier family 2
member 1` gives `SLC2A1`), but where they go, and which letters surround
them, follows conventions that differ from family to family and have changed
over time.

Measured on this data with the official metric:

- Initials of the name, digits kept (the sample submission): 0.311
- Nearest training name, its stem plus this name's trailing number: 0.365
- Hand-coded rules, no learning (nearest training family if one is close,
  otherwise initials): 0.481. Shown for reference only; rule-based
  submissions are not valid, see Rules.
- Reference (trained character transformer): 0.531

## What is new here

1. The target is a coinage decision made by a real committee, on a real
   timeline. The split is the committee's own recorded approval date. Test
   is what the committee decided next, so the mix of family extensions and
   new families is the committee's, not a construction choice of this task.
2. New-family generalization is built into the test set. 56.5% of test rows
   belong to a family whose stem never occurs in train. For these rows there
   are zero training examples of the family: the model has to coin a stem
   the way the committee would, from the name alone.
3. The answer is never in the input. Rows where the name prints the symbol
   are removed, which removes the copying that makes most short-form tasks
   easy.
4. Two sub-problems with different ceilings share one metric, and nothing in
   a row says which one it is. Extending a known family is largely learnable
   from train; founding a new one is close to irreducible from the name.
   Showing a model its nearest training sibling makes it worse (0.411
   against 0.531 without it), because on a later-in-time test set the
   nearest sibling is often from the wrong family.
5. The score reads the symbol the way the nomenclature is built: stem, then
   family, then member. Credit stops at the first wrong character.

## How it relates to prior work

The classical primitive here is character-level string transduction: mapping
one string to another character by character, as in letter-to-sound
conversion. This task uses that primitive openly and does not claim to
replace it. What it adds is a target set by a committee's decisions over
time, not by a fixed rule, evaluated forward in time, so that part of the
answer follows established convention and part of it had not been invented
when the training data ends.

Gene normalization benchmarks (BioCreative) work in the opposite direction
from this task. They take a gene mention in running text and map it to an
existing database identifier, so the answer is always an entry that already
exists and is retrieved. Here the input is a clean approved name, the output
is a symbol string that is generated, not retrieved, and for most test rows
that symbol belongs to a family with no training examples. External
databases and pretrained biomedical models are forbidden, so retrieval of
any kind is not available. Abbreviation-definition recognition in biomedical
text (Schwartz and Hearst, 2003) also runs the other way: it finds a printed
short form's long form. The committee's published guidelines (Bruford et
al., 2020) describe its conventions in prose. This task asks whether those
conventions can be learned from the decisions themselves, and where they
stop being learnable.

## Dataset

Find the data directory with this snippet:

```python
from pathlib import Path
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/gene-symbol-coinage/public"))
            if (p / "test.csv").exists())
```

- `train.csv`: 13,863 rows. `row_id` (str), `name` (str), `symbol` (str, the
  target).
- `test.csv`: 2,270 rows. `row_id` (str), `name` (str).
- `sample_submission.csv`: `row_id`, `symbol`. The initials of each name with
  its digits kept. It scores 0.311.
- `dataset_stats.json`: split sizes, median name/symbol lengths, and the rows
  removed at build time, by reason. It does not carry the seen/new-stem
  counts above — those describe the task as a whole and are not derivable
  per row from anything shipped, so they cannot be used to calibrate a
  validation split against the hidden test composition.
- `validate_submission.py`: an offline structural check of your submission
  file.

Only protein-coding genes are included. Row order and `row_id` carry no
date.

## Rules

**Use only the data directory.** The names in this task are real, public
gene names, so the answer for any test row can in principle be looked up.
Recovering a symbol by searching for the name in any gene nomenclature
resource or gene database (NCBI Gene, Ensembl, UniProt, OMIM and the like),
or through any gene-annotation package (mygene, biomaRt, pyensembl,
org.Hs.eg.db, gseapy gene-set libraries, pybiomart or equivalent), is not
solving the task. A submission built that way is invalid regardless of its
score, and so is any submission that depends on data not shipped in the data
directory. Nothing in the data can make a lookup impossible, so this rule is
what closes that route. The solver environment is expected to have no
network access, and your solution must run to completion from the data
directory alone.

**Do not use any model or resource that already knows human gene symbols.**
Pretrained language models and biomedical embeddings (BioBERT, PubMedBERT,
BioWordVec, SciBERT and the like) have read the literature in which every
approved symbol appears next to its name. Using one turns coinage back into
recall.

**[REQUIRED] Trained GPU parameters must materially drive every
prediction.** A hand-coded rule table (nearest training family if one is
close, otherwise initials) scores 0.481 on this split; the reference, a
trained character transformer, scores 0.531. A valid submission's score must
come from parameters fit by gradient descent on the GPU against the training
pairs — not from a rule table, a nearest-neighbour lookup, or a trained
model used only as a fallback behind one of those. Concretely: (a) a
solution with no `torch`/`jax`/equivalent training loop that updates weights
is invalid regardless of score; (b) a solution whose predictions are
unchanged when its trained model's output is replaced with random noise, for
rows where the rule table would have fired, is invalid; (c) reporting both a
rule-table score and a trained-model score and submitting whichever is
higher is invalid — submit the trained model's own output.

**[REQUIRED] The character vocabulary (or tokenizer) is built from the
training names and symbols only**, before `test.csv` is read for anything
other than producing the submission. Two concrete violations to avoid:
reading `test.csv` to compute a character set or vocabulary size "for
padding" before training starts; and fitting any frequency statistic — a
smoothing table, an embedding initialisation, a coverage check — on the test
names as a corpus. If your tokenizer would raise on an unseen character, use
a fixed, sufficient character set (upper- and lower-case Latin letters,
digits, space, hyphen) rather than reading `test.csv` to discover one; every
character in this task's test names already occurs in train.

**[REQUIRED] Model selection and validation must not use the held-out
answers, directly or in aggregate.** This includes: looking up any specific
test row's approved symbol while choosing between candidate models,
architectures, seeds, or ensembling strategies; computing any statistic over
`test.csv` combined with the (unpublished) answers, such as a seen/new-stem
ratio, and using that statistic to shape a validation split, early-stopping
criterion, or model choice. A valid validation split is carved out of
`train.csv` alone, for example by holding out the most recently approved
training symbols as a time-respecting validation set.

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Predict each test row from its own name plus the training data. Do not fit
  statistics on the test names as a corpus.
- Your notebook must run end to end, top to bottom: load the data, train,
  then write `submission.csv`.
- Seed everything; your run should be reproducible.

**What not to use:** pretrained models or embeddings of any kind,
task-specific or gene-annotation libraries, external word lists, gene lists
or nomenclature tables, any network access, any lookup of the answer outside
the data directory, and statistics or vocabularies fitted on the test names.

## Evaluation

Prefix agreement, averaged over all test rows. Range 0 to 1, higher is
better.

Both strings are upper-cased and stripped of whitespace. A row's score is
the number of leading characters your prediction shares with the approved
symbol, counted up to the first character that differs, divided by the
length of the longer of the two strings. A blank prediction scores 0.

- `SLC2A1` against `SLC2A1` scores 1.000: exact.
- `SLC2A3` against `SLC2A1` scores 0.833: right subfamily, wrong member.
- `SLC9` against `SLC2A1` scores 0.500: right stem only.
- `GLUT1` against `SLC2A1` scores 0.000.

Dividing by the longer length means padding past the answer costs score, and
so does stopping early after a correct stem.

Why prefix agreement. A symbol is built from the left: stem, then family
number, then subfamily, then member. Agreement from the left is agreement
down that hierarchy.

- Rejected: edit similarity. It pays for characters in the wrong place.
  Copying the whole name through scores 0.201 under it, against 0.061 under
  prefix agreement.
- Rejected: exact match. It scores `SLC2A3` and `GLUT1` the same against
  `SLC2A1`, although one found the family and the other did not.

## Submission

A CSV named `submission.csv` with exactly the columns `row_id,symbol`, one
row per `row_id` in `test.csv`, in any order.

```
row_id,symbol
te_00000,TTLL13
te_00001,ADGRE5
```

A missing row, a blank prediction, or a duplicated `row_id` scores 0 for
that row rather than voiding the submission. Unknown `row_id`s are ignored.
Case and whitespace inside a prediction are ignored. Only a file missing the
`row_id` or `symbol` column is rejected. Run
`python validate_submission.py submission.csv` to check the structure
offline.

## Compute

- 1x NVIDIA A10G (16 GB), 10 CPU cores, 62 GB RAM, 1.5 hours.
- The intended model class is a character-level encoder-decoder transformer
  trained from scratch on the 13,863 training pairs, which fits comfortably
  in the budget. A character decoder suits the metric because it writes the
  symbol left to right, the order in which prefix agreement gives credit.
