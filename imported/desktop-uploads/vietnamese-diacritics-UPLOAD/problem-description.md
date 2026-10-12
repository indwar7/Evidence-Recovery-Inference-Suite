# Restoring Tone Marks on Vietnamese Place Names Built from Unseen Syllables

Vietnamese writes vowel quality and tone with diacritical marks. When a
Vietnamese place name passes through an ASCII-only system the marks are
stripped: `A Nông` becomes `A Nong`, `Bảo Lộc` becomes `Bao Loc`, `Cần Thơ`
becomes `Can Tho`. This task asks you to put them back — given the folded
(ASCII) form of a Vietnamese place name, produce its native spelling — for
names whose syllables the training data has, in part, never shown you.

For example, given the input `A Nong` the expected output is `A Nông`; given
`Bao Loc` the expected output is `Bảo Lộc`.

## Task

Each row of the test set gives one place name in its ASCII-folded form, in a
column named `folded`. Your job is to output, for each row, the native
Vietnamese spelling of that name — the same string with tone and vowel
diacritics added at the positions where they belong. Only the letters change
their diacritics; the number and order of characters stay the same, so the
output always has exactly the same length as the input.

Each row also carries where the place is, expressed only through other names:
its province (`province`) and district (`district`) as GeoNames names them,
and the names of the five nearest settlements in the training set
(`neighbors`, separated by a vertical bar with a space on each side). No
coordinates, codes or identifiers are given. District names almost always
carry their marks; some provinces are recorded by the source as unmarked
exonyms such as `Hanoi`, so a context field can itself be plain ASCII. The
context never contains the row's answer: neighbours are drawn from the
training split only, and any context value whose folded form contains the
row's own folded name as a phrase (a district seat named after its district,
a neighbour whose name extends the row's name) is withheld.

**How the evaluation set is built.** A fixed, deterministic subset of the
corpus's less common syllables is held out. Every place name that contains
one of those syllables is an evaluation row, and no training row's name
contains any of them. So every test name carries at least one syllable that
occurs in no training name, and no folded name appears in both splits. (A
held-out syllable can still occur inside a province or district name in the
context columns, since those are administrative names rather than training
rows.)

## What makes this hard

A folded Vietnamese syllable is highly ambiguous. Stripped of its marks, a
single syllable such as `a` can stand for `a`, `à`, `á`, `ả`, `ã`, `ạ`, `â`,
`ấ`, `ầ`, and more; `o` can stand for `o`, `ô`, `ơ` and their six tones each.
Which restoration is correct is not determined by the bare letter: it depends
on the whole syllable, on the surrounding name, and on conventions of
Vietnamese place naming that have to be learned from examples.

- **Memorisation cannot reach the held-out syllables.** A table that maps each
  training syllable to its most frequent marked form has no entry for the
  held-out syllable in any test name, and a lookup from whole folded name to
  native name answers nothing at all, because no test name occurs in training.
  The marks on an unseen syllable have to be inferred from what the model has
  learned about how vowels, finals and tones combine across syllables it has
  seen — generalisation below the syllable, not recall.
- **Most names need several marks.** A name usually needs more than one
  diacritic, so getting one position right is rarely enough.
- **The context helps but does not answer.** Only a minority of the marks a
  test name needs appear verbatim, on the same syllable, in its province,
  district or neighbour names. Copying marks from the context wherever a
  syllable matches, and leaving the rest untouched, scores far below a trained
  model; the remaining marks have to be inferred from the name itself and
  from what the context implies about the region.

On this split, every hand-coded method measured — per-syllable frequency
tables (including ones conditioned on the province or district), character
n-gram context tables, and copying marks from the context columns — scores
below a character-level neural model trained from scratch on the training
rows. The intended solution is such a model: it reads the whole folded name
and predicts the diacritic for each position. The shipped reference reads the
folded name alone; conditioning it on the context names is the lever the task
leaves open to you.

## What is new here

1. **Restoration on syllables absent from training, by construction.**
   Diacritic-restoration benchmarks draw their evaluation items at random from
   running text, so nearly every test word — and every test syllable — has
   been seen in training, and a per-word or per-syllable most-frequent table is
   the strong baseline. Here the split holds out syllable types, not items:
   every evaluation name contains a syllable that occurs in no training name.
   The task therefore measures sub-syllabic generalisation of an orthography,
   which a lookup table structurally cannot provide.
2. **Proper names with no dictionary crutch.** The items are place names with
   no dictionary entry and no running-text context; a word list resolves
   nothing.
3. **Restoration conditioned on named geography.** Each row carries the names
   of its province, district and nearest training settlements — no
   coordinates, no codes, only other names. Vietnamese toponymy is regional,
   so these names carry evidence about local spelling conventions; a model has
   to learn which marks transfer from a neighbouring name to the target and
   which do not.
4. **A score that will not pay for copying, or for ignoring the context.**
   Only the positions that must change are scored, so a copy of the input
   earns nothing; and the marks a row's context could have supplied are judged
   separately from those it could not, so a name is credited only when both
   kinds are handled.
5. **A real archive, no annotation anywhere.** Both forms of every name, the
   administrative parents and the neighbour relations are the source's own
   recorded fields and coordinates; no label was created, inferred from
   unstructured text, or generated by a model.

## How it relates to prior work

The distant antecedent is **diacritic and accent restoration** — re-inserting
marks stripped from text — and, more broadly, character-level **grapheme
transduction**. Those tasks operate on running text in a known language,
benefit from a dictionary of whole words, and evaluate on randomly drawn
items, so the standard strong baseline is a per-word most-frequent-restoration
table. This task differs in input (out-of-vocabulary proper names plus named
geographic context), in the split (held-out syllable types, which null both
the word table and the syllable table on the held-out syllables), and in the
metric (only the restored positions are scored, split by whether the context
could have supplied them). It is not transliteration (no change of script)
and not phonetic transcription (the target is the recorded spelling, not the
pronunciation).

## Dataset

Discover the data directory at run time rather than hardcoding a path:

```python
from pathlib import Path
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/restoring-vietnamese-diacritics/public"))
            if (p / "test.csv").exists())
```

- `train.csv` — columns `id`, `folded`, `province`, `district`, `neighbors`,
  `name`. `folded` is the ASCII-folded place name; `province` and `district`
  are the names of its administrative parents (empty where GeoNames records
  none or where the parent's name contains the row's own name); `neighbors`
  holds the names of the nearest training settlements, separated by a spaced
  vertical bar; `name` is the native spelling of the place (the target). Use
  these rows to train.
- `test.csv` — columns `id`, `folded`, `province`, `district`, `neighbors`.
  Produce a native spelling for each `id`. Test neighbours are also drawn
  from the training set, never from other test rows.
- `sample_submission.csv` — columns `id`, `name`, with `name` set to the
  folded input (the copy baseline). Replace the `name` values with your
  predictions.

## Rules

**Use only the data directory.** The folded names are real, public place
names, so in principle an external place-name database could be consulted to
look up the native spelling. Recovering the answer that way is not solving the
task: a submission that uses any external gazetteer, place-name database, map
service or other outside resource is invalid regardless of its score, and so
is any submission that depends on data not shipped in the data directory. The
solver environment is expected to have no network access; your solution must
run to completion from the data directory alone.

**Predictions must come from parameters trained on the GPU from the training
pairs.** Hand-coded rules, character n-gram or context tables, most-frequent
per-letter or per-syllable substitution (whether or not conditioned on the
province or district), nearest-neighbour or dictionary lookup, and any other
method with no trained neural parameters are not valid solutions, and a
trained model may not be reduced to a fallback behind such a method. Copying
marks from the `province`, `district` or `neighbors` columns onto matching
syllables is such a lookup and is likewise not a valid solution on its own.
Heuristic features, including anything derived from the context columns, may
be fed to a trained model as input.

- Learn from the provided data only. No pretrained models, no pretrained
  embeddings, no LLMs, and no external data of any kind.
- Build any character vocabulary from the training data only, or use a fixed
  character set. Do not fit statistics on the test inputs as a corpus; predict
  each test row from its own input plus the training data.
- No LLM-generated outputs may be used anywhere in your solution.
- Use only libraries available in the Kaggle Python Docker image.
- Your notebook must run end to end, top to bottom, and be reproducible; seed
  everything.

What not to use: pretrained models or embeddings, external gazetteers,
place-name or geographic databases, map or geocoding services, word lists or
dictionaries, network access, any lookup of the native spelling outside the
data directory, and statistics or vocabularies fitted on the test inputs. The
names are real and public, so nothing in the data can make an external lookup
physically impossible; this rule, together with the held-out-syllable split,
is what closes that route.

## Evaluation

Scoring looks only at the positions that must change — the letters that
should receive a diacritic — and ignores the characters the folded input
already has right. A submission that leaves a name unchanged therefore earns
nothing for it, and a submission equal to the true spelling earns full credit;
in between, credit is given for each required mark supplied correctly and
withheld for each mark added where none belongs or added wrongly.

Within a name, the grader distinguishes two kinds of required mark: those
whose syllable appears, already correctly spelled, somewhere in the row's
province, district or neighbour names, and those that do not. Each kind is
judged on its own, and a name's score balances the two so that a submission
is credited only when it handles both — restoring the marks the surroundings
hand you while missing the rest earns as little as the reverse. A name whose
required marks are all of one kind is judged on that kind alone. The overall
score is the average over all graded names.

Worked example, for the true name `Tân Hòa` (folded `Tan Hoa`), where a
neighbouring training place is named `Tân Phú` (so the mark on `Tân` is one
the context could supply) and `Hòa` appears in no context value:

- `Tân Hòa` (fully correct) earns full credit.
- `Tân Hoa` gets the context-supplied mark but misses the other: no credit.
- `Tan Hòa` gets the other mark but misses the context-supplied one: no
  credit.
- `Tan Hoa` (a copy) earns nothing.

A copy of the input, a blank, and a lookup from folded form to native form
built on the training rows all score at the floor.

## Submission

Write `working/submission.csv` with exactly two columns, `id` and `name`, one
row per test `id`. `name` is your predicted native spelling for that row.

Degradation: a missing row, an unknown or duplicated `id`, a blank prediction,
or a prediction whose length differs from the folded input scores zero for
that row; the rest of the submission is still scored. Only a submission that
is missing a required column is rejected outright.

## Compute

The solver hardware is a single NVIDIA A10G GPU (16 GB), 10 CPU cores and 62 GB
RAM, with a runtime budget of 1.5 hours. The intended solution is a
character-level Transformer (or equivalent sequence model) that reads the
folded name — and, if you choose, its context names — and predicts a
diacritic for each position, trained from scratch on the training rows; it
fits comfortably within the budget. The metric rewards a model that infers
marks on syllables it has never seen from the regularities of the ones it
has, which is what a trained sequence model over characters provides and a
syllable or n-gram table does not.
