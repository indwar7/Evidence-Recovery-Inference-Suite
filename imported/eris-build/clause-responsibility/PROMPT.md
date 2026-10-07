# Finish this ERIS challenge: which sentence of a system prompt is doing the work

You are picking up a half-built ERIS benchmark challenge. The design is locked
and the code is written, but the GPU generation that produces the data has
never completed, so nothing downstream has been measured. Your job is to run
it, fix the defects listed below, measure everything, and hand back a package
that passes the platform's checks on the first upload.

Everything you need is in this folder. `ERIS-GUIDE.md` is the owner's playbook
for this platform, written from real rejections; treat it as binding. Where
this prompt and the guide disagree, this prompt is newer.

- Domain: `prompt-engineering`. This is a GPU challenge.
- You need a CUDA GPU (an A10G is the target). Do not attempt the generation
  on a laptop: it was tried on an 8 GB Mac and the machine went down.

## The task being built

A system prompt is several instructions stacked together: a persona line, a
scope rule, a tone rule, a fallback. When the assistant behaves a certain way,
which of those sentences is responsible, and which are just sitting there?

The ground truth is a real experiment, not an opinion. For each of 145 real
system prompts, a language model answers 8 probe questions under the full
prompt, then again with each sentence removed in turn. Every answer is reduced
to 8 numbers (length, hedging, refusal, bullets, and so on); no answer text is
kept. How far the behaviour moves when a sentence is removed is that
sentence's responsibility.

The solver sees a prompt's sentences in shuffled order plus the measured
behaviour under the full prompt. It submits a ranking of the sentences by
responsibility and a `necessary` / `correlated` flag for each.

## What is in this folder

```
PROMPT.md                     this file
ERIS-GUIDE.md                 the platform playbook (read before changing anything)
STATE.md                      the original design record, 2026-09-20
config.yaml                   challenge config; every anchor is still TBD
problem-description.md        written 2026-09-20 against the OLD metric
dataset-description.md        written 2026-09-20; describes raw files
rubrics.md                    written 2026-09-20
solution.ipynb                written 2026-09-20; never run on real data
dataset/
  generator/gen_raw.py        the generator, patched 2026-09-27 (see below)
  prepare.py                  builds the split; has known defects (see below)
  grade.py                    the metric; has a known defect (see below)
  raw/ public/ private/       all empty
work/corpus_cache/            the 923 source prompts, already downloaded
```

## Where things stand

| Step | State |
|---|---|
| Licence, source URL, label source | Verified 2026-09-27 |
| Source corpus downloaded | Done: 923 of 923 files |
| Corpus filter | Done: 145 prompts kept |
| Generator bugs | Fixed, not yet exercised by a full run |
| GPU generation, about 8,000 runs | **Not done** |
| Labels, split, metric | Code exists, defective, unmeasured |
| Baseline ladder and reference solution | Not started |
| Documents, notebook, zips | Stale or not started |

Verified facts you can rely on:

- The Hugging Face API reports `license: cc-by-4.0` for
  `danielrosehill/System-Prompt-Library`, ungated, and the dataset page
  returns HTTP 200.
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` is `apache-2.0`, ungated.
- The filter keeps 145 prompts and drops 440 for naming the author, 286 for
  numbered, bulleted or code-fenced structure, and 52 for sentence count
  outside 3 to 10.

Everything else in this prompt marked "by reading the code" or "by arithmetic"
has not been measured on real data. Measure it before you act on it.

## Decisions the owner has already made

Do not reopen these.

- Source corpus, its CC BY 4.0 licence, and the Hugging Face page as the
  source URL.
- SmolLM2-1.7B-Instruct as the measuring instrument, greedy decoding.
- Prompts that name the author are excluded, not rewritten.
- No response text is stored or released anywhere.
- Each item shows its sentences in shuffled order, so position in the original
  prompt carries nothing.
- The split is by topic group, never by row.
- **No TF-IDF anywhere**, in any baseline or reference. Use a trained
  transformer encoder. This is a standing rule from the owner.
- Problem descriptions are written as a full specification: overview, data
  schema, column lists, exact submission columns, the scoring formula,
  measured baselines, compute limits.
- Fix what is listed here. Do not add deliverables nobody asked for.

One decision is open and belongs to the owner: **the title**. The old title,
"Attributing Assistant Behavior to System Prompt Clauses", predates the
owner's rule that every challenge title be distinct and interesting, and
"Clause" now collides with another of their challenges ("The Vanished
Clause"). Proposed replacement: **"The Load-Bearing Sentence"**. Use it
unless the owner says otherwise, and keep the title identical in
`config.yaml`, the H1 of `problem-description.md` and `README.md`, and the
first notebook cell.

## Step 1: run the generation

```bash
pip install torch transformers accelerate numpy pandas scikit-learn
cd dataset/generator

# smoke test: 5 prompts, 248 generations. Write it somewhere disposable.
python gen_raw.py --out /tmp/smoke --smoke --batch 8

# full run: 145 prompts, about 8,000 generations
python gen_raw.py --out ../raw --batch 16
```

The corpus is read from `work/corpus_cache`, so nothing is downloaded. The run
writes a checkpoint every few batches and resumes from it if interrupted.

It must produce exactly four files in `dataset/raw/`: `ablation.json`,
`grid.json`, `meta.json`, `ATTRIBUTION.txt`. If `_checkpoint.json` is still
there at the end, the run did not finish.

What was fixed in the generator on 2026-09-27, for your information:

- It referenced an undefined name, `cells`, while writing `meta.json`. The
  run would have crashed after all the generation was done.
- It had no checkpoint; any interruption lost everything.
- The corpus cache sat inside `dataset/raw/` and would have shipped in the
  dataset zip. It now lives in `work/`.
- It wrote no attribution file, which CC BY 4.0 requires.

Before trusting the output, check that ablation changes anything at all: for
what share of (prompt, sentence, probe) cells does the ablated fingerprint
differ from the full-prompt one? If most cells are identical, the labels are
noise and the challenge cannot ship in this form. Report the number.

## Step 2: fix the label and split code (`dataset/prepare.py`)

**Distances are computed in raw units.** Found by reading the code: the
distance between two fingerprints is a plain Euclidean distance over raw
values, so word count (tens) swamps the yes/no fields (0 or 1). Standardise
each of the 8 dimensions across all per-probe cells first, then take the
distance. Print each dimension's share of the total distance before and after
to confirm the fix did what it should.

**The ten probe subsets are near-copies of each other.** Computed: the code
takes the first 10 of the 70 four-of-eight combinations, and all ten contain
probes 0 and 1. Any two share 2 or 3 of their 4 probes (mean 2.62). Replace
them with these 14, where any two share either 0 or 2 probes:

```
(0,1,2,3) (0,1,4,5) (0,1,6,7) (0,2,4,6) (0,2,5,7) (0,3,4,7) (0,3,5,6)
(1,2,4,7) (1,2,5,6) (1,3,4,6) (1,3,5,7) (2,3,4,5) (2,3,6,7) (4,5,6,7)
```

They are the four-element subsets of 0 to 7 whose values XOR to zero.

Then measure the thing `STATE.md` warned about: for each prompt, across its
items, how often does the top-ranked sentence change? If it is the same
sentence in nearly every item, the items are one answer repeated, and the
honest fix is fewer items per prompt, not more. Two disjoint subsets per
prompt gives 290 items, which still clears the test-set minimum.

**Every item needs a usable flag set.** The `necessary` threshold is the 70th
percentile of scores across the corpus. Make sure every shipped item has at
least one `necessary` sentence and at least one `correlated` one; drop the
items that do not, and report how many.

**Ship no numeric columns.** The `fp__*` columns are counts with long tails,
and the platform's profiler flags any numeric column with 5% or more extreme
values. Put the 8 measurements in one string column, `behaviour`, holding a
small JSON object, and say plainly in the dataset description why. Do not
winsorise and do not invent per-column encodings; an earlier challenge went
that way and it caused a second failure.

**Keep the seed out of public files.** The platform flagged "Generator seed in
public/dataset_stats.json" on another of the owner's builds. No file in
`public/` may contain a seed.

**Make raw-file lookup tolerant.** The platform may pass `prepare()` a root
one level above or below the local layout, and an earlier build failed with
file-not-found for exactly that reason. Look in the given directory, then
search beneath it for `ablation.json` next to `grid.json`.

**Size rules.** At least 60 test rows, test-to-train row ratio between 15% and
25%, and `prepare.py` run twice must produce byte-identical files.

## Step 3: fix the metric (`dataset/grade.py`)

By arithmetic, not yet measured: about 70% of sentences are `correlated`, so
answering "all correlated" earns about 0.70 on the flag half, and any ranking
at all earns 0.50 on the ranking half in expectation. The sample submission
would score about **0.60** while reading nothing. The platform wants agent
scores between 0.2 and 0.6, so a floor of 0.60 leaves no usable range.

Measure that number on real data first. Then correct both halves for chance:

```
ranking half   r = 1 - position_of_true_top / (n - 1)        in [0, 1]
               rank = 2r - 1                                 random gives 0

flag half      P = sentences submitted as necessary, T = truly necessary
               F1 = 2|P ∩ T| / (|P| + |T|)
               E  = 2(|P||T| / n) / (|P| + |T|)    F1 of a random same-size pick
               flag = (F1 - E) / (1 - E)

item score     0.5 * rank + 0.5 * flag
final score    max(0, mean of item scores)
```

Whatever form you settle on, these must hold on the shipped files, measured:

| Submission | Required score |
|---|---|
| The true answers | exactly 1.0 |
| Blank | 0.0 |
| Shown order, all `correlated` (the sample submission) | about 0 |
| Shown order, all `necessary` | about 0 |
| Random ranking, random flags | about 0 |

Keep the existing degradation behaviour: a missing row, unknown id, blank
field or invalid ranking scores 0 for that item and never raises. Only a
missing required column raises `InvalidSubmissionError`. One thing to change:
a duplicated `item_id` currently keeps its first row. It must instead discard
every row for that id and score 0, so a duplicate can never score as correct.

## Step 4: close the shortcut that matters most

A solver with a GPU could download the same model, ask the same 8 questions,
remove each sentence, and recompute the answer key exactly. Decoding is
greedy, so the result is reproducible. Nothing in the current documents
forbids this, and it is the single most likely reason an agent run would land
far above the reference.

Close it in layers, because no single layer is enough:

1. The probe questions appear in no public file and not in the problem
   description. They stay in the raw archive.
2. The problem description does not name the model. Say "an instruction-tuned
   language model". The name stays in the raw archive and `config.yaml`.
3. A Rules section states that generating responses from any language model to
   reproduce the measurements is invalid regardless of score.
4. A matching `[REQUIRED]` rubric.

Pretrained text encoders stay allowed; the reference uses one. The rule is
against re-running the experiment, not against pretrained models in general.

## Step 5: measure the ladder, then build the reference

Measure every one of these on the final shipped split:

- blank; the sample submission; all `necessary`
- random ranking with random flags
- shown position only: rank the sentences in the order displayed
- **sentence length only: longest sentence first**
- sentence count and topic only
- the reference
- the true answers

The length probe is the one to watch. Removing a long sentence removes more of
the prompt, so length alone may predict responsibility. If any probe that
reads no meaning lands within 0.10 of the reference, the task is a shortcut
and must not ship as it is. Stop and report; do not tune the reference until
the gap appears.

The reference: a pretrained transformer encoder over each sentence together
with the behaviour values, trained to score responsibility, on GPU. It should
land between 0.2 and 0.6 and clear the strongest probe by at least 0.10.
Tune any threshold on a slice of train that is split by prompt, never by row;
a row-wise slice puts the same prompt on both sides and reports an inflated
number.

Deliver `solution.py`, `submission.csv` and an updated `solution.ipynb` whose
final cell names the GPU explicitly.

## Step 6: rewrite the documents against the final numbers

All four documents predate these fixes. Bring each in line with what shipped.

- `problem-description.md`: section order Task, Dataset, Rules, Evaluation,
  Submission, Compute. Under Evaluation, a "Why this metric" part that names
  the alternatives rejected and the measured reason for each. Every baseline
  with its measured score. No mention of `config.yaml`.
- `dataset-description.md`: Overview, Source, Fetch method, File Structure,
  Features, Intended Use, Limitations, License. It describes only what the
  dataset upload contains, which is the four raw files and the generator. The
  split appears once, under a heading that says `prepare.py` derives it and it
  is not shipped. Intended Use states what the data is not suitable for.
  At least five limitations, each grounded in something measured.
- `rubrics.md`: at least five, specific to this task, none naming a library.
- `config.yaml`: every anchor filled from a real run. No `TBD` anywhere.
- `README.md` and `DESIGN.md`: add them. `DESIGN.md` records what was
  measured, including the numbers that turned out wrong.

For the dataset title, avoid the words "Fingerprint", "Masked" and "Items",
and leave the model name out of it. The platform's duplicate check has
rejected the owner's uploads by matching title wording against their earlier
datasets, not only the source URL, and one earlier dataset is titled "Prompt
Edit Behavioural Fingerprint".

## Step 7: package

Two zips, built from named directories, never from the package root.

```bash
zip -rq clause-raw.zip dataset/raw dataset/generator \
    -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store"

unzip -l clause-raw.zip | grep -iE "public/|private/|answers|train\.csv|test\.csv" \
    && echo LEAK || echo clean
```

Then prove the dataset zip is sufficient: extract it somewhere empty, copy in
`prepare.py`, run it, and compare every generated file with the shipped one.
All must match byte for byte. A reviewer rejected one of the owner's datasets
because the zip contained the prepared split and the answer key.

## Done means

- `dataset/raw/` holds exactly the four files, and `meta.json`'s hash matches
  `ablation.json`.
- Every row of the required-score table in Step 3 holds, measured.
- The strongest meaning-free probe sits at least 0.10 below the reference.
- The reference is between 0.2 and 0.6.
- At least 60 test rows, ratio between 15% and 25%, byte-identical rebuild.
- No numeric column and no seed in any public file.
- The only column name shared by `test.csv` and `answers.csv` is `item_id`.
- No `TBD` in `config.yaml`; every number in every document comes from a run
  on the shipped files.
- The dataset zip holds raw and generator only, and rebuilds the split
  byte for byte.
- A copy of the passing state exists (a git commit is enough).

## What to report back

1. Generation: cell count, elapsed time, device, and the share of ablated
   cells that differ from the full prompt.
2. How often the top sentence changes across a prompt's items, and how many
   items per prompt you shipped as a result.
3. The old metric's sample-submission score on real data, next to the new one.
4. The full ladder, one line per entry.
5. The reference score and its margin over the strongest probe.
6. Anything in this prompt that turned out to be wrong.

If the length probe, or any other meaning-free probe, comes within 0.10 of the
reference and you cannot open the gap, say so plainly and stop. A challenge
that reports an honest failure is worth more to the owner than one that ships
and is rejected.
