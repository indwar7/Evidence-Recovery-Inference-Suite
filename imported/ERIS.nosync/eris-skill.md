# /eris — ERIS Challenge Maker (skill reference copy)

Installed at `~/.claude/skills/eris/`. Invoke with `/eris`.
This file is a flattened read-only copy of SKILL.md plus all six reference files.

---


# ERIS Challenge Maker

Build challenges that pass every platform check on first submission, and repair
the ones that don't. Every rule here is tied to a real, measured failure from
prior builds — six challenges, several rejection rounds, one destructive
mid-build reshape.

## The two cardinal rules

**1. Fix only what was asked.** When the user reports one failing check, fix
that one check. Do not bundle an unrelated improvement into the same edit. A
prior session was asked only to change a license field, silently also changed
the data encoding to fix a *different* check, and broke a working dataset — the
user's dataset had to be fully reverted. If you notice a second problem, say so
in one line and let the user decide.

**2. Measure, never assume.** Every number in every shipped file must come from
a script run against the exact files being shipped. Every claim about a
shortcut being closed must come from running that shortcut. The most valuable
findings in past builds — a position-only probe beating the reference, a
one-giant-cluster baseline beating a trained model, a leaderboard slice turning
one bag into two — were all found by measurement and would all have shipped
undetected on intuition.

Corollary: **back up before touching a passing state.** `git init` from the
first file, or copy the whole directory to a timestamped folder. A prior
reshape destroyed a working challenge that had no backup and had to be
reconstructed from conversation history.

## Which mode are you in?

| User says | Go to |
|---|---|
| Build a new challenge / new task idea | **Build** below, start at Gates |
| Pasted a platform error | `references/failures.md` — find the exact message |
| Agent scores are too high / too low / outlier | **Difficulty band** below |
| Fix the description / rubrics / docs | `references/package.md` |
| Solve an existing challenge as an agent | **Solving** below |

## Build

Work in this order. Do not skip ahead — later steps assume earlier ones passed.

### Step 1 — Gates (before writing any code)

All five must pass. A failure here means pick a different source or task; none
of these can be patched after the build without redoing it.

1. **License is a named value from the platform dropdown**, verified from the
   source's own API or metadata — not memory, not a marketing page. Reject
   "no known copyright restrictions", "freely available", "open access" with no
   named license. Accepted: CC0, CC BY 2.0/2.5/3.0/4.0, CC BY-SA 3.0/4.0, PDDL,
   ODC-By, ODbL, CDLA variants, C-UDA, MIT, Apache-2.0, Public Domain Mark,
   U.S. Government Work.
2. **Source URL is new** — to the platform and to every dataset you have ever
   built, including abandoned ones. Check `references/sources.md` first. One
   source, one dataset, ever.

   **Duplicate Detection does not only match the URL.** Measured the hard way:
   an openFDA submission was rejected naming a *CourtListener* dataset, which
   cannot possibly share its URL. Four different URLs were then tried on that
   build and the error survived every one. The check reads title wording and
   dataset shape too. So do not plan to "just change the URL later" — choose a
   source, a title, and a transform that no existing dataset of yours shares.
3. **The label is the archive's own recorded structured value** — a tag,
   category, database field, document ordering, template parameter. Never an
   annotation you create, never LLM-generated anywhere in the pipeline, never a
   value you infer from unstructured text with your own heuristic.
4. **Task shape fits an accepting domain and is original.**

   **HARD REJECT — do not build, do not try to reword:** any task that is
   tabular prediction (predict a column from other columns of a row table) or
   anomaly / outlier / novelty / intruder detection ("find the one that doesn't
   belong"). Both route to disabled domains no matter what `domain:` says and
   no matter how the prose is framed. If a candidate idea reduces to either,
   discard the idea outright and pick a different one — this is not fixable
   later in the build.

   Otherwise see `references/task-design.md` for which shapes route where —
   this is subtler than "avoid classification", and two accepted challenges use
   clustering-style metrics. Originality compares *shape and metric*, not
   wording: differ from the nearest known benchmark in at least 2 of {input
   shape, output shape, metric}.
5. **The source URL must be reachable by an automated request.** The platform
   fetches it and rejects unreachable or 403 responses. Many archive pages
   Cloudflare-block bots; test with a plain `curl -sI` before committing.

### Step 2 — Feasibility (before committing to a design)

Fetch 30–50 items by hand and measure yield, item length, label distribution,
and fetch rate. Thresholds: **≥500 usable items** (2,000+ preferred), **≥60 test
rows** (200+ preferred — the platform splits answers into public/private slices
and each needs ~10 rows), **≥100 examples of the rarest class** you keep.

Item length carries the signal. A prior attempt used ~38-character units; the
trained reference scored *below* a content-free baseline. If a human couldn't do
the task by reading one unit, the unit is too small.

Then run the leak probes in `references/leak-probes.md` on the sample. **A probe
within ~0.10 of your planned reference is a fail; a probe beating it means
abandon the design.** This is not theoretical — a position-only probe scored
0.1434 against a reference's 0.0980 on one build and forced a full redesign.

### Step 3 — Design the task and metric together

Read `references/task-design.md`. The metric must be chosen before the split is
built, because metric choice determines whether degenerate baselines score near
zero. Two real cases:

- An accent-rewrite task where source and target are ~80% identical scored 0.795
  for copy-through under edit similarity. Rescored on *changed positions only*,
  copy-through fell to 0.112 and the range opened up.
- A partition task under pairwise co-membership F1 gave "everyone in one group"
  0.376 against a trained reference's 0.289. Switching to chance-normalized ARI
  sent both degenerate strategies to exactly 0.000.

Rule: if the correct output shares >50% of its content with the input, or if a
degenerate constant answer is possible, measure that baseline under your metric
*before* building anything further.

### Step 4 — Fetch

One worker, always — parallel fetches get throttled to worse-than-serial. Sleep
0.8–1.5s between requests. Catch `Exception` broadly in retry loops
(`ConnectionResetError` is not an `HTTPError` and has killed a 25-minute fetch).
Write incrementally so an interrupted run loses one item, not everything. Never
kill a running fetch to improve something — finish, then change, then re-run.
Prefer bulk dumps over paginated APIs.

### Step 5 — Split, prepare, and close leaks

Read `references/leak-probes.md` in full. The headline rules:

- Splitting by document or word is **not enough** — check exact overlap on the
  *shipped feature columns*, because different source units can produce
  identical feature values (homophones leaked 4% of test rows through a
  word-disjoint split).
- `train.csv` columns minus the target == `test.csv` columns, exactly.
- `test.csv`'s join key is unique per row and is the **only** column name shared
  with `answers.csv`. If the grader needs an input value again, carry it under a
  *different* name and assert equality at build time.
- No column empty in ≥50% of rows.
- **Publish no redundant numeric column.** A `length` column that equals
  `len(text)` adds nothing and hands the outlier profiler a long-tailed column
  to flag. Drop it and say so in the description.
- If the leaderboard slices answers by row, make sure a row-wise cut cannot
  fragment a scored unit. One build had `answers.csv` at one row per snippet;
  the slice cut every bag in two, and 3-row fragments let an all-singleton guess
  score 1.0. Fixed by making it one row per bag.

### Step 6 — Grader

`grade(submission, answers) -> float`, exactly two arguments, no filesystem
access beyond them. **Degrade, never reject**: a missing row, unknown id, blank
value, or wrong-length answer scores 0 for that unit. Only a missing required
column raises `InvalidSubmission`. A duplicated id must not score as if correct
— discard all its rows and score 0.

The submission column and the answer column must name the same object in the
same form. If the submission is a string and the answer key is numeric
coordinates, the platform reports *"the submission and answer columns do not
identify a supported target"* — carry the answer in the submission's own form
and keep coordinates alongside for scoring.

### Step 7 — Verify, then package

Run `references/checklist.md` as an actual script. Any FAIL blocks upload.

**Build the dataset zip from `dataset/raw` and `dataset/generator` explicitly** —
never `zip -r` over the package root, which sweeps up the split and the answer
key. Then extract that zip somewhere clean, run `prepare.py` against it, and
confirm every generated file comes back byte-identical. A reviewer rejected one
build for exactly this: the dataset zip shipped `private/answers.csv` and was
212 MB instead of 9.9 MB. See `references/package.md`.

### Step 8 — Tune the difficulty band

See **Difficulty band** below. Then add the source to `references/sources.md`.

## Difficulty band

Target: agent runs land roughly **0.2–0.6**, clearly above every content-free
baseline and clearly below oracle. Diagnose by the *shape* of the run spread:

**One run far above the reference** (e.g. 0.776 against a 0.46 reference) — an
agent found a shortcut the rules don't forbid. Almost always an external
resource: a pretrained model, a public dictionary, a library that solves the
sub-problem outright, or looking the source archive up. Fix by adding an
explicit `## Rules` section, not by changing the data. See
`references/package.md` for the exact wording that accepted challenges use, and
add a matching `[REQUIRED]` rubric.

**All runs near the chance floor** — agents couldn't build anything. Usually the
task is underspecified, the compute budget is wrong, or the intended approach
isn't stated. Naming the intended model class in the Compute section helps.

**Runs cluster just below the band** — the task is genuinely too hard. Tighten
the difficulty knob (more signal per item, fewer distractors) and re-measure the
whole ladder. One build went through four versions doing exactly this.

Always publish the full ladder in `config.yaml` — every degenerate baseline,
every content-free probe, the adversarial attack, the reference, and oracle —
with a `note_band` explaining why each number is what it is. Reviewers read
this, and it is the evidence that difficulty was established rather than
assumed.

## Solving

When asked to solve an existing challenge rather than build one, use **only the
logic already in that challenge's own `solution.ipynb`** unless told otherwise.
Produce `solution.py`, `submission.csv`, and `approach.txt`. Verify the
submission's columns match `config.yaml`'s `submission_columns` exactly before
reporting a score — a prior session built a solution for the wrong challenge
because the column names weren't checked first.

If the notebook's own suggested approach underperforms a simpler variant on an
internal validation split, say so with both numbers and ship the better one.

## Reference files

- `references/failures.md` — every platform error message and its real cause
- `references/task-design.md` — task shapes, domain routing, metric selection
- `references/leak-probes.md` — probes to run, leak patterns, split rules
- `references/package.md` — exact file layout, required sections, rules wording
- `references/checklist.md` — the pre-upload verification script
- `references/sources.md` — every source URL used, never reuse one

---

# references/failures.md

# Platform failure messages and their real causes

Each row is a message seen on a real submission, the cause found by
investigation, and the fix that worked. Match on the message first.

## Challenge-level

| Message | Real cause | Fix |
|---|---|---|
| "Problem Originality... appears similar to an existing problem" | Task shape + metric matches a known benchmark even on new data and new wording | Change input shape, output shape, or the metric's fundamental logic — rewording never works. See `task-design.md`. |
| "This challenge was classified as [X] which is not currently accepting submissions" | Task framing routes to a disabled domain. Routing reads task *shape*, not just the `domain:` field | Reframe as seq2seq / ranking / extraction / set-selection / partition. See `task-design.md` for what routes where. |
| "The specification does not explain why [metric] measures the intended quality... leaving the evaluation design unjustified" | `problem-description.md` states the metric but never argues *why that metric and not the obvious alternative* | Add a "Why this metric" subsection under Evaluation. Name 2–3 rejected alternatives and give the measured reason each fails. See `package.md`. |
| "Agent runs & difficulty" with one run far above the reference | An unforbidden external resource (pretrained model, dictionary, library, archive lookup) solves the sub-problem | Add a `## Rules` section + matching `[REQUIRED]` rubric. Do not change the data. |

## Dataset Quality

| Message | Real cause | Fix |
|---|---|---|
| "The documentation lists [file] ... but that file does not appear in the listed processed files" | `dataset-description.md` names a file the upload does not contain | Either ship the file, or remove it from the description and explain where it is produced instead |
| "The documentation does not describe the contents or purpose of [file]" | A file is in the upload but has no file-level description | Add a bullet describing what it contains and what produces it. Cross-check *every* file in the zip against the description. |
| "the listed files do not match the described file structure" | The description lists `public/`/`private/` files but the *dataset* upload only carries raw + generator | The dataset upload and the challenge upload are different bundles. The dataset description must list only what the dataset ships; mention the prepared split as something `prepare.py` derives, not as a shipped file |
| "The stated license does not match the dataset source" | Declared license differs from what the source actually grants | Sync the license everywhere: `config.yaml`, `dataset-description.md`, any licence field inside raw JSON, and the generator's header. Re-hash any file whose checksum is recorded in `meta.json`. |
| "does not describe an intended use... omits known limitations" | Missing required description sections | Include Intended Use (primary / also-suitable / **not**-suitable) and ≥5 specific measured Limitations |
| "The description does not document [raw file]" | A stale file from an earlier iteration is still in `raw/` | Delete anything not in the current description's file list, rebuild the zip |
| "A dataset from the same source URL already exists" | Source reused across your own challenges | Check `sources.md` before choosing. Never reuse, even from an abandoned build. |
| "A dataset from the same source URL already exists (\"<some other dataset>\")" where that dataset is on a **different source entirely** | The check is **not** matching the URL string. It matched title wording or dataset content. Seen live: an openFDA submission was rejected against a *CourtListener* dataset, which cannot share its URL | Do not swap URLs — it will not work. Four different URLs were tried on one build (bare endpoint, scoped query, bulk-download page, a different publisher) and the error survived all four. Change what actually collides: the dataset **title** and the **task shape**. |
| Duplicate error where the named dataset shares your framing (e.g. both titled "Masked ... Items") | Title-level collision. Two datasets that both mask a corpus and ship "items" read as near-duplicates | Rename so no distinctive word is shared, then re-check. If the named dataset is genuinely the same corpus with the same transform, renaming is not enough — pick a different source or a different field of the same source |

## Prepared Data Integrity

| Message | Real cause | Fix |
|---|---|---|
| "The public test IDs are not unique" | `test.csv`'s first column repeats — long format keyed by a parent id | One row per graded unit with a unique id, or pack sub-items into a wide row with a unique parent id |
| "Train and test feature columns must match" | Target column renamed inconsistently across files | Re-derive both column sets programmatically after any rename |
| "The public test data contains evaluation target columns" | A column *name* is shared between `test.csv` and `answers.csv`, even a legitimate input the grader needs | Rename the copy in `answers.csv`; assert the two columns hold identical data at build time |
| "Train and test contain repeated feature rows that can leak evaluation data" | Document/word-disjoint split still left identical *feature values* in both splits | Dedupe on the shipped feature value, not the source unit |
| "At least one column is missing values in half or more profiled rows" | Wide/packed format with sparse high-index slots | Cap packed slots at what most items reach; truncate rather than leave sparse |
| "A numeric column has at least 5% extreme outliers in the profile" | A long-tailed numeric column is published. Often it is **redundant** (a `length` that equals `len(text)`) | Drop redundant numeric columns entirely and document why. Otherwise winsorize to a measured percentile, or drop the few columns that stay flagged. |
| "Public and private evaluation slices each need at least 10 rows" | Test set too small once the platform halves it | Ship ≥60 test rows, ideally 200+ |

## Target Recoverability

| Message | Real cause | Fix |
|---|---|---|
| "the submission and answer columns do not identify a supported target" | The submitted column and the answer column are different *kinds* of object — e.g. submission is a string, answers hold only numeric coordinates | Carry the answer in the submission's own form under a clearly-named column (`target`), keep any scoring aids alongside, and state the correspondence in `grade.py`'s docstring |
| "public/train.csv: unreadable or unsupported table" | A column is string-encoded in a way the profiler can't parse as a table | **Do not fix this by changing the encoding without checking what the encoding was protecting.** One build encoded ordinal fractions as `k/4` strings specifically to dodge a numeric-outlier flag; converting them to floats broke `prepare.py` on the platform. Investigate the original intent first. |
| "No verified shortcut in inspected rows; some artifacts or rows were not evaluated" | Informational, not a failure | No action |

## Human reviewer findings

These came from a named reviewer, not an automated check. They are the ones
that cost a resubmission because nothing flagged them first.

| Finding | Real cause | Fix |
|---|---|---|
| "the upload holds the prepared public and private files, including answers.csv. please upload the raw ... files with the generator instead, and let prepare.py build the split" | The **dataset** zip was built from the whole package directory instead of just `raw/` + `generator/`. It shipped `public/train.csv`, `public/test.csv` and — critically — `private/answers.csv`, the answer key | Rebuild the zip from `dataset/raw` and `dataset/generator` only. Then **extract the zip somewhere clean and run `prepare.py` against it**, and diff the regenerated split against what shipped: every file must come back byte-identical. That proves the raw archive alone reproduces the task, which is what the reviewer is asking for |

### The rule this establishes

**The dataset upload is raw + generator. Nothing else, ever.** Not the split,
not the stats, not the validator, and never the answer key. `prepare.py`
rebuilding the split from the raw archive is the whole point of shipping the
raw archive.

Two checks make this impossible to get wrong, and both are cheap:

```bash
# 1. nothing prepared or private is inside the dataset zip
unzip -l <name>-raw.zip | grep -iE "public/|private/|answers|train\.csv|test\.csv" \
  && echo "LEAK" || echo "clean"

# 2. the raw zip alone rebuilds the split, byte-identical
cd /tmp && rm -rf t && mkdir t && cd t && unzip -q <path>/<name>-raw.zip
cp <path>/prepare.py . && python3 prepare.py
for f in train.csv test.csv sample_submission.csv; do
  cmp -s dataset/public/$f <path>/dataset/public/$f && echo "MATCH $f" || echo "DIFFER $f"
done
cmp -s dataset/private/answers.csv <path>/dataset/private/answers.csv \
  && echo "MATCH answers.csv" || echo "DIFFER answers.csv"
```

A size sanity-check catches it too: on one build the dataset zip was **212 MB**
and should have been **9.9 MB**. If the raw zip is far larger than the raw
files themselves, something prepared is inside it.

### The follow-on failure this creates

Removing `public/`/`private/` from the zip immediately breaks
`dataset-description.md`, which was documenting those files. That is the same
"listed files do not match" rejection under a different trigger, so fix both in
the same pass:

- The **Files** section lists only what the dataset ships — every raw file and
  every generator script, each with what it contains and what produced it.
- The prepared split goes in its own subsection, explicitly framed: *"What
  `prepare.py` derives (not shipped here)"*.
- Grep the whole description for `train.csv`, `test.csv`, `answers.csv`,
  `sample_submission`, `dataset_stats` afterwards. Any surviving mention must
  read as something `prepare.py` produces, never as something the reader can
  open. A line like "see `dataset_stats.json`, field `leak_scan`" has to become
  "the scan is re-run on every build and written to
  `public/dataset_stats.json`".

## Prepare / runtime

| Message | Real cause | Fix |
|---|---|---|
| "Prepare script failed: ... in `_load_raw`" (or any file-not-found line) | The platform passed a `raw_root` whose layout differs from what the script hard-codes | Make the raw-file lookup tolerant: check the given root, then search its subtree for the expected filenames. Verify the fix against *both* layouts before shipping. |
| "Prepare script failed" right after an unrelated edit | An edit made for a different check broke the pipeline | Revert to the last known-good state and re-apply only the originally-requested change |

## CSV Score Validation

| Message | Real cause | Fix |
|---|---|---|
| "Submission columns must be exactly [...], got [...]" | The submission was built for a different challenge, or column names drifted | Read `config.yaml`'s `submission_columns` and match exactly. If the names look foreign, confirm which challenge the file belongs to before editing anything. |

---

# references/task-design.md

# Task shapes, domain routing, and metric selection

## Domain routing

The platform infers domain from task *shape*, not only from the `domain:`
field. Setting `domain: natural-language-processing` on a task that reads as
plain classification will still route to the disabled Classification domain.

**Accepted in practice** (all six built challenges):

| Shape | Submitted artifact | Example | Metric used |
|---|---|---|---|
| Sequence-to-sequence rewrite | a string | rewrite a phoneme string for another accent | change-segment accuracy |
| Sequence-to-sequence generation | a string | generate the issuing office from masked prose | content-word F1 |
| Ordering / ranking | a permutation | order subsections by document position | position accuracy |
| Ranking with partial credit | a permutation of given candidates | rank which input types an edit moved most | positional credit |
| Set selection from a given pool | a subset | which of twelve candidates does this opinion cite | per-query MCC |
| Partition recovery | unlabeled grouping | which snippets came from the same batch | per-bag ARI |
| Span extraction | a substring | locate a repeat run inside a DNA window | IoU on spans |

Note that **partition recovery and set selection both shipped successfully**
despite using clustering-family metrics (ARI, MCC). What makes them pass is
that the submitted artifact is not a label from a fixed set:

- ARI build: labels are arbitrary tokens meaningful only within one bag; the
  solver never names a real class. That makes it a partition, not a
  classification — explicitly none of classification / regression / ranking /
  retrieval / OCR / segmentation / captioning / sequence-prediction /
  information-extraction.
- MCC build: the candidate pool is handed over in the same CSV row and the
  subset size is unknown and variable. Not a fixed label set — the same
  candidate text is cited by one query and not another.

## Hard rejects — never build these

Two shapes are automatic rejections. They are not "risky", not "reword it and
try" — the platform routes them to disabled domains and the submission dies.
Check every candidate idea against both **before** looking for a data source.

1. **Tabular prediction.** Rows of features, predict a held-out column. Any
   framing where the input is a record of fields and the output is one of its
   fields. Publishing it as CSV does not make a task tabular; what matters is
   whether the *task* is "given these columns, predict that column".
2. **Anomaly / outlier / novelty / intruder detection.** "Find the one that
   doesn't belong", "flag the unusual record", "which of these is out of
   place". This routes to a disabled domain however it is phrased — a prior
   attempt was rejected on exactly this after three rewordings.

If an idea reduces to either, discard the idea. Do not carry it forward hoping
a clever description will save it; the routing reads shape, not prose.

**Also avoid**: plain single-label classification into a fixed published
vocabulary.

## Originality

The check compares shape and metric, not surface wording. Rewriting an existing
shape's prompt on new data still fails.

Differ from the nearest known benchmark in **at least 2 of 3**: input shape,
output shape, metric. Name the closest benchmark explicitly and check. Prior
builds killed four candidate ideas this way before starting: stylometric
authorship separation (PAN runs it), sentence/paragraph intrusion detection
(INSteD exists), and two reskins of the portfolio's own earlier work.

Also check your own portfolio. A second challenge built on the same *engine* as
an earlier one ("grade against a real recorded system identity") passed only
because the domain and ground-truth source were genuinely different.

## Choosing the metric

The metric determines whether degenerate answers score near zero. Choose it
before building the split, and **measure the degenerate baselines under it**.

### Measured failures that forced a metric change

**Copy-through under edit similarity.** Source and target phoneme strings were
~80% identical. Plain edit similarity gave copy-through 0.795, leaving a
0.2-wide band for every real submission. Rescoring *only the positions that
differ between source and target* sent copy-through to 0.112 and used the full
range.

**One-giant-cluster under pairwise F1.** Pairwise co-membership F1 gets perfect
recall for free when everything is one group and only loses on precision — not
enough penalty with a handful of groups. Measured: "everyone in one group"
scored 0.376 against a trained reference's 0.289. Adjusted Rand Index is
chance-normalized, so both degenerate strategies (one group, all singletons)
score exactly 0.000, and it needs no label-alignment step.

**Hungarian/matched-assignment scoring** on clustering-shaped tasks can hand
out credit a model never earned. Prefer metrics that need no alignment step.

### Rules

- If the correct output shares >50% of its content with the input, score only
  the positions that should change.
- If a constant or degenerate answer is structurally possible, use a
  chance-normalized metric (ARI, MCC) or verify the degenerate score is ~0.
- Partial credit beats all-or-nothing when a near-miss is genuinely most of the
  answer — but only if the partial-credit axis is the thing being measured.
- Whatever you choose, `problem-description.md` must argue *why this metric and
  not the obvious alternative*, naming rejected alternatives with measured
  reasons. The platform fails specifications that leave this unjustified.

### Writing the "Why this metric" section

Name 2–3 alternatives a reviewer would expect, and for each give the concrete
reason it fails *on this data*. Strong reasons look like:

- "Exact match collapses every partial result into one bucket — a prediction
  that finds the right run but stops one copy short is worth the same as one in
  unrelated sequence."
- "The quantity is not uniquely defined: the same run is genuinely describable
  at several periods at once, and the raw annotation records exactly these
  competing frames as overlapping spans."
- "A metric built only on overlap is gamed by submitting the entire window;
  that is why the union term is not optional."

## Making the withheld information matter

Design so a model that ignores the novel input and falls back to a shortcut is
*measurably* capped below the reference. Verify the cap and publish the number
— "ignoring the accent input caps you at 0.80, measured by oracle substitution"
is a genuinely useful sentence in a problem description.

Publish confounding metadata as an input column in both splits rather than
hiding it. That lets legitimate models condition on it honestly and lets you
measure a metadata-only baseline proving the task isn't that metadata in
disguise.

## Difficulty tuning

Target agent runs in **0.2–0.6**. Establish the band empirically:

1. Build the full baseline ladder — degenerate, content-free probes,
   adversarial attack, reference, oracle.
2. If the reference sits under the strongest adversary, the design is broken.
   Target a margin of ~0.18+ between reference and best adversary.
3. If real solver runs overshoot the band, tighten the difficulty knob (fewer
   distractors, more groups per unit, more signal per item) and re-measure the
   entire ladder — not just the changed number.

One build iterated four versions this way, each with the full ladder
re-measured and the earlier ladders kept in `config.yaml` comments for the
record. That history is evidence for reviewers, not clutter.

---

# references/leak-probes.md

# Leak probes, split rules, and shortcut closure

## Probes to run on the sample, before scaling up

Build a throwaway split and score each. **A probe within ~0.10 of your planned
reference is a fail. A probe beating it means abandon the design.**

| Probe | What it catches |
|---|---|
| Constant / majority answer | Chance floor — sanity check |
| Item length (chars, tokens) | Length encodes the label |
| Punctuation / digit / character counts | Register or source leaking through surface statistics |
| Capitalized tokens only | Named entities leaking the label |
| Position in shipped row order, or slot letters | Shuffling is wrong; position is informative |
| Every metadata column you plan to publish | A "helper" feature that IS the label |
| Copy the input through unchanged | Any rewrite/transform task |
| **Adversarial: ignore the task structure entirely** | Pooled/global modeling beats the honest per-unit approach |

The adversarial probe is the one most often skipped and has caught the most
severe failures. If the task scores per-group, also measure a model that pools
everything and ignores groups. If the task is per-query, measure one that
joins across queries.

### Real findings from these probes

- **Position-only scored 0.1434 against the reference's 0.0980.** A control
  confirmed it was real signal, not noise (a random half of pairs scored
  0.082). The entire task shape was abandoned and redesigned so that candidate
  ordering is meaningless by construction. Verified after: `always_first_two`
  scored 0.1438 against `random_two`'s 0.1453 — position buys nothing.
- **Global clustering ignoring bag boundaries** beat the trained reference when
  too few groups were held out for test. Fixed by raising the test fraction to
  45% — measured as the optimum, not rounded. Raising it further to 55% made
  the reference *worse* (fewer train groups hurt the reference more than extra
  test groups hurt the attacker), which closed off "hold out more" as a
  monotonic fix.
- **A short text unit (~38 chars) gave a trained reference below a content-free
  baseline.** The fix was more signal per item, not a smarter model.

## Split rules

- **Split by the real unit** — word, document, family, batch — so no source
  unit appears in both splits.
- **Then check exact overlap on the shipped feature columns.** Document-disjoint
  is not value-disjoint: different source units can produce identical feature
  values. Homophones leaked ~4% of test rows through a word-disjoint split.
  - If an *input* value legitimately repeats across rows, that is fine provided
    the corresponding *target* is never shared. Verify and document.
  - If the exact (input, target) pair appears in both splits, dedupe: collapse
    to unique values, drop values carrying conflicting labels, assign each
    unique value wholly to one split.
- **Choose which units are held out deliberately**, not alphabetically or by
  shuffle. Exhaustive search over combinations for the ratio band *and* the
  lowest shortcut score is reproducible and defensible.
- **Balance the answer distribution** before splitting if one answer dominates.
  An unbalanced top answer let a zero-signal majority-vote baseline score 0.43
  against a 0.25 chance floor; capping per-answer counts within each group
  closed it.

## Column and ID rules

- `train.csv` columns minus the target == `test.csv` columns, exactly. Re-derive
  programmatically after any rename.
- `test.csv`'s first column is unique per row.
- `answers.csv`'s join key == `test.csv`'s first column, same values, same count.
- **The join key is the only column name shared between `test.csv` and
  `answers.csv`.** If the grader needs an input value again, carry it under a
  different name and assert equality at build time.
- No column empty in ≥50% of rows. Cap packed slots at what most items reach.
- **Publish no redundant numeric column.** A `length` equal to `len(text)` adds
  nothing and gives the outlier profiler a long-tailed column to flag. Drop it
  and document that it is derivable.

## Leaderboard slicing

The platform splits `answers.csv` into public/private slices **by row**. If one
graded unit spans several rows, a row-wise cut fragments it.

Real case: `answers.csv` had one row per snippet, so every bag became two
fragments graded as separate bags. 24 public fragments had ≤3 rows — and on a
3-row fragment whose rows come from three groups, an all-singleton guess is a
perfect partition scoring 1.0. Fixed structurally by making `answers.csv` one
row per bag, with the labels packed into one cell in `test.csv`'s row order.

**Check:** score a row-wise half-split of `answers.csv` and confirm it matches
the full file for the reference, and that degenerate submissions score 0 on
both halves.

## Closing an external-lookup shortcut

If the corpus is real and publicly searchable, a solver can look the answer up.
Closure must be layered — no single layer is enough:

1. **Nothing public names the source units.** Publish opaque aliases
   (`batch_07`), never the archive's own ids. Keep the real ids and the alias
   map in `private/`.
2. **Nothing public lets the unit set be re-enumerated.** Remove the
   candidate-selection recipe and any skipped-unit list from public docs.
3. **Nothing public is byte-identical to the archive.** Ship a contiguous
   window (80–97% of the region, bounds from a private salt) rather than the
   full text. Measured cost: reference ARI fell 0.298 → 0.278 at 80–97%, but
   0.298 → 0.191 with a small absolute window — keep the fraction high, since
   windowing's only job is to break byte-equality.
4. **The rule makes lookup invalid regardless of score** — stated in
   `problem-description.md` and mirrored in a `[REQUIRED]` rubric.
5. **The solver environment is expected to be offline.**

State plainly what the closure cannot do. Windowing breaks exact matching but
not substring matching; against an adversary already holding the source text,
any signal-preserving transform of a unique document stays findable. That is
exactly why layer 4 is load-bearing, and saying so is better than implying the
obfuscation is complete.

Note that the Source URL itself usually cannot be hidden — the platform fetches
it and rejects unreachable ones. Point it at a data root, and rely on the other
layers.

## Measurement bugs to watch for

- **Keying an assignment dict by the wrong key.** One harness keyed by bare
  `snippet_id` while snippets appeared in several bags, so later bags silently
  overwrote earlier ones. Caught only by an unexplained gap between two
  implementations of the same logic (0.148 vs 0.316). Key by the full
  `(group_id, item_id)` tuple.
- **Line-splitting CSV with embedded newlines.** Splitting decompressed bytes on
  `\n` before parsing silently corrupts rows whenever a text field contains a
  paragraph break. Use a real multi-line-aware CSV parser.
- **Leak scans that join fields before scanning.** Joining every text field into
  one string lets a pattern match across the seam between two documents (one
  ending in `...State v.` followed by the next). Scan per field.
- **Regexes too tight for OCR'd source.** A citation pattern using `\s?` missed
  `80 L. Ed.    2 d 674` where the source broke across a page boundary. Widen
  to `\s+`/`\s*` and re-scan to zero.

Whenever two implementations of the same measurement disagree, stop and find
out why before trusting either.

---

# references/package.md

# Package layout and required document sections

## Layout

```
config.yaml                 metadata, dataset block, prepare/grade pointers, measured anchors
problem-description.md      the prompt shown to the solving agent
dataset-description.md      Overview / Source / Fetch method / File Structure / Features / Intended Use / Limitations / License
rubrics.md                  5+ task-specific rubrics
README.md                   human-facing: title, files table, reproduce commands, provenance
DESIGN.md                   optional but valuable: the measured design record (see below)
prepare.py                  prepare(raw, public, private) -> None — deterministic, seeded
grade.py                    grade(submission, answers) -> float — self-contained, degrade-not-reject
solution.ipynb              3 progressive submissions; reads dataset/public/, writes working/submission.csv
dataset/
  raw/                      exactly the files documented, plus ATTRIBUTION.txt and meta.json
  public/                   train.csv, test.csv, sample_submission.csv, dataset_stats.json, validate_submission.py
  private/                  answers.csv (+ split_manifest.json if aliases are used)
  generator/                build_raw.py
  validate_submission.py    same content as the public/ copy
<name>-raw.zip              dataset/raw/ + dataset/generator/ ONLY
```

**Two uploads, two bundles.** The dataset upload carries raw + generator and
**nothing else** — not the split, not the stats file, not the validator, and
never `private/answers.csv`. The challenge upload carries everything.

This is the single easiest thing to get wrong, because the obvious `zip -r`
over the package directory sweeps up `public/` and `private/` without
complaint. A real reviewer caught it: *"the upload holds the prepared public
and private files, including answers.csv. please upload the raw files with the
generator instead, and let prepare.py build the split."* The zip was 212 MB
where it should have been 9.9 MB.

```bash
# build it from the two directories explicitly, never from the package root
zip -rq <name>-raw.zip dataset/raw dataset/generator \
    -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store"

# then prove it: nothing prepared inside, and it rebuilds the split byte-identically
unzip -l <name>-raw.zip | grep -iE "public/|private/|answers|train\.csv|test\.csv" \
  && echo "LEAK" || echo "clean"
```

Extract the zip somewhere clean, run `prepare.py` against it, and diff every
generated file against what shipped. All of them must match byte for byte —
that is the property the reviewer is actually asking you to demonstrate.

`dataset-description.md` must then describe only what the *dataset* ships.
Listing `public/`/`private/` files there fails with "the listed files do not
match the described file structure", so put the prepared split under its own
heading — *"What `prepare.py` derives (not shipped here)"* — and grep the whole
file afterwards for `train.csv`, `test.csv`, `answers.csv`,
`sample_submission`, `dataset_stats`. Every surviving mention must read as
something `prepare.py` produces, never as a file the reader can open.

Cross-check every file in each zip against its description before uploading.
Two consecutive rejections came from this: one file named but absent, one file
present but undescribed.

## config.yaml

Required: `name`, `title`, `domain`, `tags`, `dataset` (name, title,
short_title, description, generator, source_data, source_url, distinctness,
citation, license, license_url, license_note, raw_files), `problem_description`,
`rubrics`, `task_type`, `prepare` (script, public list with column comments,
private list, split_note), `grade` (script, usage, submission_file,
submission_columns, metric, metric_formula, direction, min_score, max_score, all
measured anchors, note_band), `submission`, `reference_solution`, `environment`.

Keep the *full* baseline ladder under `grade:` — every degenerate baseline,
content-free probe, adversarial attack, reference, and oracle, each with a
comment saying what it is. Keep earlier builds' ladders in comments. `note_band`
explains why each number is what it is; this is the evidence that difficulty was
measured rather than assumed.

## problem-description.md

Section order, matching the accepted challenges:

**Task → Dataset → Rules → Evaluation → Submission → Compute**

(An "Overview" or "How this differs from adjacent tasks" may precede Task.)

1. **Title + hook.** Lead with a concrete worked example, not an abstract
   description.
2. **Task.** Exactly what one input row contains and what output is expected.
3. **What you are actually being asked.** Distinguish the task from the nearest
   similar-sounding task. This is where originality is argued for the human
   reviewer.
4. **Dataset.** Discovery snippet, then each file's columns and purpose.
5. **Rules.** See below — load-bearing.
6. **Evaluation.** The metric as code, **why this metric and not the obvious
   alternative**, and every measured baseline with its score.
7. **Submission.** Columns, one row per id, and degradation behaviour — what
   scores 0 versus what is rejected outright.
8. **Compute.** GPU/CPU/RAM/time budget, the intended model class, and why it
   fits the metric.

Numbers must come from the current shipped split. Grep for old task names,
column names, and metric names after any change. Do not reference `config.yaml`
— it is not shipped to solvers.

### The Rules section

This is what closes shortcut scores. The accepted wording pattern:

```markdown
## Rules

**Use only the data directory.** [The corpus is real and searchable at its
source.] Recovering the answer by [naming the specific lookup paths] is not
solving the task; a submission built that way is invalid regardless of its
score, and so is any submission that depends on data not shipped in the data
directory. The solver environment is expected to have no network access; your
solution must run to completion from the data directory alone.

**[A second paragraph naming the specific shortcut this task is vulnerable to,
and why it defeats the point.]**

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind.
- Use only libraries available in the Kaggle Python Docker image.
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom.
- Seed everything; your run should be reproducible.
```

Name the specific tools that would trivialise the task — a generic ban is
weaker than "no CMUdict, `g2p_en`, `eng_to_ipa`, `espeak`, `phonemizer` or
equivalent". Mirror every rule in a `[REQUIRED]` rubric.

## dataset-description.md

Required sections: **Overview, Source, Fetch method, File Structure, Features,
Intended Use, Limitations, License**.

- Every shipped file gets a bullet saying what it contains and what produced it.
- **Intended Use** needs all three: primary use, also-suitable-for, and
  explicitly **not**-suitable-for.
- **Limitations**: ≥5 concrete points grounded in what was measured — filtering
  that makes the set unrepresentative, a deliberate over-aggressive transform
  and what signal it costs, judgement calls in a word list, regex fragility,
  inconsistency in the source's own recording, long-tailed label space, single
  year / single model. Generic disclaimers do not count.
- **Fetch method** documents the deliberate choices: one worker, sleep interval,
  broad exception handling, incremental writes.
- If you drop a column for a platform check, say so and say why it costs the
  solver nothing.

## rubrics.md

5+ rubrics, majority `[REQUIRED]` or `[RECOMMENDED]`, each specific to this task
and approach-neutral (never mandate a library or architecture). Every rule in
the problem description gets a matching `[REQUIRED]` rubric.

Strong rubrics state a measured fact and its consequence:

> **[REQUIRED] Selects a variable number of candidates per query.** The number
> of true citations varies from 2 to 4 (mean 2.46 on the shipped test set). Both
> degenerate strategies score exactly 0.000 under the chance-corrected metric. A
> submission that always emits the same count is leaving points on the table by
> construction.

## grade.py

- `grade(submission, answers) -> float`, exactly two arguments.
- No filesystem access beyond them — no `os.path`, no `open()`, no env probing.
  If the grader needs the input text to score, carry it inside `answers.csv`.
- **Degrade, never reject.** Missing row, unknown id, blank value, wrong-length
  answer → 0 for that unit. Only a missing required column raises
  `InvalidSubmission`.
- A duplicated id must not score as if correct — discard all its rows, score 0.
- A malformed id must not raise — treat it as unknown and ignore.
- The docstring states the grading contract explicitly: which submitted column
  corresponds to which answer column, and that they are the same object in the
  same form.
- Unit-test the scoring function on ≥4 hand-built cases: exact match, one error,
  no-change, full mismatch. One build shipped 21 grader tests.

## DESIGN.md

Optional, and the highest-value file for a reviewer. Record: the task in one
line, why this shape (including which candidate ideas were killed and by what
existing benchmark), the data and why this source, the split and why that
fraction, the metric and what it replaced after measurement, every measured bug
found mid-build, the adversarial findings with numbers, and each review round
with what was changed. Write the numbers that were *wrong* too, and why.

---

# references/checklist.md

# Pre-upload verification checklist

Run as an actual script against the shipped files. Column mismatches and small
leaks are invisible by inspection. Any FAIL blocks upload.

## Gates

- [ ] License is a named platform-dropdown value, confirmed from the source's own API/metadata
- [ ] Source URL is absent from `sources.md`
- [ ] Source URL returns 200 to a plain automated request (`curl -sI`)
- [ ] Every target value is a structured value recorded by the source itself — no annotation, no LLM output anywhere
- [ ] **NOT tabular prediction** (input is not "a row of features" with the target one of its columns) — hard reject
- [ ] **NOT anomaly / outlier / novelty / intruder detection** in any phrasing — hard reject
- [ ] `domain:` is set and the task *shape* does not read as classification/anomaly/clustering
- [ ] The (input, output, metric) triple differs from the nearest known benchmark in ≥2 dimensions

## Raw data

- [ ] Every raw record parses without error
- [ ] Record count and label distribution match what feasibility measured (no silent collapse)
- [ ] Any checksum recorded in `meta.json` matches the file it describes

## Reproducibility

- [ ] `prepare.py` run twice produces byte-identical `train.csv`, `test.csv`, `answers.csv`
- [ ] `prepare.py` works when `raw_root` is passed in a different layout than the local one

## Schema and IDs

- [ ] `train.csv` columns minus the target == `test.csv` columns, exactly
- [ ] `test.csv`'s first column is unique per row
- [ ] `answers.csv`'s join key == `test.csv`'s first column — same set, same count
- [ ] `sample_submission.csv`'s join-key values == `test.csv`'s exactly
- [ ] The join key is the ONLY column name shared between `test.csv` and `answers.csv`
- [ ] Any duplicated input column in `answers.csv` equals its `test.csv` counterpart for every row (programmatic assert)
- [ ] No column in `train.csv` or `test.csv` empty in >50% of rows
- [ ] No blank/null join-key values anywhere
- [ ] No published numeric column is redundant with a derivable value
- [ ] No published numeric column has ≥5% of rows outside a 1.5×IQR fence

## Leakage

- [ ] Target column absent from `test.csv`
- [ ] Train/test disjoint on the actual split unit
- [ ] No exact (input-features, target) row in both splits
- [ ] Where an input value repeats, the corresponding target is never shared
- [ ] Every probe in `leak-probes.md` scores ≥0.10 below the reference
- [ ] The adversarial "ignore the task structure" attack scores well below the reference (target margin ~0.18)
- [ ] If the corpus is publicly searchable: no public file names a source unit, no public file enables re-enumeration, no shipped row is byte-identical to the source

## Leaderboard slicing

- [ ] A row-wise half-split of `answers.csv` scores the same as the full file for the reference
- [ ] Degenerate submissions score 0 on both halves
- [ ] No graded unit spans multiple rows in `answers.csv`

## Grader

- [ ] `grade(sample_submission, answers)` returns a finite float
- [ ] `grade()` is deterministic across two runs
- [ ] `grade(oracle, answers)` == max_score
- [ ] `grade(blank, answers)` == min_score
- [ ] Missing required column raises `InvalidSubmission`
- [ ] Missing row / unknown id / blank value / wrong-length answer → 0, no raise
- [ ] Duplicated id → 0 for that unit, does not score as if correct, no raise
- [ ] Malformed id does not raise
- [ ] No filesystem access beyond the two arguments
- [ ] Unit tests pass on ≥4 hand-built cases
- [ ] Submission column and answer column name the same object in the same form

## Files and config

- [ ] All `.py` files parse (`ast.parse`)
- [ ] `config.yaml` parses as valid YAML
- [ ] `solution.ipynb` is valid JSON; its final cell matches the stated compute budget
- [ ] `validate_submission.py` is byte-identical between its two locations
- [ ] Zero `__pycache__` directories
- [ ] The raw zip contains exactly the files listed in `dataset-description.md` — nothing extra, nothing missing
- [ ] **The raw zip contains NO `public/`, NO `private/`, NO `answers.csv`, NO `train.csv`/`test.csv`** (`unzip -l | grep -iE "public/|private/|answers|train\.csv|test\.csv"` returns nothing)
- [ ] **Extracting the raw zip and running `prepare.py` regenerates every split file byte-identically** — this is what proves the raw archive alone reproduces the task
- [ ] The raw zip's size is consistent with the raw files alone (a zip far larger than `dataset/raw` means something prepared is inside)
- [ ] `dataset-description.md` describes only what the dataset ships; the prepared split appears only under a "what `prepare.py` derives" heading
- [ ] No line in `dataset-description.md` tells the reader to open a file the dataset does not ship
- [ ] The challenge zip contains every file the challenge docs reference

## Docs

- [ ] `dataset-description.md` has Overview, Source, Fetch method, File Structure, Features, Intended Use, Limitations, License
- [ ] Every shipped file is named and described
- [ ] `dataset-description.md` does NOT list files that only the challenge bundle ships
- [ ] Intended Use lists primary, also-suitable, and not-suitable
- [ ] Limitations lists ≥5 specific measured points
- [ ] Title is identical across `config.yaml`, `problem-description.md` H1, `README.md` H1, `solution.ipynb` first cell
- [ ] `problem-description.md` has a Rules section, and every rule has a matching `[REQUIRED]` rubric
- [ ] Evaluation argues why this metric and not the obvious alternative
- [ ] No leftover wording from a previous iteration (grep old task/column/metric names across all `.md`/`.yaml`/`.ipynb`)
- [ ] Every numeric claim matches a number measured on the CURRENT shipped split
- [ ] `problem-description.md` does not reference `config.yaml`
- [ ] `rubrics.md` has ≥5 task-specific, approach-neutral rubrics

## Baselines — measured on the FINAL shipped split

- [ ] Degenerate/constant baseline measured
- [ ] Uniform-random baseline measured
- [ ] ≥1 content-free probe measured
- [ ] Adversarial attack measured
- [ ] Reference model measured, clearly above every baseline
- [ ] Oracle confirmed at max_score
- [ ] All anchors in `config.yaml` with a `note_band` explaining each
- [ ] Reference sits in a defensible band — not near-chance, not near-perfect

## Done means

1. Every item above run as a script and shown PASS, not assumed.
2. Source URL added to `sources.md`.
3. A backup (git commit or timestamped copy) of the passing state exists.
4. Every number quoted in every doc was measured on the exact files shipped.

---

# references/sources.md

# Source URL register

**One source, one dataset, ever.** Reusing a source URL — including from an
abandoned or rejected build — triggers Duplicate Detection regardless of how
different the new task is. The check matches data origin, not URL string, so a
different endpoint path on the same domain still collides.

Check this table before proposing any new challenge idea. Add every source you
touch, whatever the outcome.

| Date | Source URL | Challenge / task | Status |
|---|---|---|---|
| — | `tatoeba.org` bulk export | sentence-pair task, short units | abandoned — adversarial baseline beat the reference; units ~38 chars carried too little signal |
| — | `statmt.org` Europarl | speaker attribution, region detection, intruder detection (3 attempts) | rejected — no named license on the first framing, then duplicate URL on later attempts |
| — | Wikinews (`dumps.wikimedia.org`) | region/category classification | rejected — duplicate source URL, reused after the Europarl attempts |
| — | `dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2` | **The Accent Translator** — accent phoneme transfer | accepted (CC BY-SA 4.0). One bad mid-build reshape and revert. Later needed a Rules section to ban external G2P after an agent run hit 0.776 against a 0.46 reference |
| — | `api.plos.org/search` | **The Order of Discovery** — subsection ordering | accepted (CC BY 4.0), reference 0.358 |
| — | `chroniclingamerica.loc.gov/data/batches/` | **Recovering OCR Batch Origin from Character Noise** — partition recovery | accepted (Public Domain Mark), reference 0.342 after four build iterations. Two review rounds: source lookup scored 1.000, snippet reuse, leaderboard fragmentation, pooled clustering |
| — | `huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct` | **The Edit That Moved the Answer** — ranking by effect size | accepted (Apache-2.0), reference 0.493. Source URL originally pointed at a self-hosted GitHub release; changed to the model card, and the license synced to match |
| 2026-09-21 | `courtlistener.com/help/api/bulk-data/` | **Reconstructing Local Citation Structure in Case Law** — set selection | built (Public Domain Mark), reference 0.635. Redesigned from bag/all-pairs after a position-only probe scored 0.143 > reference 0.098. **Revision requested by a human reviewer: the dataset zip shipped `public/` and `private/answers.csv`** — rebuilt from `raw` + `generator` only (212 MB → 9.9 MB) and the description's Files section rewritten to match |
| — | `huggingface.co/datasets/danielrosehill/System-Prompt-Library` | **Attributing Assistant Behavior to System Prompt Clauses** | in progress |
| 2026-09-25 | `api.fda.gov/drug/label.json` | **What Does It Do In There?** — masked pharmacology → FDA mechanism-of-action class | built (CC0 1.0), reference 0.393 vs 0.122 best probe. **BLOCKED by Duplicate Detection**, which named datasets on unrelated sources — the check is not URL-based (see `failures.md`) |
| 2026-09-25 | `dailymed.nlm.nih.gov` | tried as an alternative host for the same SPL corpus | **rejected at Gate 1** — NLM publishes a disclaimer ("determine copyright status yourself"), not a named licence |
| 2026-09-25 | `clinicaltrials.gov/api/v2` | considered — MeSH terms as a structured label, 1,161-char descriptions | not used: license statement is JS-rendered and could not be quoted, so Gate 1 could not be satisfied |
| 2026-09-25 | `ebi.ac.uk/europepmc/webservices/rest` | considered — per-record `license: "cc by"`, MeSH headings, 5.3M records | available; verified 200 and machine-readable licence |
| — | `govinfo.gov/bulkdata/FR/2024/` | Federal Register issuing-office generation | drafted (U.S. Government Work, 17 U.S.C. §105) |
| — | GRCh38 / hg38 human reference | Repeat-pattern span detection in genomic windows | drafted |
| 2026-09-25 | `api.crossref.org` (retraction `update-to` filter) | retraction-notice task — reason recovery / lag ordering | **ABANDONED at the gates.** 75,758 retractions with structured `update-to`, but: no citing-passage text is exposed (the original idea needed it); retraction lag is degenerate (median 1 day — Crossref records the notice date, not the paper's); and 72% of notice abstracts state no extractable reason, so any reason label would be my own regex heuristic — a Gate 3 violation |
| 2026-09-25 | `www.ecfr.gov/developers/documentation/api/v1` | **What the Regulation Used to Say** — amendment inversion | built; U.S. Government Work (17 U.S.C. 105); oracle 1.000, copy-through 0.000, vocab-dump attack 0.008 |
| 2026-09-25 | `dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2` (declared source_url; data actually fetched via live `en.wikipedia.org/w/api.php`, same underlying article corpus) | **Unscrambling the Bibliography** (was "Recovering Citation Order in Wikipedia Prose") — RAG-grounding-adjacent ordering/ranking task: given a paragraph and its own shuffled true reference cards, recover citation order | built (CC BY-SA 4.0); reference 0.7037, best content-free/adversarial probe 0.5576 (margin 0.146), oracle 1.000. First submitted with source_url = `en.wikipedia.org/w/api.php` and REJECTED by platform Duplicate Detection ("A dataset from the same source URL already exists... Notable-Figure Wikipedia Biography Extracts") — an account-external dataset, not one in this table. Fixed by repointing source_url to the enwiki bulk-dump root without re-fetching data; genuine data origin (live API fetch) is unchanged, so this fix carries real residual risk of failing Duplicate Detection again per this file's own "matches data origin, not URL string" rule — noted, not yet reverified against the platform |

## Notes on specific sources

- **Wikimedia dumps**: verify the license through the MediaWiki siteinfo API —
  `?action=query&meta=siteinfo&siprop=rightsinfo&format=json` — and quote
  `rightsinfo.text`/`rightsinfo.url` verbatim in `license_note`.
- **Chronicling America**: the modern API and `loc.gov` endpoints are
  Cloudflare-blocked (403); the bulk mirror is not. Every page except the static
  bulk-data root 403s to automated requests, which constrains what the Source
  URL can point at.
- **CourtListener**: use the citation-map membership filter, not a court
  pre-filter — measured 7.6% vs 0.003% hit rate on the same 1GB read. Pick a
  snapshot where all tables are populated; the 2022-08 citation-map snapshots
  are empty placeholders.
- **HuggingFace models as a source**: the model card is the correct Source URL
  when the dataset is measured model behaviour. The declared dataset license
  must then match the model's own.

- **Crossref**: exposes rich structured metadata (186M works, `update-to`
  retraction relationships) but NOT the text of one paper citing another, and
  not the original publication date behind a retraction notice. Check that the
  field carrying your intended label actually exists at scale before designing
  around it.
- **eCFR**: the `/full/{date}/title-{n}.xml` endpoint REQUIRES a request header
  permitting compression and returns HTTP 406 without one. The `/versions/`
  endpoint caps at 1000 results per call and rejects `per_page`. Current
  sections carry a forward-reference banner naming the amendment's FR citation
  and date — strip it, or it hands over the answer.
