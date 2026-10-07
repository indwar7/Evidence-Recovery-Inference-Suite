# ERIS Challenge Builder — Complete Playbook (NLP Domain)

A step-by-step guide for building an ERIS challenge from scratch that passes
every platform check on the first submission. Written from real attempts:
multiple challenges rejected post-build (license, duplicate source,
originality, domain routing, ID leakage, column mismatch), one reshaped mid-
build and reverted after breaking something that already worked, and one
finally accepted. Every rule below is tied to a specific, real failure.

**Domain for all challenges built from this guide: `natural-language-processing`.**
Do not use `classification`, `anomaly-detection`, `outlier-detection`, or any
other domain label — these are disabled on the platform and will fail
Domain Routing even if every other check passes. If your task is fundamentally
a classification, anomaly-detection, or clustering problem, reframe it as an
NLP task (sequence generation, sequence-to-sequence, ranking, retrieval) or
choose a different task idea. Do not fight this by relabeling the domain field
only — the platform infers domain from task shape as well as the field.

**Core principle: the platform validates AFTER a complete package is
uploaded. Every fact below must be verified BEFORE writing any file, not
discovered after.**

---

## PART 0 — Hard Gates (verify in the first 20 minutes, before any code)

Do not proceed past this section until all five pass. A failure here means
pick a different source or task — do not "fix it later," these cannot be
patched after the fact without redoing the whole build.

### 0.1 License must be a named license in the platform's dropdown

Accepted values: CC0 1.0, CC BY 2.0, CC BY 2.5, CC BY 3.0, CC BY 4.0,
CC BY-SA 3.0, CC BY-SA 4.0, PDDL, ODC-By, ODbL, CDLA Permissive 1.0/2.0,
CDLA Sharing 1.0, C-UDA, MIT, Public Domain Mark, U.S. Government Work,
Custom (conditional), Other (conditional).

**Verify from the source itself**, not from memory or a third party:
- Wikimedia projects: call the MediaWiki `siteinfo` API —
  `https://<site>/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json`
  — and read the returned `rightsinfo.text` and `rightsinfo.url` verbatim.
- Academic/publisher sources: read the `<license>` block in the actual
  per-item metadata (e.g. a PLOS article's own JATS XML), not the publisher's
  marketing page (marketing pages can 403 or describe a different policy).
- Government/API sources: read the API's terms-of-service page and look for
  an explicit named license, not "open data" or "freely available" language.

**Reject immediately if the source says any of:**
- "We are not aware of copyright restrictions" — this is a disclaimer of
  ignorance, not a license grant.
- "Released for reuse" / "freely available" / "open access" with no named
  license attached.
- "Custom" or "Unknown" with no further specificity — these are marked
  Conditional on the platform and carry rejection risk; avoid unless no
  named-license source exists for the task you want.

### 0.2 Source URL must be new — to the platform AND to every prior dataset you built

Same source URL as **any** existing dataset — including your own earlier,
even rejected, submissions — triggers a Duplicate Detection rejection
regardless of how different the new task is. This has happened repeatedly:
reusing Europarl for a second challenge failed; reusing Wikinews for a third
failed.

**Rule going forward: one source, one dataset, ever.** Keep a running list
(see Part 6) of every source URL you have used across every challenge, even
abandoned ones, and never reuse an entry from that list. Do not attempt to
disguise a reused source with a different endpoint path on the same domain —
the check appears to match on the underlying data origin, not just the exact
URL string.

### 0.3 The label must be the archive's own recorded structured value

The prediction target must already exist in the source as structured data
written by the source's own contributors/editors/system — a tag, a category
assignment, a database field, a document's own ordering, a template
parameter. It must NOT be:
- An annotation you create for this dataset.
- Any LLM-generated content, anywhere in the pipeline — this is an absolute
  platform rule, not a style preference.
- A label you infer or compute from unstructured text with your own
  heuristic (that is feature engineering, not ground truth, and belongs in a
  baseline, not the answer key).

Examples that satisfy this (from prior builds): a `{{IPA|en|...|a=RP}}`
template's accent tag in Wiktionary; a `[[Category:North America]]`
assignment in Wikinews; a `<sec><title>Results</title>` document position in
a PLOS article; a `<SPEAKER NAME="...">` tag in Europarl proceedings.

### 0.4 Task shape must be NLP, must not resemble an existing challenge, and must survive an "output shape" test

Two independent checks here, both required:

**(a) Domain fit.** The task must be a genuine natural-language-processing
problem: operating on text, producing text or a label derived from text
content, evaluated with an NLP-appropriate metric. Concretely favor:
- Sequence-to-sequence generation (text → text, one language/form → another)
- Sequence classification WHEN the classes are inherently about language
  content (not: "is this an outlier", "is this the odd one out" — those
  route to a disabled domain regardless of your framing)
- Ranking, retrieval, or ordering tasks over text units
- Structured extraction or transformation of text

Avoid entirely: outlier/anomaly/intruder detection (routes to a disabled
domain even when phrased carefully — "find the one that doesn't belong" was
rejected this way), plain single-label classification with no generative or
relational component (routed to a disabled "Classification" domain in one
attempt), clustering/partitioning tasks scored with a clustering metric like
ARI (reads as a known challenge shape even on new data).

**(b) Originality.** The platform's originality check appears to compare
*task shape and metric*, not surface description. Rewriting the prompt text
of an existing shape (e.g. describing partition-recovery-with-ARI in new
words, on new data) will still fail. To pass, change what the model must
predict and how it is scored — the input/output types, or the scoring
function's fundamental logic — not just the domain or wording.

Ask: "does any public benchmark already pose this exact input → output →
metric triple?" If you can't immediately think of one, that's a good sign,
but verify by trying to name the closest existing benchmark and checking it
differs in at least two of {input shape, output shape, metric}.

### 0.5 Do not reshape a working task to fix ONE failing check without re-verifying everything else

This is the single most expensive mistake made building these challenges. A
task passed every check except originality; instead of sharpening the
framing, the entire task was reshaped to a different input/output structure.
The reshape fixed originality but broke domain routing (landed on a disabled
domain) AND reintroduced a data-leakage bug that had already been fixed once
in the original shape. The original had to be reconstructed from scratch
because no version control was in place.

**Rules going forward:**
- Before reshaping, re-run the FULL Part 5 audit checklist mentally against
  the new shape — every item, not just the one that's failing.
- If a task is failing only ONE named check (e.g. only originality, with
  license/leakage/domain/IDs all previously green), prefer narrowing the fix
  to that check specifically: sharpen the metric, change what evidence is
  withheld from the model, add a genuinely new angle on the same input/output
  shape — rather than swapping the shape.
- **Keep a git repository from the first file you write**, or at minimum copy
  the entire working directory to a timestamped backup before any reshape.
  `git init && git add -A && git commit -m "known-good: <check status>"`
  before touching anything, every time a full audit passes.

---

## PART 1 — Feasibility Check (30–60 minutes, before committing to a design)

### 1.1 Sample before building anything

Fetch ~30–50 items by hand (a script, not the full pipeline). For each
candidate source/label combination, measure:
- Yield: usable items ÷ items fetched.
- Item length: is there enough content per unit to carry a learnable signal?
  A prior attempt used ~38-character sentence units and the trained model
  scored *below* a content-free baseline — too little signal per item. Aim
  for units with enough content that a human could plausibly do the task by
  reading them (a full sentence is a floor, not a target; several sentences
  or a paragraph is safer).
- Label distribution: how many classes/values, how balanced.
- Fetch rate and any rate-limiting behavior (see Part 3).

### 1.2 Minimum scale thresholds

- Total usable items: **at least 500**, ideally 2,000+.
- Test-set rows: **at least 60**, because the platform appears to split
  answers into a public and private leaderboard slice and each slice needs
  at least ~10 rows to pass Prepared Data Integrity.
- Per-class minimum for any categorical label: **at least 100** examples of
  the rarest class you plan to keep. Drop classes below this rather than
  ship a class the model can never learn.

### 1.3 Run every leak probe on the sample BEFORE finalizing the design

Build a throwaway train/test split on the sample and score each of these. If
any content-free probe scores within ~0.10 of your planned reference model,
the task has an exploitable shortcut and must be redesigned before scaling
up.

| Probe | What it catches |
|---|---|
| Constant / majority-class prediction | Chance floor — sanity check only |
| Item length (characters, token count) | Length itself encodes the label |
| Punctuation / digit / specific-character counts | Register or source leaking through surface statistics |
| Capitalized-token-only features | Named entities (places, people, orgs) leaking the label directly |
| Any metadata column you plan to publish (topic, category, source language) | A "helper" feature that actually IS the label in disguise |
| Position in the shipped row order / assigned slot letters | Shuffling implemented incorrectly, position itself is informative |
| A trivial "copy the input through unchanged" prediction | For any transform/rewrite task — measure this explicitly, see 1.4 |

**A probe scoring near your reference is a fail. A probe BEATING your
reference is a design failure severe enough to abandon the source.** An early
attempt in this line of work used very short text units (median ~38
characters) as the prediction unit. On that data, the trained reference model
scored *below* a content-free baseline, and a naive "ignore the task
structure, just cluster everything globally" adversarial baseline
outperformed the honest reference entirely. The source was abandoned and
rebuilt on a different one with longer, richer units — the fix was not a
smarter model, it was more signal per item (Part 1.1's length guidance exists
because of this). **Always compute an adversarial baseline that ignores your
intended task structure entirely** (e.g., if your task scores per-group,
also measure a model that pools everything and ignores groups) alongside the
content-free probes. If it beats your reference, the reference is too weak
or the task doesn't have real signal at that unit size — do not ship it
regardless of how the other checks look.

### 1.4 For any task where most of the output matches the input unchanged, choose the metric BEFORE building the split

This nearly caused a shipped-then-caught error. An accent-rewrite task where
source and target phoneme strings are ~80% identical was first scored with
plain edit similarity — under which copying the input through unchanged
scored 0.795, leaving only a 0.2-point band to separate every real
submission. The metric was changed to score *only the positions that differ
between input and output*, which sent the copy-through baseline to ~0.11 and
restored a usable [0,1] range.

**Rule:** if you expect the correct output to share more than ~50% of its
content with the input (rewriting, correcting, translating a short/related
form), measure what a copy-through baseline scores under your chosen metric
BEFORE building anything further. If it's above ~0.3, the metric is wrong —
switch to scoring only the positions/tokens that should change, or to an
exact-match variant, before proceeding.

---

## PART 2 — Task Design

### 2.1 Choose the input/output shape deliberately

For NLP-domain challenges, pick one of these shapes and stick to it through
the whole build (do not switch mid-build per Rule 0.5):

- **Sequence-to-sequence generation**: input text (with structured
  conditioning fields alongside it) → output text. Scored with an edit-based
  or task-specific string metric. This is the most defensible NLP shape and
  the one that passed every check in the accepted challenge.
- **Sequence-to-sequence with partial-credit rewrite scoring**: as above, but
  where most of the output should equal the input except at specific
  positions — use the changed-positions-only metric from 1.4.
- **Ordering / ranking**: given a shuffled set of text units drawn from one
  source item, recover their original order or ranking. Scored with a
  rank-correlation metric (Kendall's tau, Spearman) rescaled to [0,1] with an
  exact, known chance level (0.5 for a random permutation).
- **Structured extraction**: given text, produce a structured
  representation (fields, spans, a normalized value) that exists in the
  source as ground truth.

Avoid single-label sequence classification unless the classes are richly
about language content and you have verified (0.4a) that the domain routes
correctly.

### 2.2 Build in a genuine confound, and publish it as a feature rather than hide it

If some easily-available metadata (topic, source language, region, category)
correlates with your label but isn't supposed to BE the answer, publish that
metadata as an input column in both train and test. This does two things:
lets legitimate models condition on it honestly, and lets you measure a
"metadata-only" baseline to prove the task isn't secretly that metadata in
disguise (fold this into Part 1.3).

### 2.3 Make the withheld information matter

Design the task so a model that ignores the actual novel input (and instead
falls back to a shortcut — spelling instead of a given transcription,
majority class, position) is *measurably* capped below the reference score.
Verify this cap explicitly and publish the number (e.g. "ignoring the accent
input caps you at 0.80, measured by oracle substitution" — a genuinely useful
sentence to put in the problem description).

---

## PART 3 — Fetching the Raw Data

### 3.1 One worker, always

Never run multiple parallel fetch processes against the same source. This
was tried three times in one session; every time, all workers got throttled
to near-zero and the total time was worse than a single patient worker would
have taken. Use one process, with 0.8–1.5 seconds between requests, and let
it run to completion.

### 3.2 Catch broad exceptions in retry logic

A bare `except URLError` or `except HTTPError` will not catch
`ConnectionResetError`, which killed a 25-minute fetch outright. Catch
`Exception` broadly in the retry loop, log what happened, back off, and
retry a bounded number of times before giving up on that one item.

### 3.3 Never kill a running fetch to "improve" something

If you realize mid-fetch that you want to change a filter or add a field,
let the current fetch finish (or reach a natural checkpoint), THEN change the
code and re-run. Every mid-fetch kill in this session cost 20–40 minutes of
restarted work, because most simple fetch scripts only write output at the
very end.

**Better pattern:** have the fetch script write incrementally (append each
usable item to disk as it's found, not just at exit) OR support a `--start`
page/offset argument so a second run can extend a first, disjoint run and be
merged afterward.

### 3.4 Prefer bulk data dumps over paginated APIs when both exist

Official bulk dumps (e.g. Wikimedia's `dumps.wikimedia.org`) typically have
no per-request rate limit once the file is downloading, whereas paginated
REST APIs often throttle aggressively. If a bulk dump exists for your source,
use it.

### 3.5 If the source is ordered by an attribute you care about, budget per-attribute

A source ordered by language, category, or date will exhaust a flat byte
budget on the first attribute value encountered. If you need balanced
coverage across languages/categories, track bytes read per attribute
independently and stop each one at its own cap, not a shared global cap.

---

## PART 4 — Deduplication and Leakage (the check that fails most often)

This deserves its own section because it caused rejections even after
apparently correct word/document-level splitting.

### 4.1 Splitting by "document" or "word" is not enough — split by exact feature value

Word-disjoint or document-disjoint splitting prevents the same *source unit*
from appearing in both train and test. It does NOT prevent the same *feature
value* (the actual string/row the model sees) from appearing in both, if
different source units can independently produce identical feature values.
Example: two different English words can share a pronunciation
(homophones); a word-disjoint split still let one word's transcription
appear verbatim in both train and test via a different word, at ~4% of test
rows, and the platform correctly flagged this as leakage.

**Rule: after building the split, explicitly check for exact overlap on the
*shipped feature columns* between train and test — not just on the
document/word identifier.** If any overlap exists:
- For an input feature that legitimately repeats (e.g. the same source
  phoneme string can appear as an input in multiple training rows for
  different target accents), overlap is fine IF the *target* value is never
  shared. Verify this distinction explicitly and document it.
- For a case where the exact (input, target) row is duplicated across
  splits, or the raw feature value predicts the target row itself, dedupe:
  collapse to unique feature values, drop the ones that carry conflicting
  labels (unresolvable from the input alone), and assign each unique value
  wholly to one split.

### 4.2 Train and test feature columns must match exactly

The platform checks that train.csv's columns, minus the target column,
equal test.csv's columns exactly (same names, effectively same order). A
rename of the target column that isn't mirrored consistently across
train/test/answers/submission causes a "feature columns must match" failure
even when there's no real data leak. **After any column rename, re-derive
this check programmatically** (see Part 5) rather than eyeballing it.

### 4.3 row_id (the join key) between test.csv and answers.csv must be unique and must be the ONLY shared column

- `test.csv`'s first column must be a value that (a) is unique per row and
  (b) exists identically in `answers.csv` as the join key.
- If your task naturally has one row per sub-item (multiple sub-items per
  parent document), the first column must identify the sub-item uniquely —
  not the parent — or pack sub-items into a wide row with a unique parent ID
  instead of using long format. A "one row per subsection" long-format table
  keyed by a repeating `paper_id` failed this check; reshaping to one row per
  (paper, subsection) pair with a synthesized unique ID fixed it.
- If the grading metric needs to see the input value again (e.g. to align
  source and target strings), carry it in `answers.csv` under a DIFFERENT
  column name than the one used in `test.csv`, and assert programmatically
  at build time that the two columns contain identical data under their
  different names. Sharing a column NAME between test.csv and answers.csv —
  even for a legitimate input feature the grader needs — gets read as a
  leaked target column and fails the check.

### 4.4 Every packed/wide column must be filled in at least half of all rows

If sub-items are packed into a wide row (e.g. `slot_A_text`, `slot_B_text`,
...), any column filled in fewer than 50% of rows fails "column missing
values in half or more profiled rows." Cap the number of packed slots at
whatever count the majority of parent items actually reach; truncate larger
items rather than leaving high-index slots mostly empty.

---

## PART 5 — Verification Checklist (run every item, every time, before upload)

This checklist must be run as an actual script against the shipped files —
not eyeballed — because column-name mismatches and small leaks are invisible
by inspection. Treat any single FAIL as blocking.

```
GATES (Part 0)
[ ] License is a named platform-dropdown value, confirmed from the source's own API/metadata
[ ] Source URL does not appear in your running list of previously-used sources (Part 6)
[ ] Every target value is a structured value recorded by the source itself (no annotation, no LLM output anywhere)
[ ] domain: natural-language-processing is set, and the task shape does not read as classification/anomaly/clustering
[ ] Task's (input, output, metric) triple differs from the nearest known benchmark in at least 2 of those 3 dimensions

RAW DATA
[ ] Every raw record parses without error
[ ] Raw record count and class/label distribution match what Part 1 feasibility measured (no silent collapse)

REPRODUCIBILITY
[ ] prepare.py run twice produces byte-identical train.csv, test.csv, answers.csv (compare checksums)

SCHEMA AND IDS
[ ] train.csv columns minus the target column == test.csv columns, exactly
[ ] test.csv's first column is unique per row
[ ] answers.csv's join-key column == test.csv's first column, values match exactly (same set, same count)
[ ] sample_submission.csv's join-key values == test.csv's join-key values exactly
[ ] The ONLY column name shared between test.csv and answers.csv is the join key
    (if the grader needs an input value again, it must be under a different column name in answers.csv)
[ ] No column in test.csv or train.csv is empty in more than 50% of rows
[ ] No blank/null join-key values anywhere

LEAKAGE
[ ] Target column is absent from test.csv
[ ] Train/test are disjoint on the actual split unit (word/document/whatever you chose)
[ ] No exact (input-features, target) row appears in both train and test
[ ] If an input feature value legitimately repeats across rows, confirm the corresponding TARGET is never shared across those repeats
[ ] Any duplicated-answers.csv-input-column equals the corresponding test.csv column exactly, for every row (programmatic assert, not spot check)

GRADER
[ ] grade(sample_submission, answers) runs and returns a finite float
[ ] grade() is deterministic — same inputs, same output, run twice
[ ] grade(oracle_submission, answers) == max_score (usually 1.0)
[ ] grade(blank_submission, answers) == min_score (usually 0.0)
[ ] grade() raises InvalidSubmission (not a silent wrong answer) when a required column is missing
[ ] grade.py contains no filesystem access beyond its two DataFrame arguments — no os.path, no open(), no environment probing
[ ] Unit tests for the core scoring function pass on at least 3-4 hand-constructed cases (exact match, one-error, no-change, full-mismatch)

FILES AND CONFIG
[ ] prepare.py, grade.py, dataset/validate_submission.py, dataset/generator/build_raw.py all parse (ast.parse or equivalent)
[ ] config.yaml parses as valid YAML
[ ] solution.ipynb is valid JSON and its final submission cell references CUDA/GPU explicitly
[ ] dataset/validate_submission.py is byte-identical between dataset/ and dataset/public/ (wherever both copies exist)
[ ] Zero __pycache__ directories anywhere in the package
[ ] The raw-data zip contains exactly the files listed in dataset-description.md's file structure section — nothing extra (no stale files from an earlier design iteration), nothing missing

DOCS AND CONSISTENCY
[ ] dataset-description.md has every one of: Overview, Source, Fetch method, File Structure, Features, Intended Use, Limitations, License
[ ] Every raw/ file is named and described in dataset-description.md
[ ] Intended Use lists primary use, also-suitable-for, and explicitly not-suitable-for
[ ] Limitations lists at least 5 concrete, specific points grounded in what was actually measured (not generic disclaimers)
[ ] The title string is IDENTICAL across: config.yaml's title field, problem-description.md's H1, README.md's H1, solution.ipynb's first cell heading
[ ] No leftover wording from a previous design iteration anywhere (grep for old task names, old column names, old metric names across every .md/.yaml/.ipynb file)
[ ] Every numeric claim in problem-description.md (percentages, baseline scores) matches a number actually measured on the CURRENT shipped split, not a stale earlier run
[ ] No markdown pipe-tables where the target platform's renderer is known to mangle them — use bullet lists for small key/value tables if unsure
[ ] rubrics.md has at least 5 rubrics, majority REQUIRED or RECOMMENDED (not UNIVERSAL), each specific to this task (not generic ML advice), each phrased approach-neutrally (do not mandate a specific library or architecture)

BASELINES (must be measured on the FINAL shipped split, not the sample)
[ ] Majority-class / constant-output baseline measured
[ ] Uniform-random baseline measured
[ ] At least one content-free baseline (from Part 1.3's probe list) measured
[ ] A real reference model (even CPU-feasible) measured, scoring clearly above every baseline
[ ] Oracle baseline confirmed at max_score
[ ] All baselines reported in config.yaml's grade section with a note_band explaining WHY each number is what it is
[ ] Reference score sits in a defensible band for the stated difficulty (not near-chance, not near-perfect)
```

---

## PART 6 — Source Tracking (maintain this across every challenge you build)

Keep a running table, updated every time, of every source URL used —
including rejected/abandoned attempts, because Rule 0.2 forbids reuse
regardless of outcome.

| Date | Source URL | Task | Status |
|---|---|---|---|
| (example) | tatoeba.org bulk export | sentence-pair translation task, short units | abandoned — adversarial baseline beat reference (Part 1.3) |
| (example) | europarl (statmt.org) | multiple attempts: speaker attribution, region detection, intruder detection | rejected repeatedly — no named license on first framing, then duplicate URL on later attempts |
| (example) | Wikinews (dumps.wikimedia.org) | region/category classification | rejected — duplicate source URL (reused after Europarl attempts) |
| (example) | Wiktionary (dumps.wikimedia.org) | accent phoneme transfer | accepted, after one bad mid-build reshape and revert (Part 0.5) |
| 2026-09-21 | courtlistener.com/help/api/bulk-data/ (Free Law Project) | citation-structure recovery from redacted case law — first built as bag/all-pairs edge recovery, redesigned to query+12-candidate selection | built; Public Domain Mark; positional leak in the first shape (position-only probe 0.143 > reference 0.098) forced the redesign |
| 2026-09-25 | en.wikipedia.org/w/api.php (live Action API, Good/Featured articles — NOT dumps.wikimedia.org) | RAG-themed citation-order-recovery: given a paragraph and its own shuffled true reference cards, recover citation order (ordering/ranking shape, rescaled Kendall's tau) | built; CC BY-SA 4.0; reference 0.7037 vs best probe 0.5576 (margin 0.146), oracle 1.000 |
| (fill in as you go) | | | accepted / rejected / abandoned |

Never propose a new challenge idea without first checking this table for the
source you're about to use.

---

## PART 7 — Package Structure (exact layout to produce every time)

```
config.yaml                 Challenge metadata, dataset block, prepare/grade pointers, measured anchors
problem-description.md      The full prompt shown to the solving agent
dataset-description.md      Overview / Source / Fetch method / File Structure / Features / Intended Use / Limitations / License
rubrics.md                  5+ task-specific rubrics
README.md                   Human-facing summary: title, files table, reproduce commands, data provenance
prepare.py                  def prepare(raw: Path, public: Path, private: Path) -> None — deterministic, seeded
grade.py                    def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float — self-contained, degrade-not-reject
solution.ipynb              3 progressive submissions; final one uses GPU explicitly; reads ./dataset/public/, writes ./working/submission.csv
dataset/
  raw/                      Exactly the files documented in dataset-description.md, plus ATTRIBUTION.txt and meta.json
  public/                   train.csv, test.csv, sample_submission.csv, dataset_stats.json, validate_submission.py
  private/                  answers.csv
  generator/                build_raw.py (the fetch/build script) — no __pycache__ committed
  validate_submission.py    Offline structural checker, needs no answers file (same content as dataset/public/ copy)
<dataset-name>-raw.zip      dataset/raw/ + dataset/generator/ ONLY — nothing else
```

`config.yaml` required sections: `name`, `title`, `domain:
natural-language-processing`, `tags`, `dataset` (name, title, short_title,
description, generator, source_data, source_url, distinctness, citation,
license, license_url, license_note, raw_files), `problem_description`,
`rubrics`, `prepare` (script, public files list with column comments, private
files list, split_note), `grade` (script, usage, submission_file,
submission_columns, metric, metric_formula, direction, min_score, max_score,
all measured anchors, note_band, band_evidence, reference_caveat),
`submission` (filename, columns, rows), `reference_solution`, `environment`.

---

## PART 8 — `problem-description.md` Required Sections

Write in this order, every time:

1. **Title + a one-paragraph hook.** Lead with a concrete example, not an
   abstract description. State the real-world question the task answers.
2. **Task.** Exactly what one input row contains and what output is expected.
   Use bullet lists, not pipe tables, for input/output field descriptions.
3. **What you are actually being asked** (or equivalent framing section).
   Explicitly distinguish the task from the nearest similar-sounding task a
   solver might assume it is (this section is also where originality
   differentiation should be argued explicitly, for the human reviewer).
4. **Evaluation.** State the metric formula (as code, if precise), explain
   WHY that metric and not a more obvious alternative (this is where the
   1.4 metric-selection reasoning belongs), and list every measured baseline
   with its score.
5. **Dataset.** The data-discovery code snippet, then each file's columns
   and purpose.
6. **Submission.** Required columns, one-row-per-id requirement,
   degradation behavior (what scores 0 vs. what gets rejected outright).
7. **Compute.** State the GPU, CPU, RAM, and time budget; name the intended
   model class and explain briefly why it fits the metric.

---

## PART 9 — Common Failure Messages and Their Real Cause

| Platform message | Actual cause (from experience) | Fix |
|---|---|---|
| "Problem Originality... appears similar to an existing problem" | Task shape/metric matches a known challenge even with new data/wording | Change input/output shape or metric fundamentally (Part 0.4b), don't just reword |
| "This challenge was classified as [X] which is not currently accepting submissions" | Task framing routes to a disabled domain (classification, anomaly detection) | Reframe as seq2seq/ranking/extraction; verify domain fit before building (Part 0.4a) |
| "A dataset from the same source URL already exists" | Source reused across your own challenges | Check Part 6 table before choosing a source; never reuse |
| "The public test IDs are not unique" | First column of test.csv repeats (long format with a repeating parent ID) | Reshape to one row per graded unit with a unique ID, or use a genuinely unique join key |
| "Train and test feature columns must match" | Target column renamed inconsistently across files | Re-derive column sets programmatically after any rename (Part 4.2) |
| "The public test data contains evaluation target columns" | A column name is shared between test.csv and answers.csv, even if it's a legitimate input the grader needs | Rename the copy in answers.csv to a distinct name; assert equality at build time (Part 4.3) |
| "Train and test contain repeated feature rows that can leak evaluation data" | Split by document/word wasn't enough; exact feature-value overlap exists across splits | Dedupe on the actual shipped feature value, not just the source unit (Part 4.1) |
| "At least one column is missing values in half or more profiled rows" | Wide/packed format with sparse high-index slots | Cap packed-slot count at what most items reach; truncate rather than leave sparse (Part 4.4) |
| "Public and private evaluation slices each need at least 10 rows" | Test set too small once the platform splits it | Ship at least 60 test rows, ideally 200+ (Part 1.2) |
| "The description does not document [some raw file]" | Leftover file from an earlier design iteration in dataset/raw/ | Delete anything not in the current dataset-description.md's file list, rebuild the zip |
| "does not describe an intended use... omits known limitations" | dataset-description.md missing required sections | Always include Intended Use (primary/also/not-suitable) and 5+ specific Limitations |

---

## PART 10 — What "Done" Means

Do not report a challenge as ready until:
1. Every item in the Part 5 checklist has been run as an actual script and
   shown PASS, not assumed.
2. The source URL has been added to the Part 6 table.
3. A backup (git commit or timestamped copy) of the fully-passing state
   exists, before any further edits are made for polish or novelty tuning.
4. All baseline and reference numbers quoted anywhere in any doc were
   measured on the exact files being shipped, not an earlier sample or a
   prior iteration of the split.

If any of these four is not true, the challenge is not done, regardless of
how complete the files look.
