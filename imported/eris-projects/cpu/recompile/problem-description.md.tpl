# Compiling Amended Regulations

## The problem

Federal regulations are not rewritten when they change. They are *patched*. An
agency publishes a rule in the Federal Register that says, in prose, exactly what
to do to the existing text:

> Amend § 205.201 by:
> a. Removing the words "or excluded" in paragraph (a) introductory text;
> b. Revising paragraph (a)(3); and
> c. Adding paragraph (c).

Somebody then has to carry that out against the regulation as it actually stands,
and the result is what the law says the next morning.

(That is the agency's published wording. In the released data every CFR section
number is replaced by `<CITE>`, for reasons given under *what is held back*.)

That is your task. Each item gives you one real section of the Code of Federal
Regulations exactly as it read the day before a real rule took effect, together
with the amendatory instructions that rule aimed at it and whatever replacement
text the agency printed underneath them. You return the section as it read the day
the rule took effect.

The answer is not an opinion and not an annotation. It is the official text
published by the Office of the Federal Register, which did this same compilation
by hand.

> **This is a program-writing task, not a text-generation task.** @@COPIED_PCT@@ of the
> tokens in a gold answer are copied verbatim from the section you were given. The
> intended solution reads `before_text`, works out which span the instruction
> addresses, and splices the agency's printed block over that span — string
> surgery, in code. Do not write the amended sections out yourself: the test set
> is @@TOTAL_CHARS@@ characters of regulatory prose, one item alone is
> @@MAX_CHARS_ITEM@@ characters, and reproducing that by hand will exhaust your
> budget long before you finish. Reproducing it *imperfectly* is worse than not
> touching it, because every token you alter that the agency did not alter costs
> you precision.
>
> **Write a fallback `submission.csv` before you start.** Copy
> `sample_submission.csv` — it is already valid and scores 0.000 — then overwrite
> rows as your solver improves them. That way a crash or a timeout costs you the
> items you had not reached yet, not the entire run.

## Why it resists the obvious approaches

**The instruction tells you what to do, not where to do it.** "Revising paragraph
(a)(3)" presumes you can find paragraph (a)(3). Regulations are deep hierarchies
written as running prose — (a), then (1), then (i), then (A) — and the designators
that mark them are ordinary parenthesised tokens. The same tokens appear
constantly as cross-references: *"except as provided in paragraph (d) of this
section"*. Treat one of those as the start of paragraph (d) and every span after
it is wrong. @@CROSSREF_PCT@@ of the designator-shaped tokens in this dataset sit
mid-sentence rather than after a sentence boundary, which is where a genuine
paragraph opening has to be.

**Levels are ambiguous by construction.** After paragraph (h) comes paragraph
(i) — or does a third level open with roman numeral (i)? The character cannot tell
you; only what came before it can.

**The printed replacement is often partial.** When an agency revises one sentence
of a long paragraph it prints the new sentence and then `* * *`, meaning *the rest
of this paragraph is unchanged*. @@ELLIPSIS_PCT@@ of items carry such a marker.
Splice in the printed block wholesale and you silently delete the remainder.
Getting this right needs sentence boundaries in
prose that is dense with `U.S.C.`, `Pub. L.`, `Jan.` and `No.` — where splitting
on full stops does not work.

**Several instructions interact.** A section is often amended by a run of
instructions that must be applied in order, because redesignating (c) through (f)
as (d) through (g) moves the target of everything after it.

**And most of the section must survive untouched.** The amendments here are
surgical: the median item changes @@MEDIAN_CHANGE_PCT@@ of the section's tokens.
Anything that rewrites more than the agency asked for is destroying law that was
supposed to stay.

## What makes this task new

Edits are everywhere in NLP, but always as the thing to be *predicted*. Here the
edit is published, precise and legally binding, and the entire difficulty is
resolving it against a real document.

**LexGLUE** (Chalkidis et al., ACL 2022) collects legal-language understanding
tasks — ECHR, SCOTUS, EUR-Lex, LEDGAR, UNFAIR-ToS, CaseHOLD. Every one is
classification or multiple choice; none transforms a legal text.

**EURLEX57K** (Chalkidis et al., ACL 2019) and **EUR-Lex-Sum** (Aumiller et al.,
EMNLP 2022) are multi-label classification and summarization over EU legislation.
Both read legislation; neither amends it.

**CUAD** (Hendrycks et al., NeurIPS 2021) locates clauses in contracts. It is span
extraction against a fixed taxonomy, and the document is never modified.

**SARA** (Holzenberger et al., 2020) applies statutes to fact patterns — reasoning
*from* a statute whose text is given and stable.

**PANORAMA** (NeurIPS 2025 Datasets and Benchmarks) captures patent prosecution,
including the amendments applicants make to claims. There the amendment is the
unknown to be generated so that it overcomes prior art. Here the amendment is
given and the document is the unknown.

**Text-editing corpora** — WikiSplit (Botha et al., EMNLP 2018), PEER (Schick et
al., 2022), ITERATER (Du et al., EMNLP 2022) — supervise models to produce edits
from context. The instruction, where there is one, is a coarse intent label. It
never has to be resolved against a hierarchy.

The 2025 survey *Computational Law: Datasets, Benchmarks, and Ontologies*
(arXiv:2503.04305) catalogues the field's resources. Nothing in it asks a system to
carry out an amendment. Yet this is the work that keeps the Code of Federal
Regulations current, it is done largely by hand, and its ground truth is published
daily and free.

### How the data was assembled, and what is held back

Two US Government sources, both public domain:

- the bulk Federal Register XML at `https://www.govinfo.gov/bulkdata/FR`, which
  carries each final rule's amendatory instructions as `<AMDPAR>` elements
  together with the text printed beneath them, and
- the eCFR point-in-time API at `https://www.ecfr.gov/api/versioner/v1`, which
  serves any CFR part or section as it stood on any date, and records the dates on
  which each section changed.

For every final rule published in the coverage window (@@DATE_RANGE@@), each
section the rule prints is paired with eCFR's own text on the day before and the
day of the change eCFR records for it. The before/after pair is therefore the
Office of the Federal Register's compilation, not ours.

Three filters run before anything is released, and each exists for a stated reason:

- **Well-posedness.** At least 90% of the tokens the amendment introduces must
  appear in what the agency printed. Below that the gold text cannot be produced
  from the inputs by any method, and the item is usually a case where the change
  belongs to a different rule. Those are dropped. The threshold sits in a
  near-empty valley: of 191 measured candidates, 85 scored exactly 1.00 and 69
  below 0.50.
- **Surgical amendments only.** Sections the agency reprinted in full are dropped.
  For those the answer is the printed block copied out, which measures nothing.
- **One item per section, and the split is disjoint by CFR part.** Agencies re-run
  the same amendment shape year after year — the civil-penalty inflation
  adjustments most obviously — so a per-item split would put this year's version in
  training and next year's in test.

Stripped from both sides before release: eCFR's editorial pointer, which announces
the very amendment being asked about ("Link to an amendment published at 88 FR
3502, Jan. 19, 2023."); the bracketed source credits, which gain the new rule's
citation in the after text; every Federal Register citation; and every CFR section
number, replaced by `<CITE>`. That last one matters because the section number is
the lookup key for the answer.

**On holdout.** The eCFR is public, as it must be for a challenge built on public
law. What is withheld is the pairing: which section, which rule, which date. No
released file carries a section number, a Federal Register citation or a date of
enactment. Solutions run without network access.

## Scoring

The metric is **edit-operation F1**, size-weighted, averaged over test items.

Comparing your text to the gold text directly would not work. An amendment moves a
few dozen words inside a section of several hundred, so a solver that changes
nothing scores about 0.97 on any similarity measure. So the comparison is between
*edit scripts*: the token-level operations that turn the before text into yours,
against the operations that turn it into the official text. Each operation is
weighted by the number of tokens it moves, so deleting the whole section is not
one cheap mistake.

Changing nothing scores 0.000. Reproducing the official text scores 1.000.

Every figure below was measured on this dataset with the grader below:

- Return the section unchanged — **@@copy@@**
- Append the printed text to the section — **@@append_printed@@**
- Return only the text the agency printed — **@@printed_only@@**
- Find the designator by regular expression, replace to the next designator of
  any kind — **@@regex_splice@@**
- The same, but resolving addresses against a real paragraph
  hierarchy — **@@paragraph_splice@@**
- Reference solution: hierarchy plus `* * *`, deletions and
  redesignations — **@@reference@@**
- Official text — **1.000**

The step worth noticing is the third to the fourth. Both splice the agency's
printed block over the addressed paragraph; the only difference is whether the
paragraph's extent was resolved by pattern matching or by parsing the hierarchy.
That single distinction is most of the available score.

Operations are matched by position and compared by content, not required to be
identical. An amendment applied to the right paragraph but ending a token early
earns most of its credit rather than none of it: half of each operation's credit
is for getting the extent right — which tokens of the original were displaced —
and half for getting the replacement text right. Two operations are candidates for
matching only if their spans overlap, or start within a few tokens of each other,
so an edit made in the wrong place still earns nothing.

Scoring in code:

```python
def edit_ops(before, after):
    a, b = normalise(before).split(), normalise(after).split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(i1, i2, tuple(b[j1:j2]))
            for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]

def weight(op):
    i1, i2, inserted = op
    return (i2 - i1) + len(inserted)

def similarity(g, p):
    """0 unless g and p are aimed at the same place; else span IoU and token F1."""
    gi1, gi2, gins = g
    pi1, pi2, pins = p
    overlap = min(gi2, pi2) - max(gi1, pi1)
    tolerance = max(3, 0.25 * max(gi2 - gi1, pi2 - pi1, len(gins), len(pins)))
    if overlap <= 0 and abs(gi1 - pi1) > tolerance:
        return 0.0
    union = max(gi2, pi2) - min(gi1, pi1)
    span_sim = 1.0 if union <= 0 else max(overlap, 0) / union
    return 0.5 * span_sim + 0.5 * bag_f1(gins, pins)

def score_item(before, gold_after, pred_after):
    gold, pred = edit_ops(before, gold_after), edit_ops(before, pred_after)
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    # greedy one-to-one alignment, best-scoring pairs committed first
    matched = align(gold, pred)
    recall    = sum(s * weight(gold[g]) for s, g, _ in matched) \
              / sum(weight(o) for o in gold)
    precision = sum(s * weight(pred[p]) for s, _, p in matched) \
              / sum(weight(o) for o in pred)
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

`normalise` collapses whitespace and maps curly quotes and dashes to ASCII, so no
score turns on typography. `grade.py` is the authority; it ships with the data.

## The data

**Find the data before you assume where it is.** Depending on how your run is
mounted the files sit at `/data/`, at `./dataset/public/`, or under
`/kaggle/input/`. Locate them once and derive everything from that:

```python
from pathlib import Path
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/recompile-dataset/public"))
            if (p / "test.csv").exists())
```

Write your `submission.csv` into the working directory you were given, not
alongside the data.

The files themselves:

- `items/<item_id>.json` — one item: the before text and the instructions
- `train.csv` — the official after text for the @@N_TRAIN@@ training items
- `test.csv` — the @@N_TEST@@ item ids you must answer
- `sample_submission.csv` — a valid but deliberately poor entry (the section unchanged)
- `validate_submission.py` — reads a submission the way the grader will, offline
- `dataset_stats.json` — item counts

An item file is shaped like this:

```json
{"item_id": "it00042",
 "group_id": "g0117",
 "before_text": "(a) The producer or handler of a production or handling operation, except as exempt or excluded under <CITE>, intending to sell ...",
 "blocks": [{"instructions": ["6. Section <CITE> is amended by:",
                             "a. Removing the words \"or excluded\" in paragraph (a) introductory text;",
                             "b. Revising paragraph (a)(3); and",
                             "c. Adding paragraph (c)."],
             "printed": "(a)(3) A description of the monitoring practices ... (c) In addition to paragraph (a) of this section ..."}]}
```

Each block is one run of instructions together with the text the agency printed
beneath that run. `printed` is empty when the amendment only deletes.

`group_id` names the CFR part the section belongs to, as an opaque label. It is
there so you can validate honestly: the train/test split is disjoint by part, so a
validation split drawn at item level will report a score the test set does not
reproduce. Hold out whole `group_id`s instead.

Columns in `train.csv`:

- `item_id` (string) — the item
- `after_text` (string) — the official text of that section after the rule

`test.csv` carries `item_id` only.

There are @@N_ITEMS@@ items altogether, @@N_TRAIN@@ labelled and @@N_TEST@@ held
out, drawn from @@N_PARTS@@ CFR parts. The median section is @@MEDIAN_TOKENS@@
tokens long. No `group_id` appears on both sides of the split.

## Your submission

Produce a CSV called `submission.csv` with two columns:

- `item_id` (string) — copied from the test file
- `after_text` (string) — your text for that section

It should satisfy all of the following:

- Exactly **@@N_TEST@@ rows** plus a header, one per row of the test file.
- Each test `item_id` present once — no omissions, no repeats.
- `after_text` non-empty, and at most 200,000 characters.
- Written on a single line. Runs of whitespace are collapsed before scoring, so a
  stray newline will not change your score, but keeping rows on one line keeps the
  CSV readable.

**Structural slips cost points, not the run.** The grader does not throw an entry
out for being untidy. An item you omit, leave blank, or list twice is scored as
"no amendment applied" — zero for that item, which is what a blank answer is
worth — and the rest of your entry is scored normally. Rows for ids outside the
test set are ignored. So a solver that crashes on three hard items and writes out
the rest still gets full credit for the rest. The only entry that cannot be scored
is one that is not readable as a CSV with an `item_id` and an `after_text` column,
or one that shares no id at all with the test set.

That said, `after_text` is several thousand characters of prose containing commas
and quotation marks. **Write the file with `pandas.DataFrame.to_csv` or the `csv`
module**, never by pasting strings together — the grader will try hard to recover a
hand-built file, but an unquoted one loses most of its content at the first comma.

You can check an entry offline by running
`python validate_submission.py submission.csv <DATA>/test.csv`. It reports
`ERROR` for the two unscorable cases and `WARNING` for the items you are silently
scoring zero on.

A well-formed entry starts like this:

```csv
item_id,after_text
it00003,"(a) Each application shall be filed with the Director. (b) The Director shall ..."
it00007,"(a) For purposes of this subpart, the following definitions apply. ..."
```
