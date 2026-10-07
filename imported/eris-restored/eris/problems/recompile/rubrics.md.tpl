# Rubrics — Compiling Amended Regulations

Nine entries. Each has a criterion and the condition that fails it.

---

## 1. Builds the paragraph hierarchy instead of matching designators flatly
**Type:** DATA_HANDLING · **Importance:** REQUIRED

Locating "paragraph (a)(3)" requires knowing where paragraph (a)(3) ends as well
as where it starts, and that follows from the level structure: a paragraph runs
until the next designator at the same or a shallower level. A solution should
build that structure — assigning each designator a level from the sequence it
continues, not from whether the character is a letter or a digit — and use it to
resolve addresses.

*Fails if* the solution finds the designator with a regular expression and then
replaces to the next designator of any kind, or to the end of the section.

---

## 2. Separates paragraph openings from cross-references
**Type:** FEATURE_ENGINEERING · **Importance:** REQUIRED

Regulations cite their own paragraphs constantly — "except as provided in
paragraph (d) of this section" — and those citations are lexically identical to
paragraph openings. A solution must use context to tell them apart. The reliable
signal is punctuation: a real opening follows the end of the preceding paragraph.
Keying off the preceding *word* instead looks reasonable and is not — "of this
part." ends a sentence and does open the next paragraph, so a cross-reference word
list throws away genuine openings.

*Fails if* every designator-shaped token is treated as a paragraph boundary.

---

## 3. Treats `* * *` as "the rest is unchanged"
**Type:** MODELING · **Importance:** RECOMMENDED

When an agency revises part of a paragraph it prints only that part and marks the
remainder with `* * *`. The surviving text has to be carried over, which means
deciding how much of the original the printed fragment displaces — usually a
sentence boundary problem in prose full of `U.S.C.`, `Pub. L.` and `Jan.`

*Fails if* a printed block carrying `* * *` replaces the whole addressed paragraph,
or if the `* * *` marker is copied into the answer as literal text.

---

## 4. Applies a run of instructions in order, including redesignations
**Type:** MODELING · **Importance:** RECOMMENDED

Sections amended by several instructions must be processed sequentially, because
earlier operations move the targets of later ones. Redesignating (c) through (f)
as (d) through (g) is the clearest case: an address resolved against the original
text is wrong by the time it is used.

*Fails if* all addresses are resolved once against the original text, or if
instruction order is ignored.

---

## 5. Validates on a part-level split
**Type:** TRAINING · **Importance:** REQUIRED

The labelled items are there to be measured against. Because the same section and
the same amendment shape recur across years and agencies, a validation split drawn
at item level reports a score the test set will not reproduce. Held-out CFR parts
are the honest unit.

*Fails if* no held-out measurement is made, or if the split is drawn at item level
with sections or parts appearing on both sides.

---

## 6. Leaves untouched text untouched
**Type:** CODE_QUALITY · **Importance:** REQUIRED

The median amendment moves a small fraction of the section. Everything else is law
that is supposed to survive verbatim, and the metric charges for tokens moved
without being asked. A solution should change only what an instruction addresses.

*Fails if* the answer regenerates, reflows, re-cases or truncates the parts of the
section no instruction touched.

---

## 7. Beats the paragraph-splice baseline
**Type:** TRAINING · **Importance:** RECOMMENDED

Splicing the printed block over each addressed paragraph, with the paragraph's
extent resolved against a real hierarchy but no `* * *` handling, no removals and
no redesignation, scores @@paragraph_splice@@. Getting there at all requires the
hierarchy; going past it means handling partial replacements and deletions.

*Fails if* the final score does not exceed the hierarchy-splice baseline
(@@paragraph_splice@@).

---

## 8. Checks the data for shortcuts instead of assuming it is clean
**Type:** AGENT_BEHAVIOR · **Importance:** RECOMMENDED

Good practice on a corpus built from public documents is to look for the leak
before trusting the task: whether the after text can be recovered from a citation,
a date, an editorial note or a source credit left in the input. Here those were
stripped deliberately, and confirming that is cheap.

*Fails if* the solution depends on metadata that is not in the released files, or
assumes network access.

---

## 9. Produces a valid submission inside the compute budget
**Type:** CODE_QUALITY · **Importance:** REQUIRED

One row per test item, non-empty `after_text`, no duplicates, finishing well within
the time limit on CPU. `validate_submission.py` ships with the data and applies the
grader's structural checks offline.

*Fails if* the submission is malformed, incomplete, or the run does not finish.
