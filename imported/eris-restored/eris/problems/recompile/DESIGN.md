# DESIGN — Compiling Amended Regulations

Internal record: why this task, why this data, why this metric, and what was
measured rather than asserted.

---

## 1. What the challenge asks

Given one real CFR section as it stood the day before a real Federal Register rule
took effect, plus the amendatory instructions that rule aimed at it and the
replacement text the agency printed, return the section as it stood the day the
rule took effect.

Ground truth is the eCFR's own point-in-time text. It is not annotated, not
adjudicated, not model-generated: it is what the law said.

---

## 2. Occupancy scan

Five formulations were scanned before this one was chosen. Three were killed on
evidence:

- **Erratum localisation in scientific papers** — occupied by **FLAWS**
  (arXiv:2511.21843, Nov 2025), 713 paper-error pairs for error identification and
  localisation.
- **Wikipedia inline cleanup-tag prediction** — occupied by **WikiSQE**
  (arXiv:2305.05928), which uses exactly those templates as sentence-quality
  labels at scale.
- **Patent claim amendment alignment** — occupied by **PANORAMA** (NeurIPS 2025
  Datasets & Benchmarks), which aligns original claim, rejection, amendment and
  allowance, and by the European Patent Dataset (arXiv:2505.12568), which pairs
  application and granted claim versions.
- **Source-language identification / translationese** — long-occupied line of work.
- **Edit-summary to diff matching** — adjacent to **Edisum** (arXiv:2404.03428),
  and in any case the same corpus family as our accepted Palimpsest challenge, so
  ruled out to avoid self-repetition.

What survived: **applying a published amendatory instruction to the text it
amends.** Searches across legal-NLP benchmarks (LexGLUE, EURLEX57K, EUR-Lex-Sum,
CUAD, SARA) and the 2025 survey *Computational Law: Datasets, Benchmarks, and
Ontologies* (arXiv:2503.04305) turned up classification, summarization, extraction
and statutory reasoning — and no task that transforms a legal text according to a
published instruction.

The novelty claim is specific and defensible: everywhere else in NLP an edit is the
thing to be **predicted**. Here it is **given**, and the difficulty is resolving it
against a real hierarchical document. That is the compilation work that keeps the
CFR current, it is done largely by hand, and its ground truth is published daily.

---

## 3. Why this data source, and what went wrong first

**U.S. Code was the first choice and was abandoned.** `uscode.house.gov` is
unreachable from the build environment (connection failure, not a 403), and
govinfo has no `USCODE` bulk collection — only `COMPS`, `STATUTE` and `PLAW`, none
of which give a before/after pair. The CFR route has both halves available:

- `https://www.govinfo.gov/bulkdata/FR` — daily Federal Register XML. Amendatory
  instructions are `<AMDPAR>` elements; the printed replacement is the following
  `<SECTION>`. One issue carries around 255 instructions.
- `https://www.ecfr.gov/api/versioner/v1` — point-in-time CFR text.

`federalregister.gov`'s own XML endpoint bot-blocks (serves a "Request Access"
page); its **JSON** API works and is used for rule metadata only.

**Pairing on the stated effective date fails.** eCFR applies a change on the date
*it* records, which drifts a few days from the rule's stated effective date.
Instrumenting the losses over 200 rules:

- `ambiguous_date` 167, `sec_not_in_versions` 71, `date_ok` 58, `no_date_in_window` 3

Pairing on eCFR's own version record instead — exact hit on the effective date,
else the nearest recorded change within 30 days — raised the yield from 73 to 104
items over the same rules.

**Throughput, and a wrong turn worth recording.** eCFR sustains roughly 24
requests a minute regardless of request size and regardless of worker count
(measured 23/min at 6 workers, 24.5/min at 10). Whole-part fetching needs 4× fewer
requests than section fetching, which looked decisive — until parts turned out to
be several megabytes served at a few hundred KB/s, so each one blocked a worker for
20-100 seconds and throughput collapsed to 3 files a minute. Section fetching costs
about 500 bytes and one second, hits the 24/min ceiling instead of the bandwidth
one, and lets the largest CFR parts stay in the corpus rather than being skipped by
a size cap. Section fetching is what ships.

Everything is cached to disk with atomic renames, so a killed run resumes exactly
where it stopped, and `ERIS_SKIP_PREFETCH=1` rebuilds from cache without touching
the network. Because the prefetch runs in date order, stopping it early yields a
contiguous coverage window rather than a ragged sample.

---

## 4. Leaks found and closed

Every one of these was found by inspecting real items, not by reasoning about them.

1. **eCFR's editorial pointer.** Any section with a pending amendment carries
   *"Link to an amendment published at 88 FR 3502, Jan. 19, 2023."* at the top —
   it announces the very amendment being asked about. Stripped. The first regex
   stopped at the full stop in "Jan." and left `19, 2023.` glued to the front of
   68 of 104 items; the fix runs the match through the four-digit year.
2. **Source credits.** The bracketed credit line gains the new rule's citation in
   the after text, so it would hand over both the fact of the change and its date.
   Stripped from both sides.
3. **Federal Register citations** anywhere in the text — replaced by `<FRCITE>`.
4. **CFR section numbers** — replaced by `<CITE>`. The section number is the lookup
   key for the answer.
5. **Uncanonicalised instructions.** The instruction text was initially passed
   through raw, which reintroduced the section number (*"Revise § 648.146 to read
   as follows"*) after it had been masked out of the section, and left curly quotes
   that no longer matched the section's straight quotes. Instructions now go
   through the identical canonicalisation.

Verified after the fixes across the probe corpus: 0 occurrences of "Link to an
amendment", 0 raw FR citations, 0 date-tail residue, 0 undecoded HTML entities.

---

## 5. Well-posedness and selection

**Well-posedness.** At least 90% of the tokens the amendment introduces must
appear in what the agency printed. Below that, the gold text cannot be produced
from the inputs by any method — and in practice the item is one where the change
belongs to a different rule that landed in the same window. This filter is also
what guards against mis-attribution, which is why the date-pairing rule above can
afford to be permissive.

The threshold is 0.90 rather than 1.00 because the distribution is sharply
bimodal: measured over 191 candidates, 85 sat at exactly 1.00 and 69 below 0.50,
with a near-empty valley between (3 in 0.50-0.75, 17 in 0.95-1.00, 6 in
0.90-0.95). The cut therefore separates genuine amendments from mis-attributed
ones rather than tuning difficulty; the handful between 0.90 and 1.00 are cases
where eCFR renders a token slightly differently from the printed rule.

**Surgical amendments only.** Sections reprinted in full are dropped: the answer
is the printed block copied out. This filter reads the instruction text only
(`revised ... to read as follows` without a sub-provision address); it never looks
at labels or at how any solver performs.

**One item per (title, section), split disjoint by CFR part.** Agencies re-run the
same amendment shape annually — civil-penalty inflation adjustments being the
clearest case — so an item-level split would put one year's version in training and
the next year's in test.

---

## 6. Metric design

The obvious metric — similarity between the submitted and official text — is
useless here. An amendment moves a few dozen words in a section of several hundred,
so returning the section unchanged scores about 0.97.

So the comparison is between **edit scripts**. `difflib.SequenceMatcher` over
whitespace tokens gives the operations that turn the before text into each of the
two after texts; positions index the shared before text, so the scripts are
directly comparable. Score is F1 over those operations.

**Operations are weighted by the tokens they move.** This was not the first design
and the first design was wrong. With unweighted operation counts, returning only
the printed block scored **0.499** — because difflib aligns the printed paragraph
against its place in the section and emits the correct small edits *plus* one giant
deletion of everything else, and one giant deletion costs exactly one operation.
Weighting by tokens moved dropped that baseline to **0.229** and makes the score
read as "how much of the agency's text movement did you reproduce, and how much did
you invent".

**Operations are matched softly, and this was also a correction.** The first
version intersected operation tuples exactly, so an amendment applied to the right
paragraph but ending one token early scored the same as one never attempted. The
consequence was visible in the per-item distribution: on the then-current 30-item
split the reference solution scored exactly 1.000 on thirteen items and exactly
0.000 on eight, with nine anywhere in between. The mean was then essentially "what
fraction of items did you nail outright" — a near-binary quantity measured on 30
draws, which is a property of the scoring rather than of the task, and the same
bimodality that has sunk earlier designs. It also made scores collide: two
genuinely different solvers landed on the same value because they solved the same
count of items.

Gold and predicted operations are now aligned one-to-one, greedily, best pair
first. A pair earns credit only if it is aimed at the same place — spans overlap,
or start within `max(3, 0.25 * size)` tokens of each other — and then earns
`0.5 * span_IoU + 0.5 * token_F1(replacement)`. Precision and recall are weighted
by tokens moved as before, with each operation discounted by its match quality.

The correction moved the reference from 0.620 to 0.650 and, more importantly,
turned the per-item distribution continuous: 26 distinct per-item values across 47
items, against 9 non-degenerate values across 30 before. The ladder is insensitive
to the two constants — sweeping the proximity tolerance over 3–12 tokens and the
span/text split over 0.35–0.5 moves every anchor by less than 0.02 — so no anchor
depends on a tuned threshold.

Properties: returning the input unchanged scores exactly 0.000 (no operations);
returning the official text scores exactly 1.000; inventing edits costs precision;
missing them costs recall.

**Structural failure degrades, it does not abort.** The grader scores an omitted,
blank or duplicated row as "no amendment applied" — zero for that item — and
ignores rows outside the answer key. It raises only when nothing at all can be
scored. The earlier version raised on any of those, which converts a mediocre
solver into a failed run carrying no score, and a failed run measures nothing. For
a submission whose payload is 50 rows of multi-thousand-character
regulatory prose full of commas and quotation marks, the probability of some
structural imperfection is not small, and it should cost points rather than the
whole attempt. `grade.py` also retries a malformed CSV through progressively more
forgiving readers before giving up.

---

## 7. Measured anchors

All figures produced by `work/measure.py` through the shipped `grade.py`.

- Return the section unchanged — **0.000**
- Append the printed text — **0.155**
- Return only the printed text — **0.189**
- Regex address, replace to the next designator of any kind — **0.192**
- Same splice, addresses resolved against a real hierarchy — **0.400**
- Reference solution — **0.522**
- Official text (oracle) — **1.000**

**The test split is stratified by amendment complexity, and this was a
correction.** An agent scored 0.86 against a reference of 0.650 on the first
measured split, well above the intended band. The cause was visible on inspection:
that split was only 45 percent compositional against a corpus that is 58 percent,
and the reference scores 0.820 on the simple items against 0.439 on the
compositional ones. The simple case — one instruction revising one paragraph with
the whole replacement printed — is transcription once the paragraph is located, and
it was over half the test set, so most of the available score was being paid for
locating a paragraph.

An item counts as compositional when it carries a `* * *` partial replacement, a
run of more than two instructions, a redesignation, or a quoted removal. The split
now steers the test set to 65 percent compositional. The target was chosen by
sweeping it from 0.45 to 0.80 and measuring the ladder at each point, not by
picking a number that produced the wanted headline:

- at 0.45 (the accidental original) the ladder was **not** ordered — an ablation
  overtook the full reference
- past about 0.70 every solver collapses towards the same score: on compositional
  items even good solvers do badly, so the differences stop being signal
- 0.65 keeps the ladder strictly ordered with the rungs apart, which is what a
  difficulty gradient means

Train is steered too, and holds 29 compositional items of 56. Starving train of
the hard cases would make `* * *`, redesignation and removal undevelopable from
labelled data — unfair rather than merely hard.

Faithful partial implementations — each a real solver with one capability removed,
which is the shape a solver lands on when it gets part of the problem right:

- Hierarchy splice, no `* * *` handling — **0.400**
- Flat designators and no word removals — **0.419**
- Flat designators, i.e. no sentence-boundary test — **0.445**
- Full hierarchy, no word removals — **0.494**
- Reference — **0.522**

Five distinct, strictly ordered values. Dropping the hierarchy entirely falls to
**0.192**: skipping the central difficulty is supposed to cost the band, and does.

The two measured agent runs on the previous split scored 1.08x and 1.32x the
reference of that split, which projects to **0.56 - 0.69** here — inside the
intended 0.4-0.7 band at both ends, with margin.

Most of the available score sits in one step: regex splice to hierarchy splice.
The two do the same thing and differ only in how the addressed paragraph's extent
is determined, which is the point of the task. The reference adds partial
replacements, deletions and redesignations on top for a smaller further gain.

Measured per category (reference against hierarchy splice, on the probe corpus):
plain paragraph revision 0.694 -> 0.725, partial `* * *` replacement 0.423 ->
0.506, deletions 0.207 -> 0.248, redesignation 0.000 -> 0.016, tables 0.060 with
no gain. Partial replacement is where the extra machinery pays; tables are the
weak category and are documented as such below.

## 8. Reference solution

Pure Python, no learning, no dependencies beyond the standard library and pandas.

1. **Paragraph hierarchy.** Assign each designator a level from the sequence it
   continues rather than from its character class, because `(i)` after `(h)` is a
   first-level letter while `(i)` after `(1)` opens a third level.
2. **Openings versus cross-references.** A designator opens a paragraph only if the
   preceding token ends a sentence. Regulations cite their own paragraphs
   constantly; the first version of this check rejected `"of this part."` because
   "part" was on a cross-reference word list, which lost paragraph openings — the
   test has to be on punctuation, not on the word.
3. **Partial replacements.** `* * *` means the remainder survives; how much of the
   original the printed fragment displaces is a sentence-boundary problem in prose
   dense with `U.S.C.`, `Pub. L.` and `Jan.`
4. **Sequential application**, so redesignations move later targets.

Measured progression while building it, on the probe corpus: 0.439 → 0.526 after
fixing cross-reference detection and designators following `* * *`.

---

## 9. Difficulty gate

PENDING - agent gate not yet run

---

## 10. Known limitations

- Amendments to tables are the weakest category. Tables flatten into long token
  runs under canonicalisation and their addresses ("revising Table 1 to paragraph
  (b)") resolve poorly. They remain in the corpus — they are genuine amendments and
  the metric handles them consistently — but neither the baselines nor the
  reference does well on them.
- Canonicalisation to a single line discards the original paragraph breaks. This
  is deliberate: it removes a formatting channel that would otherwise mark
  paragraph boundaries for free, and makes the hierarchy something to be inferred.
- Coverage is 2021-01-05 to 2021-03-08. eCFR sustains about 24 requests a minute whatever
  the concurrency, and every item costs two point-in-time fetches, so the size of
  the corpus is set by download time rather than by the availability of data --
  the same scan run longer extends it without any change to the design.
