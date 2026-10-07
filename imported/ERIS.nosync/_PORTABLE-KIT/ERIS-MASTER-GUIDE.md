# ERIS / Shipd Challenge Builder — Master Guide

Everything learned across the whole portfolio: ~20 built challenges, every
platform rejection, every human-reviewer finding, every measured failure. Each
rule below cost a revision round. Nothing here is theoretical.

Carry this file plus `ERIS-BUILD-PROMPT.md` to any machine. Together they are
the whole build system.

---

## PART 0 — Platform hard constraints (violate any = auto-reject)

- **GPU track.** Solver hardware is **1× NVIDIA A10G (16 GB) + 10 CPU cores +
  62 GB RAM**. The task must genuinely *leverage* the GPU, not merely tolerate
  it. (This superseded an older CPU-only rule — that rule is dead.)
- **Runtime** max **1.5 h**, target ~1 h.
- **Novelty ≥ 6** to submit at all. Aim 8; see Part 6 — it is stochastic.
- **No tabular / regression challenge.** Shipping one **bars you from creating
  another**.
- Deterministic evaluation AND deterministic, seeded dataset generation.
- Public training set, hidden test labels, **CSV submission only** (exact
  filename, columns, dtypes).
- No internet and no external datasets at solve time. Self-contained.
- `domain: natural-language-processing` for text tasks. Never `classification`,
  `anomaly-detection`, `outlier-detection`, `clustering` — those route to
  disabled domains even if every other check passes.

---

## PART 1 — Five gates, before writing a single line of code

All five must pass. A failure here means **pick a different source or task**.
None of these can be patched afterwards without redoing the build.

### Gate 1 — Licence is a named value from the platform dropdown

Accepted: CC0 1.0, CC BY 2.0/2.5/3.0/4.0, CC BY-SA 3.0/4.0, PDDL, ODC-By,
ODbL, CDLA Permissive 1.0/2.0, CDLA Sharing 1.0, C-UDA, MIT, Apache-2.0,
Public Domain Mark, U.S. Government Work.

**Verify from the source's own API or metadata — never from memory or a
marketing page.**

- Wikimedia: `https://<site>/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json`
  → quote `rightsinfo.text` and `rightsinfo.url` verbatim.
- Publishers: read the `<license>` block in the actual per-item metadata (a
  PLOS article's own JATS XML), not the publisher's marketing page.
- Government/API: find an explicit named licence, not "open data".

**Reject immediately on:** "we are not aware of copyright restrictions"
(a disclaimer of ignorance, not a grant), "freely available" / "open access"
with no named licence, "Custom"/"Unknown" with no specificity.

> Measured: `dailymed.nlm.nih.gov` was **rejected at this gate** — NLM publishes
> "determine copyright status yourself", not a licence.
> `clinicaltrials.gov` was **abandoned** — its licence statement is
> JS-rendered and could not be quoted.

**Cross-check the metadata FIELD against the prose everywhere.** Palimpsest's
prose all said CC BY-SA 4.0 (correct) but the platform's declared licence field
said CC BY 4.0, silently dropping the ShareAlike obligation. Only the field is
machine-read, and they drift independently.

### Gate 2 — Source URL is new, and Duplicate Detection is NOT URL-based

One source, one dataset, **ever** — including abandoned and rejected builds.

**The critical finding:** an openFDA submission was rejected naming a
*CourtListener* dataset, which cannot possibly share its URL. **Four different
URLs were tried on that build and the error survived all four** (bare endpoint,
scoped query, bulk-download page, a different publisher).

The check reads **title wording and dataset shape**, not just the URL string.
So:

- Do not plan to "just change the URL later" — it does not work.
- Change what actually collides: the **dataset title** and the **task shape**.
- Two datasets that both mask a corpus and ship "items" read as near-duplicates
  regardless of source.
- A bare CC0 deed URL and a Gutenberg book id both collided with unrelated
  datasets — do not point `source_url` at a licence page or a single item.

> Also measured: "Unscrambling the Bibliography" was rejected with
> `source_url = en.wikipedia.org/w/api.php` against an account-external dataset
> ("Notable-Figure Wikipedia Biography Extracts"). Repointing to the enwiki
> bulk-dump root was the fix applied, but that carries real residual risk by
> this same rule.

### Gate 3 — The label is the archive's own recorded structured value

The target must already exist in the source as structured data written by the
source's own contributors, editors or system: a tag, a category, a database
field, a document's own ordering, a template parameter, an approval date.

It must **NOT** be:
- an annotation you create,
- **any** LLM-generated content, anywhere in the pipeline (absolute platform
  rule),
- a value you infer from unstructured text with your own heuristic — that is
  feature engineering, and belongs in a baseline, not the answer key.

> Measured: a Crossref retraction-reason task was **abandoned at this gate** —
> 72% of retraction-notice abstracts state no extractable reason, so any label
> would have been my own regex over free text.

Good examples that passed: a `{{IPA|en|...|a=RP}}` accent tag in Wiktionary;
a `[[Category:North America]]` assignment; a `<sec><title>Results</title>`
document position; a `<SPEAKER NAME="...">` tag; HGNC's own
`date_approved_reserved`.

### Gate 4 — Task shape: two hard rejects, then originality

**HARD REJECT — do not build, do not try to reword:**

1. **Tabular prediction.** Rows of features, predict a held-out column. What
   matters is whether the *task* is "given these columns, predict that column",
   not whether you ship CSV.
2. **Anomaly / outlier / novelty / intruder detection.** "Find the one that
   doesn't belong", "flag the unusual record". Routes to a disabled domain
   however it is phrased — one build was rejected after three rewordings.

**Also avoid:** plain single-label classification into a fixed published
vocabulary.

**Shapes that shipped successfully:**

| Shape | Submitted artifact | Metric used |
|---|---|---|
| Seq2seq rewrite | a string | change-segment accuracy |
| Seq2seq generation | a string | content-word F1 / changed-token F1 |
| Seq2seq coinage | a short string | prefix agreement |
| Ordering / ranking | a permutation | rescaled Kendall's tau, position accuracy |
| Ranking with partial credit | a permutation of given candidates | positional credit |
| Set selection from a pool | a subset | per-query MCC |
| Partition recovery | unlabeled grouping | per-bag ARI |
| Span extraction | a substring | IoU on spans |

Note partition recovery and set selection **both passed** despite
clustering-family metrics. What saves them: the submitted artifact is not a
label from a fixed set. ARI build → labels are arbitrary tokens meaningful only
within one bag. MCC build → the candidate pool is handed over in the row and
the subset size is unknown and variable.

**The submitted artifact is what makes task formulation pass.** A forest, a
partition, a permutation, a coined string, a program, an ordered op pair. A
numeric prediction reads as "just regression" no matter how it is framed.

**Originality** compares *shape and metric*, not wording. Differ from the
nearest known benchmark in **≥2 of 3**: input shape, output shape, metric. Name
the closest benchmark explicitly and check. Ideas killed this way before
starting: stylometric authorship separation (PAN), sentence/paragraph intrusion
(INSteD), and two reskins of the portfolio's own earlier work.

### Gate 5 — Source URL must answer an automated request

The platform fetches it and rejects unreachable or 403 responses. Test with
plain `curl -sI` before committing.

> Measured: every `chroniclingamerica.loc.gov` and `loc.gov` page except the
> static bulk-data root is Cloudflare-blocked (403). That constrained what the
> Source URL could point at for the whole build.

---

## PART 2 — Feasibility, before committing to a design

### 2.1 Sample 30–50 items by hand

Measure yield, item length, label distribution, fetch rate and rate-limiting.

**Item length carries the signal.** One attempt used ~38-character units; the
trained reference scored *below* a content-free baseline, and a pooled
adversarial baseline beat the honest reference outright. The source was
abandoned. The fix was **more signal per item, not a smarter model**. If a
human couldn't do the task by reading one unit, the unit is too small.

### 2.2 Scale thresholds

- **≥500 usable items**, 2,000+ preferred.
- **≥60 test rows**, 200+ preferred — the platform halves answers into
  public/private and each slice needs ~10 rows.
- **≥100 examples of the rarest class** you keep. Drop classes below this.
- **train ≥ 1000** (platform minimum).
- **train > test always.** A reviewer explicitly rejected train 101 / test 179.
- **test/train ratio 15–25%.** 66.6% was called too high.

### 2.3 Run every leak probe on the sample BEFORE scaling

| Probe | What it catches |
|---|---|
| Constant / majority answer | Chance floor — sanity check |
| Item length (chars, tokens) | Length encodes the label |
| Punctuation / digit / character counts | Register leaking through surface stats |
| Capitalised tokens only | Named entities leaking the label |
| Position in shipped row order, or slot letters | Shuffling is wrong |
| Every metadata column you plan to publish | A "helper" feature that IS the label |
| Copy the input through unchanged | Any rewrite/transform task |
| **Adversarial: ignore the task structure entirely** | Pooled modelling beats the per-unit approach |
| **Lookup/sibling analogy from train** | The task is retrieval, not generation |

**A probe within ~0.10 of your planned reference is a fail. A probe beating it
means abandon the design.**

The adversarial probe is most often skipped and has caught the most severe
failures. Real findings:

- **Position-only scored 0.1434 against the reference's 0.0980.** A control
  confirmed real signal (random half of pairs scored 0.082). The entire task
  shape was redesigned so candidate ordering is meaningless by construction.
  Verified after: `always_first_two` 0.1438 vs `random_two` 0.1453 — position
  buys nothing.
- **Global clustering ignoring bag boundaries** beat the trained reference.
  Fixed by raising the test fraction to 45% — *measured as the optimum*.
  Raising to 55% made the reference *worse*, which closed off "hold out more"
  as a monotonic fix.
- **A sibling-lookup adversary scored 0.366** on a name→symbol coinage task
  against a plain seq2seq's 0.447 — only +0.081. Fix: make the reference
  *retrieval-augmented* so the adversary becomes a strict ablation of it.
- **Corpus-consensus voting scored 0.389 against a 0.425 ceiling**, collapsing
  the advertised skill gap to +0.036. Found only by writing the attacker.

### 2.4 Copy-through must be measured before the metric is frozen

If the correct output shares >50% of its content with the input, measure what
copy-through scores under your chosen metric **first**. If it's above ~0.3 the
metric is wrong.

> Measured: an accent-rewrite task where source and target are ~80% identical
> gave copy-through **0.795** under plain edit similarity, leaving a 0.2-wide
> band for every real submission. Rescoring only the *changed positions* sent
> copy-through to **0.112** and used the full range.

---

## PART 3 — Metric design

The metric determines whether degenerate answers score near zero. **Choose it
before building the split, and measure the degenerate baselines under it.**

### Measured failures that forced a metric change

**Copy-through under edit similarity** → 0.795. Score changed positions only →
0.112.

**One-giant-cluster under pairwise F1** → 0.376 against a trained reference's
0.289. Pairwise co-membership F1 gets perfect recall free when everything is
one group. Adjusted Rand Index is chance-normalised, so both degenerate
strategies (one group, all singletons) score exactly **0.000** and no
label-alignment step is needed.

**Recall alone over changed tokens** → a vocabulary dump scored **1.000**,
identical to the oracle. Adding a precision term over what the prediction
*adds* sent the same attack to **0.008**.

**The oracle scored 0.568, not 1.000** — a real bug. Root cause:
`extra = Counter(prediction) - Counter(shown)` cancels a changed token that
also occurs elsewhere, making those tokens unreachable. Fixed by deriving both
sides through the *same* alignment. **This bug was invisible on hand-built unit
tests** (short strings, no repeated tokens) and only appeared on real text.

**Hungarian / matched-assignment scoring** hands out credit a model never
earned on anything clustering-shaped. A reviewer praised "immediate parent
exact match avoids the matched assignment trap". Prefer metrics scoring each
element's **direct, unambiguous** target over ones that find the best alignment
after the fact.

**Bimodal per-item scoring** — exact set-intersection gave 13 items at exactly
1.000 and 8 at exactly 0.000 of 30. Different solvers collide. Score overlap +
content, aligned softly.

**Prefer a metric with an attainable 1.0.** One build scored 0.39 against a
0.3965 ceiling = 98.4% of max, and the compressed sub-0.5 scale *hid* it. The
reviewer rejected it and said do **not** rescale the ceiling; make the task
harder.

### Rules

- Output shares >50% with input → score only the positions that should change.
- A constant answer is structurally possible → use a chance-normalised metric
  (ARI, MCC) or verify the degenerate score is ~0.
- Partial credit beats all-or-nothing when a near-miss is genuinely most of the
  answer — but only if the partial-credit axis is the thing being measured.
- **`problem-description.md` must argue why this metric and not the obvious
  alternative**, naming rejected alternatives with *measured* reasons. The
  platform fails specifications that leave this unjustified:
  > "The specification does not explain why [metric] measures the intended
  > quality... leaving the evaluation design unjustified"

Strong "why not X" reasons look like:

- "Exact match collapses every partial result into one bucket — a prediction
  that finds the right run but stops one copy short is worth the same as one in
  unrelated sequence."
- "A metric built only on overlap is gamed by submitting the entire window;
  that is why the union term is not optional."

---

## PART 4 — Fetching

- **One worker, always.** Parallel fetches get throttled to worse-than-serial.
  Tried three times in one session; every time total wall-clock was worse than
  a single patient worker.
- **Sleep 0.8–1.5 s between requests.**
- **Catch `Exception` broadly** in retry loops. `ConnectionResetError` is not
  an `HTTPError` and killed a 25-minute fetch.
- **Write incrementally** so an interruption loses one item, not everything.
  Or support `--start` so a second run extends the first.
- **Never kill a running fetch to improve something.** Finish, then change,
  then re-run. Every mid-fetch kill cost 20–40 minutes.
- **Prefer bulk dumps over paginated APIs.**
- **Budget per-attribute** if the source is ordered by something you care
  about — a flat byte budget exhausts on the first attribute value.
- **Stream to a `.part` file and rename on completion** so an interrupted run
  never leaves a truncated file.
- **Pin a dated snapshot + sha256** where the source republishes a rolling
  file, and verify the checksum after download.

### API quirks that cost real time

- eCFR `/full/{date}/title-{n}.xml` **requires** an `Accept-Encoding` header
  permitting compression and returns **HTTP 406** without one. Not prominently
  documented.
- eCFR `/versions/` caps at 1000 results and rejects `per_page`.
- CourtListener: use the citation-map membership filter, not a court
  pre-filter — 7.6% vs 0.003% hit rate on the same 1 GB read. The 2022-08
  citation-map snapshots are empty placeholders.
- HGNC: the archive listing is at `genenames.org/download/archive/` but files
  live on `storage.googleapis.com/public-download-files/hgnc/...`; enumerate
  with the GCS JSON API (`?prefix=...`) because guessed dates 404.

---

## PART 5 — Split, leakage, and shortcut closure

### 5.1 Split by the real unit, then check the shipped values

Split by document/word/family/batch/part so no source unit appears in both
splits. **Then check exact overlap on the *shipped feature columns*.**

> Measured: document-disjoint is not value-disjoint. Homophones leaked ~4% of
> test rows through a word-disjoint split — two different words share a
> pronunciation.

- An *input* value repeating across rows is fine **provided the target is never
  shared**. Verify and document.
- If the exact (input, target) pair appears in both splits: collapse to unique
  values, drop values carrying conflicting labels, assign each unique value
  wholly to one split.

**Split must be disjoint by group, never by row.** Row-level splits leak.

**Choose which units are held out deliberately** — not alphabetically, not by
shuffle. An exhaustive/bounded search for the ratio band *and* the lowest
shortcut score is reproducible and defensible.

**Balance the answer distribution before splitting.** An unbalanced top answer
let a zero-signal majority-vote baseline score 0.43 against a 0.25 chance
floor; capping per-answer counts within each group closed it.

**Stratify BOTH splits.** A random split drew 45% hard items from a 58%-hard
corpus and inflated every score (agent 0.86 vs a 0.650 reference). But **sweep
the knob and measure the whole ladder at each point**: past ~0.70 hard-fraction
every solver collapsed to the same score and one ablation overtook the full
reference. Keep train stratified too, or the hard cases become undevelopable
and the split is unfair rather than hard.

**Drop items unwinnable by construction** so every graded row is winnable.

> A whole split design can be wrong: a family-disjoint split for a name→symbol
> coinage task put a trained seq2seq at **0.258 against a 0.337 one-line
> initials heuristic**. A trained reference below a trivial heuristic means the
> split removed the signal along with the shortcut. Replaced with the archive's
> own approval dates: reference 0.447, best adversary 0.366.

### 5.2 A distinct rejection class: transductive / cross-row leakage

One challenge was rejected for **"violating independent single-sample inference
and using transductive test-set adaptation."** Concretely: the reference fit
global TF-IDF/N-gram models **on the test data itself**, and **grouped separate
test rows** to intersect their candidate banks — exploiting a dataset
construction artifact instead of making an independent per-row prediction.

The columns were clean. The split was clean. The **task as posed** let a solver
treat the test set as a corpus.

Check for any new design:

- Can the prediction for row *i* be computed **without looking at any other
  test row's hidden content**?
- If cross-item statistics are legitimately needed, they must come from
  **train only**, never fit on the released test set.
- Verify no public field is unique-mapped to its label across train+test
  combined — that lets a solver build a global lookup table.
- Reject any design where "intersect what's left across several test rows" is
  viable. **Attempt that attack with a real solver before shipping.**

### 5.3 Column and ID rules

- `train.csv` columns minus the target == `test.csv` columns, **exactly**.
  Re-derive programmatically after any rename.
- `test.csv`'s first column is unique per row.
- `answers.csv`'s join key == `test.csv`'s first column — same values, same
  count. **`answers.csv`'s first column must equal the submission's ID column**
  or you get "Submission missing required ID column".
- **The join key is the ONLY column name shared between `test.csv` and
  `answers.csv`.** If the grader needs an input value again, carry it under a
  *different* name and assert equality at build time. A shared name is read as
  a leaked target even for a legitimate input.
- No column empty in ≥50% of rows. Cap packed slots at what most items reach;
  truncate rather than leave sparse.
- **Publish no redundant numeric column.** A `length` equal to `len(text)` adds
  nothing and hands the outlier profiler a long-tailed column to flag.
- **Prefix categorical codes that look numeric.** CFR part numbers run 1..1357
  with a long tail, which a numeric profiler flags as ≥5% extreme outliers.
  `p155` instead of `155` makes them unambiguously categorical.

### 5.4 Leaderboard slicing

The platform splits `answers.csv` into public/private **by row**. If one graded
unit spans several rows, a row-wise cut fragments it.

> Measured: `answers.csv` had one row per snippet, so every bag became two
> fragments graded as separate bags. 24 public fragments had ≤3 rows — and on a
> 3-row fragment from three groups, an all-singleton guess is a perfect
> partition scoring **1.0**. Fixed structurally: one row per bag, labels packed
> into one cell in test.csv's row order.

**Check:** score a row-wise half-split of `answers.csv`; confirm it matches the
full file for the reference and that degenerate submissions score 0 on both
halves.

**Slices must be representative.** Order rows by a stratifying key, then
golden-ratio rekey + pairwise balance, and verify every contiguous slice scores
near the full set.

### 5.5 Closing an external-lookup shortcut

If the corpus is real and publicly searchable, a solver can look the answer up.
Closure must be **layered** — no single layer is enough:

1. **Nothing public names the source units.** Publish opaque aliases
   (`batch_07`), never the archive's ids. Keep the real ids and the alias map
   in `private/`.
2. **Nothing public lets the unit set be re-enumerated.** Remove the
   candidate-selection recipe and any skipped-unit list from public docs.
3. **Nothing public is byte-identical to the archive.** Ship a contiguous
   window (80–97% of the region, bounds from a private salt). Measured cost:
   reference ARI fell 0.298 → 0.278 at 80–97%, but 0.298 → 0.191 with a small
   absolute window — keep the fraction high, since windowing's only job is to
   break byte-equality.
4. **The rule makes lookup invalid regardless of score** — in
   `problem-description.md` and mirrored in a `[REQUIRED]` rubric.
5. **The solver environment is expected to be offline.**

**State plainly what the closure cannot do.** Windowing breaks exact matching
but not substring matching. Against an adversary already holding the source
text, any signal-preserving transform of a unique document stays findable. That
is exactly why layer 4 is load-bearing, and saying so is better than implying
the obfuscation is complete.

**Don't name the pointer.** Don't state the assembly, track, region or
coordinates that let a participant download the exact data. Cite the *method*,
not the pointer. The Source URL itself usually cannot be hidden (the platform
fetches it) — point it at a data root and rely on the other layers.

**The raw-vs-public split can resolve a licence-vs-leakage tension.** An
`attribution.csv` (titles + revision IDs, required for CC BY-SA compliance)
stayed in the RAW archive for provenance but was deliberately **excluded from
the public split**, because titles/IDs are exactly the lookup key. Real
attribution data and leak-safe public data can be the same file, scoped
differently.

### 5.6 Measurement bugs to watch for

- **Keying an assignment dict by the wrong key.** One harness keyed by bare
  `snippet_id` while snippets appeared in several bags, so later bags silently
  overwrote earlier ones. Caught only by an unexplained gap between two
  implementations of the same logic (0.148 vs 0.316). Key by the full
  `(group_id, item_id)` tuple.
- **Line-splitting CSV with embedded newlines.** Splitting decompressed bytes
  on `\n` before parsing corrupts rows whenever a text field has a paragraph
  break. Use a real multi-line-aware parser.
- **Leak scans that join fields before scanning.** Joining every text field
  into one string lets a pattern match across the seam between two documents.
  Scan per field.
- **Regexes too tight for OCR'd source.** A citation pattern using `\s?` missed
  `80 L. Ed.    2 d 674` where the source broke across a page boundary. Widen
  to `\s+`/`\s*` and re-scan to zero.
- **Target the SHAPE of a leak, not its spelling.** A fix stripped every
  template parameter whose *value* matched an ISO-date pattern rather than
  matching field names — editors misspell field names, but the date's shape is
  what leaks. Pattern-class removal beats keyword removal.

**Whenever two implementations of the same measurement disagree, stop and find
out why before trusting either.**

---

## PART 6 — Novelty (target 8, floor 6)

**Novelty is stochastic.** The same task scored 6, 4 and 3 on re-runs. Another
scored 5, 4, 4, 4 across four description rewrites of the same data.

**The displayed score runs 1–2 below the analysis text's own number.** Every
run: analysis said "capped at 6" → displayed 5; "scored at 6" → displayed 4;
"effective 6" → displayed 4. A consistent unexplained gap.

**Scoring has explicit bonus components:** `raw novelty + 1 (research-backed
adaptation) + 1 (real-world dataset provenance)`. The provenance point is
**denied for synthetic data** — "no +1 ... as no independent evidence of
operational data origin was presented". **Every fully-synthetic challenge
starts a point down.** Use real, licensed, obscure source data.

**Levers that measurably moved it:**

- A dedicated **"What is new here"** numbered section.
- A **"How it relates to prior work"** section citing **distant antecedents
  only**. Naming a close recent neighbour triggers cite-or-cap and *drops* the
  score — naming "HiCBench 2026" cost points; naming TokTrack/WikiHist and
  differentiating them *raised* one build from 6 to accepted.
- **Real data: +1.**
- Changing the **task identity**, not the wording. Reframing alone took one
  build 4 → 7; the last point needed conceding the classical primitive openly
  and naming the new capability.

**Rewriting the description does not move the number.** One reframe actively
*backfired*: emphasising "recovering lost precision from rounded readings"
pulled in Rhines et al. 2015 as new nearest-neighbour prior art.

**Cite-or-cap:** the grader names a nearest neighbour and caps at 6 unless
exhaustive absence can be proven. Adding constraints does not help — one build
had every constraint credited (closed-data rule, per-field exact match, REJECT
class, information-theoretic ceiling, distribution shift) and still capped
because the *problem family* is well represented. **Only changing the family
moves it.**

**Check novelty on a draft problem statement BEFORE building the dataset.**

---

## PART 7 — The agent-difficulty gate (the single biggest killer)

Target: agent runs land roughly **0.2–0.6** (or the band you were given),
clearly above every content-free baseline and clearly below oracle.

### Diagnose by the SHAPE of the run spread

**Agents at 1.0** — pipeline-forensics (twice), babel-gauge (3/3 at 1.0000).
Cause: every fact is exactly recoverable from the input. No amount of
format-starving or corruption fixes it; difficulty must be **irreducible loss
built into the data**.

**One run far above the reference** (0.776 vs a 0.46 reference) — an agent
found a shortcut the rules don't forbid. Almost always an external resource: a
pretrained model, a public dictionary, a library that solves the sub-problem, or
looking the source archive up. **Fix with an explicit `## Rules` section, not by
changing the data.** Add a matching `[REQUIRED]` rubric.

**All runs near the chance floor** (repeat-harvest at 0.07) — the threshold was
tuned against the *upper* bound and never stress-tested against a detector that
finds the target but localises loosely, which is exactly what real agents do.
Usually: task underspecified, compute budget wrong, or the intended approach
not stated. Naming the intended model class in the Compute section helps.

**Agents saturating the ceiling** — 0.39 against a 0.3965 ceiling = 98.4% of
max, hidden by a compressed scale. Reviewer said do **not** rescale; make the
task harder.

**All agents at the same score** — three agents at exactly 0.4. A deterministic
task where every stage has an exact algorithm converges. The real defect was
*"this is an optimization puzzle that does not rank solutions."* **At least one
stage must have no exhaustive pattern** — held-out wordings, unlearnable
boundary noise, a genuine perceptual judgment.

**Reference below a naive baseline** — reference 0.2743 vs best fixed tile
0.2778. The claimed ordering was **inverted**. Always measure the naive rungs.

**Runs cluster just below the band** — genuinely too hard. Tighten the
difficulty knob and **re-measure the whole ladder**, not just the changed
number. One build went through four versions doing exactly this.

### Agents beating the reference is FINE — if three conditions hold together

All three agents scored above the reference on Palimpsest (0.60/0.64/0.64 vs
0.552) and the reviewer called it healthy, **because**:

(a) real headroom remained beneath the oracle ceiling (0.852),
(b) the **ordering was identical on public and private** halves and the
**spread matched almost exactly** (0.0396 vs 0.0413),
(c) the public-to-private gain was small and consistent across all three.

So "reference below agents" is only a problem when it *also* shows saturation,
inverted ordering, or an inconsistent public/private split. Check all four
together, never the score alone.

### The target ladder shape

**5+ distinct, strictly ordered scores from faithful partial implementations**
— the real solver with one capability ablated — not just baselines. Those
ablations are the shapes agents actually land on.

A reviewer named all **nine** measured points by value and then **re-ran the
platform sample check himself** (got 0.2470 against the published 0.252). The
ladder is only trusted once it reproduces outside your own numbers.

**Aim for ~0.18 margin between reference and best adversary.**

### The irreducible risk

The agent gate can only be run on Shipd. Everything else is verifiable locally.
Best proxy: the ablation ladder plus a ratio extrapolation once any single real
agent score exists (`agent/reference × new reference`). **Run ONE agent, read
the number, tune, then run the rest** — do not spend three credits on an
unverified band.

---

## PART 8 — Agent runs returning "Failed / no score"

"Agent runs completed but 0/N rounds" means no valid score was produced. Four
independent causes, all found while debugging one build. Check all four.

1. **Grader rejects instead of degrading.** Raising on missing rows, blanks or
   duplicates voids the whole run. Score them 0 instead. **`InvalidSubmission`
   must subclass `ValueError`**, not `Exception` — harnesses catch `ValueError`
   as "bad entry"; a bare `Exception` reads as a crashed grader.
2. **Bimodal per-item scoring.** See Part 3.
3. **Prepare not applied.** If the platform's Prepared Dataset lists the raw
   files, solvers get no `test.csv` *and* the answer key ships. Press Rebuild;
   confirm `public/` + `private/` appear.
4. **Data path assumed, not discovered.** The decisive one. The description
   said `dataset/public/`; the agent environment mounts at **`/data/`** with
   cwd `/workspace`. Only the agent that probed for the real path scored.

**Never hardcode one location.** Always ship this discovery snippet:

```python
from pathlib import Path
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/<name>/public"))
            if (p / "test.csv").exists())
```

**Ask for an agent trajectory early.** Three rounds were spent
reverse-engineering from artifacts; one pasted command revealed `/data`
immediately.

---

## PART 9 — Package layout

```
config.yaml                 metadata, dataset block, prepare/grade pointers, measured anchors
problem-description.md      the prompt shown to the solving agent
dataset-description.md      Overview / Source / Fetch method / File Structure /
                            Features / Intended Use / Limitations / License
rubrics.md                  5+ task-specific rubrics
README.md                   human-facing: title, files table, reproduce commands, provenance
DESIGN.md                   the measured design record — highest-value file for a reviewer
prepare.py                  prepare(raw, public, private) -> None, deterministic, seeded
grade.py                    grade(submission, answers) -> float, self-contained, degrade-not-reject
solution.ipynb              3 progressive submissions; final uses GPU explicitly
run_checklist.py            the pre-upload verification script
dataset/
  raw/                      exactly the documented files + ATTRIBUTION.txt + meta.json
  public/                   train.csv, test.csv, sample_submission.csv,
                            dataset_stats.json, validate_submission.py
  private/                  answers.csv (+ split_manifest.json if aliases are used)
  generator/                build_raw.py
  validate_submission.py    same bytes as the public/ copy
<name>-raw.zip              dataset/raw/ + dataset/generator/ ONLY
```

### Two uploads, two bundles — the easiest thing to get wrong

The **dataset** upload carries raw + generator and **nothing else**. Not the
split, not the stats, not the validator, and **never `private/answers.csv`**.

> A human reviewer caught exactly this: *"the upload holds the prepared public
> and private files, including answers.csv. please upload the raw files with the
> generator instead, and let prepare.py build the split."* The zip was **212 MB**
> where it should have been **9.9 MB**. The obvious `zip -r` over the package
> root sweeps up `public/` and `private/` without complaint.

```bash
# build it from the two directories explicitly, never from the package root
zip -rq <name>-raw.zip dataset/raw dataset/generator \
    -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store"

# 1. prove nothing prepared or private is inside
unzip -l <name>-raw.zip | grep -iE "public/|private/|answers|train\.csv|test\.csv" \
  && echo "LEAK" || echo "clean"

# 2. prove the raw zip alone rebuilds the split byte-identically
cd /tmp && rm -rf t && mkdir t && cd t && unzip -q <path>/<name>-raw.zip
cp <path>/prepare.py . && python3 prepare.py
for f in train.csv test.csv sample_submission.csv; do
  cmp -s dataset/public/$f <path>/dataset/public/$f && echo "MATCH $f" || echo "DIFFER $f"
done
cmp -s dataset/private/answers.csv <path>/dataset/private/answers.csv \
  && echo "MATCH answers.csv" || echo "DIFFER answers.csv"
```

**A size sanity-check catches it too.** If the raw zip is far larger than
`dataset/raw` itself, something prepared is inside.

**Removing `public/`/`private/` from the zip immediately breaks
`dataset-description.md`**, which was documenting those files. Fix both in the
same pass: the Files section lists only what the dataset ships; the prepared
split goes under its own heading *"What `prepare.py` derives (not shipped
here)"*; then grep the whole file for `train.csv`, `test.csv`, `answers.csv`,
`sample_submission`, `dataset_stats` — every surviving mention must read as
something `prepare.py` produces, never as a file the reader can open.

**Conflicting note from an older build:** one round found the dataset zip must
be **FLAT** (the validator cannot see inside a top-level folder) and must
already contain `public/` + `private/` because `grade_entrypoint.py` reads
`dataset/public/sample_submission.csv` directly. That contradicts the reviewer
finding above. **Resolution: follow the human reviewer (raw + generator only),
and if the platform then fails to produce a prepared split, press Rebuild and
check the Prepared Dataset listing** — see Part 8 item 3.

**Cross-check every file in each zip against its description before
uploading.** Two consecutive rejections came from this: one file named but
absent, one present but undescribed.

**Before packaging, replay the platform's own entrypoint paths from the
extracted zip.** Local runs pass while the harness fails purely on output
location.

---

## PART 10 — Document requirements

### `problem-description.md`

Section order: **Task → Dataset → Rules → Evaluation → Submission → Compute**.
(An "Overview" or "How this differs from adjacent tasks" may precede Task.)

1. **Title + hook.** Lead with a concrete worked example, not an abstract
   description.
2. **Task.** Exactly what one input row contains and what output is expected.
3. **What you are actually being asked.** Distinguish the task from the nearest
   similar-sounding task. This is where originality is argued for the human
   reviewer.
4. **Dataset.** The discovery snippet, then each file's columns and purpose.
5. **Rules.** Load-bearing — see below.
6. **Evaluation.** The metric as code, **why this metric and not the obvious
   alternative**, and every measured baseline with its score.
7. **Submission.** Columns, one row per id, and degradation behaviour — what
   scores 0 versus what is rejected outright.
8. **Compute.** GPU/CPU/RAM/time budget, the intended model class, and why it
   fits the metric.

**Formatting rules learned from rejections:**

- **No markdown pipe-tables** anywhere in problem-description or
  dataset-description. The renderer eats pipe characters. Use bullet lists.
- **No tilde code fences** (`~~~`) — breaks the parser. Only triple backticks.
- **Copy from the RAW file, never a rendered preview** — rendered copies drop
  pipes and closing backticks.
- **Title: plain and descriptive.** Rejected for being too long/paper-like, too
  catchy, and for a colon subtitle / creative project name.
- **Open with a clean declarative sentence**, then short bold-labelled sections
  and bullets. Dense mid-sentence openings get rejected.
- **Do not reference `config.yaml`** — it is not shipped to solvers.
- Numbers must come from the current shipped split. Grep for old task names,
  column names and metric names after any change.

### The Rules section — accepted wording pattern

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

**Name the specific tools that would trivialise the task.** A generic ban is
weaker than "no CMUdict, `g2p_en`, `eng_to_ipa`, `espeak`, `phonemizer` or
equivalent". **Mirror every rule in a `[REQUIRED]` rubric.**

### `dataset-description.md`

Required sections: **Overview, Source, Fetch method, File Structure, Features,
Intended Use, Limitations, License**.

- Every shipped file gets a bullet saying what it contains and what produced it.
- **Intended Use** needs all three: primary use, also-suitable-for, and
  explicitly **not**-suitable-for.
- **Limitations: ≥5 concrete points grounded in what was measured** —
  filtering that makes the set unrepresentative, a deliberate over-aggressive
  transform and what signal it costs, judgement calls in a word list, regex
  fragility, inconsistency in the source's own recording, long-tailed label
  space, single year / single model. **Generic disclaimers do not count.**
- **Fetch method** documents the deliberate choices: one worker, sleep
  interval, broad exception handling, incremental writes.
- If you drop a column for a platform check, say so and say why it costs the
  solver nothing.
- **This field describes the RAW upload's files, never the prepared split.**
  Pasting problem-description content here = "fundamental mismatch".

### `rubrics.md`

5+ rubrics, majority `[REQUIRED]` or `[RECOMMENDED]` (not `[UNIVERSAL]`), each
specific to this task and **approach-neutral** — never mandate a library or
architecture. Every rule in the problem description gets a matching
`[REQUIRED]` rubric.

Strong rubrics state a measured fact and its consequence:

> **[REQUIRED] Selects a variable number of candidates per query.** The number
> of true citations varies from 2 to 4 (mean 2.46 on the shipped test set).
> Both degenerate strategies score exactly 0.000 under the chance-corrected
> metric. A submission that always emits the same count is leaving points on
> the table by construction.

### `grade.py`

- `grade(submission, answers) -> float`, **exactly two arguments**.
- **No filesystem access beyond them** — no `os.path`, no `open()`, no env
  probing. If the grader needs the input text to score, carry it inside
  `answers.csv` under a different column name.
- **Degrade, never reject.** Missing row, unknown id, blank value, wrong-length
  answer → 0 for that unit. Only a missing required column raises
  `InvalidSubmission`, which **subclasses `ValueError`**.
- A duplicated id must not score as if correct — discard all its rows, score 0.
- A malformed id must not raise — treat as unknown and ignore.
- **Must tolerate extra rows and ignore IDs absent from the slice** (the
  platform grades row-wise halves).
- The docstring states the grading contract explicitly: which submitted column
  corresponds to which answer column, and that they are the same object in the
  same form.
- **Unit-test on ≥4 hand-built cases**: exact match, one error, no-change, full
  mismatch. One build shipped 21 grader tests.

### `prepare.py`

- **Must be self-healing.** Never `assert col in df.columns` for a value that
  is deterministically reconstructable — the platform may feed a raw file
  missing derived columns.
- **Tolerate a different raw layout.** The platform may pass a `raw_root` one
  level above or below the shipped one. Check the given root, then search its
  subtree for the expected filenames. **Verify against both layouts.**
- Deterministic and seeded; running twice must give byte-identical output.
- If the split uses no randomness at all, say so and publish no seed — a
  published generator seed is a Target Recoverability risk.

### `config.yaml`

Required: `name`, `title`, `domain`, `tags`, `dataset` (name, title,
short_title, description, generator, source_data, source_url, distinctness,
citation, license, license_url, license_note, raw_files),
`problem_description`, `rubrics`, `task_type`, `prepare` (script, public list
with column comments, private list, split_note), `grade` (script, usage,
submission_file, submission_columns, metric, metric_formula, direction,
min_score, max_score, **all measured anchors**, note_band, band_evidence,
reference_caveat), `submission`, `reference_solution`, `environment`.

**Keep the FULL baseline ladder under `grade:`** — every degenerate baseline,
content-free probe, adversarial attack, reference and oracle, each with a
comment saying what it is. **Keep earlier builds' ladders in comments** —
that history is evidence for reviewers, not clutter. `note_band` explains why
each number is what it is.

**Tag the domain honestly.** `rag` on a task that retrieves nothing was
flagged.

### `solution.ipynb`

- 3 progressive submissions; the final one uses GPU explicitly.
- Reads the discovered data path, writes `working/submission.csv`.
- **`ast.parse` every cell BEFORE writing the `.ipynb`.** A literal `\n` in a
  patch silently broke a cell and burned a 40-minute run.
- Notebook must execute end-to-end.

### `DESIGN.md` — optional and the highest-value file for a reviewer

Record: the task in one line; why this shape (including which candidate ideas
were killed and by what existing benchmark); the data and why this source; the
split and why that fraction; the metric and what it replaced after measurement;
every measured bug found mid-build; the adversarial findings with numbers; and
each review round with what was changed. **Write the numbers that were *wrong*
too, and why.**

> A reviewer's praise: *"The leak work is measured rather than asserted, which
> is rarer than it should be."* Not "we removed the leak" but the actual
> before/after number: a template-date field ordered lineages at 0.64 pairwise;
> after the fix, surviving prose dates ordered at 0.497 = chance. **Quantify
> the leak's magnitude AND the fix's effect, both as numbers.**

---

## PART 11 — Pre-upload checklist (run as an actual script)

Column mismatches and small leaks are invisible by inspection. **Any FAIL
blocks upload.**

### Gates
- [ ] Licence is a named platform-dropdown value, confirmed from the source's own API/metadata
- [ ] The licence METADATA FIELD matches the prose everywhere (BY vs BY-SA)
- [ ] Source URL is absent from your source register
- [ ] Source URL returns 200 to a plain `curl -sI`
- [ ] Every target value is a structured value recorded by the source itself — no annotation, no LLM output anywhere
- [ ] **NOT tabular prediction** — hard reject
- [ ] **NOT anomaly / outlier / novelty / intruder detection** in any phrasing — hard reject
- [ ] `domain:` is set and the task *shape* does not read as classification/anomaly/clustering
- [ ] The (input, output, metric) triple differs from the nearest known benchmark in ≥2 dimensions

### Raw data
- [ ] Every raw record parses without error
- [ ] Record count and label distribution match what feasibility measured (no silent collapse)
- [ ] Any checksum in `meta.json` matches the file it describes

### Reproducibility
- [ ] `prepare.py` run twice produces byte-identical `train.csv`, `test.csv`, `answers.csv`
- [ ] `prepare.py` works when `raw_root` is passed in a different layout
- [ ] Shipped files match a fresh `prepare.py` run

### Schema and IDs
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

### Leakage
- [ ] Target column absent from `test.csv`
- [ ] Train/test disjoint on the actual split unit
- [ ] No exact (input-features, target) row in both splits
- [ ] Where an input value repeats, the corresponding target is never shared
- [ ] Every probe scores ≥0.10 below the reference
- [ ] The adversarial "ignore the task structure" attack scores well below the reference (~0.18 margin)
- [ ] **No transductive shortcut**: a solver cannot fit statistics on the released test set, or intersect multiple test rows to eliminate distractors
- [ ] If the corpus is publicly searchable: no public file names a source unit, no public file enables re-enumeration, no shipped row is byte-identical to the source

### Leaderboard slicing
- [ ] A row-wise half-split of `answers.csv` scores the same as the full file for the reference
- [ ] Degenerate submissions score 0 on both halves
- [ ] No graded unit spans multiple rows in `answers.csv`
- [ ] Every contiguous slice scores near the full set

### Grader
- [ ] `grade(sample_submission, answers)` returns a finite float
- [ ] `grade()` is deterministic across two runs
- [ ] `grade(oracle, answers)` == max_score
- [ ] `grade(blank, answers)` == min_score
- [ ] Missing required column raises `InvalidSubmission` **and it subclasses `ValueError`**
- [ ] Missing row / unknown id / blank / wrong-length → 0, no raise
- [ ] Duplicated id → 0 for that unit, does not score as if correct, no raise
- [ ] Malformed id does not raise
- [ ] Returns a **number** for ~20 deliberately damaged submissions
- [ ] No filesystem access beyond the two arguments
- [ ] Unit tests pass on ≥4 hand-built cases
- [ ] Submission column and answer column name the same object in the same form
- [ ] Reference per-item score distribution has **many distinct values**, not mostly 0s and 1s

### Files and config
- [ ] All `.py` files parse (`ast.parse`)
- [ ] `config.yaml` parses as valid YAML
- [ ] `solution.ipynb` is valid JSON; final cell matches the stated compute budget
- [ ] `validate_submission.py` byte-identical between its two locations
- [ ] Zero `__pycache__` directories
- [ ] The raw zip contains exactly the files listed in `dataset-description.md`
- [ ] **The raw zip contains NO `public/`, NO `private/`, NO `answers.csv`, NO `train.csv`/`test.csv`**
- [ ] **Extracting the raw zip and running `prepare.py` regenerates every split file byte-identically**
- [ ] The raw zip's size is consistent with `dataset/raw` alone
- [ ] `dataset-description.md` describes only what the dataset ships
- [ ] No line tells the reader to open a file the dataset does not ship
- [ ] Platform entrypoint replayed **from the extracted zip**

### Docs
- [ ] `dataset-description.md` has all 8 required sections
- [ ] Every shipped file is named and described
- [ ] Intended Use lists primary, also-suitable, and not-suitable
- [ ] Limitations lists ≥5 specific measured points
- [ ] Title identical across `config.yaml`, `problem-description.md` H1, `README.md` H1, `solution.ipynb` first cell
- [ ] `problem-description.md` has a Rules section, and every rule has a matching `[REQUIRED]` rubric
- [ ] Evaluation argues why this metric and not the obvious alternative
- [ ] No markdown pipe-tables, no tilde fences
- [ ] No leftover wording from a previous iteration (grep old task/column/metric names across all `.md`/`.yaml`/`.ipynb`)
- [ ] Every numeric claim matches a number measured on the CURRENT shipped split
- [ ] `problem-description.md` does not reference `config.yaml`
- [ ] `rubrics.md` has ≥5 task-specific, approach-neutral rubrics

### Baselines — measured on the FINAL shipped split
- [ ] Degenerate/constant baseline measured
- [ ] Uniform-random baseline measured
- [ ] ≥1 content-free probe measured
- [ ] Adversarial attack measured
- [ ] Lookup/sibling-analogy attack measured
- [ ] **≥5 ablations of the real solver, distinct and strictly ordered, reference on top**
- [ ] Reference model measured, clearly above every baseline
- [ ] Oracle confirmed at max_score
- [ ] Answer-knowing adversary cannot exceed the stated ceiling
- [ ] All anchors in `config.yaml` with a `note_band` explaining each
- [ ] Reference sits in a defensible band — not near-chance, not near-perfect
- [ ] Reference runtime well inside 1.5 h on A10G

### Done means
1. Every item above run as a script and shown PASS, not assumed.
2. Source URL added to your source register.
3. A backup (git commit or timestamped copy) of the passing state exists.
4. Every number quoted in every doc was measured on the exact files shipped.

---

## PART 12 — Platform error messages → real cause

### Challenge-level
- **"Problem Originality... appears similar to an existing problem"** → task
  shape + metric matches a known benchmark even on new data and new wording.
  Change input shape, output shape, or the metric's fundamental logic.
  Rewording never works.
- **"This challenge was classified as [X] which is not currently accepting
  submissions"** → framing routes to a disabled domain. Routing reads task
  *shape*, not the `domain:` field. Reframe as seq2seq / ranking / extraction /
  set-selection / partition.
- **"The specification does not explain why [metric] measures the intended
  quality... leaving the evaluation design unjustified"** → add a "Why this
  metric" subsection naming 2–3 rejected alternatives with measured reasons.
- **"Agent runs & difficulty" with one run far above the reference** → an
  unforbidden external resource solves the sub-problem. Add `## Rules` + a
  matching `[REQUIRED]` rubric. **Do not change the data.**

### Dataset Quality
- **"The documentation lists [file] ... but that file does not appear"** →
  ship it, or remove it from the description and say where it is produced.
- **"The documentation does not describe the contents or purpose of [file]"** →
  add a bullet. Cross-check *every* file in the zip.
- **"the listed files do not match the described file structure"** → the
  description lists `public/`/`private/` but the dataset upload only carries
  raw + generator. Two different bundles.
- **"The stated license does not match the dataset source"** → sync the licence
  everywhere: `config.yaml`, `dataset-description.md`, any licence field inside
  raw JSON, and the generator's header. Re-hash anything whose checksum is
  recorded.
- **"does not describe an intended use... omits known limitations"** → missing
  required sections.
- **"The description does not document [raw file]"** → a stale file from an
  earlier iteration is still in `raw/`. Delete and rebuild the zip.
- **"A dataset from the same source URL already exists"** → source reused.
- **"A dataset from the same source URL already exists (<other dataset>)"**
  where that dataset is on a **different source entirely** → the check is not
  matching the URL. **Do not swap URLs.** Change the dataset **title** and the
  **task shape**.

### Prepared Data Integrity
- **"The public test IDs are not unique"** → `test.csv`'s first column repeats.
  One row per graded unit with a unique id, or pack sub-items into a wide row.
- **"Train and test feature columns must match"** → target column renamed
  inconsistently. Re-derive both column sets programmatically.
- **"The public test data contains evaluation target columns"** → a column
  *name* is shared between `test.csv` and `answers.csv`. Rename the copy in
  `answers.csv`; assert equality at build time.
- **"Train and test contain repeated feature rows that can leak evaluation
  data"** → document/word-disjoint split still left identical *feature values*
  in both splits. Dedupe on the shipped value.
- **"At least one column is missing values in half or more profiled rows"** →
  wide/packed format with sparse high-index slots.
- **"A numeric column has at least 5% extreme outliers"** → a long-tailed
  numeric column, often **redundant**. Drop it, or prefix it as categorical, or
  winsorize to a measured percentile.
- **"Public and private evaluation slices each need at least 10 rows"** → ship
  ≥60 test rows, ideally 200+.

### Target Recoverability
- **"the submission and answer columns do not identify a supported target"** →
  the submitted column and the answer column are different *kinds* of object
  (submission is a string, answers hold numeric coordinates). Carry the answer
  in the submission's own form under a clearly-named column, keep scoring aids
  alongside, and state the correspondence in `grade.py`'s docstring.
- **"public/train.csv: unreadable or unsupported table"** → **do not fix this
  by changing the encoding without checking what the encoding was protecting.**
  One build encoded ordinal fractions as `k/4` strings specifically to dodge a
  numeric-outlier flag; converting them to floats broke `prepare.py` on the
  platform.
- **"No verified shortcut in inspected rows; some artifacts or rows were not
  evaluated"** → informational, no action.

### Prepare / runtime
- **"Prepare script failed: ... in `_load_raw`"** → the platform passed a
  `raw_root` whose layout differs from what the script hard-codes. Make lookup
  tolerant and verify against both layouts.
- **"Prepare script failed" right after an unrelated edit** → an edit made for
  a different check broke the pipeline. Revert to last known-good and re-apply
  only the requested change.

### CSV Score Validation
- **"Submission columns must be exactly [...], got [...]"** → the submission
  was built for a different challenge, or names drifted. Read `config.yaml`'s
  `submission_columns` and match exactly. If the names look foreign, confirm
  which challenge the file belongs to before editing anything.

### Human reviewer findings
- **"the upload holds the prepared public and private files, including
  answers.csv"** → see Part 9.
- **Reviewer re-counted rows** and found the description and the upload came
  from different generations.
- **Truth set incomplete** — 21.6% of windows contained real annotations in the
  flanks that were *not* in the answer key. A solver could be right and score 0.

---

## PART 13 — The two cardinal rules

### 1. Fix only what was asked

When the user reports one failing check, fix that one check. Do not bundle an
unrelated improvement into the same edit.

> A session was asked only to change a licence field, silently also changed the
> data encoding to fix a *different* check, and broke a working dataset. The
> dataset had to be fully reverted.

If you notice a second problem, say so in one line and let the user decide.

**Do not reshape a working task to fix ONE failing check.** The single most
expensive mistake in the portfolio: a task passed every check except
originality; instead of sharpening the framing, the entire task was reshaped.
The reshape fixed originality but **broke domain routing AND reintroduced a
data-leakage bug that had already been fixed once**. The original had to be
reconstructed from scratch because there was no version control.

If a task fails only ONE named check, narrow the fix to that check: sharpen the
metric, change what evidence is withheld, add a new angle on the same
input/output shape — rather than swapping the shape.

### 2. Measure, never assume

Every number in every shipped file must come from a script run against the
exact files being shipped. Every claim about a shortcut being closed must come
from running that shortcut.

The most valuable findings in every build — a position-only probe beating the
reference, a one-giant-cluster baseline beating a trained model, a leaderboard
slice turning one bag into two, a family split putting a trained model below a
one-line heuristic — were **all found by measurement** and **all would have
shipped undetected on intuition**.

### Corollary: back up before touching a passing state

`git init` from the first file, or copy the whole directory to a timestamped
folder. `git commit -m "known-good: <check status>"` every time a full audit
passes.

**Re-uploading an accepted dataset forces you to withdraw it.** Ship revisions
through challenge-side files only.

---

## PART 14 — Source register (never reuse an entry)

| Date | Source URL | Task | Status |
|---|---|---|---|
| — | `tatoeba.org` bulk export | sentence-pair task, short units | abandoned — adversarial baseline beat the reference; ~38-char units |
| — | `statmt.org` Europarl | speaker attribution, region detection, intruder detection (3 attempts) | rejected — no named licence, then duplicate URL |
| — | Wikinews (`dumps.wikimedia.org`) | region/category classification | rejected — duplicate source URL |
| — | `dumps.wikimedia.org/enwiktionary/...` | **The Accent Translator** | accepted (CC BY-SA 4.0). Needed a Rules section after an agent hit 0.776 vs a 0.46 reference |
| — | `api.plos.org/search` | **The Order of Discovery** | accepted (CC BY 4.0), reference 0.358 |
| — | `chroniclingamerica.loc.gov/data/batches/` | **Recovering OCR Batch Origin** | accepted (Public Domain Mark), reference 0.342 after four iterations |
| — | `huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct` | **The Edit That Moved the Answer** | accepted (Apache-2.0), reference 0.493 |
| — | `dumps.wikimedia.org/enwikivoyage/...` | **Untangling Mixed Edit Histories** (Palimpsest) | accepted — the portfolio's only fully-praising review |
| — | `zenodo.org/records/10844811` | **Observing Conditions** | built |
| — | `zenodo.org/records/7711810` | **Blind Spot** | built |
| — | `zenodo.org/records/7394851` | **Prompt Component Pair Response Atlas** | built |
| 2026-09-21 | `courtlistener.com/help/api/bulk-data/` | **Reconstructing Local Citation Structure** | built (Public Domain Mark), ref 0.635. Redesigned after a position probe scored 0.143 > ref 0.098. Reviewer rejected the 212 MB zip |
| — | `huggingface.co/datasets/danielrosehill/System-Prompt-Library` | **Attributing Assistant Behavior to System Prompt Clauses** | in progress |
| 2026-09-25 | `api.fda.gov/drug/label.json` | **What Does It Do In There?** | built (CC0), ref 0.393 vs 0.122. **BLOCKED by Duplicate Detection** naming unrelated sources |
| 2026-09-25 | `dailymed.nlm.nih.gov` | alternative host for the same SPL corpus | **rejected at Gate 1** — disclaimer, not a licence |
| 2026-09-25 | `clinicaltrials.gov/api/v2` | MeSH terms as label | not used — licence is JS-rendered, unquotable |
| 2026-09-25 | `ebi.ac.uk/europepmc/webservices/rest` | considered | available; per-record `license: "cc by"`, 5.3M records |
| — | `govinfo.gov/bulkdata/FR/2024/` | Federal Register issuing-office generation | drafted (U.S. Government Work) |
| — | GRCh38 / hg38 | repeat-pattern span detection | drafted |
| 2026-09-25 | `api.crossref.org` | retraction reason / lag | **abandoned at the gates** — no citing text, lag median 1 day, 72% no extractable reason |
| 2026-09-25 | `ecfr.gov/api/versioner/v1` | **The Vanished Clause** | built (U.S. Government Work); oracle 1.000, copy-through 0.000, vocab-dump 0.008 |
| 2026-09-25 | `dumps.wikimedia.org/enwiki/...` | **The Reference Shuffle** / Unscrambling the Bibliography | built (CC BY-SA 4.0), ref 0.7037 vs best probe 0.5576. First rejected on Duplicate Detection against an account-external dataset |
| 2026-09-27 | `genenames.org/download/archive/` | **Coining Gene Symbols from Gene Names** | in progress (CC0), ref 0.447 vs sibling-lookup 0.366; family split abandoned (ref 0.258 < 0.337 heuristic) |

Also used across earlier CPU builds: `archive.ics.uci.edu/dataset/502`
(slate-forensics), OQMD DFT (stability-frontier), EuroSAT (blind-spot),
Gutenberg (cut-the-corpus — collided on the book id).

---

## PART 15 — House style

- **The word "challenge" must not appear** anywhere in any deliverable — folder
  names or file contents. Use "task", "problem" or "evaluation". Confirmed
  across five builds by grep.
- Package dirs are named after the task, not `challenge-<task>`.
- Grep for the word before considering a deliverable done.

---

## PART 16 — Local build environment notes

- An 8 GB Apple Silicon Mac leaves ~2 GB real headroom with VS Code plus 2–3
  agent sessions. Prefer streaming/chunked processing and modest worker counts.
  Size any local compute against ~2 GB, not against 8.
- PyTorch MPS: `nn.Transformer` with `norm_first=True` hits
  `aten::_nested_tensor_from_mask_left_aligned` not implemented. Fix by
  building the encoder explicitly with
  `nn.TransformerEncoder(layer, n, enable_nested_tensor=False)` and passing it
  as `custom_encoder`.
- A 3.2M-param char transformer trains at ~35 s/epoch on MPS for ~14k rows.
  MPS is ~1.5× faster than CPU at this size — not worth fighting for more.
- Background training runs die when the agent session restarts. Write progress
  to a log file every few epochs so a restart can read the last checkpoint
  instead of starting over.
