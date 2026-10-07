# Digitization Pipeline Attribution — design record

## The task in one line

Given a shuffled bag of real historical-newspaper OCR text snippets, each from
a DIFFERENT article/topic, recover which real digitization batch each snippet
came from — using only the OCR engine's character-error fingerprint, never the
words themselves.

## Why this shape

An earlier occupancy scan across candidate GPU-native NLP tasks killed four
spaces: stylometric authorship-vs-topic separation (PAN 2026 runs this exact
shared task), sentence/paragraph intrusion detection (INSteD, 170k+ real docs,
same idea), and a text-steering-hidden-classifier idea that was a reskin of
this portfolio's own prior work (blind-spot / galaxy-blind-spot). The one
candidate that came back with no direct hit: recovering which real
**digitization pipeline** produced a piece of OCR'd text, from its error
pattern, ignoring content. This instantiates the same "grade against a real
recorded system identity" engine as blind-spot, in a new domain, with the
ground truth being the archive's own ingest record (a `batch_id`) rather than
a hidden model's measured behaviour.

## Data

Real Library of Congress NDNP (National Digital Newspaper Program) newspaper
pages, fetched from the bulk mirror `chroniclingamerica.loc.gov/data/batches/`
(the modern API/loc.gov endpoints are Cloudflare-blocked, 403 — this mirror is
not). Each page's ALTO XML carries genuinely noisy OCR text AND real recorded
vendor metadata (`softwareCreator`/`softwareName`/`processingAgency`) — a
literal recorded pipeline fingerprint, not inferred. 35 batches, 8,417
snippets (one per `TextBlock`, i.e. one per article/column region), 34
distinct institution prefixes. `software_creator` populated for all 35
batches; `processing_agency` for 17/35 (some awardees never recorded it).

Fetch throughput was rate-limited by the LOC mirror (~1 batch per 2-5 min); 35
batches was the practical ceiling within the build's time budget, not a
target. `build_raw.py` is resumable and 250+ further verified candidate batch
IDs remain untried if the corpus needs to grow later.

## Split: batch-disjoint (open-set), 45% test by batch

`train.csv` is flat and fully labeled (`snippet_id, text, batch_id`) from 19
batches. `test.csv` is bag-structured and unlabeled from the other 16 batches
— **test batches never appear in train.** This forces a solver to learn a
GENERAL notion of "what does this OCR pipeline's noise look like," not
memorize a closed vocabulary of known batch IDs.

The 45% test fraction (19 train / 16 test batches) was NOT the first choice —
it is the empirically safest point found after three measured attempts (see
"the adversarial finding" below). Standard practice elsewhere in this
portfolio holds out ~25-30% for test; here more of the batch pool had to go to
test specifically to blunt a shortcut that measurement caught, not intuition.

## Task artifact: a partition, not a label

Each bag draws 3-5 DISTINCT test batches and 3-8 snippets per chosen batch —
both randomized per bag, so a fixed bag shape can never itself be a tell.
Snippet order within a bag is shuffled. Submission is `(bag_id, snippet_id,
group_label)` where `group_label` is an arbitrary string/int, meaningful only
within that bag — the solver never has to name the real batch, only recover
which snippets travelled together. This keeps the graded artifact a
**partition**, which is none of classification / regression / ranking /
retrieval / OCR / segmentation / captioning / sequence-prediction /
information-extraction — the review-committee's excluded-task list.

## Metric: mean per-bag Adjusted Rand Index (ARI), clipped to [0, 1]

**This was not the first metric shipped, and the reason it changed is the
single most important measured finding in this build.**

The first version scored mean per-bag PAIRWISE CO-MEMBERSHIP F1 (chosen
specifically to avoid a "matched assignment" trap — a prior challenge's
review in this portfolio flagged that Hungarian-style best-alignment scoring
can silently hand out credit a model never earned on clustering-shaped
tasks). Measuring the actual baseline ladder caught a DIFFERENT problem with
that choice: raw pairwise F1 structurally rewards "put everyone in one
group" — it gets perfect recall on every true same-batch pair for free and
only loses on precision, which for bags with a handful of groups isn't
enough of a penalty. Measured: with an earlier bag composition (5-9 groups
per bag), "everyone in one group" scored **0.376** under pairwise F1 —
comfortably beating a properly trained embedding-based reference (0.289).
This is exactly the failure this project's rejection register warns about
under "reference below a naive baseline."

Adjusted Rand Index fixes this by construction: it is chance-normalized, so
both degenerate strategies (one giant group, or every snippet its own
singleton) score ~0.0, and only genuine above-chance agreement with the true
partition earns credit — re-measured, both degenerate baselines now score
**exactly 0.000**. ARI still needs no group-label alignment step, so it keeps
the original anti-Hungarian-matching property.

## A measurement bug found and fixed mid-build

Before trusting any of the numbers below: an early version of the baseline
harness (`work/baselines.py`) keyed its `assignments` dict by bare
`snippet_id`. Because bags deliberately reuse snippets across DIFFERENT bags
(sampling is with replacement across bags, without replacement within one
bag — see "Split" above), a snippet appearing in two bags had its
second-computed label silently overwrite the first in that flat dict, so
every bag containing that snippet got graded against whichever bag happened
to be processed last. This was caught by a large, unexplained gap between
the harness's number for the trained reference (0.148) and a freshly written
`reference_solution.py`'s number on the same data (0.316) — the two
computed the same clustering logic but only one indexed by `(bag_id,
snippet_id)`. Fixed by keying every assignment dict that way throughout.
The corrected numbers below are uniformly HIGHER and more separated than the
buggy ones — including the safety margin against the adversarial attack,
which went from a worryingly thin 0.009 to a comfortable 0.177. The
qualitative finding that motivated the 45% test-split fix (see next section)
was still directionally correct even under the buggy measurement; only the
absolute numbers and margin size were wrong.

## The adversarial finding: a real transductive/cross-bag shortcut, measured and fixed

A rejection this project received on a different challenge flagged: *"it fits
global models on the test data, and groups separate test rows together to
intersect their candidate banks, exploiting a dataset artifact... instead of
relying on a generalizable model."* Before freezing this spec, that exact
attack was built and run against this design: **ignore bag boundaries
entirely, run one unsupervised clustering pass over the WHOLE test corpus at
once**, using zero train data.

First measurement (9 test batches, the original 30%-test split, BEFORE the
scoring bug above was found): this attack, given the true number of test
batches as `k`, scored an ARI above both trained references measured at the
time. The exact historical numbers from that pass are superseded by the bug
fix and not reproduced here, but the qualitative conclusion held up after
the fix was applied and re-measured: with too few held-out test batches, the
shortcut is genuinely competitive with or ahead of a trained reference. The
shortcut was real, not theoretical — only the precise margins reported at
each stage needed correcting.

Root cause: with very few real batches held out for test, an unsupervised
pass can trivially discover "the" small number of true clusters without
learning anything about the fingerprint signal itself. Fix attempts and
their measured effect:
- Raising the test-batch fraction to 45% (16 test batches, 19 train): the
  **realistic** version of the attack — a solver estimating cluster count
  itself via silhouette score, no cheating — sits at 0.072, comfortably below
  the trained reference (0.316, post bug-fix). The ORACLE version of the
  attack (told the true k) reaches 0.139, still well below the reference.
- Raising it further to 55% (19 test / 16 train) was tested and rejected: it
  made the trained reference WORSE, not better (fewer train batches hurt the
  reference more than the extra test batches hurt the unsupervised
  attacker), the opposite of the intended effect. This closed off "just hold
  out more batches" as a monotonic fix and is why 45%, not the largest tried
  fraction, is what shipped.
- 45% was kept as the final setting: it is the measured optimum among the
  fractions tried, not a rounded compromise.

**Honest residual note.** Against an OMNISCIENT version of the attack — where
the solver is simply told the exact true number of test batches, which is
never published and has no legitimate way to be derived — the reference still
wins comfortably: 0.316 (reference) vs 0.139 (oracle-k global clustering), and
0.329 vs 0.139 in a fair oracle-vs-oracle comparison where the reference is
ALSO given the true k. Margin ~0.18-0.19 in both cases. This is treated the
way this project's other challenges treat ceiling/oracle disclosures (e.g.
repeat-harvest's answer-knowing-adversary checks): documented plainly here
rather than hidden, on the grounds that a real solver has no channel to learn
the true global batch count in the first place — it is not a column in any
released file, not derivable from train (train batches are disjoint from
test), and not something an unsupervised method can obtain "for free" the way
the diagnostic version of this check hands it over.

## Measured ladder (final, shipped configuration, post bug-fix)

| Strategy | ARI |
|---|---|
| all-singleton / all-same-group | 0.000 |
| artifact attack (position + length only, no content read) | 0.021 |
| random partition | 0.024 |
| adversarial global clustering, REALISTIC (silhouette-estimated k) | 0.072 |
| single cheap noise scalar, k-means | 0.096 |
| full hand-feature set, unsupervised per-bag clustering | 0.099 |
| char-CNN embedding (supervised-contrastive), per-bag clustering | 0.133 |
| adversarial global clustering, ORACLE (true k given) | 0.139 |
| **reference: RandomForest-probability embedding, per-bag clustering** | **0.316** |
| oracle (true batch_id revealed) | 1.000 |

Note the char-CNN embedding (0.133) is still measurably WEAKER than the
simpler hand-crafted-feature RandomForest approach (0.316) at this data
scale, though both improved substantially after the bug fix. This is
reported honestly rather than smoothed over — with only 19 train batches, a
from-scratch contrastive character encoder does not yet out-learn a
classical-ML baseline on hand-engineered noise features. This is itself a
legitimate, disclosed finding about where the current skill ceiling sits, and
leaves real headroom for a stronger encoder or more data to climb further,
which is what makes the task non-saturated.

## What is new here

- Ground truth is a real digitization pipeline's own identity, not a human
  annotation or another model's output — the same "grade against a recorded
  real system" principle as blind-spot/galaxy-blind-spot, in a domain (OCR
  pipeline forensics) with no prior benchmark found.
- The graded artifact is a partition recovered from content-disjoint items,
  open-set across pipelines never seen in training.
- The metric (per-bag ARI) and the split ratio (45% test by batch) were both
  set by adversarially attacking the design before freezing it, not by
  convention.

## Relation to prior work (cite distant antecedents only)

- PRNU/sensor-noise camera fingerprinting (Lukas, Fridrich & Goljan, 2006) is
  the nearest conceptual neighbour — recovering device identity from
  low-level imaging noise. This task inverts the axis (which processing
  PIPELINE handled real text, not which physical sensor captured an image)
  and the domain (OCR error patterns in text, not photographic sensor noise).
- OCR error analysis / post-OCR correction literature treats noise as
  something to REMOVE; this task treats the noise pattern itself as the
  signal to be classified.
