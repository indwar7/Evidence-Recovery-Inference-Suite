# Methodology

How the benchmarks in this suite are built, and the discipline that separates a
benchmark from a dataset with a score attached.

---

## 1. The label must be a record

Every target in this suite was written down by someone, or measured from a real run,
*before* the benchmark existed and for reasons unrelated to it.

| Benchmark | The label is | Recorded by |
|---|---|---|
| The Accent Translator | An accent-tagged IPA transcription | A Wiktionary editor, in the article's own `{{IPA\|en\|...\|a=}}` template |
| Recovering OCR Batch Origin | A scanning batch's identity | The Library of Congress NDNP programme, in the batch's own ALTO XML |
| The Order of Discovery | The sequence of experiments | The paper's authors, in their own Results section ordering |
| The Edit That Moved the Answer | A behavioural delta ranking | Derived arithmetically from 4,608 real greedy generations |
| Retrieval Sufficiency in RAG | A measured retrieval rank | Measured from a real BGE-small retrieval run, never asserted |

This rules out three failure modes at the source. There is no annotator agreement to
report, because there were no annotators. There is no teacher-model bias to inherit,
because no LLM was asked to produce a target. And there is no construction-intent leak,
because in every case where intent and measurement could disagree, **the measurement
wins**: an item built to be "sufficient" whose gold passage actually measures outside
top-*k* is relabelled by what was measured, not by what was intended.

---

## 2. Split by group, never by row

Row-level random splits leak. Every split in this suite is disjoint on the unit that
actually carries the signal:

| Benchmark | Split unit | Why |
|---|---|---|
| The Accent Translator | **Word** | Otherwise a word's sibling accent sits in train and a model copy-edits instead of generating |
| The Edit That Moved the Answer | **Prompt framing** | Otherwise framing-specific response style transfers across the split |
| Recovering OCR Batch Origin | **Batch (open-set)** | The 16 test batches never appear in train's 19, forcing a general notion of fingerprint similarity rather than a closed vocabulary |
| The Order of Discovery | **Paper** | One item *is* one paper, so disjointness holds by construction |
| Retrieval Sufficiency in RAG | **Topic** | Otherwise a topic's corpus is learnable from its own train items |

Two tasks go further and **choose which groups to hold out by exhaustive search**, not
by seed. Candidate same-size group combinations are enumerated, and the combination
selected is the one that jointly lands the train/test ratio inside the 15–25% band
*and* minimises a measured shortcut baseline on the resulting test set. Within each
group, items are capped by outcome class before the search runs, because an unbalanced
outcome distribution let a zero-signal majority-vote baseline score well above chance.

---

## 3. Choose the metric after measuring it

A metric is a hypothesis about what a score means. Two of these hypotheses were tested
and failed, and both were replaced.

**The Accent Translator** originally used edit similarity. Measurement showed that
copying the source transcription scored `0.795` and a rule baseline `0.914`: the ~80% of
every string that never changes was dominating the score, compressing every real model
into a 0.2-wide band. Change-segment accuracy scores only positions that actually
differ. Copy-source fell to `0.112` and the full range came back.

**Recovering OCR Batch Origin** originally used pairwise co-membership F1. Measurement
showed that "put everyone in one group" scored `0.376` — *above* a trained reference's
`0.289`. Adjusted Rand Index chance-corrects exactly this failure, and both degenerate
submissions now score `0.000` by construction.

**The Order of Discovery** originally used rescaled Kendall's tau, whose bottom half no
submission can reach. Three agent runs scored 0.56 / 0.82 / 0.90 against a 0.5 floor.
Position accuracy keeps task and data identical and spreads the same submissions across
the full `[0, 1]` range from a 0.218 floor.

In all three cases the rejected metric and its measured numbers are **retained** in
`config.yaml`. A benchmark that hides its rejected metrics is asking to be trusted about
the one it kept.

---

## 4. Attack the task before shipping it

Every benchmark ships a ladder of adversarial baselines that were *written and run*, not
merely anticipated. The classes attacked:

- **Degenerate submissions** — all-one-class, all-singleton, identity, copy-through.
- **Content-free statistics** — length, position, ordering by size, majority class.
- **Ablations** — reading the content but not the signal, and vice versa, to prove which
  one carries the task.
- **Replica shortcuts** — recomputing the released features under the suspected label
  rule. If a solver can re-derive the label from published numbers alone, the label rule
  is the leak.
- **Closed-set argmax** — treating an open-set task as classification over known classes.
- **Pooled attacks** — ignoring the per-item boundaries the task defines, e.g.
  clustering the entire test corpus at once instead of bag by bag.
- **Source lookup** — locating the original archive record from the shipped text.

Attacks that succeed are not quietly dropped. The OCR task's pooled-clustering attack
scores `0.094`, is **banned by an explicit rule** in the problem statement and rubrics,
and its score is published so a reader can judge whether the ban is load-bearing. An
earlier claim that per-bag scoring *structurally* blocked the attack was found to be
false and was corrected in the solver-facing text.

---

## 5. When a leak cannot be closed by construction, close it by rule and say so

Two corpora in this suite are verbatim published text, and full-text search over their
sources is public. No signal-preserving transformation can prevent an adversary holding
the archive from finding the source record — the text is its own search key.

The suite's position is to state this plainly rather than to imply a guarantee it cannot
make:

- Nothing shipped names the source record. Identities are opaque aliases; DOIs, titles,
  authors and archive batch names are stripped from every solver-visible file.
- Shipped text is windowed (80–97% of the source region) so that no test row is
  byte-identical to any archive region.
- External data and network access are prohibited by an explicit `[REQUIRED]` rubric
  line, with the solver environment expected offline.
- The residual exposure is documented in `dataset-description.md` under its own heading,
  and any run scoring at or near `1.000` is flagged in `config.yaml` as source lookup
  rather than ability.

Mitigation is layered and the limits of each layer are named. That is a weaker claim
than "leak-proof" and a considerably more useful one.

---

## 6. Difficulty is a measured property, tuned until it holds

The target is a task where the reference clears every adversarial rung by a wide margin
while leaving real headroom to the oracle. Both edges were enforced by measurement and,
where they failed, by redesign.

The OCR task is the worked example. Independent solver runs on v2 sat at `0.40`–`0.43`
against a `0.276` reference — the bottom edge of the band. The bag shape was swept
across seven configurations on the reference (`0.276` / `0.289` / `0.295` / `0.296` /
`0.325` / `0.326` / `0.328`) and the best shipped as v3. Best-of-three runs on v3 then
reached only `0.38` against a `0.326` reference — a ratio of 1.17×, well below the ~1.5×
seen on v2 — so the shape was swept again across six more configurations and v4 shipped
at a `0.342` reference. The entire ladder was re-measured on each build, and the ladders
for v1, v2 and v3 are retained in `config.yaml`.

A deliberate rejection from that sweep is instructive: a 2–3-group shape scored well but
would have made 53% of bags two-group, close enough to binary to become a tell. The
2–4-group shape with a full 33/32/45 spread was shipped instead.

---

## 7. Graders degrade, they do not crash

Every grader in this suite is a deterministic pure function of (submission, answers) and
follows one contract:

- **Structural violations are rejected outright** — a missing column, a duplicated
  identifier where uniqueness is load-bearing. These raise `InvalidSubmissionError`, a
  `ValueError` subclass.
- **Content violations degrade to zero for the affected item only** — a missing row, a
  non-permutation where a permutation is required, a wrong label count. These never
  raise, so one malformed item never zeroes an otherwise valid submission.
- **Answer keys are sliceable by row.** Where the scoring unit is a group rather than a
  row, the answer key carries one row per group, so any contiguous slice of the key
  scores whole units. This was a real defect once: a per-snippet key let every bag be cut
  into two fragments and scored separately, and fragments of three rows or fewer made ARI
  nearly free.

Each grader ships with its own unit tests covering perfect, empty, partial, NaN,
duplicated and malformed submissions.

---

## 8. Reproducibility is a deliverable

- Every preparation script is seeded and exposes `prepare(raw, public, private)`.
- Every grader exposes `grade(submission, answers)` and a CLI with identical semantics.
- Every corpus ships with `meta.json` recording the fetch date, seed, filters,
  normalisation policy and a SHA-256 of the generated artifact.
- Every corpus ships with `ATTRIBUTION.txt` recording source, licence and retrieval date.
- `tools/verify.py` re-scores every published anchor from shipped data on CPU, offline,
  in under 30 seconds, and exits non-zero if any anchor moves.

Where reproduction is *not* bit-exact, the reason is named and the tolerance is
published — see the two disclosed tolerances in
[BENCHMARKS.md § Verification log](BENCHMARKS.md#verification-log).

---

## 9. Licensing is verified at the source, not assumed

Every corpus licence was confirmed against an authoritative endpoint and recorded with
the evidence, not inferred from a project's reputation. The Wiktionary licence, for
instance, is confirmed through the MediaWiki `siteinfo` API's `rightsinfo` response
rather than from a wiki page. Where a licence carries an attribution obligation, the
attribution line is worded identically in every prose file and cross-checked word for
word against the declared licence field before packaging.

Full provenance, obligations and evidence per corpus: **[LICENSES.md](LICENSES.md)**.
