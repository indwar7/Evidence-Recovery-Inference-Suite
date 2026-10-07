# Real Genomic Sequences with Tandem Repeats Finder Ground Truth

## Overview

This dataset supports a benchmark that scores a tandem-repeat *recovery policy* directly, on **real human genome data**. It contains 1700 DNA sequences over the alphabet `A, C, G, T`. Each sequence is a window of the **GRCh38 (hg38) human reference genome**, containing one tandem-repeat locus embedded in real genomic flanks. The true repeat extents are real **Tandem Repeats Finder** (Benson, 1999) calls. The task is to output, per sequence, the one interval that best recovers the tandem array.

A `split` column marks each sequence as `train` (1,400 sequences) or `test` (300 sequences) — the supervised split is deliberately 4.7× the evaluation split. This raw `sequences.csv` carries the full ground truth (`true_intervals`) for every row; the challenge's `prepare.py` later withholds the test extents in the public release.

A single tandem array is annotated by Tandem Repeats Finder **redundantly** — as several **overlapping** calls (period 2 vs 4, eight copies vs twelve). Because a submission scores only one interval per sequence, and one interval can match only one call, each sequence can be credited at most its **heaviest** call while the denominator counts them all. The exact best-possible harvest is therefore `sum(heaviest call) / sum(all calls)` = **0.4147** corpus-wide and **0.4160** on the evaluation split — a hard ceiling below 0.5 that no submission can exceed, verified against an answer-knowing adversary. Because that ceiling, not 1.0, is the top of the scale, scores are best read as a fraction of it: an unsupervised maximal-segment reference detector scores 0.2881 (69% of ceiling), the best answer-blind fixed-interval guess — swept over every width and offset — scores 0.0929 (22%), and the sample one-tile submission scores 0.0246 (6%). The achievable band is a verified property of the real data.

A second property is guaranteed by construction: **the truth set is complete**. A window is released only if every `simpleRepeat` call inside it belongs to that window's locus, so no real tandem repeat inside a released sequence is missing from the answer key.

## Source

- **Sequence:** the GRCh38 / hg38 human reference assembly, via the UCSC Genome Browser ([conditions of use](https://genome.ucsc.edu/conditions.html)).
- **Annotation:** the UCSC `simpleRepeat` track — Tandem Repeats Finder run genome-wide.
- **Citation:** Benson G. "Tandem repeats finder: a program to analyze DNA sequences." *Nucleic Acids Research* 27(2):573–580, 1999. DOI [10.1093/nar/27.2.573](https://doi.org/10.1093/nar/27.2.573) · PMID 9862982 · free full text [PMC148217](https://pmc.ncbi.nlm.nih.gov/articles/PMC148217/).

The exact regions, window coordinates and per-row provenance are held in the
build's private artifacts rather than published here: the released corpus is real
data, but pointing at the precise coordinates would hand a participant the
annotator's own output for the evaluation sequences.

### The ceiling mechanism, on one real row

Row `seq_0001` is 239 bases long and carries three genuine, mutually overlapping
annotator calls at a single array — the annotator's own multi-period framings of
the same physical repeat:

- `138–177`, period 18, consensus `AAAAAGAAAGAAAGAAAG` (39 bases)
- `140–179`, period 4, consensus `AAAG` (39 bases)
- `144–173`, period 14, consensus `AAAGAAAGAAAGAA` (29 bases)

One reported interval matches at most one of them, so this row's maximum credit
is 39 of its 107 total annotated bases = 0.364. The corpus-wide ceiling of 0.4147
is that same arithmetic aggregated over 1700 rows — a measured property of the
annotator's real redundancy, not a design choice.

Every row's provenance (assembly coordinate, source region, the live annotator
calls it came from) is recorded in the build's private artifacts, so the corpus
can be audited against the reference assembly without publishing the pointers
that would let a participant look up the evaluation answers.

## Provenance and license

- **Sequences:** real windows of the GRCh38 / hg38 human reference genome, fetched from the **UCSC Genome Browser** REST API (`getData/sequence`). The 60 source regions span every autosome and chrX, and include real disease-expansion loci (the *HTT* / Huntington region, *C9orf72*).
- **Ground truth:** real **Tandem Repeats Finder** calls from the UCSC `simpleRepeat` track (`getData/track`), which is TRF (Benson, *Nucleic Acids Research* 1999) run genome-wide. Each call carries a period, copy number and consensus motif; overlapping multi-period calls at one array are the intended candidate framings.
- Both the sequence and the annotation are bundled in `raw_windows.json`, so the generator is self-contained (no network needed to build). `fetch_ucsc.py` re-derives that file from the UCSC REST API for full reproducibility, issuing 120 requests (60 regions × 2 endpoints) and caching the raw responses locally.
- **License:** CC0 1.0 Universal (public domain dedication). The UCSC hg38 assembly and the simpleRepeat track are freely available and redistributable ([conditions of use](https://genome.ucsc.edu/conditions.html)); the repeat motifs are public factual biology.
- **Determinism:** hg38 is a static assembly, and window flanks and the train/test split are drawn with `numpy.random.default_rng` (PCG64) on a fixed seed (20260716); two runs produce byte-identical files.

## File Structure

The dataset contains these files, all at the top level:

- `sequences.csv` — the corpus with full ground truth (1700 data rows + header).
- `str_motifs.json` — real observed motif and period statistics from the corpus.
- `meta.json` — source attribution and verified score anchors emitted by the generator.
- `generate.py` — the deterministic generator that emitted the three files above. It formats `raw_windows.json`, the build artifact holding the real DNA, the annotator's calls and each window's provenance; that artifact and the fetch script that rebuilds it are kept with the challenge bundle rather than published here, since they carry the coordinates that would expose the evaluation answers.
- `dataset-description.md` — this document.

## Features: sequences.csv

One row per sequence. Each column, its data type, and its meaning:

- `seq_id` — string. Sequence identifier, `seq_0001` onward. Unique per row.
- `split` — string, one of `train` or `test`. Marks the supervised split (extents revealed) versus the evaluation split.
- `length` — integer. Length of the sequence in characters.
- `sequence` — string over `{A, C, G, T}`. The real hg38 DNA window (upper-cased).
- `true_intervals` — string. Pipe-separated real TRF extents, each written `start-end` as half-open character indices `[start, end)` into this sequence (example: `318-472|318-410`). Overlapping extents at one locus are the annotator's multi-period framings of that array.
- `n_true` — integer. Number of TRF extents in `true_intervals`.

## Features: str_motifs.json

A single JSON object summarizing the real motifs in the corpus. Its fields:

- `description` — string. Provenance note.
- `source` — string. The UCSC / TRF source attribution.
- `n_distinct_motifs` — integer. Number of distinct consensus motifs observed (thousands — a genuine genome-wide spread, not a small planted set).
- `top_consensus_motifs` — array of `{motif, count}` objects: the most frequent TRF consensus units observed — the empirical distribution, dominated by common genome-wide microsatellites such as `A`, `AC`, `AG`, `AT`, and `TG`. Nothing is curated or planted.
- `period_histogram` — array of `{period, count}` objects: the distribution of TRF repeat-unit periods in the corpus.

## Features: meta.json

A single JSON object. Its fields:

- `source` — string. UCSC hg38 simpleRepeat / Tandem Repeats Finder attribution.
- `assembly` — string. `hg38`.
- `track` — string. `simpleRepeat`.
- `n_seq` / `n_train` / `n_test` — integers. 1700 / 1400 / 300.
- `budget_k` — integer. Intervals scored per sequence (1).
- `iou_threshold` — float. IoU at which an interval harvests an extent (0.50).
- `verified_ceiling` — float. `sum(heaviest call) / sum(all calls)` over the whole corpus — the hard ceiling no submission can exceed (0.4147).
- `verified_ceiling_eval_split` — float. The same ceiling computed on the evaluation split alone (0.4160) — the number a submission's score should be read against.
- `notes` — string. Explanation of the ceiling.

## Generation process

1. `fetch_ucsc.py` fetches, for 60 fixed hg38 regions, the real DNA and the real `simpleRepeat` (TRF) calls from the UCSC REST API (cached locally).
2. TRF calls within 20 bp are clustered into loci. A locus is kept when its heaviest single call is 25–70% of the locus's total called extent — i.e. it carries genuine overlapping multi-period annotation but still has a dominant call.
3. A window is cut around each locus with a total flank of 4.0–8.0× the locus span, split at a uniform-ish left/right ratio, so the repeat occupies ≤ 20% of the window (median 14%) at an unpredictable offset (no fixed-position or whole-window guess substitutes for detection).
4. A window is dropped if any `simpleRepeat` call from a *neighbouring* locus falls inside it, so every real tandem repeat in a released sequence appears in its `true_intervals` (verified: 0 of 1700 windows carry an unannotated call).
5. Ceiling: for each sequence the heaviest call is the max recoverable weight; aggregated over the corpus, `sum(heaviest)/sum(all)` is the provable ceiling (0.4147 corpus-wide, 0.4160 on the evaluation split).
6. The train/test split is deterministic and stratified by the heaviest/total ratio, drawn at exactly 1,400 train and 300 test. The challenge's private answer file is written in a slice-balanced order: every contiguous row slice (a public/private leaderboard split) carries the same ceiling as the full set — measured 0.4159–0.4160 against the full 0.4160 at 2, 5 and 10 slices — and an equal share of the metric's denominator.

## Skills this dataset tests

Detecting periodic (tandem-repeat) structure in real genomic DNA and distinguishing it from real interspersed repeats and low-complexity tracts; localizing the dominant array tightly enough to clear a strict 50% IoU overlap test against a real annotator's boundaries (an interval twice the true length already fails); and calibrating that localization on a supervised split — rather than sequence classification or regression. The score is smooth and rewards better localization continuously, and the ceiling is a proven property of the real annotation's redundancy.
