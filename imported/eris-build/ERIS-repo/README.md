<div align="center">


### Evidence Recovery Inference Suite

**Eleven adversarially-hardened machine-learning benchmarks in which every label is a recorded fact, not a human annotation and not a model output.**

[![Benchmarks](https://img.shields.io/badge/benchmarks-8%20shipped%20%2B%203%20in%20build-1f6feb)](docs/BENCHMARKS.md)
[![Anchors](https://img.shields.io/badge/score%20anchors-80%2B%20measured-2ea043)](docs/BENCHMARKS.md)
[![Reproducible](https://img.shields.io/badge/reproducible-one%20command-2ea043)](#reproduce-every-number)
[![Data licence](https://img.shields.io/badge/data-commercial%20use%20permitted-8957e5)](docs/LICENSES.md)
[![Code licence](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)

**Authored by [Abhaydeep Indwar](https://linkedin.com/in/abhay-indwar7)** · abhayindwar7@gmail.com

</div>

---

## What this is

A benchmark suite for evaluating whether a model can **recover a hidden generative process from the trace it left behind**.

Each task hands the solver an artifact — a phoneme string, a shuffled Results section, a bag of OCR snippets, a set of response fingerprints, an amended regulation, a drug label with its mechanism masked out — and asks it to reconstruct the process that produced it: which dialect, which author's argument, which scanning pipeline, which prompt edit, which wording came before, which mechanism.

Three properties separate this suite from the benchmark commons:

| | |
|---|---|
| **Labels are records, not opinions.** | The accent tag is the Wiktionary editor's own `{{IPA\|en\|...\|a=}}` value. The batch identity is the Library of Congress's own recorded batch. The experiment order is the authors' own section order. Nobody was paid to annotate anything, and no LLM was used to invent a target. |
| **Every difficulty claim is measured.** | Each task ships a baseline ladder — up to fourteen scored rungs from degenerate floor to oracle ceiling. Adversarial shortcut attacks were written, run and scored *before* the spec was frozen, and are published with their scores whether or not they were flattering. |
| **The audit trail is public.** | Metrics that failed measurement were replaced and the failure is documented. A reviewer who scored 1.000 by source lookup is written up, with the redesign that closed it. Confounds that could not be removed are disclosed in the solver-facing text rather than buried. |

---

## The benchmarks

| Benchmark | Task | Metric | Floor | Reference | Ceiling | Data |
|---|---|---|---|---|---|---|
| [**The Accent Translator**](benchmarks/accent-transfer/) | Rewrite a real IPA transcription from one English accent into another | Change-segment accuracy | `0.112` | `0.768` | `1.000` | 29,108 rows / 8,960 words |
| [**The Edit That Moved the Answer**](benchmarks/prompt-edit-attribution/) | Given two prompts differing by one edit, rank which input kind the edit moved most | Positional credit | `0.337` | `0.493` | `1.000` | 1,900 items / 4,608 generations |
| [**Recovering OCR Batch Origin**](benchmarks/pipeline-attribution/) | Cluster unlabelled OCR snippets by the digitization pipeline that produced them | Mean per-bag Adjusted Rand Index | `0.000` | `0.342` | `1.000` | 7,881 snippets / 35 batches |
| [**The Order of Discovery**](benchmarks/experimental-order/) | Reconstruct the authors' experimental sequence from a shuffled Results section | Mean per-paper position accuracy | `0.285` | `0.358` | `1.000` | 259 papers / 1,216 subsections |
| [**Reconstructing Local Citation Structure**](benchmarks/citation-structure/) | From twelve candidate court opinions, select the ones a query opinion cites, with every citation string stripped | Mean per-query MCC | `0.141` | `0.672` | `1.000` | 2,354 queries / 4,151 opinions |
| [**The Reference Shuffle**](benchmarks/reference-order/) | Recover the order in which a paragraph cited its own three references | Mean rescaled Kendall's tau | `0.541` | `0.725` | `1.000` | 492 paragraphs / 296 articles |
| [**What Does It Do In There?**](benchmarks/mechanism-recovery/) | From masked drug pharmacology, select the mechanism classes the FDA assigned | Chance-corrected set F1 | `0.193` | `0.462` | `1.000` | 630 labels / 22 classes |
| [**The Vanished Clause**](benchmarks/amendment-reversal/) | Reconstruct how a federal regulation read before it was amended | Gap-anchored changed-token F1 | `0.012` | `0.002` † | `1.000` | 763 pairs / 9 CFR titles |
| [**Retrieval Sufficiency in RAG**](benchmarks/rag-sufficiency/) | Emit a joint `(diagnosis, action, reason_code)` trace over a real retrieval step | Joint field accuracy | *in build* | *in build* | `1.000` | 1,536 items / 24 topics |
| [**Coining Gene Symbols**](benchmarks/gene-symbol-coinage/) | Produce the symbol a nomenclature committee coined for a gene, given its name | Prefix agreement | *in build* | *in build* | `1.000` | 16,133 genes |
| [**System Prompt Clause Responsibility**](benchmarks/clause-responsibility/) | Rank a system prompt's sentences by how much each one drives the measured behaviour | Joint ranking and flag score | *in build* | *in build* | `1.000` | 145 prompts, not yet generated |

"Floor" is the strongest content-free submission measured. "Reference" is a shipped, runnable solution — a working baseline, deliberately **not** a ceiling. The gap between them is the headroom the benchmark exists to measure.

† **The Vanished Clause has no working baseline yet.** Every shipped approach scores at or below the floor. The data, split, metric and ladder are measured and reproducible, and two diagnostic ceilings show the score is reachable: perfectly restoring one sentence per section is worth `0.749`. It is listed as shipped because its numbers reproduce, not because it is solved.

Full ladders, every intermediate rung, and the adversarial attacks: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**

---

## Reproduce every number

No number in this repository has to be taken on trust.

```bash
pip install -r requirements.txt

python tools/verify.py               # re-scores every published floor            (~5 seconds)
python tools/verify.py --reference   # retrains and re-scores every CPU reference  (~95 seconds)
```

```
accent-transfer          The Accent Translator — change_segment_accuracy
  [PASS] copy source IPA (sample_submission)             0.1125  (published 0.112)
  [PASS] per-segment context rewrite table (CPU)         0.7681  (published 0.768)
...
14/14 anchors reproduced.
```

Everything `verify.py` runs is on CPU, offline, against data already in this repository. No API keys, no downloads, no GPU. Two references are not part of that run because they need packages outside `requirements.txt`: `citation-structure` downloads a pre-trained sentence encoder and `reference-order` imports `torch`. Their floors are verified; their reference scores are as published in their `config.yaml`.

---

## Capabilities

**As an evaluation suite.** Eight held-out test sets with sealed answer keys and deterministic graders. Metrics are chance-normalised or floor-anchored, so a score reads as ability rather than as a property of the label distribution. Degenerate submissions score zero by construction, not by convention.

**As a model-selection instrument.** Each task isolates one capability that generic leaderboards blend away: conditional sequence transduction under a categorical transfer direction; causal attribution of a behavioural delta; open-set clustering on style-invariant noise; long-range discourse ordering under presupposition; set selection over long legal and biomedical text with the giveaway vocabulary removed; inverting an edit from the text it left behind. A model that clears the reference on all of them is demonstrating that many different things.

**As a red-team target.** The shortcut attacks are shipped, not just described — pooled global clustering, closed-set argmax, length-only features, majority-combination voting, threshold replicas. Each is runnable and each has a published score. A new attack that beats the reference is a finding, and the harness will tell you so.

**As a dataset-construction reference.** Group-disjoint splitting by word, framing, batch and paper; exact-search split selection against a measured shortcut objective; metric selection *after* measurement rather than before; label derivation from measured facts rather than construction intent. The methodology is written up in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

**As licensed corpora.** Wiktionary and Wikipedia (CC BY-SA 4.0), PLOS (CC BY 4.0), Library of Congress NDNP and CourtListener case law (Public Domain Mark), openFDA drug labels and the HGNC gene set (CC0 1.0), the eCFR (U.S. Government Work), and original measured model behaviour (CC0 / CC BY 4.0). Commercial use is permitted throughout, with attribution. The licence of every byte is traced in **[docs/LICENSES.md](docs/LICENSES.md)**.

---

## Repository layout

```
eris-ml/
├── benchmarks/
│   ├── accent-transfer/            The Accent Translator
│   ├── prompt-edit-attribution/    The Edit That Moved the Answer
│   ├── pipeline-attribution/       Recovering OCR Batch Origin from Character Noise
│   ├── experimental-order/         The Order of Discovery
│   ├── citation-structure/         Reconstructing Local Citation Structure in Case Law
│   ├── reference-order/            The Reference Shuffle
│   ├── mechanism-recovery/         What Does It Do In There?
│   ├── amendment-reversal/         The Vanished Clause
│   ├── rag-sufficiency/            Predicting Retrieval Sufficiency in RAG Pipelines   (in build)
│   ├── gene-symbol-coinage/        Coining Gene Symbols from Gene Names                (in build)
│   └── clause-responsibility/      Attributing Assistant Behavior to System Prompt Clauses (in build)
├── docs/
│   ├── BENCHMARKS.md               every measured anchor, every adversarial attack
│   ├── METHODOLOGY.md              how the tasks are built and hardened
│   ├── USAGE.md                    integration, submission format, scoring
│   ├── LICENSES.md                 provenance and licence for every corpus
│   └── ROADMAP.md                  status of the in-build benchmarks
├── tools/verify.py                 one-command reproduction of every anchor
└── archive/                        superseded builds, kept for provenance
```

Every benchmark directory is self-contained and identically shaped:

```
<benchmark>/
├── config.yaml                     the spec: metric, anchors, split policy, provenance
├── problem-description.md          what the solver is given and asked for
├── dataset-description.md          what the data is and where it came from
├── rubrics.md                      the rules a submission must satisfy
├── prepare.py                      raw corpus  ->  public/private split (seeded)
├── grade.py                        the official scorer
├── solution.ipynb / solution.py    the runnable reference
└── dataset/
    ├── generator/build_raw.py      source fetch/measurement  ->  raw corpus
    ├── raw/                        the corpus as collected, with ATTRIBUTION.txt
    ├── public/                     train.csv, test.csv, sample_submission.csv, stats
    └── private/                    answers.csv — the sealed key
```

---

## Quick start

```bash
git clone https://github.com/indwar7/eris-ml.git
cd eris-ml && pip install -r requirements.txt

cd benchmarks/experimental-order
python solution.py --out submission.csv        # run the reference
python grade.py submission.csv dataset/private/answers.csv
# 0.3585
```

Integration details, submission schemas and grader contracts: **[docs/USAGE.md](docs/USAGE.md)**

---

## Citation

```bibtex
@misc{indwar2026eris,
  author       = {Indwar, Abhaydeep},
  title        = {{ERIS}: Evidence Recovery Inference Suite --
                  Eleven Adversarially-Hardened Benchmarks for
                  Latent-Process Recovery},
  year         = {2026},
  howpublished = {\url{https://github.com/indwar7/eris-ml}}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Each corpus carries its own upstream citation requirement — see [docs/LICENSES.md](docs/LICENSES.md).

---

## Licence

Harness code, graders, generators, reference solutions and documentation: **MIT** ([LICENSE](LICENSE)).

Datasets retain their upstream licences (CC BY-SA 4.0, CC BY 4.0, Public Domain Mark, CC0 1.0, U.S. Government Work). All permit commercial use; five require attribution and two of those also require share-alike. The obligations are set out per corpus in [docs/LICENSES.md](docs/LICENSES.md).

---

<div align="center">

**Abhaydeep Indwar** · [abhayindwar7@gmail.com](mailto:abhayindwar7@gmail.com) · [linkedin.com/in/abhay-indwar7](https://linkedin.com/in/abhay-indwar7)

</div>
