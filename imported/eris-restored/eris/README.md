# Eris-ML

Machine-learning challenge packages built for Project ERIS — each one a
self-contained task with a generated dataset, a held-out answer key, a
grader, and a reference solution.

## Layout

```
problems/            one directory per challenge
docs/                programme notes, authoring guide, rejection register
dataset-builders/    tooling that produces source corpora
```

A typical problem package contains:

| Path | Contents |
|---|---|
| `problem-description.md` | the task as a solver sees it |
| `dataset/` | generator scripts and raw source data |
| `prepared/public/` | train / test splits released to solvers |
| `prepared/private/` | held-out answers, never released |
| `grade.py` | scoring against the private answers |
| `rubrics.md` | grading thresholds |
| `solution.ipynb` / `reference_solution.py` | a worked solution |

## Problems

26 challenge packages. Two further scaffolds, `prompt-challenge` and
`session-attribution`, exist locally but hold only a virtualenv — no
authored content yet — so they are not committed.

| Problem | Summary |
|---|---|
| [babel-gauge](problems/babel-gauge) | Six legacy logging systems each report the same six physical facts about a machine in mutually incompatible formats. |
| [babel-kinship](problems/babel-kinship) | Cross-lingual intent kinship over the MASSIVE corpus. |
| [bird-song-syntax](problems/bird-song-syntax) | Next-phrase prediction over stochastic, individual-specific birdsong grammars. |
| [blind-spot](problems/blind-spot) | Characterise a hidden image classifier you can never inspect, from its responses alone. |
| [budget-index](problems/budget-index) | A search system cannot index everything it stores — choose what to embed under a cost budget. |
| [cross-lingual-organization-name-normalization](problems/cross-lingual-organization-name-normalization) | Match institution names across scripts and languages to a curated index. |
| [edrmi](problems/edrmi) | Single-GPU entity-resolution solve under a 1.5-hour runtime bound. |
| [feature-trace](problems/feature-trace) | Trace feature provenance through a transformation pipeline. |
| [galaxy-blind-spot](problems/galaxy-blind-spot) | Automated morphology catalogues are built by running a classifier over millions of galaxies — find where it fails. |
| [habitat-contact](problems/habitat-contact) | Species-habitat contact inference. |
| [law-tree](problems/law-tree) | Hierarchical structure recovery over legal text. |
| [model-routing](problems/model-routing) | Route queries across models under a cost/quality tradeoff. |
| [palimpsest](problems/palimpsest) | A wiki page is never written once — recover the edit history layered inside it. |
| [phantom-track](problems/phantom-track) | Data association across the instant between two observations. |
| [pipeline-attribution](problems/pipeline-attribution) | Attribute digitized newspaper scans to the Library of Congress batch that produced them. |
| [prompt-atlas](problems/prompt-atlas) | Predict how a hidden news classifier reacts to different wordings of the same question. |
| [prompt-delta](problems/prompt-delta) | Measure behavioural drift between prompt variants. |
| [pump-failure](problems/pump-failure) | Train a failure-risk model from scratch on hourly condition-monitoring telemetry. |
| [recompile](problems/recompile) | Federal regulations are patched, not rewritten — reconstruct the current text from the patch stream. |
| [repeat-harvest](problems/repeat-harvest) | Reverse-engineer a length-preserving string function over DNA. |
| [restock-radar](problems/restock-radar) | Predict grocery restock demand from view / add-to-cart / purchase logs. |
| [ror-names](problems/ror-names) | Normalise research-institution names against the ROR registry. |
| [slate-forensics](problems/slate-forensics) | Reconstruct a UK giftware retailer's "You may also like" recommender from transaction records. |
| [storm-genealogy](problems/storm-genealogy) | Storm lineage reconstruction. |
| [unfluke](problems/unfluke) | Separate skill from luck across 6,080 systematic trading records. |
| [valencia-o3](problems/valencia-o3) | Calibrate two low-cost IoT sensors against a regulatory air-quality station over 238 days. |

## Working with this repository

Large assets (archives, `.npy`/`.npz`, model weights, and several oversized
CSV/JSON datasets) are stored in **Git LFS**. Clone with LFS available:

```bash
git lfs install
git clone https://github.com/indwar7/Eris-ML.git
```

Without `git lfs install`, those paths clone as small pointer files rather
than real data.

### History note

Earlier revisions of this repository used a `cpu/` and `gpu/` split and had
removed six challenge folders as superseded. Both layouts are merged here:
everything now lives under `problems/`, and the six removed packages are
restored — each is complete, with a grader and a reference solution.
