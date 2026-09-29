# Attributing Assistant Behavior to System Prompt Clauses

**Domain:** prompt-engineering · **Task type:** structured prediction
**Metric:** joint ranking and flag score · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC BY 4.0

> **Status: in build (~25%).** The design is locked, the source corpus is
> fetched and filtered to 145 prompts, and the generator, preparation script
> and grader are written. **The generation has not been run, so no data ships
> and no anchors are published** — `config.yaml` carries `TBD` in every anchor
> slot rather than an estimate. The preparation script and the grader have
> known defects, listed below. Full status and remaining work:
> [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md).

## The task

A system prompt is several instructions stacked together: a persona line, a
scope rule, a tone rule, a fallback. When the assistant behaves a certain
way, which of those sentences is responsible, and which are only sitting
there?

The ground truth is a real experiment. For each prompt, a language model
answers 8 probe questions under the full prompt, then again with each
sentence removed in turn. Every answer is reduced to 8 numbers; no answer
text is kept. How far the behaviour moves when a sentence is removed is that
sentence's responsibility.

You are shown a prompt's sentences in shuffled order and the measured
behaviour under the full prompt. You submit a ranking of the sentences by
responsibility and a `necessary` or `correlated` flag for each.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration; anchor slots marked `TBD` until measured |
| `STATE.md` | The design log: task shape, prior-art scan, label rule, corpus filter |
| `problem-description.md` | The prompt an agent sees; written against the first metric, to be revised |
| `dataset-description.md` | Source, measurement method, schema |
| `rubrics.md` | The rules a submission must satisfy |
| `prepare.py` | Label derivation from measured ablation, split by topic group |
| `grade.py` | `grade(submission, answers) -> float` |
| `dataset/generator/gen_raw.py` | The generator: corpus fetch, filter, and the ablation grid |

## Build it

```bash
pip install torch transformers accelerate
cd dataset/generator
python gen_raw.py --out /tmp/smoke --smoke --batch 8     # 5 prompts, 248 generations
python gen_raw.py --out ../raw --batch 16                # 145 prompts, about 8,000
cd ../.. && python prepare.py
```

The run needs a CUDA GPU. It writes a checkpoint every few batches and
resumes from it if interrupted.

## Known defects, not yet fixed

Each was found by reading the code or by arithmetic. None has been measured
on real data, because no real data exists yet.

| Defect | Where | Consequence |
|---|---|---|
| The metric has a high floor | `grade.py` | About 70% of sentences are `correlated`, so answering "all correlated" in any order would score about 0.60 without reading anything |
| Distances are in raw units | `prepare.py` | Word count, in the tens, swamps the yes/no measurements |
| Probe subsets overlap | `prepare.py` | The ten subsets share 2 or 3 of their 4 probes, so a prompt's items are near-copies |
| The experiment can be re-run | documents | A solver with the same model and questions could recompute the answer key; no rule forbids it yet |

## Data

System prompts from `danielrosehill/System-Prompt-Library` on Hugging Face,
923 prompts filtered to 145: 440 are dropped for naming their author, 286 for
numbered or bulleted structure whose steps depend on each other, and 52 for
having fewer than 3 or more than 10 sentences. No prompt wording is
rewritten.

**CC BY 4.0** — read from the dataset's own Hugging Face metadata. The
measuring instrument is `HuggingFaceTB/SmolLM2-1.7B-Instruct`, Apache-2.0.
Commercial use permitted with attribution.
