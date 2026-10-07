# The Edit That Moved the Answer

**Domain:** natural-language-processing · **Task type:** ranking
**Metric:** positional credit · **Runtime:** 1x NVIDIA A10G, <1.5h (reference: ~1.5 s, CPU)
**Licence:** CC0-1.0

## The task

A prompt is edited in exactly one way — a persona is added, a format is forced, a hedge
is inserted, a chain-of-thought instruction appears. That single edit moves the model's
behaviour by different amounts depending on what it is asked about.

You are given the measured behaviour fingerprints of the model's responses **before and
after** the edit, across four candidate input kinds. Rank the four by how much the edit
moved the model on each. Credit is `[1.0, 0.33, 0, 0]` by where the true most-affected
candidate lands in your ordering.

Unranked candidates score `0.337`, a uniform random permutation `0.350`, the strongest
measured shortcut `0.418`, and the ridge-regression reference `0.493`. This is the widest
headroom in the suite.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and measured baseline anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, measurement method, schema, intended use, limitations |
| `rubrics.md` | The rules a submission must satisfy |
| `prepare.py` | Deterministic raw → public/private split (seed 20260912), framing-disjoint |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution — ridge regression over fingerprints |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |
| `STATE.md` | Construction log, including the label rule that had to be rewritten |

## Reproduce

```bash
python grade.py dataset/public/sample_submission.csv \
                --answers dataset/private/answers.csv     # -> 0.3371 (floor)
```

## Data

4,608 real greedy generations from `HuggingFaceTB/SmolLM2-1.7B-Instruct` across 16
hand-written prompt framings × 12 conditions (11 edit operations plus the unedited
baseline) × 6 hand-written input kinds × 4 hand-written probes. Every response is reduced
to a 16-dimensional numeric behaviour fingerprint **at generation time**.

**No response text is stored or released anywhere.**

Split is **by prompt framing**: every item from one framing lands entirely in train or
entirely in test, and which three of sixteen framings are held out was chosen by
exhaustive search over all same-size combinations, optimising for the test ratio *and*
the lowest measured majority-vote shortcut on the resulting test set.

**CC0-1.0** for the measured fingerprints, grid definition and derived splits. Apache-2.0
governs the model's weights and code, not the text it generates. See `../../docs/LICENSES.md`.
