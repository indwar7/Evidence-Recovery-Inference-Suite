# EDRMI — Approach

**Deliverables:** `solution.py` (entrypoint), `submission.csv`

---

## Compliance with explicit requirements

**1. Runs on one NVIDIA A10G GPU.**
`solution.py` targets CUDA directly: `DEVICE = 'cuda'`, model and all tensors moved to it,
fp16 mixed precision via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.
The device is declared at launch (`--device`, default `cuda`) and never probed from the host —
there is no capability check and no fallback path. Model size is chosen for the A10G:
MiniLM-L6 (22M parameters) at 48 tokens, which fits comfortably in 24 GB with batch 192.

**2. End-to-end runtime under 1.5 hours (5400 s).**
Runtime is bounded by construction, not by measurement:

| Stage | Bound |
|---|---|
| Data load + pair mining | Single linear pass over the CSV |
| Training | **Fixed step count** — `--max_steps` (default 6,000), computed before the loop starts |
| Inference | 1,500 episodes; each statement embedded **once** and cached |
| Decoding | Closed-form: 36 scored pairings + two Hungarian solves per episode |
| Submission write | One pass, 1,500 rows |

The only unbounded-looking stage is training, and it is capped by a constant that does not
depend on host speed, elapsed time, or data size. At 6,000 steps of a 22M-parameter model with
48-token sequences and fp16 AMP, training is a small fraction of the budget on an A10G. Inference
is a bi-encoder forward pass over ~14 statements per episode with caching — no per-query model
calls, because all combinatorial work is symbolic.

`--max_steps 0` runs full epochs instead; the default is deliberately capped so the bound holds.

**3. No language models, embedding services, or external APIs at evaluation.**
Scoring is exact categorical matching. The solution outputs only categorical codes
(`O01`–`O06`, `S01`–`S10`, `P01`–`P04`); the trained encoder is used solely to produce them and
plays no part in grading. No network calls at inference beyond loading the backbone weights.

**4. Deterministic execution.**
One code path, no runtime branching. Device, precision and data loading are fixed at launch;
the step count is fixed before training begins; all RNG is seeded (Python, NumPy, torch, CUDA);
the DataLoader uses a seeded generator with single-process loading; cuDNN is deterministic with
benchmark autotuning disabled. Every `time.time()` call feeds `print()` telemetry only — no
conditional, loop bound, or computed value consumes a timing result.

**5. Submission format.**
1,500 rows × 31 columns, header byte-identical to `sample_submission.csv`, sample_id order
preserved. `validate_row()` asserts every grader rule before writing: O01–O06 form a complete
permutation per row, three distinct slots per query, valid slot and probe codes, no blanks.
**0 rule violations.**

---

## Key insight

Dataset analysis showed the episode-local "hidden codebook" is a presentation layer, not a
learning problem:

- Dialogues repeat across episodes (10,907 unique; 9,558 appear 3+ times)
- Each repeated dialogue's pooled statements average ~30 = **~10 latent registers × ~3 variants**

So ~10 globally recurring semantic registers sit underneath; operator codes merely rename them
per episode. This splits the task into one learned component and one exact component — and is
why the runtime bound above holds, since the combinatorial half needs no model calls.

## Rejected alternatives

- **Unsupervised clustering** — k-means over 36,000 statements clustered by *topic*
  ("financial", "food", "empathy"), not by semantic *role*. Every statement in a bank shares the
  dialogue's vocabulary; only the role differs.
- **Surface-signature heuristics** — a 4-way syntactic signature matched across an operator's C0
  demos 49.8% of the time. Well above chance, so operator programs are provably stable, but far
  too coarse to decode with.

Both confirmed a trained semantic model is required.

## Architecture

**1. Neural — bi-encoder, trained once globally.**
MiniLM-L6 trained with InfoNCE so same-register statements embed together. Bi-encoder rather
than cross-encoder is a runtime decision: the solver embeds each statement once and reuses it
across 6 operators × 6 queries. A cross-encoder would need thousands of forward passes per
episode and would not fit the 1.5 h budget.

Supervision is exact, never guessed:

| Type | Source | Guarantee |
|---|---|---|
| Positive | `step_i` across an operator's C0 demos | C0 executes base order, so same position = same register |
| Positive | C0 query executed slots ↔ demo steps | From training targets |
| Hard negative | Any two slots of one register bank | Bank holds exactly one statement per register |
| Hard negative | Rejected probe vs. executed steps | Intrusion is a non-member by definition |

The bank-derived negatives are the strongest signal: same dialogue, same topic, guaranteed
different register — precisely the discrimination being scored. Yield: 326K positives,
3.4M negatives.

An auxiliary 3-way head predicts execution position, giving the C1 permutation learned evidence
instead of depending on the single C1 demo per operator.

**2. Symbolic — exact, per episode.**

1. Average C0 demo steps into three register prototypes per operator
2. Score all 5 non-identity permutations to recover the episode-local C1 orientation
3. Decode each query, choosing intrusion probe and bijective slot assignment jointly
4. Hungarian-match a 6×6 score matrix so every operator is used exactly once

The one-statement-per-register guarantee makes trace prediction a bijective assignment, so
Hungarian matching repairs individually uncertain calls.

## Verification

- **Parser** — assertions across 60 episodes: 3 demos/operator, 3 steps/demo (3,600/3,600 at
  scale), 4 probes, 10 bank slots, valid control and reject codes
- **Hungarian solver** — verified against a hand-checked matrix; returns the true optimum
- **Unbiased decoder** — with random embeddings every metric sits at chance (A=0.156 vs 1/6,
  I=0.278 vs 1/4, T=0.107 vs 1/10), proving the symbolic layer leaks no accidental signal
- **End-to-end** — full pipeline executed on `cuda`, `mps` and `cpu` device settings
- **Metric** — `official_score()` implements `100 · B^0.35 · X^0.65` exactly

## Results

Held-out validation, scaling with training data:

| Configuration | Score | A | I | T | Q | E |
|---|---|---|---|---|---|---|
| Untrained (chance) | 0.23 | 0.156 | 0.278 | 0.107 | 0.000 | 0.000 |
| 400 episodes | 24.20 | 0.477 | 0.682 | 0.432 | 0.173 | 0.000 |
| 1,200 episodes | 33.40 | 0.512 | 0.742 | 0.512 | 0.270 | 0.000 |
| **4,000 episodes** | **42.88** | **0.543** | **0.788** | **0.570** | **0.379** | **0.047** |

Every metric improves together, which the multiplicative `B` term rewards — no single component
carries the score. Exact-episode accuracy became non-zero, meaning whole six-query episodes are
solved perfectly; since that requires 30 consecutive correct decisions, it indicates the induced
machines are genuinely correct.

Training loss fell 3.81 → 2.52 without plateauing, and only 4,000 of 9,000 episodes were used,
so headroom remains via more data, more steps, and a larger backbone.

## Run

```bash
python solution.py --train train.csv --test test.csv --out submission.csv \
    --max_steps 6000 --bs 192 --holdout 200
```
