# Task state

## Current stage

`Stage 3 — built and locally validated`. Not yet run through `harbor`, not yet
difficulty-probed. Two author-owned fields still need the contributor's words.

## Task summary

A small in-memory view store (input cells, formula cells, hooks) ships with a
recalculation engine that is correct on values but recomputes the entire graph
on every write. The agent rewrites `/app/src/store.py` so it obeys twelve
written rules and evaluates only what is actually needed. Graded by driving the
store from outside on unseen programs and comparing every observable event in
order — hook runs, cleanups, reads and their values, disposals, exception
classes — plus how many times each formula was evaluated, against a sealed
reference; then one timed run.

## Why it is hard

Two rules fight: a formula is evaluated only when something needs it (rule 1),
and a formula that recomputes to an equal value has not changed, so nothing
below it may be evaluated because of it (rule 3). Together, the set of cells
needing work cannot be decided at write time — it depends on results that do not
exist yet, some of which come back "no change" and stop the walk. The eager
design is correct on values but evaluates cells nobody asked for and fails the
timed run; the naive lazy pull evaluates a shared cell twice down a diamond and
can show two branches different moments.

- Expert time estimate: 7 hours
- Failure mode targeted: iterative trial and error against a rule bundle where
  every design choice trades one rule against another
- Difficulty mechanisms used (docs/DIFFICULTY.md): 1 (laziness and the
  no-change rule invert the memorized eager/push default), 2 (the concept is
  described operationally — the term of art appears nowhere in the instruction,
  the spec, or the starter), 3 (no single published engine satisfies this exact
  bundle), 4 (fenced both sides: cases that must *not* re-evaluate and cases
  that must), 6 (timed run bans the naive-but-correct engine), 7 (twelve rules
  that must hold at once), 8 (one editable file, frozen API, stdlib only),
  9 (16 hardcoded corner cases + 50 randomised differential programs + timed
  run, all-or-nothing)
- Assistant's attack on the plan: I can name the shape of the correct design
  (two grades of staleness, mark cheaply, resolve on demand, notify only on a
  real change) because it is roughly how modern incremental-computation runtimes
  work. What I would get wrong on a first pass, and did while building the
  reference: a hook that writes a cell it reads. Change propagation during a
  pull and invalidation from a write look identical at the notify site, and
  treating them the same either loses the re-trigger or re-evaluates a shared
  cell in a diamond. That took a real fix, not a guess. Evaluation *counts*
  being graded — not just values — is what makes the near-misses fail.
- Estimated solves out of 8: 3
- Originality check: not searched yet. **Do this before submitting** — the
  underlying problem (incremental recomputation with equality cutoff) is
  well-documented in framework source, so the similarity screen matters. The
  defence is that the rule bundle is not any one published engine's semantics,
  and the grader compares evaluation counts.

## Local validation (all passing)

| candidate | result |
|---|---|
| `solution/store.py` (reference) | 83 passed, reward **1**, 0.25s |
| `environment/src/store.py` (starter, = nop agent) | 63 failed / 20 passed, reward **0** |
| `cheat/cheat_store.py` | 57 failed / 26 passed, reward **0** |
| `scripts/preflight.py` | all mechanical checks passed |

Timed run: reference 0.03s, starter 46.6s, ceiling 25s.

Re-verified 2026-08-10 by rebuilding the verifier image and running all three
candidates through `tests/test.sh` in the container. The starter's timed-run
margin over the ceiling measures 1.9x, not the ~3x an earlier note implied — see
the open item below.

## Still to do

1. **`relevant_experience` in `task.toml` is a placeholder.** The contributor
   must replace it with their own real experience; a human reviewer reads it.
   This is the only blocking item left.
2. Search for public write-ups of this exact rule bundle (originality gate).
3. `harbor check` (the oracle / nop / cheat runs were done directly with Docker
   and all three gave the right reward).
4. **Timed-run margin is 1.9x, not 3x.** The starter finishes the timed schedule
   in 46.6s against a 25s ceiling. On grading hardware roughly twice as fast per
   core, an eager-but-correct engine would slip under the ceiling. Impact is
   contained — such an engine still fails `lazy-never`, `cutoff-absorbs` and the
   randomised programs on evaluation counts, so the reward stays 0 — but the
   timed run is meant to be the rule-12 gate on its own. Widening it means
   raising `TIMED_IDLE`, which changes numbers the contributor stated in their
   own `instruction.md` ("ten thousand formula cells"), so it is their call, not
   an edit to make silently.

## Done

- **`instruction.md` is the contributor's own text** (2026-08-10), written from
  a fact sheet rather than edited from an assistant draft. Checked against the
  fact sheet: every graded behaviour is stated, including the evaluation-count
  comparison and the timed run's exact shape (10000 idle cells, 5000 writes,
  three reads, 25s). Numbers verified against `tests/gen.py`. Required suffix
  present, blank line before it, single trailing newline.
