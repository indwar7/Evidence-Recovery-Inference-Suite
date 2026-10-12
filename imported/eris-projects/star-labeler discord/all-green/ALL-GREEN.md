# All Green

**Frontier-Bench (Extended Terminal-Bench) task — build plan**
Status: design draft, not yet built · Owner: Abbu · Started 2026-08-08
Supersedes `../cold-mirror/COLD-MIRROR.md` (retired against the pitfall list — see that file's header)

---

## 1. The name

**`all-green`**

CI is green. The test suite is green. The eval report is green and the numbers sit exactly where you'd expect. The model is wrong. The name *is* the failure mode, which is the point.

---

## 2. The one deep thing

**Point-in-time correctness.**

Not five mechanisms. One, and it goes all the way down.

The repo trains an incident-escalation classifier for a freight operations team. The training table was built by flattening a live operational database — so several features carry values that were only written **after** the outcome they're predicting. The flagship offender is `route_replan_count`: the ops system increments it when a planner reroutes a shipment, which happens for ordinary reasons *and* automatically whenever an escalation opens. In the training snapshot it is a partial echo of the label. At serving time it is present, plausible, and quietly poisoned.

Everything downstream is consistent with the corruption, which is why nothing looks broken:

- `data/val.csv` is a **random** split, not a temporal one, so the leak is present on both sides and never shows.
- The eval report lands in the range the team expects, so nobody re-derives it.
- The test suite checks schema, null rates, and row counts — all true, all green, none of them about behavior.

The fix is not "drop the leaky columns." Those features are genuinely predictive *before* an escalation; dropping them costs enough accuracy to miss the floor. The agent has to find the raw event log, reconstruct what each feature's value **was as of the prediction timestamp**, rebuild the training set from that, and retrain. That is the hours-to-days part, it is iterative, it requires exploring an environment rather than reading a fact, and none of it is findable online because the corruption is specific to this database.

> **Depth knob, held in reserve.** A prior commit added `calibration.py` — temperature scaling plus a hand-tuned threshold shift, commit message *"fix overconfidence on escalations."* It exists to compensate for leak-inflated confidence. Once the leak is gone it over-corrects and pushes the headline metric **down**, so the obvious response is to revert the real fix. This is the "obvious approach quietly breaks something later" beat.
>
> **Do not ship it in v1.** Build the minimum, measure at Gate 3, and add this only if frontier agents are passing. Adding depth up front is how the stacking pitfall gets you.

---

## 3. Why the failure is real

Named before anything else was designed:

**The agent trusts the harness it was given.**

It will run the pipeline, read the eval report, see healthy numbers, and start looking for bugs in the *code*. The code is fine. The corruption is in how the training data was assembled from a database the agent has to go find. An agent that never questions `val.csv` never sees anything wrong — and an agent that does notice the leak will reach for the cheap fix (drop the columns) and land under the accuracy floor.

Both failure paths are substantive. Neither is trivia, missing context, or a handicap.

---

## 4. Why this grades to a clean 0/1

This is the constraint that kills most ML tasks: binary reward plus a continuous outcome forces a tuned threshold, and a tuned threshold is either luck or a wall. All Green avoids it because the primary check is an **invariance**, not a score.

**Check 1 — point-in-time agreement (the real check).**
The verifier replays K hidden incidents twice: once with features as of the prediction timestamp, once with final-state features. **Every decision must match.** A model that uses no post-hoc information scores exactly 100% by construction; a leaked model disagrees on a large fraction. There is no band to luck into — the metric is bimodal by design, which is what makes it honest under binary grading.

**Check 2 — accuracy floor (a guardrail, not a difficulty knob).**
Point-in-time accuracy on hidden held-out incidents must clear a floor set *generously* — comfortably reachable by any competent reconstruction, unreachable by a constant predictor or by dropping every implicated column. Its only job is to stop degenerate solutions from trivially satisfying Check 1.

Both are stated in the brief in full.

---

## 5. What the brief states — and what it doesn't

The pitfall list is explicit: anything the verifier checks goes in the brief. Difficulty comes from the work, not from guessing the rules.

**Stated in `instruction.md`:**

- The deliverable: a retrained model plus a serving entrypoint at a fixed path, stdlib + the pinned env.
- **The full point-in-time contract**, in plain terms: predictions must depend only on information available at the prediction timestamp, and the verifier will score hidden incidents from both a point-in-time snapshot and the final-state record and require identical decisions.
- The accuracy floor, the exact metric, and the hidden-set size.
- The runtime cap and that retraining must complete inside it.
- That the existing eval harness and test suite are the team's, not ground truth.

**Not stated — because it is the work, not a rule:**

- Which features violate the contract.
- That the event log exists, or how to reconstruct history from it.
- That the provided split is random rather than temporal.

The contract is stated in general form precisely so it can be complete without naming the offender. The agent knows exactly what winning means from sentence one. Finding *which* features break it, and rebuilding them rather than deleting them, is the task.

---

## 6. Build plan

### Phase 1 — The database, then the corruption

Build the operational database first: an event log with real temporal structure — incidents, planner actions, status transitions, timestamps. Then derive the flattened training table from it *with the leak*, exactly the way a rushed data engineer would: a `GROUP BY` over the full history of each incident with no `AS OF` filter.

The event log must contain enough to reconstruct every feature's point-in-time value. If it doesn't, the task is unsolvable and the difficulty is artificial.

> **Gate 1:** an honest point-in-time rebuild is possible from the shipped event log alone, and the resulting clean model clears the accuracy floor with real margin. Prove it by building it. This is the [[dataset-first-principle]] beat — the data and the eval are the task; the prose is packaging.

### Phase 2 — The green surface

Random-split `val.csv`, an eval report with plausible numbers, a passing test suite that checks schema and null rates and nothing behavioral, and a clean `git log`. Everything must survive an agent's first hour of poking without looking staged.

> **Gate 2:** run the pipeline end to end from a clean container. No warnings, no anomalies, no comment or filename that points at `route_replan_count`.

### Phase 3 — Difficulty calibration

The gate that decides whether this ships.

Three independent frontier-agent attempts (Claude Opus, GPT-5.x), full time budget, real `instruction.md`, scored by the real verifier.

> **Gate 3:** **0/3 pass**, and each failure is one of the two designed paths — trusted the harness, or dropped instead of reconstructed. A failure like "couldn't find the database" or "ran out of time on setup" means the environment is wrong, not that the task is hard.
>
> If any attempt passes: add the `calibration.py` knob from §2 and re-measure. If attempts still pass after that, the mechanism class is too shallow — pivot, don't keep stacking.

### Phase 4 — Reference solution

`solution/solve.sh` solves it the honest way: locate the event log, reconstruct point-in-time features, rebuild, retrain, serve. Keep the transcript as evidence for reviewers that the reconstruction is derivable.

> **Gate 4:** scores 1 from a clean container, three runs, three separate hidden sets.

### Phase 5 — Verifier

`tests/test.sh` builds the hidden incidents from seeds the agent never sees, runs the dual scoring of §4, and writes the binary reward. Ground truth is generated at image build time and baked in. Runtime-capped so a search-at-verify-time can't win.

> **Gate 5:** deterministic across 10 runs; rejects every cheat in Phase 6.

### Phase 6 — `cheat/`

| Cheat | Why it fails |
| --- | --- |
| Drop every feature that differs between snapshots | Passes Check 1, misses the accuracy floor |
| Return a constant / majority class | Same |
| Hardcode predictions for the visible incidents | Hidden incidents are freshly seeded |
| Keep the leak, add a wrapper that ignores the field at serve time | Field still shapes the learned weights; point-in-time agreement fails |
| Retrain on the flattened table with a "leakage-aware" regularizer | Doesn't restore point-in-time values; fails Check 1 |

### Phase 7 — Package & submit

`task.toml`, `README.md`, zip as `all-green.zip`, post to `#task-forum`.

---

## 7. Kill criteria

- **Gate 3 passes twice** (once bare, once with the calibration knob) → the mechanism is too shallow. Pivot to a Cold Mirror shell rather than stacking more.
- The event log makes the leak *obvious* — e.g. a column literally named `post_escalation_*`. Rebuild the schema.
- Reference solution needs more than ~60% of the agent's budget. Retraining cost is the thing to watch; keep the model small so the loop is fast and the difficulty stays in the diagnosis.
- Public terminal-bench task list already contains a train/serve-skew or leakage task. **Check this first — it is the cheapest kill.**

---

## 8. Immediate next actions

1. Scan the public terminal-bench task list for data-leakage / point-in-time / train-serve-skew overlap.
2. Design the event-log schema and build the honest point-in-time rebuild. Clear Gate 1 before writing a word of `instruction.md`.
3. Build the green surface, clear Gate 2, then go straight to the 3-attempt measurement. Nothing gets polished before Gate 3.
