# Cold Mirror

**Frontier-Bench (Extended Terminal-Bench) task — build plan**
Status: **SUPERSEDED 2026-08-08** — see `../all-green/ALL-GREEN.md` · Owner: Abbu

> **Why it was retired.** Audited against the platform's pitfall list and failed two checks.
> (1) *"Stacking lots of hidden requirements instead of making one thing genuinely deep"* — the design
> required five separate mechanisms (debounce, hysteresis, latching, drift, priority masking) to all be
> recovered under a binary reward. That is the stated anti-pattern, and it flips the task to never-solved.
> (2) *"Hiding information to create difficulty"* — black-box cloning's difficulty is structurally that the
> mechanism is hidden, so the task's survival would rest on winning an argument with a reviewer.
> Kept on disk because the **verifier design is still good** (sealed environment kills the proxy cheat;
> exact behavioral match, no surface checks) and the pivot shells in §8 remain live if All Green dies.

---

## 1. The name

**`cold-mirror`**

A mirror that reflects a system it can no longer see. The agent gets bounded, live access to a black-box device, and must ship a reimplementation that reproduces it *exactly* after the device is taken away — a cold mirror.

Directory / zip / `task.toml` name: `cold-mirror`.

---

## 2. Premise (one paragraph)

The container ships a stripped, undocumented binary — `/opt/bms/faultd` — that emulates the fault-diagnostics coprocessor of a battery-management system. It accepts raw ADC telemetry frames over a Unix socket and returns a 32-bit diagnostic word per frame. There is no source, no datasheet, no symbols. The agent must probe it under a hard query budget and produce `/app/mirror.py`, a standalone reimplementation that emits the identical diagnostic word for every frame of hidden telemetry sequences it has never seen — **with the coprocessor absent**.

Domain: hardware / embedded reverse-engineering. Real professional work (undocumented ECU and BMS diagnostic protocols get reverse-engineered for a living), programmatically verifiable, and nothing about it is findable online.

---

## 3. Start with the failure

Per the brief's own guidance: name the failure *before* designing anything.

**The expected failure mode: the agent fits a stateless function.**

An agent will probe the oracle, collect (frame → word) pairs, notice that the diagnostic word correlates strongly with the current frame's channel values, and build a per-frame decision function. That reproduces roughly 90–97% of frames and **scores 0**, because the coprocessor is a latching state machine:

| Hidden mechanism | Why per-frame fitting breaks on it |
| --- | --- |
| **Debounce counters** — a fault asserts only after N consecutive out-of-range frames | Output depends on run-length history, not the current frame |
| **Hysteresis** — de-assert threshold ≠ assert threshold | The same frame maps to two different words depending on prior state |
| **Latching** — some faults stay set until an explicit clear frame | Output depends on arbitrarily distant history |
| **Drift accumulator** — calibration offset integrates slowly across the session | Identical frames give different words early vs. late in a sequence |
| **Priority masking** — a higher-severity fault suppresses lower-severity bits | Bits are not independent; per-bit fitting is inconsistent |

Second failure lever: **the query budget.** `faultd` refuses more than a fixed number of frames per session and requires a session reset that scrambles internal state. Exhaustive enumeration of the input space is impossible. The agent must *design experiments* — isolate one channel, sweep a threshold by bisection, measure a debounce length by run-length probing, verify hysteresis by approaching a boundary from both directions. That is the hours-to-days part for a human expert, and it is iteration, not recall.

Third lever: **the obvious-path trap.** First-order behavior resembles a textbook threshold comparator with a CRC-ish checksum in the high byte. It isn't. The checksum is over the *post-mask* word plus a session-varying salt derived from frame index. An agent that pattern-matches to a known CRC polynomial from memory will lock onto the wrong hypothesis and burn its budget.

**If the agent cannot name and measure all five mechanisms, it cannot hit 100%.** That is the task.

---

## 4. Platform contract (Frontier-Bench rules, condensed)

Non-negotiables from the channel brief — check every one before zipping.

- **Binary reward.** The verifier scores exactly `0` or `1`. Pass iff the instruction is completed.
- **Beyond-frontier difficulty.** Expert: hours to days. Frontier agents must fail for *substantive* reasons — not trivia, not missing context, not artificial handicaps.
- **Open internet.** Network stays on. Difficulty must not depend on disabling it, and the solution must not be findable online.
- **Everything the verifier checks must be stated in `instruction.md`.** Hidden *rules* are a pitfall; hidden *work* is the point. State the contract, withhold the mechanism.
- **No re-skins.** Similarity checks compare the underlying problem, not the topic.
- **Anti-cheat probe will run.** If a shortcut reaches a passing state without doing the work, it will be found.
- Automated check runs within ~5 min of submission, up to 2 hours.

### Required structure

```
cold-mirror/
├── instruction.md              # the brief the agent reads
├── task.toml                   # config + metadata
├── environment/
│   ├── Dockerfile              # agent container — bakes faultd, no source
│   └── data/                   # sample telemetry, protocol stub
├── solution/
│   └── solve.sh                # reference solution — MUST score 1
├── tests/
│   ├── Dockerfile              # verifier image, bakes /tests
│   ├── test.sh                 # entry point — writes the binary reward
│   ├── sequences/              # hidden telemetry + ground-truth words
│   └── ...
├── cheat/                      # a deliberate cheating attempt (never executed)
└── README.md
```

---

## 5. Build plan

Seven phases. Each has a **gate** — do not start the next phase until the gate passes. Phases 1–3 are the real work; the rest is packaging.

### Phase 1 — Write the coprocessor spec (the secret)

Author `spec/coprocessor.md` (never shipped) fixing every mechanism numerically:

- Frame format: N channels, endianness, header, frame index.
- Per-channel assert/de-assert thresholds (hysteresis gap per channel).
- Per-fault debounce length (vary: 1, 3, 7 frames — so the agent can't assume one constant).
- Latch set: which faults latch, what clears them.
- Drift model: accumulator step, saturation point, reset condition.
- Priority mask table: which bits suppress which.
- High-byte check: post-mask word + frame-index salt.

**Rule:** every constant must be *discoverable by probing within budget*. Write down the probe procedure that recovers each one, and its query cost. Sum the costs — that sum sets the budget.

> **Gate 1:** the sum of minimal probe costs is between ~35% and ~70% of the query budget. Too cheap → the task is easy. Too expensive → the task is impossible and the difficulty is artificial.

### Phase 2 — Implement `faultd`

Reference implementation in C (or Rust), compiled `-O2 -s`, stripped, no debug info, no strings that leak thresholds (compute constants, don't `#define` them into a table that `strings` dumps).

- Unix socket at `/run/bms/faultd.sock`, length-prefixed frames.
- Session semantics: `RESET` scrambles state deterministically from a session counter; budget enforced per session; over-budget returns a defined error, not a crash.
- Fully deterministic given (session index, frame sequence). No wall-clock, no `/dev/urandom`.

> **Gate 2:** `strings`, `objdump -d`, `ltrace`, and a Ghidra pass do not hand over the constants faster than probing does. If static analysis trivially wins, add computed constants / control-flow that resists it — but never make it *impossible*, only not-cheaper. RE is a legitimate solve path; it just shouldn't be a five-minute one.

### Phase 3 — Difficulty calibration (the gate that actually matters)

This is where the task lives or dies. Measure before freezing — same discipline as every Eris challenge that survived review.

1. Build a throwaway container with the real `faultd` and the real `instruction.md`.
2. Run a frontier agent (Claude Opus, GPT-5.x) end-to-end, 3 independent attempts, full time budget.
3. Score each with the real verifier.

> **Gate 3:** **0/3 pass.** If any attempt scores 1, the task is not beyond-frontier — go back to Phase 1 and deepen *one* mechanism (longer latch chains, a second drift channel). Do **not** patch difficulty by stacking more hidden requirements — the brief calls that out as a pitfall that flips the task from always-solved to never-solved.
>
> Also record *how* each attempt failed. The failure must be "reconstructed the state machine incompletely," never "misread the frame format" or "ran out of tokens on boilerplate."

### Phase 4 — Reference solution

`solution/solve.sh` must score 1 and must solve it *the honest way*: probe within budget, recover constants, emit `mirror.py`. It doubles as proof the task is solvable at all.

Keep a second copy of the probe transcript — it is the evidence for reviewers that the constants are recoverable.

> **Gate 4:** `solve.sh` scores 1 on three separate hidden sequence sets, from a clean container, three times running.

### Phase 5 — Verifier

`tests/test.sh`:

1. Copy `/app/mirror.py` into a **sealed** environment: `faultd` absent, socket path gone, no network. Kills the proxy cheat outright.
2. Run it against **K hidden sequences** (K ≥ 8, ~2000 frames each) generated from seeds never exposed to the agent, covering: cold start, mid-session latch, drift saturation, hysteresis approached from both sides, priority-mask collisions, budget-boundary frames, and one adversarial sequence built from the *complement* of the probe-friendly region.
3. Compare word-for-word. **Any mismatch → reward 0.** No partial credit.
4. Enforce a runtime cap so a brute-force search-at-verify-time can't win.

Ground truth is generated by the reference `faultd` at build time and baked into the verifier image. The agent never sees the sequences or the words.

> **Gate 5:** verifier is deterministic across 10 runs, and rejects each cheat in Phase 6.

### Phase 6 — `cheat/`

Ship deliberate cheating attempts (never executed by the harness, but they document the threat model):

| Cheat | Why it fails |
| --- | --- |
| `mirror.py` shells out to `faultd` | Sealed verifier — binary and socket absent |
| Memorize probe I/O in a lookup table | Hidden sequences are freshly seeded, disjoint |
| Stateless classifier fit on probe data | Fails debounce / latch / drift frames |
| Reimplement 6 of 7 mechanisms | Binary reward, exact match — 99.9% is 0 |
| Copy the binary into the deliverable and exec it | Verifier rejects non-`.py` payloads / bans exec of baked blobs |

### Phase 7 — Package & submit

`task.toml` metadata, `README.md`, zip as `cold-mirror.zip`, post to `#task-forum`. Automated check runs within ~5 minutes, up to 2 hours.

---

## 6. What `instruction.md` must state (and must not)

**Must state** — everything the verifier checks:

- Deliverable path and form: `/app/mirror.py`, standalone, stdlib-only.
- It will be run **without `faultd` present and without network.**
- Exact I/O contract: reads a sequence file, writes one diagnostic word per frame, format specified byte-for-byte.
- **Exact match on every frame of every hidden sequence is required. Partial correctness scores zero.**
- Output depends on sequence history, not only the current frame — the mirror is fed whole sequences from a cold start and must reproduce the coprocessor's own cold-start behavior.
- The query budget, and that `RESET` exists and what it costs.
- The runtime cap.

**Must not state** — the work itself: thresholds, debounce lengths, which faults latch, the drift model, the priority table, the checksum construction. None of these are "rules the verifier checks" in the pitfall sense; they are the artifact under reconstruction.

The line to hold: *the agent should never be confused about what winning means, only about how to win.*

---

## 7. Kill criteria

Abandon or redesign if any of these hit:

- **Gate 3 fails twice.** Two frontier passes after one deepening pass means the mechanism class isn't deep enough — pivot, don't patch.
- Static analysis of `faultd` recovers all constants in under ~15 minutes of expert effort. The task becomes an RE-tooling exercise, not a systems-inference one.
- The reference solution needs more than ~60% of the agent's time budget. Too tight; the failure becomes "ran out of time," which reads as an artificial handicap.
- Similarity check flags overlap with an existing terminal-bench task. Check the public task list before Phase 2.

---

## 8. Fallback names / pivots

If `cold-mirror` dies at Gate 3, the same *shape* (bounded probing of a live black box → exact offline reimplementation) survives with a different oracle:

- **`understudy`** — a stateful rate-limiter / billing meter with tiered rollover.
- **`dead-reckoning`** — an undocumented dead-reckoning filter with clamped integrator and fault fallback.
- **`quarantine`** — a packet-classification engine with stateful connection tracking.

Same failure mode (agents fit stateless functions), same verifier design, different surface. Note: this is a legitimate pivot *within design*, not a re-skin of a submitted task.

---

## 9. Immediate next actions

1. Check the public terminal-bench task list for black-box-reimplementation overlap. *(Do this first — it's the cheapest kill.)*
2. Write `spec/coprocessor.md` and the per-constant probe-cost table. Clear Gate 1.
3. Implement `faultd`; clear Gate 2.
4. Stand up the throwaway container and run the 3-attempt difficulty measurement. **Nothing gets polished before Gate 3 passes.**
