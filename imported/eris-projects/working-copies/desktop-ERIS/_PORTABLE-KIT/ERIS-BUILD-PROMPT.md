# The Prompt

Copy everything between the lines into a fresh Claude Code session, with
`ERIS-MASTER-GUIDE.md` in the working directory.

---

Read `ERIS-MASTER-GUIDE.md` in full before doing anything else. It is the
accumulated record of ~20 built ERIS/Shipd tasks and every rejection they
earned. Every rule in it cost a revision round. Treat it as binding, not
advisory.

Build me one new ERIS task, complete and submission-ready.

## What I need

- Novelty **8** (floor is 6, and the guide's Part 6 explains why the displayed
  number runs 1–2 below the analysis text — design for 8 to land above 6).
- Agent scores in the band **0.3–0.7**.
- **GPU track**: 1× A10G, 10 cores, 62 GB, under 1.5 h. The task must genuinely
  need the GPU.
- **Real source data** with a **named licence from the platform dropdown** and a
  **reachable source URL**. Synthetic data forfeits a novelty point outright.
- A source, title and task shape that collide with **nothing** in the guide's
  Part 14 register — Duplicate Detection reads title wording and dataset shape,
  not just the URL.
- Not seq2seq specifically unless that's what fits; pick the shape the data
  supports from the guide's Part 1 Gate 4 table.

## How to work

**Do exactly what I asked, in the simplest way that works.** No clever
alternatives, no unrequested extras, no deliverables I didn't ask for. If you
hit a real blocker, pick a sane default and keep moving — don't stop to ask me
unless the answer genuinely changes what you build.

**Measure, never assume.** Every number in every shipped file comes from a
script you actually ran against the exact files being shipped. Every claim
about a shortcut being closed comes from running that shortcut. If you write a
number you did not measure, you have failed the task.

**`git init` and commit before touching any passing state.** Commit
`known-good: <status>` every time a full audit passes.

Work in this order and do not skip ahead:

### 1. Gates — before writing any code

Run all five gates from the guide's Part 1. Verify the licence from the
source's own API or metadata and quote it verbatim. `curl -sI` the source URL.
Confirm the label is the archive's own structured value. Check the task shape
against both hard rejects and against the Part 14 register.

Report the five gate results to me in a short block before continuing. If any
gate fails, pick a different source — do not carry the idea forward.

### 2. Feasibility

Fetch 30–50 items. Measure yield, item length, label distribution, fetch rate.
Check the scale thresholds. Then run **every** leak probe in the guide's Part
2.3 including the adversarial "ignore the task structure" probe and the
lookup/sibling-analogy probe.

A probe within 0.10 of your planned reference is a fail. A probe beating it
means abandon the design and tell me why.

### 3. Metric, chosen before the split is built

Measure copy-through and every degenerate strategy under the candidate metric
before freezing it. If a degenerate answer scores above ~0.3, the metric is
wrong. Write down which alternatives you rejected and the measured number that
killed each one — I need that for the Evaluation section.

### 4. Fetch

One worker, sleep 0.8–1.5 s, broad `except Exception`, incremental writes, pin
a dated snapshot with a sha256 where the source republishes a rolling file.
Never kill a running fetch to improve something.

### 5. Split, prepare, close leaks

Split by the real group unit, then check exact overlap on the **shipped**
columns. Verify no transductive shortcut. Check the leaderboard row-wise half
split. Close any external-lookup path with all five layers and state plainly
what the closure cannot do.

If the split turns out to have removed the signal along with the shortcut — the
signal that a trained reference lands below a trivial heuristic — say so and
change the split, don't tune the model.

### 6. Grader

`grade(submission, answers) -> float`, two arguments, no filesystem access.
Degrade never reject. `InvalidSubmission` subclasses `ValueError`. Unit-test on
≥4 hand-built cases and against ~20 deliberately damaged submissions.

### 7. The ladder

Build **5+ ablations of the real solver**, distinct and strictly ordered, with
the reference on top — not just baselines. Target ~0.18 margin between
reference and best adversary. Confirm oracle == max_score and every degenerate
== min_score. Print the reference's per-item distribution and count distinct
values; mostly 0s and 1s means fix the matching first.

### 8. Package and verify

Produce every file in the guide's Part 9 layout. Build the raw zip from
`dataset/raw` + `dataset/generator` **only**, then extract it somewhere clean,
run `prepare.py`, and prove every generated file comes back byte-identical.

Write `run_checklist.py` implementing the guide's Part 11 checklist as an
actual script and run it. Report PASS/FAIL counts. **Any FAIL blocks the
build** — fix it and re-run.

### 9. Report

Give me:
- the five gate results,
- the full measured ladder as a table,
- the PASS/FAIL count from `run_checklist.py`,
- the one-line task statement and why its (input, output, metric) triple
  differs from the nearest named benchmark in ≥2 of 3,
- anything you measured that turned out to contradict a design assumption.

## Non-negotiables

- The word **"challenge"** appears nowhere in any file or folder name. Use
  "task", "problem" or "evaluation".
- No markdown pipe-tables and no tilde code fences in
  `problem-description.md` or `dataset-description.md`.
- No LLM-generated content anywhere in the data pipeline.
- The data-path discovery snippet, never a hardcoded path:
  ```python
  DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                          Path("/kaggle/input/<name>/public"))
              if (p / "test.csv").exists())
  ```
- `dataset-description.md` describes only what the **dataset** ships. The
  prepared split goes under "What `prepare.py` derives (not shipped here)".
- Every rule in `problem-description.md`'s Rules section has a matching
  `[REQUIRED]` rubric.
- The title is identical across `config.yaml`, `problem-description.md` H1,
  `README.md` H1 and `solution.ipynb`'s first cell.
- `config.yaml` carries the full ladder with a `note_band` explaining every
  number, and earlier iterations' ladders kept in comments.

Start with the gates. Report them before you build anything.

---

## Shorter variant — when you already know the source

Read `ERIS-MASTER-GUIDE.md` in full first; it is binding.

Build one ERIS task on `<SOURCE URL>`. Novelty 8, agent band 0.3–0.7, GPU
track (A10G, <1.5 h), `<LICENCE>`.

Same workflow: gates → feasibility with every leak probe → metric chosen by
measurement → fetch → split → grader → 5+ ablation ladder → package →
`run_checklist.py` all PASS. Measure every number you write. `git init` first
and commit each passing state. Report the gates before building, and the ladder
plus checklist counts at the end.

Do exactly what I asked and nothing more.

---

## Repair variant — when a submission comes back rejected

Read `ERIS-MASTER-GUIDE.md` in full. Part 12 maps platform error messages to
their real causes.

Here is the rejection:

```
<PASTE THE EXACT PLATFORM MESSAGE>
```

Find that exact message in Part 12 and apply the fix it names.

**Fix only that check.** Do not bundle an unrelated improvement into the same
edit — a session that silently also changed the data encoding while fixing a
licence field destroyed a working dataset. If you notice a second problem, tell
me in one line and let me decide.

**Do not reshape the task.** A reshape to fix originality once broke domain
routing and reintroduced an already-fixed leak, and the original had to be
rebuilt from scratch. If only one check is failing, narrow the fix to that
check.

`git commit` the current state before touching anything. Then re-run
`run_checklist.py` in full — not just the failing item — and show me the
before/after on whatever number the fix moved.
