# settle-order

A Software / Frontend task for Frontier Bench. The agent is given a small
in-memory view store whose recalculation engine is correct but recomputes the
whole graph on every write, and must rewrite it to obey a precise set of rules
while evaluating only what is actually needed.

The hard core is that two rules pull against each other: a formula is evaluated
only when something needs its value, and a formula that recomputes to an equal
value has not changed, so nothing below it may be evaluated because of it.
Together they mean the set of cells needing work cannot be decided at write
time. The obvious eager design is correct on values but evaluates cells nobody
asked for and fails the timed run; the obvious lazy pull evaluates a shared cell
twice down a diamond and can show it two different moments at once.

- `instruction.md` — the task as the agent sees it.
- `environment/` — the agent container. `src/store.py` is the shipped engine
  (the only file the agent edits); `README.md` is the full rule spec;
  `harness.py` is a scratch harness.
- `solution/` — the reference engine installed by `solve.sh`.
- `tests/` — the sealed verifier: a reference engine, a neutral program driver,
  a generator (hand-derived corner cases, randomised differential programs, and
  a timed schedule), and the grader.
- `cheat/` — a documented cheating attempt the verifier rejects (never executed).

Verification drives the agent's `Store` from outside and compares every
observable event in order, plus the number of times each formula was evaluated,
against a sealed reference. Comparing evaluation counts is what stops an engine
that reaches the right values by computing far too much. Agent code runs only
inside an unprivileged subprocess that cannot read the reference or the answers;
the root grader never imports agent code and sets the reward from its own exit
status.

## Local numbers

Measured on the author's machine while building the task:

| | corner cases | random programs | timed run |
|---|---|---|---|
| reference | 16/16 match hand-derived answers | — | 0.03s |
| shipped engine (starter) | differs on 12/16 | differs on 50/50 | 46.6s |

The timed ceiling is 25s. Through the containerised verifier the reference
scores 1 with 83 passed; the shipped engine scores 0 with 63 failed; the
documented cheat scores 0 with 57 failed.
