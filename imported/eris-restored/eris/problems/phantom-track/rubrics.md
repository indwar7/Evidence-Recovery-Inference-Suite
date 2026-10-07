# Rubrics — Phantom Track: Appearance-Free Identity Re-Linking

Task-specific criteria for good engineering on this challenge. Evaluated alongside the
link-accuracy score.

## Rubric 1 — REQUIRED

**Solves a global one-to-one assignment, not per-object nearest neighbour.** The solution
links frame-A to frame-B objects with a mutually-exclusive matching (e.g. the Hungarian
algorithm / minimum-cost bipartite matching), rather than picking each object's closest
candidate independently. Greedy nearest-neighbour double-books popular destinations and
scores ≈ 0.46; a global assignment scores ≈ 0.59. This gap is the point of the task.

## Rubric 2 — REQUIRED

**Matches on real motion geometry, with no appearance shortcut.** Because objects carry
only 2-D positions in two frames — no colour, size, id, or descriptor — the solution must
derive every link from the geometry of where objects were and where they went. It uses the
real motion statistics (e.g. the empirical displacement distribution, recoverable from the
supervised split) to define its matching cost, and does not rely on point index order,
which is scene-shuffled and carries no information.

## Rubric 3 — REQUIRED

**Accepts the irreducible ambiguity rather than fighting it.** The solution recognises that
some objects are unresolvable from position alone — in particular the coincident hard pairs,
which are a provable coin flip — and does not distort the rest of the matching trying to
force them. No submission can exceed ≈ 0.882 (the hard-pair cap) or, realistically, ≈ 0.60
(the position-only optimum); a solver that reaches the mid-band and stops is correct.

## Rubric 4 — REQUIRED

**Produces a valid submission by construction.** Exactly one row per `(scene_id, a_id)` in
the test frame-A set, with an integer `b_id` in `0..33`, columns `scene_id, a_id, b_id`.
The matching should be a per-scene permutation (no double-booked `b_id`), since duplicates
guarantee at least one wrong link. Malformed rows are scored incorrect leniently and never
crash the grade.

## Rubric 5 — RECOMMENDED

**Runs deterministically and cheaply.** The pipeline is seeded and reproducible, uses only
CPU and standard scientific Python (numpy and scipy suffice), and completes well within the
time budget (the reference solution runs in ~2 seconds). No hardcoded id-to-answer mappings
and no attempt to recover held-out links from anything but the released public files.
