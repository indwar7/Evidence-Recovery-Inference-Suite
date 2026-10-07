# Phantom Track: Appearance-Free Identity Re-Linking

## Overview

A tracker's hardest moment is the instant between two observations: the same objects
are seen twice, a fraction of a second apart, and *nothing about how they look* tells
you which is which — you have only where they were and where they now are. This
challenge isolates that moment and makes it the whole task. You are given a **scene**:
`N` anonymous objects observed at two instants — **frame A** (time `t`) and **frame B**
(time `t + 15 s`) — as nothing but 2-D positions. Identity between the frames is
hidden. You output the **re-linking**: for each frame-A object, which frame-B object is
the same one. It is a pure **data-association** problem on real motion, scored against
the true correspondence, with a provable ceiling below 1.

**The motion is real.** Every object's displacement from frame A to frame B is a real
15-second GPS displacement of a **Porto taxi**, drawn from the ECML/PKDD-2015 Porto taxi
dataset (442 taxis, GPS pings every ~15 s; `kraina/porto_taxi`, CC-BY-4.0). The
distribution of speeds, turns, and stops that governs how confusable two objects are is
the real distribution of a city's traffic — median step 148 m, with a heavy tail of fast
and near-stationary vehicles. Nothing about the *motion* is invented. The only synthetic
element is where each object is placed within a scene, which is what lets the difficulty
be dialled to a target band and the ceiling be certified.

**The task.** For each scene you are given two sets of `N = 34` points: frame-A points
(the query, indexed `0..N-1`) and frame-B points (the candidates, indexed `0..N-1` in a
**scene-shuffled order** that carries no information). Each frame-A object moved to
exactly one frame-B point. Output, for every frame-A index `a_id`, the frame-B index
`b_id` you believe is the same object — a one-to-one re-linking.

**How you are scored.** **link accuracy** = the fraction of frame-A objects, across all
evaluation scenes, whose submitted `b_id` equals the true one. It is a smooth number in
`[0, 1]`; higher is better.

**Why it is not nearest-neighbour.** The naive move — link each frame-A point to its
closest frame-B point — **double-books** popular targets: two objects that both end up
near the same place both grab it, and one is wrong. Because objects move different real
distances (a stopped taxi barely moves; a fast one jumps 300 m), the closest frame-B
point to `a` is frequently *another object's* destination. The task is a **global,
mutually-exclusive assignment**: the right answer is the one-to-one matching that is
jointly most consistent, not `N` independent nearest-neighbour picks. Replacing greedy
matching with a global assignment is the measured skill — it moves the score from
**≈ 0.46 to ≈ 0.59**.

**Why it has a provable ceiling below 1.** In every scene, four of the objects are
arranged as two coincident **hard pairs**: the two objects of a pair share an *identical*
frame-A position. When two frame-A points are the same point, *which* one moved to *which*
of the two frame-B destinations is information-theoretically a coin flip — no solver, not
even one that knows exactly how the scene was built, can do better than 1/2 on such a
pair. Four hard-pair objects per scene therefore cap the **expected** link accuracy at
`(34 − 4) / 34 = 0.882`, a rigorous bound no submission can exceed. The realistic ceiling
is lower still: even among the ordinary objects, confusable ones cannot all be resolved
from position alone, so the best achievable accuracy from the observations is **≈ 0.60** —
which the reference solution essentially reaches. The whole challenge lives in the gap
between naive nearest-neighbour (≈ 0.46) and that limit.

**Why appearance-free matters.** Objects carry *no* features — no colour, no size, no id,
no descriptor — only 2-D positions in two frames. So no object detector, no learned
re-identification embedding, and no pretrained model can help: the only signal is
geometry and the one-to-one constraint. This is what makes the task a clean test of
**data association** itself, the primitive at the heart of every multi-object tracker,
rather than a test of perception.

## What is new here

Multi-object tracking benchmarks (MOT17, KITTI, DanceTrack) score *end-to-end* tracking,
where association is entangled with detection and appearance, and the graded object is a
set of trajectories. Here the graded object is a **single frame-to-frame re-linking**,
appearance is removed by construction, and the data is real *motion* rather than real
*pixels*. Point-set registration and optimal-transport matching study the same primitive
but on synthetic or rigid point clouds; this challenge supplies real, heterogeneous urban
motion and a certified ceiling. The task is a **permutation-recovery** problem — not
classification, regression, ranking, retrieval, or detection — and it does not overlap
any standard competition format.

## Dataset

Real Porto-taxi motion, synthetic placement. Files:

- `points_train.csv` — 4000 supervised scenes. Columns: `scene_id` (string), `frame`
  (`A` or `B`), `point_id` (integer, 0-based within its frame and scene), `x`, `y`
  (floats, metres in a local planar frame). Each scene has 34 frame-A rows and 34
  frame-B rows.
- `matches_train.csv` — the revealed links for the training scenes: `scene_id`, `a_id`,
  `b_id` (the true frame-B index for each frame-A object). Use these to calibrate and
  validate your matcher.
- `points_test.csv` — 2500 evaluation scenes, positions only (same schema as
  `points_train.csv`, no matches).
- `sample_submission.csv` — a valid identity baseline (`b_id = a_id`); because frame-B is
  shuffled, it scores at the ~random level.

The training split is larger than the evaluation split, so there is ample supervised data
to measure your association accuracy and study the real motion distribution before you
submit.

## Submission Format

Submit a CSV named `submission.csv` with exactly three columns, one row per frame-A
object in `points_test.csv`:

```
scene_id,a_id,b_id
test_00000,0,17
test_00000,1,4
test_00000,2,29
```

- `scene_id` — the scene, matching `points_test.csv`.
- `a_id` — a frame-A `point_id` in that scene (`0 .. 33`).
- `b_id` — the frame-B `point_id` you link it to (`0 .. 33`).

Validation and edge cases:

- One row per `(scene_id, a_id)`; on a duplicated pair the last row wins.
- A missing `(scene_id, a_id)` is scored as an incorrect link (0 for that object).
- A malformed or non-integer `b_id` is dropped and scored incorrect; it never crashes the
  grade. Extra rows for unknown scenes/objects are ignored.
- A submission need not be a strict permutation per scene — but double-booking a frame-B
  index guarantees at least one of those links is wrong, so a valid one-to-one matching is
  always at least as good.

## Evaluation

**link accuracy** = fraction of the 85,000 evaluation query objects (2500 scenes × 34)
whose submitted `b_id` equals the true frame-B index. Range 0 to 1; higher is better.

Score anchors, all measured on this exact dataset:

- identity baseline (`b_id = a_id`) scores ≈ **0.030** (≈ 1/N, i.e. chance),
- naive nearest-neighbour matching scores ≈ **0.462**,
- the reference **global assignment** (Hungarian on Euclidean cost) scores ≈ **0.591**,
- the position-only optimum is ≈ **0.60**, and no submission can exceed the **provable
  ceiling ≈ 0.882** set by the coincident hard pairs.

A competent solver recognises the problem as a minimum-cost bipartite matching and lands
in the **0.5–0.7** band; the distance from 0.46 to 0.59 is exactly the value of solving it
globally rather than greedily.

## Rules and Environment

- CPU only: 10 cores, 62 GB RAM, at most 1.5 hours end to end. The reference solution runs
  in about 2 seconds using numpy and scipy.
- Use only libraries in the Kaggle Python Docker image (numpy, pandas, scipy,
  scikit-learn, and similar).
- No internet access; no external data beyond the released files.
- No LLM-generated outputs.
- No hardcoded id-to-answer mappings, and no attempt to reconstruct held-out links from
  anything but the released public files.
- Your notebook must run end to end — load the scenes, match each one, write
  `submission.csv` — deterministically and seeded.
