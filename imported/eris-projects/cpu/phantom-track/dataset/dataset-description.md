# Dataset — Phantom Track

## Source (real)

Every object's motion between the two frames is a **real 15-second GPS displacement of a
Porto taxi**, taken from the ECML/PKDD-2015 Porto Taxi dataset (`kraina/porto_taxi` on the
Hugging Face Hub; 442 taxis operating July 2013 – June 2014; GPS pings every ~15 s;
license **CC-BY-4.0**). From the raw pings we extract every consecutive same-taxi ping pair
spaced 14–16 s apart, convert longitude/latitude to metres in a local planar frame, and
keep the displacement vector `(dx, dy)` of every genuinely moving object (step > 5 m). A
deterministic subsample of **200,000** such real displacement vectors is bundled as
`raw/motion_pool.npz`, so scene generation is fully self-contained and needs no network.

The real displacement distribution (median 148 m, IQR ≈ 94–227 m, heavy-tailed) is what
makes the association problem realistic: objects move very different distances, so an
object's true destination is often *not* its nearest frame-B point.

`dataset/generator/fetch_porto.py` re-derives `motion_pool.npz` from the Hub if you want to
rebuild it from source; `dataset/generator/generate.py` builds the scenes from the bundled
pool and is deterministic (seed `20260718`).

## What is synthetic

Only the **placement** of objects within a scene. For each of `N = 34` objects we draw a
real displacement from the pool; soft objects get an independent start position in a
2000 m × 2000 m square (the density that sets confusability), and four objects are arranged
as two coincident hard pairs (identical start positions) to certify the ceiling. Frame-B
positions are `start + real_displacement`, emitted in a scene-shuffled order. The
correspondence is the identity permutation by construction and is stored as the ground
truth.

## Files (raw)

- `motion_pool.npz` — bundled real Porto displacement vectors (`xy`, `dxy`, metres).
- `points_train.csv`, `matches_train.csv` — 4000 supervised scenes and their true links.
- `points_test.csv`, `matches_test.csv` — 2500 evaluation scenes and their hidden links.
- `meta.json` — generation parameters and verified score anchors.

## Licensing

Source taxi data is CC-BY-4.0 (attribution: Moreira-Matias et al., ECML/PKDD 2015; HF
release by Kraina, Wrocław University of Science and Technology). The bundled pool and the
generated scenes are derivative works distributable under CC-BY-4.0. No personal data is
present: only anonymised vehicle displacement vectors, stripped of taxi id, time, and
absolute location.
