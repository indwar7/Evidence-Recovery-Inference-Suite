"""grade.py - scorer for the Vietnamese place-name orthographic-transfer task.

    grade(submission, answers) -> float

Both arguments are pandas DataFrames (no filesystem access is performed).

Grading contract
----------------
* ``submission`` has columns ``id`` and ``name``: for each test id, the
  predicted native Vietnamese spelling of the place name.
* ``answers`` has columns ``id``, ``name``, ``folded_src``, ``province_src``,
  ``district_src`` and ``neighbors_src``:
  - ``name``         = the true native spelling (the target).
  - ``folded_src``   = the ASCII-folded input that was shown to the solver.
  - ``province_src``, ``district_src``, ``neighbors_src`` = the same context
    the solver saw in test.csv's ``province``/``district``/``neighbors``
    columns, carried here under ``_src`` names so that ``id`` is the only
    column name shared with the public test file.
  The submitted ``name`` and the answer ``name`` are the same kind of object
  (the native string) compared position-by-position against ``folded_src``.

Score
-----
This task is framed as orthographic transfer, not generic diacritic
restoration: a syllable's marks are "context-attributable" when its exact
native spelling appears verbatim in the row's province, district or neighbour
names, and "context-independent" otherwise (31.2% of changed positions are
context-attributable on the shipped split). Per row we compute changed-
position F1 (precision/recall over positions that should receive a diacritic)
separately on each subset, then combine them as the harmonic mean of the two
subset scores -- so a submission that only gets the easier, context-
independent positions right cannot earn a high score by volume alone; it must
also demonstrate the transfer capability specifically. A row with changed
positions of only one kind is scored on that subset alone.
  precision = correct diacritic edits / all edits the prediction made
  recall    = correct diacritic edits / all edits the true name requires.
A row whose prediction equals the folded input (no edits) scores 0; a perfect
prediction scores 1. The task is a 1:1, length-preserving substitution, so a
prediction whose length differs from the input cannot align and scores 0.
The returned score is the mean over all graded rows, in [0, 1].

Degradation (never raises on bad data; only a missing required column raises):
* unknown / missing id, blank prediction, length mismatch -> 0 for that row
* a duplicated id          -> that id scores 0 (not treated as correct)
* extra submission rows / ids absent from ``answers`` -> ignored
* ``answers`` may be any row-wise subset of the full test set (public/private
  leaderboard halves are graded this way).
"""
from __future__ import annotations


class InvalidSubmission(ValueError):
    """Raised only when the submission lacks a required column."""


def _context_syllables(province: str, district: str, neighbors: str) -> set[str]:
    pool = set()
    if province:
        pool.update(province.split())
    if district:
        pool.update(district.split())
    if neighbors:
        for nm in neighbors.split(" | "):
            pool.update(nm.split())
    return pool


def _f1(correct: int, n_pred: int, n_true: int) -> float:
    if n_true == 0:
        return None  # no positions of this kind in the row
    if n_pred == 0:
        return 0.0
    precision = correct / n_pred
    recall = correct / n_true
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _row_score(folded: str, target: str, pred: str,
                province: str, district: str, neighbors: str) -> float:
    if pred is None:
        return 0.0
    pred = str(pred)
    if len(pred) != len(folded):
        return 0.0
    changed = [i for i in range(len(folded)) if target[i] != folded[i]]
    if not changed:                       # nothing to restore -> exact match only
        return 1.0 if pred == target else 0.0

    # classify each changed position by whether its syllable's exact native
    # form is recoverable from the row's own context columns
    ctx_pool = _context_syllables(province, district, neighbors)
    f_syl, t_syl = folded.split(), target.split()
    ctx_pos: set[int] = set()
    if len(f_syl) == len(t_syl):
        pos = 0
        for f, t in zip(f_syl, t_syl):
            if t in ctx_pool:
                ctx_pos.update(range(pos, pos + len(f)))
            pos += len(f) + 1   # +1 for the joining space
    changed_ctx = [i for i in changed if i in ctx_pos]
    changed_indep = [i for i in changed if i not in ctx_pos]

    pred_changes = [i for i in range(len(folded)) if pred[i] != folded[i]]
    pred_ctx = [i for i in pred_changes if i in ctx_pos]
    pred_indep = [i for i in pred_changes if i not in ctx_pos]

    correct_ctx = sum(1 for i in changed_ctx if pred[i] == target[i])
    correct_indep = sum(1 for i in changed_indep if pred[i] == target[i])

    f1_ctx = _f1(correct_ctx, len(pred_ctx), len(changed_ctx))
    f1_indep = _f1(correct_indep, len(pred_indep), len(changed_indep))

    if f1_ctx is None:
        return f1_indep
    if f1_indep is None:
        return f1_ctx
    if f1_ctx == 0 or f1_indep == 0:
        return 0.0
    return 2 * f1_ctx * f1_indep / (f1_ctx + f1_indep)


def _as_records(df, cols):
    """Return list of dict rows for the given columns; tolerant of DataFrame
    or list-of-dict. Raises InvalidSubmission if a required column is absent."""
    if hasattr(df, "columns"):            # pandas DataFrame
        have = list(df.columns)
        for c in cols:
            if c not in have:
                raise InvalidSubmission(f"missing required column: {c!r}")
        return df.to_dict("records")
    # assume iterable of mappings
    recs = list(df)
    if recs:
        for c in cols:
            if c not in recs[0]:
                raise InvalidSubmission(f"missing required column: {c!r}")
    return recs


def grade(submission, answers) -> float:
    ans = _as_records(answers, ["id", "name", "folded_src"])
    sub = _as_records(submission, ["id", "name"])

    # map submission id -> prediction; a duplicated id is poisoned (score 0)
    pred = {}
    dup = set()
    for r in sub:
        rid = r.get("id")
        if rid is None:
            continue
        rid = str(rid)
        if rid in pred or rid in dup:
            dup.add(rid)
            pred.pop(rid, None)
            continue
        pred[rid] = r.get("name")

    total = 0.0
    n = 0
    for r in ans:
        rid = r.get("id")
        if rid is None:
            continue
        rid = str(rid)
        folded = str(r.get("folded_src", ""))
        target = str(r.get("name", ""))
        province = str(r.get("province_src") or "")
        district = str(r.get("district_src") or "")
        neighbors = str(r.get("neighbors_src") or "")
        n += 1
        if rid in dup or rid not in pred:
            continue                      # missing / duplicated -> 0
        p = pred[rid]
        if p is None or (isinstance(p, float) and p != p):   # NaN
            continue
        total += _row_score(folded, target, str(p), province, district, neighbors)
    return total / n if n else 0.0


if __name__ == "__main__":
    # tiny smoke test
    answers = [
        {"id": "a", "name": "A Nội", "folded_src": "A Noi",
         "province_src": "", "district_src": "", "neighbors_src": "Nội Bài"},
        {"id": "b", "name": "Bảo", "folded_src": "Bao",
         "province_src": "", "district_src": "", "neighbors_src": ""},
    ]
    perfect = [{"id": "a", "name": "A Nội"}, {"id": "b", "name": "Bảo"}]
    copy = [{"id": "a", "name": "A Noi"}, {"id": "b", "name": "Bao"}]
    print("oracle:", grade(perfect, answers))   # 1.0
    print("copy:  ", grade(copy, answers))       # 0.0
