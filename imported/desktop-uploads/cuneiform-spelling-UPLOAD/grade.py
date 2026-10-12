"""Grader for "Cuneiform Spelling of Unseen Akkadian Words".

Contract
--------
grade(submission, answers) -> float in [0, 1]

submission : pandas.DataFrame with columns ``row_id`` and ``spelling``
answers    : pandas.DataFrame with columns ``row_id`` and ``spelling_true``
             (plus any extra columns, which are ignored)

``spelling`` and ``spelling_true`` are the same object in the same form: one
transliterated Akkadian word in ORACC convention, signs joined by hyphens,
determinatives in curly braces, e.g. ``{LU₂}GAL-u₂-ra-sa-ni`` or ``iš-pur-u-ni``.

Score per row = normalized sign-level edit similarity between the predicted
and true sign sequences:

    sim = 1 - levenshtein(pred_signs, true_signs) / max(len(pred_signs), len(true_signs))

where a word is split into sign tokens by taking each ``{...}`` determinative
as one token and splitting the rest on ``-``, ``.``, ``:`` and whitespace.
Comparison is exact on the sign token string (so ``u`` and ``u₂`` are different
signs, as they are on the tablet). The final score is the mean over all rows in
``answers``.

Degrade, never reject: a missing row, an unknown or malformed id, a blank
prediction, or a duplicated id scores 0 for that row. Only a missing required
column raises ``InvalidSubmission``.
"""
import re
import unicodedata

import pandas as pd

_TOKEN_RE = re.compile(r"\{[^}]*\}|[^{}\-.:\s]+")


class InvalidSubmission(ValueError):
    pass


def sign_tokens(form):
    """Split an ORACC transliterated word into sign tokens."""
    if form is None:
        return []
    s = unicodedata.normalize("NFC", str(form)).strip()
    if not s or s.lower() == "nan":
        return []
    return _TOKEN_RE.findall(s)


def _levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sign_similarity(pred, true):
    """Normalized sign-level edit similarity in [0, 1]; 1.0 only for an exact sign match."""
    p, t = sign_tokens(pred), sign_tokens(true)
    if not t and not p:
        return 1.0
    if not p or not t:
        return 0.0
    return 1.0 - _levenshtein(p, t) / max(len(p), len(t))


def grade(submission, answers):
    for col in ("row_id", "spelling"):
        if col not in submission.columns:
            raise InvalidSubmission(f"submission is missing required column '{col}'")
    for col in ("row_id", "spelling_true"):
        if col not in answers.columns:
            raise InvalidSubmission(f"answers is missing required column '{col}'")
    if len(answers) == 0:
        return 0.0

    sub = submission[["row_id", "spelling"]].copy()
    sub["row_id"] = sub["row_id"].astype(str).str.strip()
    dup = sub["row_id"].duplicated(keep=False)
    sub = sub[~dup]  # every row of a duplicated id is discarded -> scores 0
    pred = dict(zip(sub["row_id"], sub["spelling"]))

    ans_ids = answers["row_id"].astype(str).str.strip().tolist()
    total = 0.0
    for rid, true in zip(ans_ids, answers["spelling_true"].tolist()):
        p = pred.get(rid)
        if p is None or (isinstance(p, float) and pd.isna(p)):
            continue
        total += sign_similarity(p, true)
    return float(total / len(answers))
