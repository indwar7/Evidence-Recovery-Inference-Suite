"""Grader: edit-operation F1 between the amendments you made and the ones the
Office of the Federal Register actually made.

Usage:  python grade.py submission.csv answers.csv

`answers.csv` carries item_id, before_text, after_text.  `submission.csv` carries
item_id, after_text.

Why not just compare the two texts?  Because an amendment usually touches a few
dozen words in a section of several hundred.  A solver that changes nothing at all
would score around 0.97 on any character- or token-similarity measure, which makes
the measure useless.  Scoring the *edits* instead puts "changed nothing" at 0.0,
gives partial credit for the amendments you did apply, and charges you for the
ones you invented.

Operations are matched *softly*.  An earlier version of this grader intersected
edit-operation tuples exactly, so an amendment applied to the right paragraph but
ending one token early scored the same as one not attempted at all.  That made the
per-item score effectively binary -- on the shipped split the reference solution
scored exactly 1.000 on thirteen items and exactly 0.000 on eight, with almost
nothing in between -- which is a property of the scoring, not of the task.  Gold
and predicted operations are now aligned by position and compared by content, so
landing the right edit in the right place with the wrong extent earns most of the
credit rather than none of it.

Structural problems degrade; they do not abort.  A submission that omits an item,
leaves one blank or repeats a row is *scored*, with the affected items counted as
"no amendment applied", i.e. 0.  Only a submission that cannot be read at all, or
that carries none of the required columns, is rejected outright.  A grader that
raises on any imperfection turns a mediocre solver into a failed run with no score,
which tells nobody anything.
"""
import io
import sys
import difflib
import pandas as pd


class InvalidSubmission(ValueError):
    """Signals a submission that cannot be scored at all.

    Deliberately a `ValueError`.  Harnesses commonly catch `ValueError` to tell
    "the entry was bad" apart from "the grader itself broke"; a bare `Exception`
    subclass escapes that catch and gets recorded as a crashed run with no score.
    """


MAX_CHARS = 200_000

# How far apart two operations may start and still be considered attempts at the
# same amendment.  Three tokens covers the usual near-miss -- a paragraph
# designator included or excluded from the replaced span.  The tolerance grows
# with the size of the operation, because starting eight tokens early on a
# sixty-token replacement is plainly the same amendment, while eight tokens is a
# different sentence when the amendment moves three words.
PROXIMITY = 3
PROXIMITY_FRACTION = 0.25

# Split of per-operation credit between getting the *extent* right (which tokens
# of the original were displaced) and getting the *content* right (what replaced
# them).  Both halves matter and neither alone is the amendment.
W_SPAN = 0.5
W_TEXT = 0.5

UNI = {"‘": "'", "’": "'", "“": '"', "”": '"',
       "–": "-", "—": "-", "−": "-", " ": " ",
       "…": "...", "¶": " "}


def normalise(text):
    """The canonical form both sides are compared in.

    Whitespace and the handful of typographic characters below carry no legal
    meaning, so nobody should win or lose points on them.
    """
    text = "" if text is None else str(text)
    for k, v in UNI.items():
        text = text.replace(k, v)
    return " ".join(text.split())


def edit_ops(before, after):
    """The token-level edit script turning `before` into `after`.

    Positions are indices into `before`, which both the gold and the predicted
    text share, so the two scripts are directly comparable.
    """
    a, b = normalise(before).split(), normalise(after).split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(i1, i2, tuple(b[j1:j2]))
            for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def weight(op):
    """How much text an operation moves: tokens removed plus tokens added.

    Operations are weighted rather than counted because otherwise deleting the
    entire section would cost exactly as much as deleting one stray word -- a
    single operation either way.  Weighting makes the score read as "how much of
    the agency's text movement did you reproduce, and how much did you invent".
    """
    i1, i2, inserted = op
    return (i2 - i1) + len(inserted)


def _bag_f1(x, y):
    """Token-level F1 between two sequences, compared as multisets."""
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    shared = 0
    pool = list(y)
    for tok in x:
        if tok in pool:
            pool.remove(tok)
            shared += 1
    if not shared:
        return 0.0
    p, r = shared / len(y), shared / len(x)
    return 2 * p * r / (p + r)


def similarity(g, p):
    """How much of gold operation `g` predicted operation `p` accounts for.

    Zero unless the two operations are aimed at the same place: their replaced
    spans overlap, or they start within PROXIMITY tokens of each other.  Pure
    insertions have an empty span, which is why proximity is needed at all.
    """
    gi1, gi2, gins = g
    pi1, pi2, pins = p

    overlap = min(gi2, pi2) - max(gi1, pi1)
    tolerance = max(PROXIMITY, PROXIMITY_FRACTION * max(
        gi2 - gi1, pi2 - pi1, len(gins), len(pins)))
    if overlap <= 0 and abs(gi1 - pi1) > tolerance:
        return 0.0

    union = max(gi2, pi2) - min(gi1, pi1)
    if union <= 0:                       # two pure insertions at the same point
        span_sim = 1.0
    else:
        span_sim = max(overlap, 0) / union

    return W_SPAN * span_sim + W_TEXT * _bag_f1(gins, pins)


def _align(gold, pred):
    """Greedy one-to-one alignment of predicted operations onto gold ones.

    Best-scoring pairs are committed first, so a predicted operation is spent on
    the amendment it explains best rather than the first one it happens to touch.
    """
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            s = similarity(g, p)
            if s > 0:
                pairs.append((s, gi, pi))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_g, used_p, out = set(), set(), []
    for s, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((s, gi, pi))
    return out


def score_item(before, gold_after, pred_after):
    """Size-weighted F1 between the predicted edit script and the official one."""
    gold = edit_ops(before, gold_after)
    pred = edit_ops(before, pred_after)

    if not gold and not pred:
        return 1.0                        # nothing to do, nothing done
    if not gold or not pred:
        return 0.0

    matched = _align(gold, pred)
    got_gold = sum(s * weight(gold[gi]) for s, gi, _ in matched)
    got_pred = sum(s * weight(pred[pi]) for s, _, pi in matched)
    if not got_gold or not got_pred:
        return 0.0

    recall = got_gold / sum(weight(o) for o in gold)
    precision = got_pred / sum(weight(o) for o in pred)
    return 2 * precision * recall / (precision + recall)


def _frame(obj, required, what, strict):
    """Accept an already-loaded DataFrame or a path to a CSV.

    The platform hands the grader DataFrames; the command line hands it paths.
    Supporting both is what keeps `python grade.py sub.csv answers.csv` working
    without the harness having to write temporary files.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        df = None
        # `after_text` is several thousand characters of regulatory prose, full of
        # quoted phrases and commas, so a submission written by hand rather than by
        # a CSV writer is a live possibility.  Try progressively more forgiving
        # readers before giving up on it.
        #
        # The text is pulled into memory first because each attempt has to start
        # from the beginning.  Handing the same file object to pandas twice reads
        # an exhausted stream the second time, which turns every fallback into the
        # same "No columns to parse" and silently disables the ladder.
        try:
            if hasattr(obj, "read"):
                if hasattr(obj, "seek"):
                    obj.seek(0)
                raw = obj.read()
            else:
                with open(obj, encoding="utf8", errors="replace") as fh:
                    raw = fh.read()
        except Exception as e:                                 # noqa: BLE001
            raise InvalidSubmission(f"could not open {what}: {e}") from e

        if isinstance(raw, bytes):
            raw = raw.decode("utf8", errors="replace")

        attempts = (dict(dtype=str, keep_default_na=False),
                    dict(dtype=str, keep_default_na=False, engine="python"),
                    dict(dtype=str, keep_default_na=False, engine="python",
                         on_bad_lines="skip"),
                    # Last resort: a hand-written file whose text was never
                    # quoted, so every comma inside the regulation looks like a
                    # column break.  Split on the first comma only.
                    dict(dtype=str, keep_default_na=False, engine="python",
                         sep=",", quoting=3, names=list(required), header=0,
                         on_bad_lines="skip"))
        errors = []
        for kwargs in attempts:
            try:
                cand = pd.read_csv(io.StringIO(raw), **kwargs)
            except Exception as e:                             # noqa: BLE001
                errors.append(str(e))
                continue
            if all(c in cand.columns for c in required):
                df = cand
                break
            errors.append(f"columns {list(cand.columns)}")
        if df is None:
            raise InvalidSubmission(
                f"could not read {what} as CSV with columns {list(required)}: "
                f"{errors[-1]}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InvalidSubmission(
            f"{what} is missing column(s) {missing}; found {list(df.columns)}")

    out = df[list(required)].copy()
    for c in required:
        out[c] = out[c].fillna("").astype(str)

    if strict:
        # Duplicates are a malformed entry, but a recoverable one: the first row
        # for an item is the answer and the rest are ignored.
        out = out.drop_duplicates(subset=["item_id"], keep="first")
        # An over-long field is truncated rather than fatal.  Nothing legitimate
        # comes near the cap; the cap exists to stop a runaway generation from
        # making the grader chew through megabytes.
        out["after_text"] = out["after_text"].str.slice(0, MAX_CHARS)
    return out


def evaluate(submission, answers):
    """Mean over answer items of the size-weighted edit-script F1.

    Items the submission does not cover, or covers with an empty field, score 0 --
    they are amendments that were not applied.  They are not an error.
    """
    pred = dict(zip(submission["item_id"], submission["after_text"]))
    total = 0.0
    for item_id, before, gold_after in zip(
            answers["item_id"], answers["before_text"], answers["after_text"]):
        got = pred.get(item_id, "")
        if not got.strip():
            got = before                  # no amendment applied
        total += score_item(before, gold_after, got)
    return float(total / len(answers)) if len(answers) else 0.0


def grade(submission, answers) -> float:
    """Score a submission against ground truth. Returns weighted edit-op F1 in [0,1].

    Both arguments may be DataFrames (as the platform passes them) or paths to
    CSV files (as the command line passes them).
    """
    sub = _frame(submission, ["item_id", "after_text"], "submission", strict=True)
    ans = _frame(answers, ["item_id", "before_text", "after_text"], "answers",
                 strict=False)

    # The one thing that really is unscorable: a submission aimed at some other
    # challenge, sharing not a single identifier with the answer key.
    if not set(sub["item_id"]) & set(ans["item_id"]):
        raise InvalidSubmission(
            "no submitted item_id appears in the answer key; expected ids like "
            f"{sorted(ans['item_id'])[:3]}, got {sorted(sub['item_id'])[:3]}")

    return evaluate(sub, ans)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    try:
        print(f"{grade(sys.argv[1], sys.argv[2]):.6f}")
    except InvalidSubmission as e:
        print(f"InvalidSubmission: {e}", file=sys.stderr)
        raise SystemExit(1)
