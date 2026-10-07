#!/usr/bin/env python3
"""Build the public/private split for "The Vanished Clause".

RAW INPUT (produced by dataset/generator/build_raw.py)
------------------------------------------------------
    raw/amendments.jsonl   one record per (section, amendment) pair:
        record_id, title, part, section, section_name, subpart,
        date_before, date_after, text_before, text_after

THE TASK
--------
The solver is shown a regulation section as it reads AFTER an amendment, and
must reconstruct how it read BEFORE. The amendment boundary is the archive's
own recorded `amendment_date`; `text_before` and `text_after` are the agency's
own published wording on those two dates. Nothing is annotated or generated.

    input   : text_after   (the current wording)
    output  : text_before  (the prior wording)

METRIC CHOICE, MEASURED BEFORE THE SPLIT WAS BUILT
---------------------------------------------------
Consecutive versions of a regulation share most of their text -- a typical
amendment touches one paragraph of a long section. Scored with plain edit
similarity, "submit the input unchanged" is therefore worth most of the
maximum, which leaves no usable range (this exact failure is why a prior
challenge in this line rescored on changed positions only).

So the metric scores ONLY the tokens that actually differ between after and
before. grade.py aligns the two token sequences, takes the positions where
the true `before` departs from the shown `after`, and asks how many of those
the prediction got right. Copying the input through therefore scores ~0 by
construction, and the full [0, 1] range is spent on the one thing being
measured: knowing what the amendment took away.

WHY THE AMENDMENT NOTE IS NOT PUBLISHED
----------------------------------------
The Federal Register publishes an instruction alongside each amendment ("in
paragraph (b)(2), remove the words ..."). Those instructions are deliberately
NOT shipped: with the instruction in hand the task degenerates into executing
an explicit edit script, which is a string-manipulation exercise rather than a
question about regulatory language. The solver gets the amended text only, and
must infer the change from how the prose reads.

WHY SOURCE NOTES AND BANNER REMNANTS ARE CUT
---------------------------------------------
The fetcher strips the forward-reference banner ("Link to an amendment
published at 89 FR 47080, Nov. 21, 2024.") and editorial notes with a
pattern that stops at the first period -- so it stops at the "." of a month
abbreviation and leaves "21, 2024." behind. Measured: that remnant sits in
677 of 1,404 raw text_before values, and a content-free dump of frequent
tokens (",", ".", "2024", ...) scored 0.148 by recovering it. The OFR source
note ("[88 FR 30042, May 10, 2023, as amended at 88 FR 69882, Oct. 10,
2023]") names the amending Federal Register document outright, and whatever
follows it is an editorial or effective-date note that can narrate the change
("§ 11.201(b) table was revised; ..."). None of this is regulatory wording.
So, identically on both sides, before anything else:
  - the text is cut at the first bracketed FR source note (the note and all
    notes after it go; measured, nothing but notes ever follows one), and
  - every free-standing "DD, YYYY." banner remnant is removed.

WHY THE SECTION'S IDENTITY IS NOT PUBLISHED
--------------------------------------------
The CFR title, part and section number together address the section in the
eCFR's own version history, which returns every prior wording directly. So
none of them is published: no title, part or section_name column, and the
section's own number is masked to `[SECTION]` wherever it occurs in the text
(its heading, and self-references such as "§ [SECTION](a)"). The mask is
applied identically to text_before, so it is never itself a changed token.
Cross-references to OTHER sections are left as published: they are part of
the regulation's wording and are often exactly what an amendment changes.

SPLIT
-----
By CFR PART, never by row. Every pair drawn from one part lands entirely in
train or entirely in test, so a solver cannot have seen a neighbouring section
of the same part -- parts share drafting conventions, defined terms and
cross-references heavily, and a part-mixed split would let those be memorised
rather than generalised. Which parts are held out is chosen by a deterministic
search for about TARGET_TEST_ROWS test rows.

ONE ROW PER SECTION IN TEST. A section amended several times yields a chain
of pairs (v1->v2), (v2->v3), ... and the text_after of one pair IS the
text_before of the next. With both in test, one row's shown text is another
row's exact answer: measured on the previous build, 10 of 161 test rows were
answered this way, worth 0.062 against a 0.104 reference. Even non-adjacent
pairs of one section share almost all their wording. So each held-out section
contributes exactly ONE pair (its latest), and the rest of its chain is
dropped rather than moved to train, where it would leak the same way.
Train keeps every pair: its answers are public anyway.

Additionally:
  - A section never appears in both splits (implied by part-disjointness, and
    asserted anyway).
  - Exact-duplicate `text_after` values are collapsed before splitting: two
    different sections can carry identical boilerplate, and a duplicate input
    spanning the split is a leak even when the sections differ.

  - No test text_before equals any train text (asserted).
  - Test rows are ordered by a hash of their record id, so row order carries
    no information about title, part or section.

OUTPUTS
-------
    public/train.csv               row_id, text_after, text_before
    public/test.csv                row_id, text_after
    public/sample_submission.csv   row_id, text_before  (input copied through)
    public/dataset_stats.json
    public/validate_submission.py
    private/answers.csv            row_id, shown_text, text_before

    `shown_text` is `test.csv`'s `text_after` under a DIFFERENT name. The
    grader needs the shown version to locate which tokens changed, and a
    column NAME shared between test.csv and answers.csv is read by the
    platform as a leaked target column. The two are asserted equal at build
    time.

No numeric feature column is published: a character count would be derivable
from the text itself and its long tail reads as an extreme-outlier column to
the platform's profiler.

Deterministic: seeded, sorted iteration throughout.

Usage:
    python prepare.py [--raw dataset/raw] [--public dataset/public]
                      [--private dataset/private]
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Retained for provenance only; the build uses NO randomness, so this seeds
# nothing and is deliberately NOT written to any public file.
_UNUSED_SEED = 20260925
TARGET_TEST_ROWS = 350     # one row per held-out section
MAX_HELD_OUT_SHARE = 0.50  # at most half of all pairs leave train
SECTION_MASK = "[SECTION]"
MIN_CHANGED_TOKENS = 3      # a pair whose change is trivially small is dropped
MAX_CHANGED_FRACTION = 0.60  # a near-total rewrite is not an "amendment"

csv.field_size_limit(10_000_000)

TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(t):
    return TOKEN.findall(t or "")


SOURCE_NOTE = re.compile(r"\[[^\[\]]*\b\d+ FR \d+[^\[\]]*\]")
# Banner variants that survive the fetcher's cut, e.g. "This amendment was
# delayed until Apr. 28, 2025, at 90 FR 10592, Feb.", "The effective date of
# this amendment was corrected to read Jan. 4, 2025, at 89 FR 106364, Dec." and
# the correction banner "Link to a correction published at 87 FR 73246, Nov."
# (the fetcher's pattern only knew the amendment banner).
BANNER_TAIL = re.compile(
    r"(?:Link to an? (?:amendment|correction)|This amendment|"
    r"The effective date of this amendment)[^()\[\]]*?"
    r"\b(?:at )?\d+ FR \d+(?:, [A-Z][a-z]{2,4}\.?)? ?"
    r"(?:\d{1,2}, (?:19|20)\d\d\.)?\s*")
# A source note printed without brackets, ending the section:
# "79 FR 21097, Apr. 14, 2014, as amended at 83 FR 49465, Oct. 2, 2018; ..."
BARE_SOURCE_NOTE = re.compile(
    r"\s\d+ FR \d+, [A-Z][a-z]{2,4}\.? \d{1,2}, (?:19|20)\d\d"
    r"(?:, (?:as amended|unless otherwise noted|redesignated)\b.*)?\.?$")
DATE_REMNANT = re.compile(r"(?:(?<=^)|(?<=[.:;)\]] ))(?:\d{1,2}, (?:19|20)\d\d\. ?)+")


def strip_notes(text):
    """Cut the OFR source note and everything after it; drop banner remnants."""
    t = text or ""
    m = SOURCE_NOTE.search(t)
    if m:
        t = t[:m.start()]
    t = BANNER_TAIL.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = BARE_SOURCE_NOTE.search(t)
    if m:
        t = t[:m.start()]
    t = DATE_REMNANT.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def mask_section(text, section):
    """Replace the section's own number, wherever it stands alone.

    Anchored so `1.2` does not touch `11.2`, `1.20`, `1.25` or `21.1.2`, but
    does match `§ 1.2`, `§ 1.2(a)` and a sentence-final `1.2.`.
    """
    pat = r"(?<![\w.])" + re.escape(str(section)) + r"(?![\w]|\.\d)"
    return re.sub(pat, SECTION_MASK, text or "")


def lcs_changed_positions(after_toks, before_toks):
    """Positions in `before` whose token is not matched through from `after`.

    Standard LCS backtrace. Returns the set of indices into before_toks that
    an aligner would call inserted/substituted relative to after_toks -- i.e.
    exactly the tokens the amendment removed when going after -> before.
    """
    n, m = len(after_toks), len(before_toks)
    if n == 0:
        return set(range(m))
    if m == 0:
        return set()
    # Row-wise DP; O(n*m) time, O(m) space for lengths, then backtrace with
    # a compact parent encoding.
    prev = [0] * (m + 1)
    choices = []
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        row = bytearray(m + 1)
        ai = after_toks[i - 1]
        for j in range(1, m + 1):
            if ai == before_toks[j - 1]:
                cur[j] = prev[j - 1] + 1
                row[j] = 1                      # diagonal match
            elif prev[j] >= cur[j - 1]:
                cur[j] = prev[j]
                row[j] = 2                      # up
            else:
                cur[j] = cur[j - 1]
                row[j] = 3                      # left
        choices.append(row)
        prev = cur
    matched = set()
    i, j = n, m
    while i > 0 and j > 0:
        c = choices[i - 1][j]
        if c == 1:
            matched.add(j - 1)
            i -= 1
            j -= 1
        elif c == 2:
            i -= 1
        else:
            j -= 1
    return set(range(m)) - matched


def load_raw(raw: Path):
    """Read every amendments.jsonl under `raw`, de-duplicating by record_id.

    The fetcher can be run as several workers over disjoint CFR titles, each
    writing its own shard (raw/, raw_shard/, ...). They are merged here rather
    than concatenated by hand so a partially-complete fetch is always usable
    and a re-run that overlaps an earlier one cannot double-count a pair.
    Different layouts are also tolerated because the platform may hand
    prepare() a root one level above or below the shipped one.
    """
    candidates = []
    direct = raw / "amendments.jsonl"
    if direct.exists():
        candidates.append(direct)
    for alt in sorted(raw.rglob("amendments.jsonl")):
        if alt not in candidates:
            candidates.append(alt)
    # Also look beside `raw` for sibling shards (raw_shard/, raw_2/, ...).
    if raw.parent.exists():
        for sib in sorted(raw.parent.glob("raw*/amendments.jsonl")):
            if sib not in candidates:
                candidates.append(sib)
    if not candidates:
        raise FileNotFoundError(
            f"amendments.jsonl not found under {raw}. Run "
            f"dataset/generator/build_raw.py --out {raw} first.")

    by_id, n_lines = {}, 0
    for p in candidates:
        with p.open(encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                rec = json.loads(line)
                by_id[rec["record_id"]] = rec
    if len(candidates) > 1 or n_lines != len(by_id):
        print(f"merged {len(candidates)} shard(s): {n_lines} lines -> "
              f"{len(by_id)} unique records")
    return [by_id[k] for k in sorted(by_id)]


def choose_test_parts(by_part):
    """Deterministically pick held-out parts giving ~TARGET_TEST_ROWS test rows.

    Test contributes one row per SECTION, train one row per PAIR, so a part
    costs train `rows` and gives test `sections`. Parts are taken in order of
    sections-per-row (then size, then id), which reaches the target while
    giving up the fewest training pairs. Sorted iteration, no randomness.
    """
    rows = {p: len(v) for p, v in by_part.items()}
    secs = {p: len({r["section"] for r in v}) for p, v in by_part.items()}
    order = sorted(by_part, key=lambda p: (-secs[p] / rows[p], -secs[p],
                                           str(p)))
    # Never give up more than MAX_HELD_OUT_SHARE of the pairs: on a smaller
    # raw archive the absolute target would otherwise empty train entirely.
    total = sum(rows.values())
    sel, n, lost = [], 0, 0
    for p in order:
        if n >= TARGET_TEST_ROWS:
            break
        if lost + rows[p] > MAX_HELD_OUT_SHARE * total:
            continue
        sel.append(p)
        n += secs[p]
        lost += rows[p]
    if n < TARGET_TEST_ROWS:
        print(f"WARNING: only {n} test rows reachable from {total} pairs "
              f"(target {TARGET_TEST_ROWS}); the raw archive is smaller than "
              f"the shipped one.")
    return set(sel)


def prepare(raw, public, private):
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    records = load_raw(raw)
    print(f"raw records: {len(records)}")

    # ---- filter to pairs with a real, bounded amendment --------------- #
    kept, dropped = [], defaultdict(int)
    seen_after = {}
    for r in sorted(records, key=lambda r: r["record_id"]):
        r["text_after"] = mask_section(strip_notes(r["text_after"]),
                                       r["section"])
        r["text_before"] = mask_section(strip_notes(r["text_before"]),
                                        r["section"])
        after, before = r["text_after"], r["text_before"]
        if after == before:
            dropped["identical"] += 1
            continue
        at, bt = tokenize(after), tokenize(before)
        if not at or not bt:
            dropped["empty"] += 1
            continue
        changed = lcs_changed_positions(at, bt)
        if len(changed) < MIN_CHANGED_TOKENS:
            dropped["change_too_small"] += 1
            continue
        if len(changed) / len(bt) > MAX_CHANGED_FRACTION:
            dropped["near_total_rewrite"] += 1
            continue
        # Collapse duplicate inputs: identical shown text across two records
        # would let one row's answer inform the other across the split.
        key = hashlib.sha256(after.encode("utf8")).hexdigest()
        if key in seen_after:
            dropped["duplicate_input"] += 1
            continue
        seen_after[key] = r["record_id"]
        r["_changed"] = len(changed)
        r["_before_tokens"] = len(bt)
        kept.append(r)

    print(f"kept {len(kept)} pairs; dropped {dict(dropped)}")
    if len(kept) < 100:
        raise SystemExit(
            f"only {len(kept)} usable pairs -- fetch more titles/dates before "
            f"building (build_raw.py --titles ... --since ...)")

    # ---- split by CFR part -------------------------------------------- #
    by_part = defaultdict(list)
    for r in kept:
        by_part[(r["title"], r["part"])].append(r)
    test_parts = choose_test_parts(by_part)
    train = [r for r in kept if (r["title"], r["part"]) not in test_parts]
    held = defaultdict(list)
    for r in kept:
        if (r["title"], r["part"]) in test_parts:
            held[(r["title"], r["section"])].append(r)
    # one pair per section: the latest amendment; the rest of the chain drops
    test = [max(v, key=lambda r: (r["date_after"], r["record_id"]))
            for v in held.values()]
    dropped["chain_sibling_in_test"] = sum(len(v) for v in held.values()) \
        - len(test)
    train.sort(key=lambda r: r["record_id"])
    test.sort(key=lambda r: hashlib.sha256(
        r["record_id"].encode("utf8")).hexdigest())
    if not train or not test:
        raise SystemExit(f"empty split: {len(train)} train / {len(test)} test")
    ratio = len(test) / len(train)
    print(f"parts: {len(by_part)} total, {len(test_parts)} held out "
          f"(test rows {len(test)}, train rows {len(train)}, "
          f"ratio {ratio:.1%})")

    # ---- assert disjointness ------------------------------------------ #
    tr_sec = {(r["title"], r["section"]) for r in train}
    te_sec = [(r["title"], r["section"]) for r in test]
    assert not (tr_sec & set(te_sec)), "section overlap across splits"
    assert len(te_sec) == len(set(te_sec)), "section repeated in test"
    tr_in = {r["text_after"] for r in train}
    te_in = {r["text_after"] for r in test}
    assert not (tr_in & te_in), "identical shown text across splits"
    tr_all = tr_in | {r["text_before"] for r in train}
    assert not any(r["text_before"] in tr_all for r in test), \
        "a test answer appears verbatim in train"
    assert not any(r["text_before"] in te_in for r in test), \
        "a test answer is another test row's shown text"
    own = [r for r in train + test for t in (r["text_after"], r["text_before"])
           if mask_section(t, r["section"]) != t]
    assert not own, "own section number survived masking"
    for r in train + test:
        for t in (r["text_after"], r["text_before"]):
            assert not SOURCE_NOTE.search(t), "FR source note survived"
            assert not DATE_REMNANT.search(t), "banner remnant survived"
            assert not BANNER_TAIL.search(t), "banner variant survived"
            assert not BARE_SOURCE_NOTE.search(t), "bare source note survived"

    def rid(i, split):
        return f"{split}_{i:05d}"

    for i, r in enumerate(train):
        r["row_id"] = rid(i, "tr")
    for i, r in enumerate(test):
        r["row_id"] = rid(i, "te")

    cols_common = ["row_id", "text_after"]

    with (public / "train.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols_common + ["text_before"])
        w.writeheader()
        for r in train:
            w.writerow({"row_id": r["row_id"],
                        "text_after": r["text_after"],
                        "text_before": r["text_before"]})

    with (public / "test.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols_common)
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["row_id"],
                        "text_after": r["text_after"]})

    # answer-blind baseline: the shown text, unchanged
    with (public / "sample_submission.csv").open(
            "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "text_before"])
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["row_id"], "text_before": r["text_after"]})

    # private answers; shown_text under a DIFFERENT name than in test.csv
    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "shown_text",
                                          "text_before"])
        w.writeheader()
        for r in test:
            w.writerow({"row_id": r["row_id"], "shown_text": r["text_after"],
                        "text_before": r["text_before"]})

    # programmatic assert: shown_text == test.csv's text_after, row for row
    with (public / "test.csv").open(encoding="utf8") as f:
        t_rows = list(csv.DictReader(f))
    with (private / "answers.csv").open(encoding="utf8") as f:
        a_rows = list(csv.DictReader(f))
    assert len(t_rows) == len(a_rows)
    for t, a in zip(t_rows, a_rows):
        assert t["row_id"] == a["row_id"], "row_id misalignment"
        assert t["text_after"] == a["shown_text"], "shown_text mismatch"
    assert len({r["row_id"] for r in t_rows}) == len(t_rows), "ids not unique"

    shared = (set(t_rows[0].keys()) & set(a_rows[0].keys())) - {"row_id"}
    assert not shared, f"columns shared with answers.csv: {shared}"

    ch = [r["_changed"] for r in test]
    bt = [r["_before_tokens"] for r in test]
    stats = {
        "n_train": len(train),
        "n_test": len(test),
        "test_train_ratio": round(ratio, 4),
        "n_parts_train": len(by_part) - len(test_parts),
        "n_parts_test": len(test_parts),
        "split_unit": "cfr_part",
        "test_rows_per_section": 1,
        "section_number_masked_as": SECTION_MASK,
        "median_changed_tokens_test": sorted(ch)[len(ch) // 2],
        "median_before_tokens_test": sorted(bt)[len(bt) // 2],
        "median_changed_fraction_test": round(
            sorted(c / b for c, b in zip(ch, bt))[len(ch) // 2], 4),
        "dropped_at_build": dict(dropped),
        "metric": ("gap-anchored changed-token F1: tokens where the true "
                   "prior wording departs from the shown wording, each keyed "
                   "by the gap it sits in, matched against the tokens the "
                   "prediction adds; F1 per row, mean over rows"),
        # NOTE: no generator seed is published. The split uses no randomness at
        # all -- it is a deterministic, sorted, bounded search over CFR parts --
        # so a seed would be both meaningless and, per Target Recoverability,
        # a public generator input a solver could try to replay.
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    here = Path(__file__).parent
    for cand in (here / "dataset" / "validate_submission.py",
                 here / "validate_submission.py"):
        if cand.exists():
            (public / "validate_submission.py").write_text(
                cand.read_text(encoding="utf8"), encoding="utf8")
            break

    print(f"public/train.csv              {len(train):,} rows")
    print(f"public/test.csv               {len(test):,} rows")
    print(f"public/sample_submission.csv  {len(test):,} rows")
    print(f"private/answers.csv           {len(test):,} rows")
    print(f"median changed tokens (test): {stats['median_changed_tokens_test']}"
          f"  of {stats['median_before_tokens_test']} "
          f"({stats['median_changed_fraction_test']:.1%})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    p.add_argument("--public", default=os.path.join(here, "dataset", "public"))
    p.add_argument("--private",
                   default=os.path.join(here, "dataset", "private"))
    a = p.parse_args()
    prepare(Path(a.raw), Path(a.public), Path(a.private))


if __name__ == "__main__":
    main()
