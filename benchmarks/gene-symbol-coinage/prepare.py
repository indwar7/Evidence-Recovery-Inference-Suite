#!/usr/bin/env python3
"""Build the public/private split for "Coining Gene Symbols from Gene Names".

RAW INPUT (fetched by dataset/generator/build_raw.py)
-----------------------------------------------------
    raw/hgnc_complete_set_2026-07-07.txt   the HGNC complete gene set, a dated
                                           archive snapshot, stored byte for
                                           byte. Tab-separated, 54 columns.

Six of its columns are read: `symbol` and `name` (the pair), `status` and
`locus_type` (the filter), and `date_approved_reserved` and
`date_symbol_changed` (the split). `hgnc_id` is used as a stable sort key.
Only `name` and `symbol` are ever published.

THE TASK
--------
    input   : name     the HGNC approved gene NAME, e.g.
                       "solute carrier family 2 member 1"
    output  : symbol   the HGNC approved gene SYMBOL, e.g. "SLC2A1"

Both are the archive's own recorded values. Nothing is annotated or generated.

THE SPLIT IS BY DATE, AND THE DATE IS THE ARCHIVE'S OWN
-------------------------------------------------------
Every HGNC record carries the date its symbol was approved and, if the symbol
was later replaced, the date of that change. The later of the two is the day
the CURRENT symbol came into force -- its effective date.

    train : symbols in force before  CUTOFF
    test  : symbols that came into force on or after CUTOFF

So the training set is what the committee had already decided, and the test
set is what it decided next. That is the situation the committee itself is in
every time a gene needs a symbol, and it fixes the mixture of easy and hard
rows without this build choosing it: some new symbols extend a family that
already existed (a stem seen in train), the rest found a new one.

A FAMILY-DISJOINT SPLIT WAS BUILT FIRST AND ABANDONED ON MEASUREMENT
-------------------------------------------------------------------
Holding out whole gene families (connected components over symbol stem and
name skeleton) looked like the cleaner test of coinage. Measured on that
split, a character sequence-to-sequence model trained for 15 epochs scored
0.258 prefix agreement against 0.337 for "take the first letter of each
word". A trained reference below a one-line heuristic means the split had
removed the signal along with the shortcut, so it was dropped.

ROWS REMOVED, AND WHY
---------------------
  printed_answer   the name already contains the symbol, or its stem, as a
                   token ("BRCA1 DNA repair associated" -> BRCA1). That is
                   copying, not naming, so these rows are removed from both
                   splits.
  mentioned        a TEST symbol or stem that appears as a token inside any
                   other shipped name ("TP53 target 1" names TP53). Left in, a
                   solver could harvest capitalised tokens across the whole
                   public directory and match them back. Removed from test.
  bad_symbol       symbols with characters outside [A-Za-z0-9-].
  duplicate_name   a name that occurs twice after case-folding.
  no_date          no recorded approval date, so the row cannot be placed.

OUTPUTS
-------
    public/train.csv               row_id, name, symbol
    public/test.csv                row_id, name
    public/sample_submission.csv   row_id, symbol   (initial letters of the name)
    public/dataset_stats.json
    public/validate_submission.py
    private/answers.csv            row_id, symbol

No date and no numeric column is published. Row order and row_id come from a
salted hash, so neither carries the approval date.

Deterministic: no randomness anywhere, so no seed exists and none is
published.

Usage:
    python prepare.py [--raw dataset/raw] [--public dataset/public]
                      [--private dataset/private]
"""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

SNAPSHOT_GLOB = "hgnc_complete_set_*.txt"
SALT = "gene-symbol-coinage|v2"
CUTOFF = "2010-01-01"
RATIO_BAND = (0.15, 0.25)

SYMBOL_OK = re.compile(r"^[A-Za-z0-9-]+$")
DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WORD = re.compile(r"[A-Za-z0-9]+")
SMALL_WORDS = frozenset("of and the to in for with a an or".split())


def stem_of(symbol):
    """Leading letters of a symbol, upper-cased: SLC2A1 -> SLC."""
    m = re.match(r"^([A-Za-z]+)", symbol)
    return m.group(1).upper() if m else symbol.upper()


def name_tokens(name):
    return [t.upper() for t in WORD.findall(name)]


def printed_answer(name, symbol):
    """True if the name already spells the symbol or its stem."""
    raw = WORD.findall(name)
    toks = {t.upper() for t in raw}
    sym = symbol.upper().replace("-", "")
    st = stem_of(symbol)
    if sym in toks:
        return True
    if len(st) >= 3 and st in toks:
        return True
    for t in raw:
        if len(t) >= 3 and re.search(r"[A-Z]", t) and sym.startswith(t.upper()):
            return True
    return False


def initials(name):
    """The sample submission: first letter of each word, digits kept."""
    out = []
    for t in re.findall(r"[A-Za-z]+|\d+", name):
        if t.isdigit():
            out.append(t)
        elif t.lower() not in SMALL_WORDS:
            out.append(t[0].upper())
    return "".join(out)


def salted(key):
    return hashlib.sha256(f"{SALT}|{key}".encode("utf8")).hexdigest()


def find_snapshot(raw):
    """Locate the snapshot under `raw`, tolerating a shifted root."""
    raw = Path(raw)
    hits = sorted(raw.glob(SNAPSHOT_GLOB))
    if not hits:
        hits = sorted(raw.rglob(SNAPSHOT_GLOB))
    if not hits and raw.parent.exists():
        hits = sorted(raw.parent.rglob(SNAPSHOT_GLOB))
    if not hits:
        raise FileNotFoundError(
            f"no {SNAPSHOT_GLOB} under {raw}. Run "
            f"dataset/generator/build_raw.py --out {raw} first.")
    return hits[0]


def load_raw(raw):
    path = find_snapshot(raw)
    rows = []
    with path.open(encoding="utf8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t", quotechar='"')
        for r in rd:
            rows.append({
                "hgnc_id": r["hgnc_id"].strip(),
                "symbol": r["symbol"].strip(),
                "name": " ".join(r["name"].split()),
                "status": r["status"].strip(),
                "locus_type": r["locus_type"].strip(),
                "approved": r["date_approved_reserved"].strip(),
                "changed": r["date_symbol_changed"].strip(),
            })
    return path, rows


def effective_date(r):
    """The day the CURRENT symbol came into force, or '' if unrecorded."""
    dates = [d for d in (r["approved"], r["changed"]) if DATE_OK.match(d)]
    return max(dates) if dates else ""


def prepare(raw, public, private):
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    path, records = load_raw(raw)
    print(f"raw records: {len(records):,} ({path.name})")

    dropped = Counter()
    kept = []
    for r in sorted(records, key=lambda r: r["hgnc_id"]):
        if r["status"] != "Approved":
            dropped["not_approved"] += 1
            continue
        if r["locus_type"] != "gene with protein product":
            dropped["not_protein_coding"] += 1
            continue
        if not r["name"] or not r["symbol"]:
            dropped["empty"] += 1
            continue
        if not SYMBOL_OK.match(r["symbol"]):
            dropped["bad_symbol"] += 1
            continue
        r["effective"] = effective_date(r)
        if not r["effective"]:
            dropped["no_date"] += 1
            continue
        kept.append(r)

    # the same input carrying two different answers cannot be resolved from
    # the input, so colliding names are dropped rather than split
    by_name = defaultdict(list)
    for r in kept:
        by_name[r["name"].lower()].append(r)
    kept = []
    for name in sorted(by_name):
        grp = by_name[name]
        if len(grp) > 1:
            dropped["duplicate_name"] += len(grp)
            continue
        kept.append(grp[0])

    named = []
    for r in sorted(kept, key=lambda r: r["hgnc_id"]):
        if printed_answer(r["name"], r["symbol"]):
            dropped["printed_answer"] += 1
            continue
        r["stem"] = stem_of(r["symbol"])
        named.append(r)

    # every token of every shipped name, to find test symbols named elsewhere
    token_owner = defaultdict(set)
    for r in named:
        for t in set(name_tokens(r["name"])):
            token_owner[t].add(r["hgnc_id"])

    def is_mentioned(r):
        sym = r["symbol"].upper().replace("-", "")
        if token_owner.get(sym, set()) - {r["hgnc_id"]}:
            return True
        st = r["stem"]
        return len(st) >= 3 and bool(token_owner.get(st, set()) - {r["hgnc_id"]})

    train_rows = [r for r in named if r["effective"] < CUTOFF]
    test_rows = []
    for r in named:
        if r["effective"] < CUTOFF:
            continue
        if is_mentioned(r):
            dropped["mentioned"] += 1
            continue
        test_rows.append(r)

    ratio = len(test_rows) / len(train_rows)
    assert RATIO_BAND[0] <= ratio <= RATIO_BAND[1], f"ratio {ratio:.3f}"
    assert len(train_rows) > len(test_rows)
    assert len(train_rows) >= 1000

    train_rows.sort(key=lambda r: salted("row|" + r["hgnc_id"]))
    test_rows.sort(key=lambda r: salted("row|" + r["hgnc_id"]))
    for i, r in enumerate(train_rows):
        r["row_id"] = f"tr_{i:05d}"
    for i, r in enumerate(test_rows):
        r["row_id"] = f"te_{i:05d}"

    # ---- disjointness, asserted on the shipped values ------------------ #
    assert max(r["effective"] for r in train_rows) < CUTOFF
    assert min(r["effective"] for r in test_rows) >= CUTOFF
    assert not ({r["hgnc_id"] for r in train_rows}
                & {r["hgnc_id"] for r in test_rows}), "gene overlap"
    assert not ({r["name"].lower() for r in train_rows}
                & {r["name"].lower() for r in test_rows}), "name overlap"
    assert not ({r["symbol"].upper() for r in train_rows}
                & {r["symbol"].upper() for r in test_rows}), "symbol overlap"
    shipped_tokens = set()
    for r in train_rows + test_rows:
        shipped_tokens.update(name_tokens(r["name"]))
    for r in test_rows:
        assert r["symbol"].upper().replace("-", "") not in shipped_tokens, \
            f"test symbol printed in a shipped name: {r['symbol']}"
        assert len(r["stem"]) < 3 or r["stem"] not in shipped_tokens, \
            f"test stem printed in a shipped name: {r['stem']}"

    with (public / "train.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row_id", "name", "symbol"])
        for r in train_rows:
            w.writerow([r["row_id"], r["name"], r["symbol"]])

    with (public / "test.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row_id", "name"])
        for r in test_rows:
            w.writerow([r["row_id"], r["name"]])

    with (public / "sample_submission.csv").open(
            "w", newline="", encoding="utf8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row_id", "symbol"])
        for r in test_rows:
            w.writerow([r["row_id"], initials(r["name"]) or "X"])

    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row_id", "symbol"])
        for r in test_rows:
            w.writerow([r["row_id"], r["symbol"]])

    with (public / "test.csv").open(encoding="utf8", newline="") as f:
        t_rows = list(csv.DictReader(f))
    with (private / "answers.csv").open(encoding="utf8", newline="") as f:
        a_rows = list(csv.DictReader(f))
    assert [t["row_id"] for t in t_rows] == [a["row_id"] for a in a_rows]
    assert len({t["row_id"] for t in t_rows}) == len(t_rows), "ids not unique"
    shared = (set(t_rows[0]) & set(a_rows[0])) - {"row_id"}
    assert not shared, f"columns shared with answers.csv: {shared}"

    train_stems = {r["stem"] for r in train_rows}
    n_seen = sum(1 for r in test_rows if r["stem"] in train_stems)
    sym_len = sorted(len(r["symbol"]) for r in test_rows)
    name_len = sorted(len(r["name"]) for r in test_rows)
    stats = {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_train_ratio": round(ratio, 4),
        "split_unit": "effective date of the current approved symbol",
        "train_is": "symbols in force before the cutoff",
        "test_is": "symbols that came into force on or after the cutoff",
        "n_symbol_stems_train": len(train_stems),
        "test_rows_stem_seen_in_train": n_seen,
        "test_rows_stem_new": len(test_rows) - n_seen,
        "share_test_stem_seen_in_train": round(n_seen / len(test_rows), 4),
        "median_symbol_chars_test": sym_len[len(sym_len) // 2],
        "median_name_chars_test": name_len[len(name_len) // 2],
        "dropped_at_build": dict(sorted(dropped.items())),
        "metric": "prefix agreement: shared leading characters of the "
                  "predicted and the approved symbol, over the longer of the "
                  "two lengths",
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    here = Path(__file__).resolve().parent
    for cand in (here / "dataset" / "validate_submission.py",
                 here / "validate_submission.py"):
        if cand.exists():
            (public / "validate_submission.py").write_bytes(cand.read_bytes())
            break

    print(f"dropped: {dict(sorted(dropped.items()))}")
    print(f"public/train.csv              {len(train_rows):,} rows")
    print(f"public/test.csv               {len(test_rows):,} rows")
    print(f"private/answers.csv           {len(test_rows):,} rows")
    print(f"test/train ratio              {ratio:.1%}")
    print(f"test rows on a stem in train  {n_seen:,} "
          f"({n_seen / len(test_rows):.1%})")


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
