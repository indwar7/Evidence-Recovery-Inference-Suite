"""Build the public/ and private/ splits from the raw before/after corpus.

Deterministic: fixed seed, no wall-clock, no network.  Running it twice produces
byte-identical output.

An ITEM is one real CFR section together with the amendatory instructions one real
final rule aimed at it.  The label is the official text of that section on the day
the rule took effect.

Two selection rules do real work here and both are documented in DESIGN.md:

  * Sections whose entire text the agency reprinted ("§ X is revised to read as
    follows:") are dropped.  For those the answer is the printed block copied out,
    which measures nothing.  What is kept are the surgical amendments, where the
    instruction names a sub-provision and you have to find it.

  * The split is disjoint by CFR *part*, not by item.  Agencies re-run the same
    amendment shape year after year -- the civil-penalty inflation adjustments are
    the clearest case -- so an item-level split would put this year's version in
    train and next year's in test.
"""
from pathlib import Path
import json, random, re, csv

SEED = 20260810
# Deliberately near half, not the usual 0.7.  Nothing here is trained: the task is
# resolving an instruction against a document, and the reference solution fits no
# parameters, so labelled items buy development and validation rather than fitting
# power.  Test items, by contrast, buy score resolution -- with a small pool, a
# 30-item test set makes the score a coarse grid on which two genuinely different
# solutions collide on the same value.
TRAIN_FRACTION = 0.52

# Share of the test set that should be compositional -- a `* * *` partial
# replacement, a run of more than two instructions, a redesignation, or a quoted
# removal.  The corpus is 58 percent compositional and a plain random part split
# drew only 45 percent, which inflated every score measured on it.  0.65 is a mild
# tilt towards the cases the task is about; see the split code for why it is not
# higher.
TEST_COMPOSITIONAL_TARGET = 0.65
MAX_ITEMS = 1600
MIN_BEFORE_TOKENS = 25
MAX_BEFORE_TOKENS = 1200
MAX_OPS = 30

# "§ 19.2 is revised to read as follows" with no sub-provision named: the agency
# reprints the whole section, so applying it is a copy.  The same instruction WITH
# a paragraph address ("revising paragraph (b)") is the interesting case and stays.
WHOLE_SECTION = re.compile(
    r"(?:is\s+(?:amended\s+and\s+)?revised|\brevises?\b)[^.]{0,60}?"
    r"(?:to\s+read\s+as\s+follows|\bpart\s+\d)", re.I)
SUBPROVISION = re.compile(
    r"paragraph|subparagraph|sentence|introductory|heading|table|"
    r"definition|clause|item|appendix|concluding|undesignated", re.I)


REDESIGNATE = re.compile(r"redesignat", re.I)
REMOVE = re.compile(r"\bremov", re.I)


def is_compositional(rec):
    """True when the amendment needs more than "find the paragraph, paste the block".

    Four things make an item compositional, and each is a distinct capability:
    a `* * *` marker means the printed text replaces only part of the paragraph and
    the remainder survives; a run of more than two instructions has to be applied
    in order; a redesignation moves the target of every later instruction; and a
    quoted removal edits words in place with nothing printed to paste.
    """
    instructions = " ".join(i for b in rec["blocks"] for i in b["instructions"])
    n_instructions = sum(len(b["instructions"]) for b in rec["blocks"])
    return bool(
        any("* * *" in (b["printed"] or "") for b in rec["blocks"])
        or n_instructions > 2
        or REDESIGNATE.search(instructions)
        or REMOVE.search(instructions))


VALIDATOR = r'''"""Check a submission offline, the way the grader will read it.

    python validate_submission.py submission.csv test.csv

Two levels of message:

  ERROR    the grader cannot score this at all -- exits 1.
  WARNING  the grader will score it, but the flagged items count as "no amendment
           applied", i.e. zero.  Exits 0.

The grader is deliberately forgiving about structure: an entry that omits an item,
leaves one blank or repeats a row is *scored low*, not thrown out.  So a warning
here is a silent loss of points rather than a failed run, and is still worth
fixing.
"""
import sys
import pandas as pd

MAX_CHARS = 200_000


def main(sub_path, test_path):
    try:
        sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    except Exception as e:                                     # noqa: BLE001
        print(f"ERROR: could not read {sub_path} as CSV: {e}")
        print("  after_text is long prose full of commas and quotation marks;"
              " write the file with pandas or the csv module, never by hand.")
        return 1
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)

    errors, warnings = [], []
    for col in ("item_id", "after_text"):
        if col not in sub.columns:
            errors.append(f"missing column '{col}'; found {list(sub.columns)}")
    if errors:
        print("ERROR:\n  " + "\n  ".join(errors))
        return 1

    if not set(sub["item_id"]) & set(test["item_id"]):
        print("ERROR:\n  no submitted item_id appears in the test set; "
              f"expected ids like {sorted(test['item_id'])[:3]}")
        return 1

    if sub["item_id"].duplicated().any():
        dup = sub.loc[sub["item_id"].duplicated(), "item_id"].unique()[:5]
        warnings.append(f"duplicate item_id(s) {list(dup)} -- the first row for "
                        "an item is used and the rest are ignored")

    missing = set(test["item_id"]) - set(sub["item_id"])
    if missing:
        warnings.append(f"{len(missing)} test item(s) absent, e.g. "
                        f"{sorted(missing)[:5]} -- each scores 0")

    extra = set(sub["item_id"]) - set(test["item_id"])
    if extra:
        warnings.append(f"{len(extra)} item_id(s) not in the test set, e.g. "
                        f"{sorted(extra)[:5]} -- ignored")

    empty = sub.loc[sub["after_text"].str.strip() == "", "item_id"]
    if len(empty):
        warnings.append(f"empty after_text for {len(empty)} item(s), e.g. "
                        f"{list(empty[:3])} -- each scores 0")

    long = sub.loc[sub["after_text"].str.len() > MAX_CHARS, "item_id"]
    if len(long):
        warnings.append(f"after_text over {MAX_CHARS} characters for "
                        f"{len(long)} item(s), e.g. {list(long[:3])} -- truncated")

    if warnings:
        print("WARNING:\n  " + "\n  ".join(warnings))
        print(f"\nScorable: {len(sub)} rows, "
              f"{len(set(sub['item_id']) & set(test['item_id']))}"
              f"/{len(test)} test items covered.")
        return 0
    print(f"OK: {len(sub)} rows, {len(test)} test items covered.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
'''


def instruction_text(rec):
    return " ".join(i for b in rec["blocks"] for i in b["instructions"])


def is_whole_section_rewrite(rec):
    ins = instruction_text(rec)
    return bool(WHOLE_SECTION.search(ins)) and not SUBPROVISION.search(ins)


def load(raw: Path):
    out = []
    with open(raw / "items.jsonl", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def eligible(rec):
    n = len(rec["before"].split())
    if not MIN_BEFORE_TOKENS <= n <= MAX_BEFORE_TOKENS:
        return False
    if not 1 <= rec["n_ops"] <= MAX_OPS:
        return False
    if not instruction_text(rec).strip():
        return False
    if is_whole_section_rewrite(rec):
        return False
    return True


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    recs = [r for r in load(raw) if eligible(r)]
    # deterministic input order, independent of the order the scan happened to run
    recs.sort(key=lambda r: r["uid"])

    # One item per (title, section): the same section amended by several rules
    # yields near-identical items, and keeping them all would let a solver
    # memorise a section rather than learn to apply an instruction.
    seen, uniq = set(), []
    for r in recs:
        key = (r["cfr_title"], r["section"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    # Disjoint by CFR part, and stratified by amendment complexity.
    #
    # A plain random part split gave a test set that was only 45% compositional
    # against a corpus that is 58%, and the consequence was measured: the
    # reference scores 0.820 on the simple cases and 0.439 on the compositional
    # ones, so an unlucky-easy test set inflates every score on it.  The simple
    # case -- one instruction revising one paragraph, with the whole replacement
    # printed -- is transcription once the paragraph is located.  What the task is
    # actually about is the compositional cases: partial `* * *` replacements,
    # runs of instructions applied in order, redesignations that move later
    # targets, and quoted removals.  So test is drawn from the parts richest in
    # those, while train keeps enough of them to develop against.
    parts = sorted({(r["cfr_title"], r["cfr_part"]) for r in uniq})
    by_part = {p: [] for p in parts}
    for r in uniq:
        by_part[(r["cfr_title"], r["cfr_part"])].append(r)

    def hard_fraction(p):
        recs = by_part[p]
        return sum(is_compositional(r) for r in recs) / len(recs)

    # Deterministic: complexity first, then a seeded shuffle order to break ties
    # without letting part numbering carry any signal.
    rng.shuffle(parts)
    order = sorted(parts, key=lambda p: (-hard_fraction(p), parts.index(p)))

    # Parts are added to test while steering the running compositional share
    # towards TEST_COMPOSITIONAL_TARGET, taking from the complex pile when the
    # share is below target and the simple pile when it is above.  Steering rather
    # than simply taking the hardest parts is deliberate, and the difference was
    # measured: sweeping the target from 0.45 to 0.80 showed that past about 0.7
    # every solver collapses into the same score -- on the compositional items
    # even good solvers do poorly, so the differences become noise and one
    # ablation overtook the full reference. At 0.65 the ladder stays strictly
    # ordered and the rungs stay apart, which is what a difficulty gradient means.
    hard_pile = sorted([p for p in parts if hard_fraction(p) >= 0.5],
                       key=lambda p: (-hard_fraction(p), parts.index(p)))
    easy_pile = sorted([p for p in parts if hard_fraction(p) < 0.5],
                       key=lambda p: (hard_fraction(p), parts.index(p)))

    target = int(len(uniq) * (1 - TRAIN_FRACTION))
    test_parts, n_test_items, n_test_hard = set(), 0, 0
    hi = ei = 0
    while n_test_items < target and (hi < len(hard_pile) or ei < len(easy_pile)):
        below = (n_test_hard / n_test_items if n_test_items else 0.0)
        take_hard = below < TEST_COMPOSITIONAL_TARGET
        if take_hard and hi < len(hard_pile):
            p = hard_pile[hi]; hi += 1
        elif ei < len(easy_pile):
            p = easy_pile[ei]; ei += 1
        elif hi < len(hard_pile):
            p = hard_pile[hi]; hi += 1
        else:
            break
        test_parts.add(p)
        n_test_items += len(by_part[p])
        n_test_hard += sum(is_compositional(r) for r in by_part[p])
    train_parts = set(parts) - test_parts

    rng.shuffle(uniq)
    uniq = uniq[:MAX_ITEMS]
    uniq.sort(key=lambda r: r["uid"])

    # An opaque label for "same CFR part".  Solvers need this: the split is
    # part-disjoint, so an item-level validation split reports a score the test
    # set will not reproduce -- and without a grouping key there would be no way
    # to validate honestly.  It is a renumbering, not the part number, so it
    # identifies nothing outside the dataset.
    part_ids = {p: f"g{n:04d}" for n, p in enumerate(sorted(parts))}

    items = []
    for i, r in enumerate(uniq):
        part = (r["cfr_title"], r["cfr_part"])
        items.append(dict(
            item_id=f"it{i:05d}",
            group_id=part_ids[part],
            before_text=r["before"],
            after_text=r["after"],
            blocks=r["blocks"],
            _part=part,
            _rec=r,
            _uid=r["uid"],
        ))

    train = [it for it in items if it["_part"] in train_parts]
    test = [it for it in items if it["_part"] not in train_parts]

    item_dir = public / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        payload = {"item_id": it["item_id"],
                   "group_id": it["group_id"],
                   "before_text": it["before_text"],
                   "blocks": it["blocks"]}
        with open(item_dir / f"{it['item_id']}.json", "w", encoding="utf8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)

    def write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    write_csv(public / "train.csv", ["item_id", "after_text"],
              [[it["item_id"], it["after_text"]] for it in train])
    write_csv(public / "test.csv", ["item_id"], [[it["item_id"]] for it in test])
    write_csv(public / "sample_submission.csv", ["item_id", "after_text"],
              [[it["item_id"], it["before_text"]] for it in test])
    write_csv(private / "answers.csv", ["item_id", "before_text", "after_text"],
              [[it["item_id"], it["before_text"], it["after_text"]] for it in test])

    # The validator is written out here rather than copied, so the prepared split
    # is self-contained wherever prepare.py runs.  The problem description tells
    # solvers to run it; it has to exist.
    (public / "validate_submission.py").write_text(VALIDATOR, encoding="utf8")

    meta = dict(seed=SEED, n_items=len(items), n_train=len(train), n_test=len(test),
                n_parts=len(parts), n_train_parts=len(train_parts),
                n_test_compositional=sum(
                    1 for it in test if is_compositional(it["_rec"])),
                n_train_compositional=sum(
                    1 for it in train if is_compositional(it["_rec"])),
                median_before_tokens=sorted(
                    len(it["before_text"].split()) for it in items)[len(items) // 2])
    with open(public / "dataset_stats.json", "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=1)
    with open(private / "provenance.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "uid"])
        w.writerows([[it["item_id"], it["_uid"]] for it in items])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    prepare(Path(a[0] if a else "dataset/raw"),
            Path(a[1] if len(a) > 1 else "dataset/public"),
            Path(a[2] if len(a) > 2 else "dataset/private"))
