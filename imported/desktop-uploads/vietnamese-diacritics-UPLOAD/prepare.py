"""prepare.py - build the public/private split for the Vietnamese diacritic
restoration task from the raw GeoNames Vietnam dump.

Contract:
    prepare(raw_root, public_dir, private_dir) -> None

Deterministic and seedless: the train/test assignment is a pure function of the
source records (a SHA-256 bucket over syllable types), so running twice yields
byte-identical output and no random seed is published. The task is to recover
a place name's native Vietnamese orthography (tone and vowel diacritics) from
its ASCII-folded form, with its named surroundings as additional evidence.
A deterministic subset of the corpus's less common syllables is held out:
every name containing one of them is an evaluation row and no training row
contains any of them, so every evaluation name carries at least one syllable
that never occurs in training.

Only records whose ASCII folding is a clean 1:1, length-preserving mapping of the
native name are kept, so input and target align position-by-position and the
changed-position metric is well defined.

Every row also carries context built from the same dump, expressed only through
other names:
  province, district - the GeoNames name of the row's ADM1 / ADM2 record. When
      the record's alternatenames list the same name with more marks (ignoring
      English generics such as "Province"), that spelling is used; otherwise
      the recorded name stays as is.
  neighbors - the names of the NEIGHBORS nearest TRAINING settlements by
      great-circle distance.
A context value is withheld when its folded form contains the row's own folded
name as a whole-word phrase (a district seat named after its district, a
neighbour whose name extends the row's name), so no row's context contains its
own answer. Neighbours never come from the test split. Coordinates and admin
codes are consumed here and not published. Standard library only.
"""
from __future__ import annotations
import csv, hashlib, json, math, sys, unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path

FEATURE_CLASS = "P"          # populated places only
LEN_MIN, LEN_MAX = 3, 40
HELD_FREQ_MIN, HELD_FREQ_MAX = 2, 50   # corpus-frequency band of syllable types eligible to be held out
HELD_PERMILLE = 400          # share (per mille) of eligible syllable types held out, by SHA-256 bucket
ALLOWED = set("abcdefghijklmnopqrstuvwxyz '-")
NEIGHBORS = 5                # nearest training settlements given as context
SEP = " | "


def fold(s: str) -> str:
    """ASCII-fold by dropping Unicode combining marks (NFKD)."""
    nk = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nk if not unicodedata.combining(c))


def _marks(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", s) if unicodedata.combining(c))


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def _clean(s: str) -> bool:
    return all(c.lower() in ALLOWED for c in s)


def _phrase(s: str) -> str:
    """lower-cased, folded, single-spaced, padded: for whole-word containment tests"""
    return " " + " ".join(fold(s).lower().split()) + " "


def _contains(context_value: str, folded_name: str) -> bool:
    """does the context value contain the row's folded name as a whole-word phrase?"""
    return _phrase(folded_name) in _phrase(context_value)


GENERIC = {"province", "city", "municipality"}      # English generics in GeoNames admin names
VN_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323", "\u0302", "\u0306", "\u031b"}


def _vn_marked(s: str) -> bool:
    """every combining mark in s is a Vietnamese tone or vowel mark"""
    return all(c in VN_MARKS for c in unicodedata.normalize("NFKD", s) if unicodedata.combining(c))


def _core(s: str) -> list:
    """tokens for matching admin names: folded, lower-cased, D-with-stroke treated as D,
    English generic words removed"""
    return [t for t in fold(s.replace("Đ", "D").replace("đ", "d")).lower().split() if t not in GENERIC]


def _native(name: str, alternates: str) -> str:
    """the most fully marked alternate spelling of the same admin name (same tokens
    once marks and English generics are ignored), else `name` itself. Deterministic:
    strict improvement only, first wins on ties."""
    best, best_marks, key = name, _marks(name), _core(name)
    if len(key) < 2:                      # single-token names are exonyms (Hanoi); keep as recorded
        return name
    for c in alternates.split(","):
        c = " ".join(t for t in c.split() if fold(t).lower() not in GENERIC)
        if c and _core(c) == key and _vn_marked(c) and _marks(c) > best_marks:
            best, best_marks = c, _marks(c)
    return best


def _find_raw(raw_root: Path) -> Path:
    """Locate VN.txt tolerantly: the given root, or anywhere in its subtree."""
    raw_root = Path(raw_root)
    direct = raw_root / "VN.txt"
    if direct.exists():
        return direct
    for p in raw_root.rglob("VN.txt"):
        return p
    # fall back to any GeoNames-style country file
    for p in raw_root.rglob("*.txt"):
        if p.name[:2].isalpha():
            return p
    raise FileNotFoundError(f"VN.txt not found under {raw_root}")


def _extract(raw_root) -> list[dict]:
    path = _find_raw(Path(raw_root))
    seen = set()
    rows = []
    admin = {}               # (level, admin1, admin2) -> name
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            name, ascii_, fclass = p[1].strip(), p[2].strip(), p[6]
            alts = p[3] if len(p) > 3 else ""
            a1 = p[10] if len(p) > 10 else ""
            a2 = p[11] if len(p) > 11 else ""
            if fclass == "A" and p[7] == "ADM1" and a1:
                admin.setdefault(("ADM1", a1, ""), _native(name, alts))
            elif fclass == "A" and p[7] == "ADM2" and a1 and a2:
                admin.setdefault(("ADM2", a1, a2), _native(name, alts))
            if fclass != FEATURE_CLASS:
                continue
            if not name or not ascii_ or name == ascii_:
                continue
            if not (LEN_MIN <= len(name) <= LEN_MAX) or len(name) != len(ascii_):
                continue
            if not _is_ascii(ascii_) or not _clean(ascii_):
                continue
            if fold(name) != ascii_:
                continue
            if sum(1 for a, b in zip(name, ascii_) if a != b) == 0:
                continue
            key = (ascii_, name)
            if key in seen:
                continue
            seen.add(key)
            try:
                lat, lon = float(p[4]), float(p[5])
            except (ValueError, IndexError):
                lat, lon = float("nan"), float("nan")
            rows.append({"folded": ascii_, "name": name, "lat": lat, "lon": lon, "a1": a1, "a2": a2,
                         "geonameid": p[0]})
    for r in rows:
        for col, key in (("province", ("ADM1", r["a1"], "")), ("district", ("ADM2", r["a1"], r["a2"]))):
            v = admin.get(key, "") if key[1] and (col == "province" or key[2]) else ""
            # withhold a parent whose name contains the row's own name (district seats)
            r[col] = "" if v and _contains(v, r["folded"]) else v
    return rows


def _unit(lat: float, lon: float) -> tuple:
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def _add_neighbors(rows: list[dict], pool: list[dict]) -> None:
    """Attach the names of the NEIGHBORS nearest pool rows by great-circle distance,
    skipping any pool row whose folded name equals or contains the row's folded
    name. Exact: a latitude window grows until the 5th-best angular distance is
    within the window half-width, so nothing outside the window can be nearer.
    Ties break on the pool's stable order. Rows without coordinates get none."""
    cand = sorted((q["lat"], i) for i, q in enumerate(pool) if q["lat"] == q["lat"])
    lats = [c[0] for c in cand]
    unit = {i: _unit(pool[i]["lat"], pool[i]["lon"]) for _, i in cand}
    for r in rows:
        if r["lat"] != r["lat"]:
            r["neighbors"] = ""
            continue
        x, y, z = _unit(r["lat"], r["lon"])
        own = r["folded"]
        delta = 0.05
        while True:
            lo, hi = bisect_left(lats, r["lat"] - delta), bisect_right(lats, r["lat"] + delta)
            best = []
            for k in range(lo, hi):
                i = cand[k][1]
                if pool[i]["folded"] == own or _contains(pool[i]["folded"], own):
                    continue
                ux, uy, uz = unit[i]
                best.append(((ux - x) ** 2 + (uy - y) ** 2 + (uz - z) ** 2, i))
            best.sort()
            best = best[:NEIGHBORS]
            if hi - lo == len(cand):
                break                                   # whole pool scanned
            if len(best) == NEIGHBORS:
                theta = 2.0 * math.asin(min(1.0, math.sqrt(best[-1][0]) / 2.0))
                if theta <= math.radians(delta):
                    break                               # nothing outside can be nearer
            delta *= 2.0
        r["neighbors"] = SEP.join(pool[i]["name"] for _, i in best)


def _held_bucket(syllable: str) -> int:
    """deterministic 0-999 bucket of a folded syllable type; no random seed"""
    return int(hashlib.sha256(("held::" + syllable).encode("utf-8")).hexdigest(), 16) % 1000


def _shuffle_key(folded: str) -> str:
    """deterministic, seedless shuffle key: unrelated to the native-name sort
    order used for id assignment, so holding this out as test cannot leak the
    native spelling via id order."""
    return hashlib.sha256(("vn-diacritics-shuffle::" + folded).encode("utf-8")).hexdigest()


def prepare(raw_root, public_dir, private_dir) -> None:
    public_dir = Path(public_dir)
    private_dir = Path(private_dir)
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    rows = _extract(raw_root)
    # stable order: by folded then name, so the split and files are deterministic
    rows.sort(key=lambda r: (r["folded"], r["name"]))

    # split by held-out syllables. A syllable type (lower-cased folded token)
    # whose corpus frequency lies in [HELD_FREQ_MIN, HELD_FREQ_MAX] is held out
    # when its SHA-256 bucket falls under HELD_PERMILLE. Every row whose folded
    # name contains a held-out syllable is a test row; every other row is a
    # training row. Because all rows sharing a folded string share its
    # syllables, the folded string is atomic (never in both splits), and every
    # test name carries at least one syllable absent from train by
    # construction. An earlier design split by whole-string hash, which left
    # nearly every test syllable in train and let a per-syllable lookup table
    # match or beat the trained reference.
    syl_freq: dict[str, int] = {}
    for r in rows:
        for t in r["folded"].lower().split():
            syl_freq[t] = syl_freq.get(t, 0) + 1
    held = {t for t, c in syl_freq.items()
            if HELD_FREQ_MIN <= c <= HELD_FREQ_MAX and _held_bucket(t) < HELD_PERMILLE}

    train, test = [], []
    for r in rows:
        (test if any(t in held for t in r["folded"].lower().split()) else train).append(r)

    # context: nearest training settlements (train only, never the same folded string)
    _add_neighbors(train, train)
    _add_neighbors(test, train)

    # opaque ids: shuffle each split before assigning ids, so id order cannot
    # leak the answer. The shuffle key is salted with the row's GeoNames
    # geonameid (never published, unique per record, unrelated to the native
    # spelling's alphabetic order) rather than with `name` itself: without
    # this, homonym groups (same folded form, several distinct native
    # spellings) would have their rows' ids assigned in native-name sort
    # order, since that was the stable sort key used for the split above.
    train = sorted(train, key=lambda r: _shuffle_key(f"id-tr::{r['geonameid']}"))
    test = sorted(test, key=lambda r: _shuffle_key(f"id-te::{r['geonameid']}"))
    for i, r in enumerate(train):
        r["id"] = f"loc_tr_{i:06d}"
    for i, r in enumerate(test):
        r["id"] = f"loc_te_{i:06d}"

    ctx = ["province", "district", "neighbors"]

    # ---- public/train.csv : id, folded, context, name (target) ----
    with open(public_dir / "train.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "folded"] + ctx + ["name"])
        for r in train:
            w.writerow([r["id"], r["folded"]] + [r[c] for c in ctx] + [r["name"]])

    # ---- public/test.csv : id, folded, context (train columns minus target) ----
    with open(public_dir / "test.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "folded"] + ctx)
        for r in test:
            w.writerow([r["id"], r["folded"]] + [r[c] for c in ctx])

    # ---- public/sample_submission.csv : id, name (copy baseline) ----
    with open(public_dir / "sample_submission.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name"])
        for r in test:
            w.writerow([r["id"], r["folded"]])   # copy = folded, scores 0

    # ---- private/answers.csv : id, name (target), folded_src (input copy),
    # province_src/district_src/neighbors_src (context copy) ----
    # Every column carried from test.csv is renamed with an _src suffix so the
    # only shared column name between test.csv and answers.csv is "id". The
    # context columns are repeated here (not just folded) so the grader can
    # classify each changed position as context-attributable (the syllable's
    # exact native form appears in province/district/neighbors) without
    # re-deriving test.csv from train.csv at grading time.
    with open(private_dir / "answers.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "folded_src", "province_src", "district_src", "neighbors_src"])
        for r in test:
            w.writerow([r["id"], r["name"], r["folded"], r["province"], r["district"], r["neighbors"]])

    # ---- private/split_manifest.json ----
    with open(private_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "method": "held-out syllables: syllable types with corpus frequency in "
                      f"[{HELD_FREQ_MIN}, {HELD_FREQ_MAX}] whose SHA-256 bucket is under "
                      f"{HELD_PERMILLE}/1000 are held out; every row whose folded name "
                      "contains a held-out syllable is a test row, every other row a "
                      "training row; deterministic, no random seed",
            "held_freq_band": [HELD_FREQ_MIN, HELD_FREQ_MAX],
            "held_permille": HELD_PERMILLE,
            "n_held_syllable_types": len(held),
            "feature_class": FEATURE_CLASS,
            "n_train": len(train),
            "n_test": len(test),
            "split_unit": "folded string (no folded string appears in both splits); "
                          "every test name contains at least one syllable that occurs "
                          "in no training name",
            "id_assignment": "ids are assigned after an independent shuffle of each "
                       "split, keyed on each row's GeoNames geonameid (never "
                       "published) rather than on the folded/name sort order used "
                       "for the split, so id order does not reveal the alphabetic "
                       "rank of a row's native spelling within a homonym group.",
            "context": "province and district = name of the row's GeoNames ADM1/ADM2 record "
                       "(most-marked same-spelling alternate when listed); "
                       f"neighbors = names of the {NEIGHBORS} nearest training settlements. "
                       "A value whose folded form contains the row's folded name as a "
                       "phrase is withheld.",
        }, fh, indent=2)

    # ---- public/dataset_stats.json ----
    def changed(r):
        return sum(1 for a, b in zip(r["folded"], r["name"]) if a != b)
    all_rows = train + test
    stats = {
        "n_total": len(all_rows),
        "n_train": len(train),
        "n_test": len(test),
        "test_fraction": round(len(test) / max(1, len(all_rows)), 4),
        "mean_name_length": round(sum(len(r["folded"]) for r in all_rows) / max(1, len(all_rows)), 2),
        "mean_changed_positions": round(sum(changed(r) for r in all_rows) / max(1, len(all_rows)), 2),
        "distinct_output_chars": len(sorted({c for r in all_rows for c in r["name"]})),
        "neighbors_per_row": NEIGHBORS,
        "rows_with_province": sum(1 for r in all_rows if r["province"]),
        "rows_with_district": sum(1 for r in all_rows if r["district"]),
        "rows_with_marked_province": sum(1 for r in all_rows if _marks(r["province"])),
        "rows_with_marked_district": sum(1 for r in all_rows if _marks(r["district"])),
    }
    with open(public_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # ---- public/validate_submission.py (shipped copy) ----
    _write_validator(public_dir / "validate_submission.py")


def _write_validator(dest: Path) -> None:
    dest.write_text(VALIDATOR_SRC, encoding="utf-8")


VALIDATOR_SRC = '''"""Validate a submission file for the Vietnamese diacritic restoration task.

Usage: python validate_submission.py <submission.csv> <test.csv>
Checks columns, ids, and that no row is empty. Does not score.
"""
import csv, sys

REQUIRED = ["id", "name"]

def main(sub_path, test_path):
    with open(test_path, encoding="utf-8", newline="") as fh:
        ids = {row["id"] for row in csv.DictReader(fh)}
    with open(sub_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if cols[:2] != REQUIRED:
            raise SystemExit(f"columns must start with {REQUIRED}, got {cols}")
        seen = set()
        for row in reader:
            rid = row["id"]
            if rid in seen:
                raise SystemExit(f"duplicate id: {rid}")
            seen.add(rid)
        missing = ids - seen
        if missing:
            print(f"warning: {len(missing)} test ids missing from submission (scored 0)")
    print("OK: submission is well formed")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    raw = sys.argv[1] if len(sys.argv) > 1 else here / "dataset" / "raw"
    pub = sys.argv[2] if len(sys.argv) > 2 else here / "dataset" / "public"
    priv = sys.argv[3] if len(sys.argv) > 3 else here / "dataset" / "private"
    prepare(raw, pub, priv)
    print("prepare: wrote split to", pub, "and", priv)
