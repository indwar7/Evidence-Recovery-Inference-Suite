"""Build the public/private split from dataset/raw/sources.jsonl.

    python prepare.py [raw_root] [public_dir] [private_dir]

For every Python file in the raw snapshot, the code is cut into logical lines (blank and
comment-only lines dropped, trailing comments cut, continuation lines joined with one
space, all leading whitespace removed) and each line's block depth is recorded. One row
per file.

Pool and split. A file enters the pool only if it has at least MIN_EVENTS scored block
closures (the scoring rule is the grader's, repeated below). The same threshold applies
to training and test files, so both sets are drawn from one population. The pool is then
split by DIRECTORY, not by file: every file whose path has the same parent directory is
one group, and a group goes wholly to test or wholly to train, so no directory (package)
contributes to both sides. Groups are visited in salted-hash order and assigned to test
until the test side holds TEST_SHARE of the pool's files; a group that would push the
test side past TEST_CAP is left in train (this keeps one very large directory from
swallowing the test set). Deterministic: no random numbers are drawn.

Ladder. The build also scores the content-free baselines (never close, always close one
block, close at random at the training closure rate) on the test side with the grader's own
rule, and records them in private/build_audit.json; nothing else is uploaded, so this is
where the ladder is measured.

Outputs. public/ holds only what a solver needs: train.csv, test.csv,
sample_submission.csv and validate_submission.py. Everything derived from the labels of
the test files (closure counts, depths, build statistics) goes to private/.

The logical-line splitter below is a small self-contained lexer rather than Python's
`tokenize`, whose output differs between Python versions; the split must be byte-identical
wherever it is rebuilt.
"""
import csv, hashlib, json, random, sys
from pathlib import Path, PurePosixPath
import re

SALT = "pyindent-2026-10-11"
MIN_EVENTS = 3          # scored closures a file needs to enter the pool (train and test alike)
TEST_SHARE = 1 / 3      # target fraction of pool files on the test side
TEST_CAP = 0.40         # hard ceiling on that fraction

# ---------------------------------------------------------------- logical lines
_PREFIX = re.compile(r"(?i)(rb|br|fr|rf|b|r|u|f)?('''|\"\"\"|'|\")")


def logical_lines(src):
    """[(depth, text)] for every logical code line of `src`, or None if it cannot be split."""
    lines = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out, stack = [], [0]
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        ind = len(raw.expandtabs(8)) - len(raw.expandtabs(8).lstrip())
        parts, depth_br, quote, j = [], 0, None, i
        while True:
            if j >= n:
                return None
            line = lines[j]
            k, cut, cont = 0, None, False
            while k < len(line):
                c = line[k]
                if quote:
                    if c == "\\":
                        k += 2
                        continue
                    if line.startswith(quote, k):
                        k += len(quote)
                        quote = None
                        continue
                    k += 1
                    continue
                if c == "#":
                    cut = k
                    break
                if c in "'\"" or (k == 0 or not (line[k - 1].isalnum() or line[k - 1] == "_")):
                    m = _PREFIX.match(line, k)
                    if m:
                        quote = m.group(2)
                        k = m.end()
                        continue
                if c in "([{":
                    depth_br += 1
                elif c in ")]}":
                    depth_br -= 1
                k += 1
            seg = line if cut is None else line[:cut]
            if quote is None and seg.rstrip().endswith("\\"):
                seg = seg.rstrip()[:-1]
                cont = True
            if quote in ("'", '"'):
                tail = len(line) - len(line.rstrip("\\"))
                if tail % 2 == 0:
                    return None          # unterminated single-quoted string
                cont = True              # backslash-newline inside a string continues it
            parts.append(seg.strip())
            j += 1
            if quote is None and depth_br <= 0 and not cont:
                break
        text = " ".join(p for p in parts if p)
        if ind > stack[-1]:
            stack.append(ind)
        else:
            while ind < stack[-1]:
                stack.pop()
            if ind != stack[-1]:
                return None              # inconsistent dedent
        if text:
            out.append((len(stack) - 1, text))
        i = j
    return out


# ---------------------------------------------------------------- scoring rule (same as grade.py)
TERMINAL = {"return", "raise", "break", "continue", "pass"}
OPENERS = {"elif", "else", "except", "finally", "case", "def", "class", "async", "@"}


def first_word(t):
    m = re.match(r"[A-Za-z_]\w*|\S", t)
    return m.group(0) if m else ""


def is_header(t):
    return t.rstrip().endswith(":")


def eligible(T, i):
    return (not is_header(T[i - 1]) and first_word(T[i - 1]) not in TERMINAL
            and first_word(T[i]) not in OPENERS)


def n_closures(D, T):
    return sum(1 for i in range(1, len(D)) if eligible(T, i) and D[i] != D[i - 1])


def header_only_depths(T):
    d = [0]
    for i in range(1, len(T)):
        d.append(d[-1] + (1 if is_header(T[i - 1]) else 0))
    return d


# ---------------------------------------------------------------- baseline ladder (grader's file_score, repeated)
def file_score(pred, true, T):
    S = [i for i in range(1, len(true)) if eligible(T, i)
         and (true[i] != true[i - 1] or pred[i] != pred[i - 1])]
    if not S:
        return 1.0 if pred == true else 0.0
    return sum(pred[i] - pred[i - 1] == true[i] - true[i - 1] for i in S) / len(S)


def always_close_depths(T):
    """Header rule, plus close one block at every eligible line that has a block to close."""
    d = [0]
    for i in range(1, len(T)):
        if is_header(T[i - 1]):
            d.append(d[-1] + 1)
        elif eligible(T, i) and d[-1] > 0:
            d.append(d[-1] - 1)
        else:
            d.append(d[-1])
    return d


def random_close_depths(T, p, rng):
    """Header rule, plus close one block at an eligible line with probability p (content-free)."""
    d = [0]
    for i in range(1, len(T)):
        if is_header(T[i - 1]):
            d.append(d[-1] + 1)
        elif eligible(T, i) and d[-1] > 0 and rng.random() < p:
            d.append(d[-1] - 1)
        else:
            d.append(d[-1])
    return d


def measure_ladder(tr, te):
    """Content-free baselines on the test side, scored exactly as grade.py scores a submission."""
    n_elig = sum(1 for r in tr for i in range(1, len(r["T"])) if eligible(r["T"], i))
    p = sum(r["events"] for r in tr) / n_elig if n_elig else 0.0
    rng = random.Random(SALT + ":ladder")
    mean = lambda f: round(sum(f(r) for r in te) / len(te), 4) if te else 0.0
    return dict(
        never_close=mean(lambda r: file_score(header_only_depths(r["T"]), r["D"], r["T"])),
        always_close_one=mean(lambda r: file_score(always_close_depths(r["T"]), r["D"], r["T"])),
        random_close_at_train_rate=mean(lambda r: file_score(random_close_depths(r["T"], p, rng), r["D"], r["T"])),
        train_closure_rate_per_eligible_line=round(p, 4),
        oracle=1.0,
    )


# ---------------------------------------------------------------- the offline validator shipped to solvers
VALIDATOR = r'''"""Offline structural check of a submission. Needs test.csv, not the answers.

    python validate_submission.py submission.csv [--test path/to/test.csv]

By default test.csv is looked for next to this script, then next to the submission, then
in the current directory. The checks mirror what the grader scores 0 for: a test file
missing from the submission, a file_id given more than once, a depths value that is blank,
not a list of integers, of the wrong length, or containing a negative number. Exit code 0
when every test file passes, 1 otherwise.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--test", default=None)
    a = ap.parse_args()
    sub_path = Path(a.submission)
    cands = ([Path(a.test)] if a.test else
             [Path(__file__).resolve().parent / "test.csv", sub_path.resolve().parent / "test.csv", Path("test.csv")])
    test_path = next((c for c in cands if c.exists()), None)
    if test_path is None:
        print("test.csv not found; pass --test path/to/test.csv")
        return 2
    _, test = read_rows(test_path)
    want = {r["file_id"]: len(json.loads(r["lines"])) for r in test}
    cols, rows = read_rows(sub_path)
    bad = []
    for c in ("file_id", "depths"):
        if c not in cols:
            print(f"FAIL: submission is missing required column '{c}' (columns: {cols})")
            return 1
    seen = {}
    for r in rows:
        seen[r["file_id"]] = seen.get(r["file_id"], 0) + 1
    dup = {k for k, v in seen.items() if v > 1}
    vals = {r["file_id"]: r["depths"] for r in rows}
    for fid, n in want.items():
        if fid not in vals:
            bad.append((fid, "missing"))
            continue
        if fid in dup:
            bad.append((fid, f"given {seen[fid]} times"))
            continue
        v = vals[fid]
        try:
            d = [int(x) for x in v.split()]
        except ValueError:
            bad.append((fid, "depths is not a space-separated list of integers"))
            continue
        if len(d) != n:
            bad.append((fid, f"{len(d)} depths for {n} lines"))
        elif d and min(d) < 0:
            bad.append((fid, "negative depth"))
    extra = sorted(set(vals) - set(want))
    for fid, why in bad[:50]:
        print(f"  {fid}: {why}")
    if len(bad) > 50:
        print(f"  ... {len(bad) - 50} more")
    if extra:
        print(f"note: {len(extra)} file_id(s) not in test.csv are ignored by the grader")
    print(f"{len(want) - len(bad)} of {len(want)} test files pass; {len(bad)} would score 0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------- io helpers
def _h(s):
    return hashlib.sha256((SALT + s).encode("utf-8")).hexdigest()


def group_of(path):
    """The split group of a file: its parent directory. A file with no directory is its own group."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    return str(p.parent) if str(p.parent) not in ("", ".") else "file:" + p.name


def _looks_like_sources(path):
    """True if the first non-blank line is a JSON object with 'path' and 'text'."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    return isinstance(rec, dict) and "path" in rec and "text" in rec
    except Exception:
        return False
    return False


def find_raw(raw_root):
    """Locate the sources file under whatever root the platform passes.

    Tries the documented name at the usual layouts first, then any file whose
    content matches the expected record shape (so a renamed or re-extracted
    archive still works), and only then fails -- with a listing of what IS
    there, so the error is self-diagnosing.
    """
    raw_root = Path(raw_root)
    names = ["sources.jsonl", "sources.ndjson", "sources.json", "raw.jsonl", "data.jsonl"]
    bases = [raw_root, raw_root / "raw", raw_root / "dataset" / "raw", raw_root.parent]
    for b in bases:
        for n in names:
            cand = b / n
            if cand.is_file():
                return cand
    for n in names:
        hits = sorted(raw_root.rglob(n))
        if hits:
            return hits[0]
    # Fall back to content: any json-lines file whose rows carry path + text.
    for ext in ("*.jsonl", "*.ndjson", "*.json"):
        for cand in sorted(raw_root.rglob(ext)):
            if _looks_like_sources(cand):
                return cand
    listing = []
    for p in sorted(raw_root.rglob("*")):
        if len(listing) >= 60:
            listing.append("... (truncated)")
            break
        listing.append(str(p.relative_to(raw_root)) + ("/" if p.is_dir() else f"  ({p.stat().st_size} B)"))
    raise FileNotFoundError(
        "sources.jsonl not found under " + str(raw_root) + ". Contents:\n  " + "\n  ".join(listing)
    )


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def assign_groups(pool):
    """Mark r['test'] for every pool file. Groups in salted-hash order fill the test side."""
    groups = {}
    for r in pool:
        groups.setdefault(r["group"], []).append(r)
    order = sorted(groups, key=lambda g: _h("group:" + g))
    target, cap, n_test = TEST_SHARE * len(pool), TEST_CAP * len(pool), 0
    for g in order:
        if n_test >= target:
            break
        if n_test + len(groups[g]) > cap:
            continue
        for r in groups[g]:
            r["test"] = True
        n_test += len(groups[g])
    for r in pool:
        r.setdefault("test", False)


def prepare(raw, public, private):
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    src_path = find_raw(raw)
    files, seen = [], set()
    dropped = {"unsplittable": 0, "under_3_lines": 0, "duplicate": 0, "names_own_path": 0,
               f"under_{MIN_EVENTS}_scored_closures": 0}
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["path"] in rec["text"]:
                dropped["names_own_path"] += 1      # the code itself would publish its location
                continue
            L = logical_lines(rec["text"])
            if L is None:
                dropped["unsplittable"] += 1
                continue
            if len(L) < 3:
                dropped["under_3_lines"] += 1
                continue
            T = [t for _, t in L]
            key = hashlib.sha256("\n".join(T).encode("utf-8")).hexdigest()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            D = [d for d, _ in L]
            ev = n_closures(D, T)
            if ev < MIN_EVENTS:
                dropped[f"under_{MIN_EVENTS}_scored_closures"] += 1
                continue
            files.append(dict(path=rec["path"], group=group_of(rec["path"]), T=T, D=D, events=ev))
    assign_groups(files)
    for r in files:
        r["order"] = _h("order:" + r["path"])
    files.sort(key=lambda r: r["order"])
    tr = [r for r in files if not r["test"]]
    te = [r for r in files if r["test"]]
    for k, r in enumerate(tr):
        r["id"] = f"tr_{k:05d}"
    for k, r in enumerate(te):
        r["id"] = f"te_{k:05d}"
    assert not ({r["group"] for r in tr} & {r["group"] for r in te}), "a directory is on both sides"
    js = lambda T: json.dumps(T, ensure_ascii=False)
    ds = lambda D: " ".join(map(str, D))
    write_csv(public / "train.csv", ["file_id", "lines", "depths"], [[r["id"], js(r["T"]), ds(r["D"])] for r in tr])
    write_csv(public / "test.csv", ["file_id", "lines"], [[r["id"], js(r["T"])] for r in te])
    write_csv(public / "sample_submission.csv", ["file_id", "depths"],
              [[r["id"], ds(header_only_depths(r["T"]))] for r in te])
    (public / "validate_submission.py").write_text(VALIDATOR, encoding="utf-8")
    stale = public / "dataset_stats.json"           # written by earlier builds; label-derived, never public
    if stale.exists():
        stale.unlink()
    write_csv(private / "answers.csv", ["file_id", "depths", "lines_ref"],
              [[r["id"], ds(r["D"]), js(r["T"])] for r in te])
    stats = dict(train_files=len(tr), test_files=len(te),
                 train_groups=len({r["group"] for r in tr}), test_groups=len({r["group"] for r in te}),
                 train_lines=sum(len(r["T"]) for r in tr), test_lines=sum(len(r["T"]) for r in te),
                 train_scored_closures=sum(r["events"] for r in tr),
                 test_scored_closures=sum(r["events"] for r in te),
                 min_scored_closures_per_file=MIN_EVENTS,
                 max_depth=max(max(r["D"]) for r in files) if files else 0,
                 dropped_at_build=dropped, source_files=len(files) + sum(dropped.values()),
                 ladder=measure_ladder(tr, te))
    (private / "build_audit.json").write_text(json.dumps(stats, indent=2) + "\n")
    (private / "split_manifest.json").write_text(
        json.dumps({r["id"]: r["path"] for r in tr + te}, indent=1, sort_keys=True) + "\n")
    return stats


if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 3
    print(json.dumps(prepare(a[0] or "dataset/raw", a[1] or "dataset/public", a[2] or "dataset/private"), indent=2))
