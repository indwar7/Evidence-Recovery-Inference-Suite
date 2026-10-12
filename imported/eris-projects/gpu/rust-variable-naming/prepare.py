"""prepare(raw_root, public_dir, private_dir) -> None  (Rust Toolchain Local Variable Names)

Standard library only. Tokenises every .rs file of the pinned rust-lang/rust
snapshot, takes each `fn` item with a body (outside macro bodies, not nested in
another fn), masks every parameter and `let`-bound local name as VAR0, VAR1, ...
at every usage site, and drops a function whenever masking could be ambiguous or
leaky: a re-bound name, a masked name that still appears as a plain identifier,
in a comment (whole word, >= 3 chars) or in a string literal, or fewer than
MIN_VARS names. Deduplicates by masked skeleton, then splits BY SOURCE FILE
(SHA-256 of the repo-relative path). Deterministic, seedless, tolerant of the
raw_root layout.
"""
import csv, hashlib, json, re
from bisect import bisect_left
from pathlib import Path

ROOTS = ("library/core", "library/alloc", "library/std", "library/coretests",
         "library/alloctests", "compiler", "src/tools/rust-analyzer/crates")
FIXTURE_DIRS = {"test_data"}
TEST_FRACTION, MAX_FN_CHARS, MIN_FN_CHARS, MIN_VARS = 0.2, 4000, 60, 2
KW = set("""as async await box break const continue crate dyn else enum extern false fn for if
impl in let loop match mod move mut pub ref return self Self static struct super trait true
type unsafe use where while yield try macro_rules""".split())
MODS = {"pub", "const", "async", "unsafe", "extern", "default", "safe"}
TOK = re.compile(r"""\s*(?:
 (?P<lc>//[^\n]*)
|(?P<bc>/\*)
|(?P<rs>(?:br|cr|b|c|r)\#*")
|(?P<st>(?:b|c)?"(?:[^"\\]|\\[\s\S])*")
|(?P<ch>b?'(?:[^'\\\n]|\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]{1,6}\}|[\s\S]))')
|(?P<lt>'[A-Za-z_][A-Za-z0-9_]*)
|(?P<nu>\d[\dA-Za-z_]*(?:\.\d[\dA-Za-z_]*)?)
|(?P<id>(?:r\#)?[A-Za-z_][A-Za-z0-9_]*)
|(?P<pu>::|->|=>|\.\.=|\.\.\.|\.\.|<<=|>>=|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|%=|\^=|&=|\|=|<<|>>|[^\sA-Za-z0-9_"'])
)""", re.X)


def tokenize(text):
    """-> (tokens, comments): tokens are (kind, text, start, end) without whitespace/comments."""
    toks, comments, pos, n = [], [], 0, len(text)
    while pos < n:
        m = TOK.match(text, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        k = m.lastgroup
        if k is None:
            pos = m.end()
            continue
        s = m.start(k)
        if k == "bc":
            depth, e = 1, s + 2
            while depth and e < n:
                o, c = text.find("/*", e), text.find("*/", e)
                if c < 0:
                    e = n
                    break
                if 0 <= o < c:
                    depth, e = depth + 1, o + 2
                else:
                    depth, e = depth - 1, c + 2
            comments.append((s, text[s:e]))
        elif k == "rs":
            hashes = m.group(k).count("#")
            e = text.find('"' + "#" * hashes, m.end())
            e = n if e < 0 else e + 1 + hashes
            toks.append(("st", text[s:e], s, e))
        else:
            e = m.end()
            if k == "lc":
                comments.append((s, text[s:e]))
            else:
                toks.append((k, text[s:e], s, e))
        pos = e
    return toks, comments


def _bracket_map(T, lo, hi):
    """open-index -> close-index for (), [], {} inside T[lo:hi]; None if unbalanced."""
    close, stack = {}, []
    for i in range(lo, hi):
        s = T[i][1]
        if s in "([{":
            stack.append((s, i))
        elif s in ")]}":
            if not stack or "([{"[")]}".index(s)] != stack[-1][0]:
                return None
            close[stack.pop()[1]] = i
    return None if stack else close


def _functions(T):
    """(start_idx, name_idx, param_lo, param_hi, body_open, body_close) for every fn item with a
    body that is not inside a macro body and not nested in another fn."""
    out, stack, skip_to, n = [], [], -1, len(T)
    for i in range(n):
        s = T[i][1]
        if s in "([{":
            macro = (i > 0 and T[i - 1][1] == "!") or (i > 1 and T[i - 2][1] == "!") or (stack and stack[-1])
            stack.append(bool(macro))
        elif s in ")]}":
            if stack:
                stack.pop()
        elif s == "fn" and T[i][0] == "id" and i >= skip_to and not (stack and stack[-1]) \
                and i + 1 < n and T[i + 1][0] == "id" and T[i + 1][1] not in KW:
            j = i + 2
            if j < n and T[j][1] == "<":
                depth = 0
                while j < n:
                    depth += {"<": 1, ">": -1, ">>": -2}.get(T[j][1], 0)
                    j += 1
                    if depth <= 0:
                        break
            if j >= n or T[j][1] != "(":
                continue
            depth, k = 0, j
            while k < n:
                depth += {"(": 1, ")": -1}.get(T[k][1], 0)
                if depth == 0:
                    break
                k += 1
            p_lo, p_hi, b = j + 1, k, k + 1
            depth = 0
            while b < n and not (depth == 0 and T[b][1] in "{;"):
                depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(T[b][1], 0)
                b += 1
            if b >= n or T[b][1] == ";":
                continue
            depth, e = 0, b
            while e < n:
                depth += {"{": 1, "}": -1}.get(T[e][1], 0)
                if depth == 0:
                    break
                e += 1
            if e >= n:
                continue
            st = i
            while st > 0:
                if T[st - 1][1] in MODS:
                    st -= 1
                elif T[st - 1][0] == "st" and st > 1 and T[st - 2][1] == "extern":
                    st -= 2
                elif T[st - 1][1] == ")" and st > 3 and T[st - 4][1] == "pub" and T[st - 3][1] == "(":
                    st -= 4
                else:
                    break
            out.append((st, i + 1, p_lo, p_hi, b, e))
            skip_to = e + 1
    return out


def _is_field_label(T, i):
    """T[i] is an identifier followed by ':' — is it a `Struct { field: ... }` label?"""
    depth, j = 0, i - 1
    while j >= 0:
        s = T[j][1]
        if s in ")]}":
            depth += 1
        elif s in "([{":
            if depth == 0:
                return s == "{"
            depth -= 1
        elif depth == 0 and (s in ("|", ";", "let", "=>", "=") or s == "::"):
            return False
        j -= 1
    return False


def _pattern_bindings(T, lo, hi):
    """Lower-case identifiers bound by a pattern (no paths, constructors, field labels, consts)."""
    out = []
    for i in range(lo, hi):
        k, s = T[i][0], T[i][1]
        if k != "id" or s in KW or s == "_" or s[:1].isupper():
            continue
        nxt = T[i + 1][1] if i + 1 < hi else ""
        prv = T[i - 1][1] if i > lo else ""
        if nxt in ("(", "{", "::", "!") or prv == "::" or (nxt == ":" and _is_field_label(T, i)):
            continue
        out.append(i)
    return out


def mask_function(text, T, comments, fn):
    """-> {"masked", "answers", "num_vars"} or None when masking would be ambiguous or leaky."""
    st, name_i, p_lo, p_hi, b_open, b_close = fn
    close = _bracket_map(T, st, b_close + 1)
    if close is None:
        return None
    binds = []  # (token_idx, scope_lo, scope_hi)
    seg, depth = p_lo, 0
    for i in range(p_lo, p_hi + 1):
        s = T[i][1] if i < p_hi else ","
        depth += {"(": 1, "[": 1, "{": 1, "<": 1, ")": -1, "]": -1, "}": -1, ">": -1, ">>": -2}.get(s, 0)
        if s == "," and depth == 0:
            pat = [j for j in range(seg, i)]
            colon = next((j for j in pat if T[j][1] == ":" ), None)
            if colon is not None:
                pat = [j for j in pat if j < colon and T[j][1] != "mut"]
                if len(pat) == 1 and T[pat[0]][0] == "id" and T[pat[0]][1] not in KW \
                        and T[pat[0]][1] != "_" and not T[pat[0]][1][:1].isupper():
                    binds.append((pat[0], st, b_close))
            seg = i + 1
    braces = []
    for i in range(b_open, b_close + 1):
        s = T[i][1]
        if s == "{":
            braces.append(i)
        elif s == "}":
            braces.pop()
        elif s == "let" and T[i][0] == "id" and T[i - 1][1] not in ("if", "while", "&&", "||"):
            j, depth = i + 1, 0
            while j < b_close and not (depth == 0 and T[j][1] in ("=", ":", ";", "else")):
                depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(T[j][1], 0)
                j += 1
            scope_hi = close[braces[-1]] if braces else b_close
            binds += [(b, i, scope_hi) for b in _pattern_bindings(T, i + 1, j)]
    binds.sort()
    names = [T[b][1] for b, _, _ in binds]
    if len(names) < MIN_VARS or len(set(names)) != len(names) or any(n in ("self", "Self") for n in names):
        return None
    nameset = set(names)
    attr = set()
    for i in range(st, b_close):
        if T[i][1] == "#" and (T[i + 1][1] == "[" or (T[i + 1][1] == "!" and T[i + 2][1] == "[")):
            o = i + 1 if T[i + 1][1] == "[" else i + 2
            attr.update(range(i, close.get(o, o) + 1))

    def usage(i):
        prv, nxt = T[i - 1][1], T[i + 1][1] if i + 1 <= b_close else ""
        return i not in attr and prv not in (".", "::") and nxt not in ("::", "!") \
            and not (nxt == ":" and _is_field_label(T, i))

    spans = {}
    for slot, (b, lo, hi) in enumerate(binds):
        name = T[b][1]
        for i in range(lo, hi + 1):
            if T[i][0] == "id" and T[i][1] == name and (i == b or usage(i)):
                spans[i] = slot
    for i in range(st, b_close + 1):
        if T[i][0] == "id" and (T[i][1] in nameset and i not in spans or re.fullmatch(r"VAR\d+", T[i][1])):
            return None  # the name survives somewhere (method, field, path, other binding): would leak
    f_lo, f_hi = T[st][2], T[b_close][3]
    for i in range(st, b_close + 1):
        if T[i][0] in ("st", "ch") and any(re.search(r"\b%s\b" % re.escape(n), T[i][1]) for n in names):
            return None
    for c in comments:
        if any(len(n) >= 3 and re.search(r"\b%s\b" % re.escape(n), c) for n in names):
            return None
    parts, cur = [], f_lo
    for i in sorted(spans):
        parts += [text[cur:T[i][2]], f"VAR{spans[i]}"]
        cur = T[i][3]
    parts.append(text[cur:f_hi])
    return {"masked": "".join(parts), "answers": names, "num_vars": len(names)}


def _find_base(raw_root):
    cands = [raw_root, raw_root / "dataset" / "raw", raw_root / "raw"]
    cands += sorted(p.parent for p in raw_root.rglob("library") if p.is_dir())
    for c in cands:
        if all((c / r).is_dir() for r in ROOTS):
            return c
    have = [str(p.relative_to(raw_root)) for p in raw_root.rglob("*") if p.is_dir() and len(p.relative_to(raw_root).parts) <= 2][:12]
    raise FileNotFoundError(f"raw snapshot incomplete under {raw_root}: need {', '.join(ROOTS)} "
                            f"(upload rust-variable-naming-raw.zip, 17 MB / 6,210 files); found dirs: {have}")


def _collect_items(raw_root):
    base = _find_base(Path(raw_root))
    seen, items = set(), []
    for r in ROOTS:
        for f in sorted((base / r).rglob("*.rs")):
            parts = f.relative_to(base).parts
            if FIXTURE_DIRS & set(parts):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            T, comments = tokenize(text)
            starts = [c[0] for c in comments]
            for fn in _functions(T):
                lo, hi = T[fn[0]][2], T[fn[5]][3]
                if not MIN_FN_CHARS <= hi - lo <= MAX_FN_CHARS:
                    continue
                own = [c[1] for c in comments[bisect_left(starts, lo):bisect_left(starts, hi)]]
                res = mask_function(text, T, own, fn)
                if res is None:
                    continue
                skel = re.sub(r"\s+", " ", res["masked"]).strip()
                if skel in seen:
                    continue
                seen.add(skel)
                items.append({"source_file": "/".join(parts), **res})
    items.sort(key=lambda it: (it["source_file"], it["masked"]))
    return items


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare(raw_root, public_dir, private_dir):
    public_dir, private_dir = Path(public_dir), Path(private_dir)
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    items = _collect_items(raw_root)
    if len(items) < 500:
        raise RuntimeError(f"only {len(items)} usable items found, expected >= 500")
    for idx, it in enumerate(items):
        it["id"] = f"rvn_{idx:05d}"

    # Whole files go to one side: a per-function split let a same-file
    # most-common-name probe score 0.178 vs 0.034 file-blind.
    by_file = {}
    for it in items:
        by_file.setdefault(it["source_file"], []).append(it)
    ranked = sorted(by_file, key=lambda f: int(hashlib.sha256(f.encode("utf8")).hexdigest()[:16], 16))
    target, n_test, test_files = max(200, int(round(TEST_FRACTION * len(items)))), 0, set()
    for f in ranked:
        if n_test >= target:
            break
        test_files.add(f)
        n_test += len(by_file[f])

    train, test, answers = [], [], []
    for it in items:
        row = {"id": it["id"], "masked": it["masked"], "num_vars": it["num_vars"]}
        names = "|".join(it["answers"])
        if it["source_file"] in test_files:
            test.append(row)
            answers.append({"id": it["id"], "names": names})
        else:
            train.append({**row, "names": names})
    _write_csv(public_dir / "train.csv", train, ["id", "masked", "num_vars", "names"])
    _write_csv(public_dir / "test.csv", test, ["id", "masked", "num_vars"])
    _write_csv(public_dir / "sample_submission.csv",
               [{"id": r["id"], "names": "|".join(["x"] * r["num_vars"])} for r in test], ["id", "names"])
    _write_csv(private_dir / "answers.csv", answers, ["id", "names"])
    nv = [it["num_vars"] for it in items]
    stats = {"n_train": len(train), "n_test": len(test), "n_total_items": len(items),
             "num_vars_min": min(nv), "num_vars_max": max(nv), "num_vars_mean": sum(nv) / len(nv)}
    with open(public_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--private", required=True)
    a = ap.parse_args()
    prepare(a.raw, a.public, a.private)
