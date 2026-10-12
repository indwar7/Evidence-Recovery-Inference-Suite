"""prepare.py - build the public/private split for "Spelling Unseen Akkadian
Words in Cuneiform Signs" from the raw Oracc extraction.

Contract:
    prepare(raw_root, public_dir, private_dir) -> None

Raw input (written by dataset/generator/build_raw.py from the ORACC JSON
exports): ``lines.jsonl`` (one record per text line with its lemmatised words)
and ``texts.jsonl`` (one record per text with catalogue metadata). The script
looks for them directly under raw_root and then anywhere below it.

Deterministic and seedless. Text-level hold-out: a text is an evaluation text
when the SHA-256 bucket of its Oracc text id falls under TEST_PERMILLE/1000.
Every lemmatised Akkadian word in a training text with no restored or
unreadable sign is a training row (capped per normalized-form/spelling pair). An evaluation row is a word in an evaluation text whose
normalized form never occurs in any training text, whose lemma does occur in
training, that is common vocabulary (part of speech is not a proper-noun tag),
has no missing (restored) sign, and is not a bare numeral. "Never occurs" is
checked against every word of every training text, scorable or not. Held-out
texts that largely duplicate one training text (at least half of their lines of
4+ words occur verbatim, by spelling, in that training text; stock lines found
in more than 50 training texts never count) contribute no evaluation rows. Rows whose spelling is identical to the normalized form are not evaluation
rows. Evaluation rows are
capped per lemma and per normalized form so no single verb dominates, then
sub-sampled to at most TEST_MAX rows in a second, independent salted order (which
is also the row-id order), so neither the cut nor the ids track lemma frequency.

Row ids are assigned after a shuffle keyed on a salt derived from the raw file
contents, so id order carries no information about text, line, or spelling.
Text ids are published only as opaque aliases built from the same salt, and
only for training rows (train_docs.csv).
Standard library only.
"""
from __future__ import annotations
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

TEST_PERMILLE = 200          # share of texts held out, by SHA-256 bucket of the text id
TEST_MAX = 4000              # at most this many evaluation rows
CAP_PER_LEMMA = 6            # evaluation rows per lemma
CAP_PER_NORM = 2             # evaluation rows per normalized form
TRAIN_CAP_PER_PAIR = 40      # training rows per (normalized form, spelling) pair
ATTESTED_K = 4               # attested (normalized form = spelling) pairs of the lemma given per row
NAME_POS = re.compile(r"^[A-Z]N$")   # PN, DN, GN, SN, RN, ... proper-noun tags
SKIP_POS = {"n", "u", "X"}            # numerals, unlemmatised, unknown
TOKEN_RE = re.compile(r"\{[^}]*\}|[^{}\-.:\s]+")
DUP_MIN_LINES = 3            # a held-out text needs this many qualifying lines to be tested for duplication
DUP_MIN_WORDS = 4            # a qualifying line has at least this many words
DUP_SHARE = 0.5              # dropped from evaluation if >= this share of its qualifying lines occur verbatim in ONE training text
DUP_COMMON = 50              # line spellings found in more than this many training texts are stock formulae and never count
# strings pandas reads as NaN by default; a row whose spelling or normalized form is one of these is dropped
PANDAS_NA = {"NA", "N/A", "NULL", "nan", "None", "-nan", "#N/A", "<NA>", "n/a", "null", "NaN", "-NaN", "#NA",
             "1.#IND", "1.#QNAN", "-1.#IND", "-1.#QNAN", "#N/A N/A", ""}


def _find_raw(raw_root: Path, name: str) -> Path:
    direct = raw_root / name
    if direct.exists():
        return direct
    for p in raw_root.rglob(name):
        return p
    raise FileNotFoundError(f"{name} not found under {raw_root}")


def _bucket(key: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest()[:8], 16) % 1000


def _hex(key: str, salt: str) -> str:
    return hashlib.sha256((salt + key).encode("utf-8")).hexdigest()


def _scorable(w: dict) -> bool:
    return bool(w.get("norm")) and str(w.get("lang", "")).startswith("akk") \
        and int(w.get("missing", 0)) == 0 and not w.get("x") \
        and (w.get("pos") or "") not in SKIP_POS and bool(w.get("pos")) and int(w.get("n_signs", 0)) > 0 \
        and bool(TOKEN_RE.findall(w.get("form", ""))) \
        and w["form"].strip() not in PANDAS_NA and w["norm"].strip() not in PANDAS_NA \
        and "[" not in w["norm"] and "]" not in w["norm"]


def _ctx(words: list, i: int) -> str:
    out = []
    for j, w in enumerate(words):
        tok = w.get("norm") or ("?" if (w.get("missing") or w.get("x") or not w.get("norm")) else w.get("form"))
        if not w.get("norm") and (w.get("pos") or "") == "n":
            tok = w.get("form") or "?"
        out.append(f"[{tok}]" if j == i else tok)
    return " ".join(out)


def prepare(raw_root, public_dir, private_dir) -> None:
    raw_root, public_dir, private_dir = Path(raw_root), Path(public_dir), Path(private_dir)
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    lines_path = _find_raw(raw_root, "lines.jsonl")
    texts_path = _find_raw(raw_root, "texts.jsonl")
    raw_bytes = lines_path.read_bytes()
    salt = hashlib.sha256(raw_bytes).hexdigest()[:16]   # never published; derived from raw content

    texts = {}
    with open(texts_path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                t = json.loads(ln)
                texts[t["textid"]] = t
    is_test_text = {tid: _bucket(tid) < TEST_PERMILLE for tid in texts}

    train_rows, test_cand = [], []
    train_norms, train_cf = set(), set()
    train_norms_any = set()                    # every normalized form on ANY word of a training text
    train_line_texts = defaultdict(set)        # spelling hash of a qualifying training line -> training text ids
    heldout_lines = defaultdict(list)          # held-out text id -> spelling hashes of its qualifying lines
    with open(lines_path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            L = json.loads(ln)
            tid = L["textid"]
            if tid not in texts:
                continue
            meta = texts[tid]
            words = L["words"]
            if len(words) >= DUP_MIN_WORDS:
                h = hashlib.blake2b("\x1f".join(w.get("form") or "" for w in words).encode("utf-8"), digest_size=12).digest()
                if is_test_text[tid]:
                    heldout_lines[tid].append(h)
                else:
                    train_line_texts[h].add(tid)
            if not is_test_text[tid]:
                for w in words:
                    if w.get("norm"):
                        train_norms_any.add(w["norm"])
            for i, w in enumerate(words):
                if not _scorable(w):
                    continue
                row = {
                    "textid": tid, "line": L.get("line_label") or "", "pos_in_line": i,
                    "norm": w["norm"], "lemma": w.get("cf") or "", "gloss": w.get("gw") or "",
                    "pos": w.get("pos") or "", "context": _ctx(words, i),
                    "period": meta.get("period") or "", "genre": meta.get("genre") or "",
                    "provenience": meta.get("provenience") or "", "dialect": meta.get("dialect") or "",
                    "spelling": w["form"],
                }
                if is_test_text[tid]:
                    test_cand.append(row)
                else:
                    train_rows.append(row)
                    train_norms.add(w["norm"])
                    if w.get("cf"):
                        train_cf.add(w["cf"])

    # ---- attested spellings per lemma, from training texts only (most frequent first)
    pair_counts = defaultdict(Counter)
    for r in train_rows:
        if r["lemma"]:
            pair_counts[r["lemma"]][(r["norm"], r["spelling"])] += 1
    lemma_att = {cf: [p for p, _ in c.most_common()] for cf, c in pair_counts.items()}

    def attested(lemma: str, exclude_norm: str) -> str:
        out = []
        for n, f in lemma_att.get(lemma, []):
            if n == exclude_norm:      # a row never sees its own normalized form spelled out
                continue
            out.append(f"{n}={f}")
            if len(out) >= ATTESTED_K:
                break
        return " ; ".join(out)

    # ---- held-out texts that largely duplicate one training text (re-editions under a new id,
    #      duplicate composites, parallel exemplars) contribute no evaluation rows
    dup_eval = set()
    for tid, hs in heldout_lines.items():
        if len(hs) < DUP_MIN_LINES:
            continue
        cnt = Counter()
        for h in hs:
            owners = train_line_texts.get(h)
            if owners and len(owners) <= DUP_COMMON:
                cnt.update(owners)
        if cnt and max(cnt.values()) >= DUP_SHARE * len(hs):
            dup_eval.add(tid)

    # ---- evaluation rows: normalized form on no word of any training text, seen lemma, common vocabulary
    cand = [r for r in test_cand
            if r["norm"] not in train_norms_any and r["lemma"] in train_cf
            and not NAME_POS.match(r["pos"]) and r["pos"] not in SKIP_POS
            and r["spelling"] != r["norm"]            # a spelling identical to its normalized form is a giveaway
            and r["textid"] not in dup_eval]
    cand.sort(key=lambda r: _hex(f"{r['textid']}|{r['line']}|{r['pos_in_line']}", salt))
    per_lemma, per_norm, capped = Counter(), Counter(), []
    for r in cand:
        if per_lemma[r["lemma"]] >= CAP_PER_LEMMA or per_norm[r["norm"]] >= CAP_PER_NORM:
            continue
        per_lemma[r["lemma"]] += 1
        per_norm[r["norm"]] += 1
        capped.append(r)
    # the cut to TEST_MAX and the row order use a second, independent key, so neither tracks lemma frequency
    capped.sort(key=lambda r: _hex(f"pick|{r['textid']}|{r['line']}|{r['pos_in_line']}", salt))
    test_rows = capped[:TEST_MAX]

    # ---- training rows: cap per (norm, spelling) pair, then shuffle
    train_rows.sort(key=lambda r: _hex(f"{r['textid']}|{r['line']}|{r['pos_in_line']}", salt))
    pair = Counter()
    kept = []
    for r in train_rows:
        k = (r["norm"], r["spelling"])
        if pair[k] >= TRAIN_CAP_PER_PAIR:
            continue
        pair[k] += 1
        kept.append(r)
    train_rows = kept

    for n, r in enumerate(train_rows, 1):
        r["row_id"] = f"a{n:06d}"
    for n, r in enumerate(test_rows, 1):
        r["row_id"] = f"t{n:05d}"
    alias = {}
    def doc_alias(tid: str) -> str:
        if tid not in alias:
            alias[tid] = "doc_" + _hex(tid, salt)[:10]
        return alias[tid]

    feat = ["row_id", "norm", "lemma", "gloss", "pos", "attested", "context", "period", "genre", "provenience"]
    def out_row(r, with_target):
        d = {"row_id": r["row_id"], "norm": r["norm"], "lemma": r["lemma"],
             "gloss": r["gloss"], "pos": r["pos"], "attested": attested(r["lemma"], r["norm"]),
             "context": r["context"], "period": r["period"], "genre": r["genre"], "provenience": r["provenience"]}
        if with_target:
            d["spelling"] = r["spelling"]
        return d

    with open(public_dir / "train.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=feat + ["spelling"], lineterminator="\n")
        wr.writeheader()
        for r in train_rows:
            wr.writerow(out_row(r, True))
    # text grouping is published for training rows only, so validation can hold out whole texts;
    # evaluation rows carry no grouping key
    with open(public_dir / "train_docs.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row_id", "doc"], lineterminator="\n")
        wr.writeheader()
        for r in train_rows:
            wr.writerow({"row_id": r["row_id"], "doc": doc_alias(r["textid"])})
    with open(public_dir / "test.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=feat, lineterminator="\n")
        wr.writeheader()
        for r in test_rows:
            wr.writerow(out_row(r, False))
    with open(public_dir / "sample_submission.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row_id", "spelling"], lineterminator="\n")
        wr.writeheader()
        for r in test_rows:
            wr.writerow({"row_id": r["row_id"], "spelling": r["norm"]})   # copy baseline
    with open(private_dir / "answers.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row_id", "spelling_true", "norm_src", "lemma_src", "pos_src", "textid_src"], lineterminator="\n")
        wr.writeheader()
        for r in test_rows:
            wr.writerow({"row_id": r["row_id"], "spelling_true": r["spelling"], "norm_src": r["norm"],
                         "lemma_src": r["lemma"], "pos_src": r["pos"], "textid_src": r["textid"]})

    stats = {
        "texts_total": len(texts),
        "texts_train": sum(1 for t in texts if not is_test_text[t]),
        "texts_eval": sum(1 for t in texts if is_test_text[t]),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "eval_texts_dropped_as_duplicates": len(dup_eval),
        "eval_candidates_before_caps": len(cand),
        "eval_candidates_after_caps": len(capped),
        "train_docs": len({r["textid"] for r in train_rows}),
        "test_docs": len({r["textid"] for r in test_rows}),
        "train_distinct_norm": len({r["norm"] for r in train_rows}),
        "train_distinct_lemma": len({r["lemma"] for r in train_rows}),
        "test_distinct_lemma": len({r["lemma"] for r in test_rows}),
        "test_pos": dict(Counter(r["pos"] for r in test_rows).most_common()),
        "train_pos": dict(Counter(r["pos"] for r in train_rows).most_common()),
        "test_period": dict(Counter(r["period"] for r in test_rows).most_common()),
        "train_period": dict(Counter(r["period"] for r in train_rows).most_common()),
        "test_norm_on_no_training_word": all(r["norm"] not in train_norms_any for r in test_rows),
        "test_rows_with_attested": sum(1 for r in test_rows if attested(r["lemma"], r["norm"])),
        "train_rows_with_attested": sum(1 for r in train_rows if attested(r["lemma"], r["norm"])),
    }
    with open(public_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    with open(private_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"method": "text-level SHA-256 hold-out; evaluation rows = normalized forms on no word of any training text, lemma seen; held-out texts duplicating a training text excluded",
                   "duplicate_rule": {"min_lines": DUP_MIN_LINES, "min_words": DUP_MIN_WORDS, "share": DUP_SHARE, "common_line_texts": DUP_COMMON},
                   "test_permille": TEST_PERMILLE, "caps": {"lemma": CAP_PER_LEMMA, "norm": CAP_PER_NORM,
                   "train_pair": TRAIN_CAP_PER_PAIR}, "test_max": TEST_MAX, "counts": stats}, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw"
    pub = sys.argv[2] if len(sys.argv) > 2 else "dataset/public"
    prv = sys.argv[3] if len(sys.argv) > 3 else "dataset/private"
    prepare(raw, pub, prv)
