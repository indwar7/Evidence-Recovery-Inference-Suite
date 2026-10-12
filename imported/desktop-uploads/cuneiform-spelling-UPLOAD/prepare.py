"""prepare.py - build the public/private split for "Cuneiform Spelling of
Unseen Akkadian Words" from the raw ORACC extraction.

Contract:
    prepare(raw_root, public_dir, private_dir) -> None

Raw input (written by dataset/generator/build_raw.py from the ORACC JSON
exports): ``lines.jsonl`` (one record per text line with its lemmatised words)
and ``texts.jsonl`` (one record per text with catalogue metadata). The script
looks for them directly under raw_root and then anywhere below it.

Deterministic and seedless. Text-level hold-out: a text is an evaluation text
when the SHA-256 bucket of its ORACC text id falls under TEST_PERMILLE/1000.
Every lemmatised Akkadian word in a training text with no restored or
unreadable sign is a training row (capped per normalized-form/spelling pair). An evaluation row is a word in an evaluation text whose
normalized form never occurs in any training text, whose lemma does occur in
training, that is common vocabulary (part of speech is not a proper-noun tag),
has no missing (restored) sign, and is not a bare numeral. Rows whose spelling is identical to the normalized form are not evaluation
rows. Evaluation rows are
capped per lemma and per normalized form so no single verb dominates, then
sub-sampled to at most TEST_MAX rows by SHA-256 bucket.

Row ids are assigned after a shuffle keyed on a salt derived from the raw file
contents, so id order carries no information about text, line, or spelling.
Text ids are published only as opaque aliases built from the same salt.
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
        and w["form"].strip() not in PANDAS_NA and w["norm"].strip() not in PANDAS_NA


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

    # ---- evaluation rows: unseen normalized form, seen lemma, common vocabulary
    cand = [r for r in test_cand
            if r["norm"] not in train_norms and r["lemma"] in train_cf
            and not NAME_POS.match(r["pos"]) and r["pos"] not in SKIP_POS
            and r["spelling"] != r["norm"]]          # a spelling identical to its normalized form is a giveaway
    cand.sort(key=lambda r: _hex(f"{r['textid']}|{r['line']}|{r['pos_in_line']}", salt))
    per_lemma, per_norm, test_rows = Counter(), Counter(), []
    for r in cand:
        if per_lemma[r["lemma"]] >= CAP_PER_LEMMA or per_norm[r["norm"]] >= CAP_PER_NORM:
            continue
        per_lemma[r["lemma"]] += 1
        per_norm[r["norm"]] += 1
        test_rows.append(r)
    test_rows = test_rows[:TEST_MAX]

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

    feat = ["row_id", "doc", "norm", "lemma", "gloss", "pos", "attested", "context", "period", "genre", "provenience"]
    def out_row(r, with_target):
        d = {"row_id": r["row_id"], "doc": doc_alias(r["textid"]), "norm": r["norm"], "lemma": r["lemma"],
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
        "eval_candidates_before_caps": len(cand),
        "train_distinct_norm": len({r["norm"] for r in train_rows}),
        "train_distinct_lemma": len({r["lemma"] for r in train_rows}),
        "test_distinct_lemma": len({r["lemma"] for r in test_rows}),
        "test_pos": dict(Counter(r["pos"] for r in test_rows).most_common()),
        "test_period": dict(Counter(r["period"] for r in test_rows).most_common()),
        "train_period": dict(Counter(r["period"] for r in train_rows).most_common()),
        "test_norm_unseen_in_train": all(r["norm"] not in train_norms for r in test_rows),
        "test_rows_with_attested": sum(1 for r in test_rows if attested(r["lemma"], r["norm"])),
        "train_rows_with_attested": sum(1 for r in train_rows if attested(r["lemma"], r["norm"])),
    }
    with open(public_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    with open(private_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"method": "text-level SHA-256 hold-out; evaluation rows = unseen normalized forms with seen lemma",
                   "test_permille": TEST_PERMILLE, "caps": {"lemma": CAP_PER_LEMMA, "norm": CAP_PER_NORM,
                   "train_pair": TRAIN_CAP_PER_PAIR}, "test_max": TEST_MAX, "counts": stats}, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw"
    pub = sys.argv[2] if len(sys.argv) > 2 else "dataset/public"
    prv = sys.argv[3] if len(sys.argv) > 3 else "dataset/private"
    prepare(raw, pub, prv)
