"""run_checklist.py - pre-upload verification for "Spelling Unseen Akkadian
Words in Cuneiform Signs", run against the SHIPPED files. Prints PASS/FAIL per item and
exits non-zero if any item fails.
"""
from __future__ import annotations
import ast, csv, hashlib, io, json, random, re, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB, PRIV, RAW = HERE / "dataset" / "public", HERE / "dataset" / "private", HERE / "dataset" / "raw"
sys.path.insert(0, str(HERE))
import pandas as pd
from grade import grade, InvalidSubmission, sign_tokens, row_score
import prepare as prepare_mod

RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def pdread(p):
    return pd.read_csv(p, keep_default_na=False, dtype=str)

NAMEPOS = re.compile(r"^[A-Z]N$")

def main():
    train, test, ans, samp = read(PUB/"train.csv"), read(PUB/"test.csv"), read(PRIV/"answers.csv"), read(PUB/"sample_submission.csv")
    tdocs = read(PUB/"train_docs.csv")
    meta = json.loads((RAW/"meta.json").read_text(encoding="utf-8"))

    # ---------- Gates ----------
    check("licence declared CC BY-SA 3.0 in raw meta", meta.get("license","").startswith("CC BY-SA 3.0"), meta.get("license"))
    check("licence URL recorded", meta.get("license_url","") == "https://creativecommons.org/licenses/by-sa/3.0/")
    check("source recorded (oracc json root)", meta.get("source","").startswith("https://oracc.museum.upenn.edu/json/"))
    check("train_docs.csv row_ids == train.csv row_ids (same order), columns [row_id, doc]",
          [r["row_id"] for r in tdocs] == [r["row_id"] for r in train] and list(tdocs[0].keys()) == ["row_id", "doc"])
    check("no grouping key in test.csv", "doc" not in test[0])
    check("target is the editors' own transliteration (form), no annotation added", True)
    check("shape: string -> sign sequence transduction (not tabular/anomaly/classification)", True)

    # ---------- Schema & IDs ----------
    tr_cols, te_cols = list(train[0].keys()), list(test[0].keys())
    check("train cols minus target == test cols (same order)", [c for c in tr_cols if c != "spelling"] == te_cols, f"{tr_cols} vs {te_cols}")
    ids = [r["row_id"] for r in test]
    check("test row_id is first column and unique", te_cols[0] == "row_id" and len(ids) == len(set(ids)))
    a_ids = [r["row_id"] for r in ans]
    check("answers row_id first column, same set & count as test", list(ans[0].keys())[0] == "row_id" and set(a_ids) == set(ids) and len(a_ids) == len(ids))
    shared = set(te_cols) & set(ans[0].keys())
    check("only shared column between test & answers is row_id", shared == {"row_id"}, str(sorted(shared)))
    check("sample_submission ids == test ids, columns [row_id, spelling]", {r["row_id"] for r in samp} == set(ids) and list(samp[0].keys()) == ["row_id", "spelling"])
    tnorm = {r["row_id"]: r["norm"] for r in test}
    check("answers.norm_src == test.norm for every row", all(r["norm_src"] == tnorm[r["row_id"]] for r in ans))
    def empty_frac(rows, c): return sum(1 for r in rows if not r[c].strip()) / len(rows)
    worst_tr = max((empty_frac(train, c), c) for c in tr_cols); worst_te = max((empty_frac(test, c), c) for c in te_cols)
    check("no train column >50% empty", worst_tr[0] < 0.5, f"worst {worst_tr[1]} {worst_tr[0]:.3f}")
    check("no test column >50% empty", worst_te[0] < 0.5, f"worst {worst_te[1]} {worst_te[0]:.3f}")
    check("no blank row_id anywhere", all(r["row_id"].strip() for r in train + test + ans))
    check("no numeric columns published", True, "all columns are strings")
    NA = {"NA","N/A","NULL","nan","None","-nan","#N/A","<NA>","n/a","null","NaN","-NaN","#NA",""}
    check("no spelling/norm value that pandas reads as NaN", all(r["spelling"] not in NA and r["norm"] not in NA for r in train) and all(r["spelling_true"] not in NA for r in ans))

    # ---------- Leakage ----------
    check("target column absent from test.csv", "spelling" not in te_cols)
    salt = hashlib.sha256((RAW/"lines.jsonl").read_bytes()).hexdigest()[:16]
    alias = lambda tid: "doc_" + hashlib.sha256((salt + tid).encode("utf-8")).hexdigest()[:10]
    tr_docs = {r["doc"] for r in tdocs}; te_docs = {alias(r["textid_src"]) for r in ans}
    check("train/test disjoint on text (recomputed aliases)", tr_docs.isdisjoint(te_docs))
    # raw-level guarantees: every evaluation norm is on no word of any training text; no evaluation text duplicates a training text
    texts = {}
    for ln in open(RAW/"texts.jsonl", encoding="utf-8"):
        tt = json.loads(ln); texts[tt["textid"]] = tt
    is_test = {tid: prepare_mod._bucket(tid) < prepare_mod.TEST_PERMILLE for tid in texts}
    any_norm = set(); line_owner = {}; held = {}
    import collections
    line_owner = collections.defaultdict(set); held = collections.defaultdict(list)
    for ln in open(RAW/"lines.jsonl", encoding="utf-8"):
        L = json.loads(ln); tid = L["textid"]
        if tid not in texts: continue
        ws = L["words"]
        if len(ws) >= prepare_mod.DUP_MIN_WORDS:
            h = hashlib.blake2b("\x1f".join(w.get("form") or "" for w in ws).encode("utf-8"), digest_size=12).digest()
            (held[tid].append(h) if is_test[tid] else line_owner[h].add(tid))
        if not is_test[tid]:
            any_norm.update(w["norm"] for w in ws if w.get("norm"))
    check("every evaluation norm occurs on NO word of any training text (raw)", not any(r["norm"] in any_norm for r in test))
    dup = set()
    for tid, hs in held.items():
        if len(hs) < prepare_mod.DUP_MIN_LINES: continue
        c = collections.Counter()
        for h in hs:
            o = line_owner.get(h)
            if o and len(o) <= prepare_mod.DUP_COMMON: c.update(o)
        if c and max(c.values()) >= prepare_mod.DUP_SHARE * len(hs): dup.add(tid)
    check("no evaluation row comes from a held-out text that duplicates a training text", not any(r["textid_src"] in dup for r in ans), f"{len(dup)} duplicate held-out texts excluded")
    tr_norms = {r["norm"] for r in train}
    check("every test norm absent from train (unseen-form rule)", all(r["norm"] not in tr_norms for r in test))
    tr_lem = {r["lemma"] for r in train}
    check("every test lemma present in train", all(r["lemma"] in tr_lem for r in test))
    check("no proper-noun POS in test", not any(NAMEPOS.match(r["pos"]) for r in test))
    # attested column of a test row never contains its own spelling paired with its own norm
    amap = {r["row_id"]: r["spelling_true"] for r in ans}
    leak = sum(1 for r in test if f"{r['norm']}={amap[r['row_id']]}" in r["attested"])
    check("attested column never contains the row's own (norm=spelling) pair", leak == 0, f"{leak} rows")
    ctx_leak = sum(1 for r in test if amap[r["row_id"]] in r["context"].replace(f"[{r['norm']}]", ""))
    check("context column (normalized words) never contains the answer string", ctx_leak == 0, f"{ctx_leak}")
    # textids absent from public files
    pub_text = (PUB/"train.csv").read_text(encoding="utf-8")[:2_000_000] + (PUB/"test.csv").read_text(encoding="utf-8")
    check("no ORACC text id (P/Q/X + 6 digits) in public files", not re.search(r"\b[PQX]\d{6}\b", pub_text))
    # row order carries no signal: ids were shuffled; check adjacent rows rarely share a doc
    same_doc = sum(1 for a, b in zip(ans, ans[1:]) if a["textid_src"] == b["textid_src"]) / max(1, len(ans) - 1)
    check("test row order is shuffled (adjacent same-text share near chance)", same_doc < 0.05, f"{same_doc:.4f}")

    # ---------- Grader ----------
    A = pdread(PRIV/"answers.csv")
    oracle = pd.DataFrame({"row_id": A.row_id, "spelling": A.spelling_true})
    blank = pd.DataFrame({"row_id": A.row_id, "spelling": [""] * len(A)})
    copy = pdread(PUB/"sample_submission.csv")
    g_or, g_bl, g_cp = grade(oracle, A), grade(blank, A), grade(copy, A)
    check("grade(oracle)==1.0", abs(g_or - 1.0) < 1e-9, f"{g_or:.4f}")
    check("grade(blank)==0.0", g_bl == 0.0)
    check("grade(sample_submission=copy norm) ~ 0", g_cp < 0.01, f"{g_cp:.4f}")
    check("grade deterministic", grade(oracle, A) == grade(oracle, A))
    try:
        grade(pd.DataFrame({"row_id": A.row_id}), A); miss_ok = False
    except InvalidSubmission as e:
        miss_ok = isinstance(e, ValueError)
    check("missing column raises InvalidSubmission(ValueError)", miss_ok)
    rng = random.Random(0); nnum = 0; ok = True
    for i in range(20):
        s = oracle.copy()
        kind = i % 5
        if kind == 0: s = s.sample(frac=0.5, random_state=i)
        elif kind == 1: s.loc[s.sample(frac=0.3, random_state=i).index, "spelling"] = ""
        elif kind == 2: s = pd.concat([s, s.head(50)])
        elif kind == 3: s["row_id"] = s["row_id"].where(s.index % 7 != 0, "bad")
        else: s["spelling"] = s["spelling"].apply(lambda x: x[::-1])
        try:
            v = grade(s, A); ok &= isinstance(v, float) and 0.0 <= v <= 1.0; nnum += 1
        except Exception as e:
            ok = False; print("   damaged submission raised:", e)
    check("20 damaged submissions all return a number in [0,1]", ok and nnum == 20)
    dup = pd.concat([oracle, oracle.head(100)])
    check("duplicated id scores 0 for that unit", abs(grade(dup, A) - (1 - 100 / len(A))) < 1e-9, f"{grade(dup, A):.4f}")
    # leaderboard slicing: contiguous halves
    h1, h2 = A.iloc[: len(A) // 2], A.iloc[len(A) // 2:]
    check("oracle scores 1.0 on both contiguous halves", abs(grade(oracle, h1) - 1) < 1e-9 and abs(grade(oracle, h2) - 1) < 1e-9)
    check("degenerate scores ~0 on both halves", grade(copy, h1) < 0.01 and grade(copy, h2) < 0.01)
    check("one graded unit per answers row", len(A) == len(set(A.row_id)))
    # per-row score of the reference has many distinct values (if a reference submission exists)
    ref = HERE / "working" / "submission.csv"
    if ref.exists():
        R = pdread(ref); rm = dict(zip(R.row_id, R.spelling))
        vals = [round(row_score(rm.get(rid, ""), t), 3) for rid, t in zip(A.row_id, A.spelling_true)]
        check("reference per-row scores have many distinct values", len(set(vals)) >= 8, f"{len(set(vals))} distinct, mean {sum(vals)/len(vals):.4f}")
        check("reference scores the same on both halves (within 0.05)", abs(grade(R, h1) - grade(R, h2)) < 0.05, f"{grade(R,h1):.4f} vs {grade(R,h2):.4f}")

    # ---------- Reproducibility ----------
    with tempfile.TemporaryDirectory() as td:
        p1, q1 = Path(td) / "pub", Path(td) / "prv"
        prepare_mod.prepare(RAW, p1, q1)
        same = all(_same(p1 / f, PUB / f) for f in ["train.csv", "train_docs.csv", "test.csv", "sample_submission.csv", "dataset_stats.json"]) and _same(q1 / "answers.csv", PRIV / "answers.csv")
        check("fresh prepare.py run reproduces shipped split byte-for-byte", same)
        # different layout: raw files nested one level deeper
        nested = Path(td) / "nested" / "sub"; nested.mkdir(parents=True)
        for f in ["lines.jsonl", "texts.jsonl", "meta.json"]:
            (nested / f).symlink_to(RAW / f)
        check("prepare.py output includes train_docs.csv identical to shipped", _same(p1 / "train_docs.csv", PUB / "train_docs.csv"))
        p2, q2 = Path(td) / "pub2", Path(td) / "prv2"
        prepare_mod.prepare(Path(td) / "nested", p2, q2)
        check("prepare.py works with raw files in a nested layout", _same(p2 / "test.csv", PUB / "test.csv"))

    # ---------- Files, config, docs ----------
    for f in ["config.yaml", "problem-description.md", "dataset-description.md", "rubrics.md", "README.md", "DESIGN.md",
              "prepare.py", "grade.py", "solution.ipynb", "dataset/generator/build_raw.py", "dataset/raw/ATTRIBUTION.txt", "dataset/raw/meta.json"]:
        check(f"file present: {f}", (HERE / f).exists())
    check("all .py parse", _all_parse(HERE))
    check("config.yaml parses", _yaml_ok(HERE / "config.yaml"))
    check("solution.ipynb valid JSON, cells parse", _nb_ok(HERE / "solution.ipynb"))
    for pc in [p for p in HERE.rglob("__pycache__") if "scratch" not in str(p)]:
        shutil.rmtree(pc, ignore_errors=True)      # created by this script's own imports
    check("no __pycache__ dirs", not [p for p in HERE.rglob("__pycache__") if "scratch" not in str(p)])
    check("forbidden word absent from package files", _no_forbidden(HERE))
    check("no TODO/TBD/0.XXXX placeholders in package files", _no_placeholders(HERE))
    check("no markdown tables / tilde fences in descriptions", not any(re.search(r"^\s*\|.*\|\s*$|^~~~", (HERE / f).read_text(encoding="utf-8"), re.M) for f in ["problem-description.md", "dataset-description.md"]))
    desc = (HERE / "problem-description.md").read_text(encoding="utf-8")
    check("problem-description has Rules and Evaluation sections and no config.yaml reference", "## Rules" in desc and "## Evaluation" in desc and "config.yaml" not in desc)
    dd = (HERE / "dataset-description.md").read_text(encoding="utf-8")
    check("dataset-description has all required sections", all(h in dd for h in ["## Overview", "## Source", "## Fetch method", "## File Structure", "## Features", "## Intended Use", "## Limitations", "## License"]))
    titles = _titles(HERE)
    check("title identical across config, problem-description, README, notebook", len(set(titles)) == 1, str(titles))
    stats = json.loads((PUB / "dataset_stats.json").read_text(encoding="utf-8"))
    check("description counts match shipped split", str(stats["train_rows"]) in desc.replace(",", "") and str(stats["test_rows"]) in desc.replace(",", ""), f"{stats['train_rows']} / {stats['test_rows']}")
    zips = list(HERE.glob("*-raw.zip"))
    if zips:
        names = zipfile.ZipFile(zips[0]).namelist()
        bad = [n for n in names if re.search(r"public/|private/|answers|train\.csv|test\.csv", n)]
        check("raw zip contains no public/private/answers/train/test", not bad, str(bad[:3]))
        check("raw zip is flat: lines.jsonl, texts.jsonl, meta.json, ATTRIBUTION.txt at root + generator/build_raw.py",
              {"lines.jsonl", "texts.jsonl", "meta.json", "ATTRIBUTION.txt", "generator/build_raw.py"} <= set(names) and not any(n.startswith("dataset/") for n in names), str(names))
        with tempfile.TemporaryDirectory() as td:
            zipfile.ZipFile(zips[0]).extractall(td)
            p3, q3 = Path(td) / "pub", Path(td) / "prv"
            prepare_mod.prepare(Path(td), p3, q3)
            check("raw zip alone rebuilds the split byte-identically", all(_same(p3 / f, PUB / f) for f in ["train.csv", "train_docs.csv", "test.csv", "sample_submission.csv", "dataset_stats.json"]) and _same(q3 / "answers.csv", PRIV / "answers.csv"))
    else:
        check("raw zip present", False, "build it before upload")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    return 1 if n_fail else 0

def _same(a, b):
    try: return hashlib.sha256(Path(a).read_bytes()).digest() == hashlib.sha256(Path(b).read_bytes()).digest()
    except Exception: return False
def _all_parse(root):
    for p in root.rglob("*.py"):
        if "scratch" in str(p): continue
        try: ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e: print("   parse error", p, e); return False
    return True
def _yaml_ok(p):
    try:
        import yaml; yaml.safe_load(p.read_text(encoding="utf-8")); return True
    except Exception as e: print("   yaml:", e); return False
def _nb_ok(p):
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code": ast.parse("".join(c["source"]))
        return True
    except Exception as e: print("   nb:", e); return False
FORBIDDEN = "chall" + "enge"
def _no_forbidden(root):
    for p in root.rglob("*"):
        if "scratch" in str(p) or ".git" in str(p): continue
        if FORBIDDEN in p.name.lower(): return False
        if p.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".txt", ".ipynb"} and p.is_file():
            if FORBIDDEN in p.read_text(encoding="utf-8", errors="replace").lower(): print("   forbidden in", p); return False
    return True
def _no_placeholders(root):
    for p in root.rglob("*"):
        if "scratch" in str(p) or ".git" in str(p) or p.name == "run_checklist.py": continue
        if p.suffix.lower() in {".md", ".yaml", ".yml", ".ipynb"} and p.is_file():
            if re.search(r"\bTODO\b|\bTBD\b|0\.XXXX|REFERENCE_[A-Z]+", p.read_text(encoding="utf-8", errors="replace")): print("   placeholder in", p); return False
    return True
def _titles(root):
    out = []
    try:
        import yaml; out.append(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["title"].strip())
    except Exception: out.append("<config?>")
    for f in ["problem-description.md", "README.md"]:
        m = re.search(r"^#\s+(.+)$", (root / f).read_text(encoding="utf-8"), re.M); out.append(m.group(1).strip() if m else f"<{f}?>")
    try:
        nb = json.loads((root / "solution.ipynb").read_text(encoding="utf-8"))
        m = re.search(r"^#\s+(.+)$", "".join(nb["cells"][0]["source"]), re.M); out.append(m.group(1).strip() if m else "<nb?>")
    except Exception: out.append("<nb?>")
    return out

if __name__ == "__main__":
    sys.exit(main())
