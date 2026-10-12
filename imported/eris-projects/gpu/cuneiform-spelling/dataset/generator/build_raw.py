#!/usr/bin/env python3
"""build_raw.py - fetch the ORACC JSON corpus exports and extract the raw
word-level records used by this dataset.

Usage:
    python3 build_raw.py --out <raw_dir> [--zip-dir <dir>] [--no-download]

Source: https://oracc.museum.upenn.edu/json/<project>.zip (one zip per ORACC
project/volume; the list is PROJECTS below). Each zip holds one JSON file per
text under corpusjson/ with the editors' lemmatised transliteration (the
"cdl" tree), and catalogue.json with the catalogue metadata.

Fetch method: one worker, 1 s sleep between requests, three attempts per file
with broad exception handling, each zip verified with zipfile.testzip before
use, incremental (a file that already verifies is not re-fetched).

Output (raw_dir):
    lines.jsonl   one record per text line: textid, project, line_label,
                  line_idx, words[] with form, norm, cf, gw, pos, lang,
                  n_signs, missing, damaged, x
    texts.jsonl   one record per text: textid, project, period, genre,
                  subgenre, provenience, dialect, language, script,
                  object_type, ruler, designation, n_lines
    meta.json     build date, project list, counts, SHA-256 of both files
"""
import argparse, collections, hashlib, json, pathlib, re, sys, time, urllib.request, zipfile

BASE = "https://oracc.museum.upenn.edu/json/"
PROJECTS = """saao-saa01 saao-saa02 saao-saa03 saao-saa04 saao-saa05 saao-saa06 saao-saa07 saao-saa08 saao-saa09 saao-saa10
saao-saa11 saao-saa12 saao-saa13 saao-saa14 saao-saa15 saao-saa16 saao-saa17 saao-saa18 saao-saa19 saao-saa20 saao-saa21
saao-saas2 saao-aebp saao-knpp
rinap-rinap1 rinap-rinap2 rinap-rinap3 rinap-rinap4 rinap-rinap5 rinap-rinap5p1
riao-ria1 riao-ria2 riao-ria3 riao-ria4 riao-ria5
ribo-babylon2 ribo-babylon3 ribo-babylon4 ribo-babylon5 ribo-babylon6 ribo-babylon7 ribo-babylon8 ribo-babylon10
cams-gkab cams-akno hbtin blms ccpo aemw-amarna aemw-ugarit aemw-alalakh-idrimi ckst suhu obta obmc
cmawro-cmawr1 cmawro-cmawr2 cmawro-cmawr3 cmawro-maqlu adsd-adart1 adsd-adart2 adsd-adart3 adsd-adart5 adsd-adart6
babcity borsippa btto btmao caspo-akkpm dccmt glass nimrud oimea qcat rimanum asbp-ninmed asbp-rlasb""".split()


def fetch(name, zip_dir):
    zip_dir.mkdir(parents=True, exist_ok=True)
    path = zip_dir / f"{name}.zip"
    if path.exists() and path.stat().st_size > 0:
        try:
            if zipfile.ZipFile(path).testzip() is None:
                return path
        except Exception:
            pass
    for attempt in range(3):
        try:
            req = urllib.request.Request(BASE + f"{name}.zip", headers={"User-Agent": "eris-build/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if zipfile.ZipFile(path).testzip() is None:
                return path
        except Exception as e:  # broad on purpose: ConnectionResetError is not an HTTPError
            print(f"  attempt {attempt + 1} failed for {name}: {e}", file=sys.stderr)
            time.sleep(3)
    return None


def gdl_flags(gdl):
    n = miss = dam = 0
    hasx = False
    def rec(items):
        nonlocal n, miss, dam, hasx
        for s in items:
            if not isinstance(s, dict):
                continue
            if "v" in s or "s" in s or "n" in s:
                n += 1
                b = s.get("break")
                if b == "missing":
                    miss += 1
                elif b == "damaged":
                    dam += 1
                val = s.get("v") or s.get("s") or ""
                if val in ("x", "X") or s.get("x"):
                    hasx = True
            for key in ("seq", "group", "gdl"):
                if isinstance(s.get(key), list):
                    rec(s[key])
    rec(gdl)
    return n, miss, dam, hasx


def walk_text(cdl):
    ev = []
    def rec(n):
        if isinstance(n, dict):
            if n.get("node") == "d" and n.get("type") == "line-start":
                ev.append(("LINE", n.get("label")))
            elif n.get("node") == "l":
                ev.append(("W", n))
            for v in n.values():
                if isinstance(v, (list, dict)):
                    rec(v)
        elif isinstance(n, list):
            for v in n:
                rec(v)
    rec(cdl)
    return ev


META_KEYS = ["period", "genre", "subgenre", "provenience", "dialect", "language", "script", "object_type", "ruler", "designation"]


def extract(zip_paths, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = collections.Counter()
    with open(out_dir / "lines.jsonl", "w", encoding="utf-8") as fl, open(out_dir / "texts.jsonl", "w", encoding="utf-8") as ft:
        for zp in zip_paths:
            z = zipfile.ZipFile(zp)
            names = z.namelist()
            cat = {}
            for c in (n for n in names if n.endswith("/catalogue.json")):
                try:
                    cat.update(json.loads(z.read(c).decode("utf-8")).get("members", {}))
                except Exception as e:
                    print("catalogue error", zp, c, e, file=sys.stderr)
            for n in sorted(n for n in names if "/corpusjson/" in n and n.endswith(".json")):
                try:
                    d = json.loads(z.read(n).decode("utf-8"))
                except Exception:
                    stats["json_err"] += 1
                    continue
                tid = d.get("textid") or pathlib.Path(n).stem
                if tid in seen:
                    stats["dup_text"] += 1
                    continue
                proj = d.get("project", "")
                meta = cat.get(tid, {})
                lines, cur, words = [], None, []
                for e in walk_text(d.get("cdl", [])):
                    if e[0] == "LINE":
                        if cur is not None and words:
                            lines.append((cur, words))
                        cur, words = e[1], []
                    else:
                        f = e[1].get("f", {})
                        form = f.get("form")
                        if not form:
                            continue
                        ns, miss, dam, hasx = gdl_flags(f.get("gdl", []))
                        words.append({"form": form, "norm": f.get("norm"), "cf": f.get("cf"), "gw": f.get("gw"),
                                      "pos": f.get("pos"), "lang": f.get("lang", ""), "n_signs": ns,
                                      "missing": miss, "damaged": dam, "x": hasx})
                if cur is not None and words:
                    lines.append((cur, words))
                if not lines:
                    stats["no_lines"] += 1
                    continue
                seen.add(tid)
                rec = {"textid": tid, "project": proj}
                rec.update({k: meta.get(k) for k in META_KEYS})
                rec["n_lines"] = len(lines)
                ft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for li, (label, ws) in enumerate(lines):
                    fl.write(json.dumps({"textid": tid, "project": proj, "line_label": label, "line_idx": li, "words": ws}, ensure_ascii=False) + "\n")
                    stats["lines"] += 1
                    stats["words"] += len(ws)
            stats["texts"] = len(seen)
            print(f"{pathlib.Path(zp).name}: texts {len(seen)} lines {stats['lines']}", flush=True)
    return stats


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip-dir", default=None)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    zip_dir = pathlib.Path(a.zip_dir) if a.zip_dir else out.parent / "oracc_zips"
    paths = []
    for name in PROJECTS:
        p = zip_dir / f"{name}.zip"
        if a.no_download:
            if p.exists():
                paths.append(p)
            else:
                print("missing", p, file=sys.stderr)
        else:
            got = fetch(name, zip_dir)
            if got:
                paths.append(got)
                print("fetched", name, flush=True)
            time.sleep(1.0)
    stats = extract(paths, out)
    meta = {"source": BASE, "projects": PROJECTS, "zips_used": [p.name for p in paths],
            "built": time.strftime("%Y-%m-%d"), "counts": dict(stats),
            "sha256": {"lines.jsonl": sha256(out / "lines.jsonl"), "texts.jsonl": sha256(out / "texts.jsonl")},
            "license": "CC BY-SA 3.0 (ORACC project pages and per-text credits); the JSON exports themselves state CC0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/"}
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
