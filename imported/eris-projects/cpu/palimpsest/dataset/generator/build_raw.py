"""Build the raw Palimpsest dataset from the Wikivoyage full-history dump.

    python build_raw.py <hist.xml.7z> <out_dir> [max_articles]

Streams the dump once (via `7z e -so`), selects one churn window per eligible
article, and writes:

    windows.jsonl     the selected revision windows, with text
    attribution.csv   article + revision ids used (CC BY-SA attribution)
    meta.json         dump provenance, selection parameters, counts

Deterministic: articles are emitted in dump order and, if capped, the cap is
applied to a deterministic ordering by title. No randomness, no wall-clock.
"""
import sys, os, json, csv, subprocess, hashlib
import xml.etree.ElementTree as ET

W = 14                                # revisions per window
MIN_SIZE, MAX_SIZE = 2500, 40000      # median bytes across the window
MIN_REVS = 20                         # article must have a real history
MIN_CHURN = 200                       # bytes of total movement
DUMP_URL = ("https://dumps.wikimedia.org/enwikivoyage/latest/"
            "enwikivoyage-latest-pages-meta-history.xml.7z")


def strip_ns(tag):
    return tag.rsplit("}", 1)[-1]


def pick_window(sizes, w=W):
    """Consecutive run of `w` revisions maximising churn / (1 + net drift)."""
    if len(sizes) < w:
        return None
    best, best_i = -1.0, None
    for i in range(len(sizes) - w + 1):
        s = sizes[i:i + w]
        churn = sum(abs(b - a) for a, b in zip(s, s[1:]))
        if churn < MIN_CHURN:
            continue
        score = churn / (1.0 + abs(s[-1] - s[0]))
        if score > best:
            best, best_i = score, i
    return (best_i, best) if best_i is not None else None


def iter_articles(dump):
    """Yield (title, [revisions]) for every main-namespace non-redirect page."""
    proc = subprocess.Popen(["7z", "e", "-so", dump], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=1 << 20)
    title = ns = None
    is_redirect = False
    revs, cur = [], None
    for event, elem in ET.iterparse(proc.stdout, events=("start", "end")):
        tag = strip_ns(elem.tag)
        if event == "start":
            if tag == "page":
                title, ns, is_redirect, revs = None, None, False, []
            elif tag == "revision":
                cur = {}
            elif tag == "redirect":
                is_redirect = True
            continue
        if tag == "title" and cur is None:
            title = elem.text
        elif tag == "ns" and cur is None:
            ns = elem.text
        elif cur is not None:
            if tag == "timestamp":
                cur["ts"] = elem.text
            elif tag == "text":
                cur["text"] = elem.text or ""
                cur["size"] = int(elem.get("bytes") or len(cur["text"].encode("utf8")))
            elif tag == "id" and "revid" not in cur:
                cur["revid"] = elem.text
            elif tag == "comment":
                cur["comment"] = (elem.text or "")[:200]
            elif tag == "revision":
                if "text" in cur:
                    revs.append(cur)
                cur = None
                elem.clear()
        elif tag == "page":
            if ns == "0" and not is_redirect and len(revs) >= MIN_REVS:
                yield title, revs
            revs = []
            elem.clear()
    try:
        proc.kill()
    except Exception:
        pass


def main(dump, out_dir, max_windows=None):
    """Stream windows to disk as they are found. `max_windows` stops the scan
    early after that many windows -- the kept set is then "the first
    max_windows eligible articles in dump (page-id) order", which is
    deterministic and reproducible for a given dump file."""
    os.makedirs(out_dir, exist_ok=True)
    stream_path = os.path.join(out_dir, "windows_stream.jsonl")
    kept = scanned = 0
    with open(stream_path, "w", encoding="utf8") as out:
        for title, revs in iter_articles(dump):
            scanned += 1
            sizes = [r["size"] for r in revs]
            pick = pick_window(sizes)
            if not pick:
                continue
            i, score = pick
            win = revs[i:i + W]
            med = sorted(r["size"] for r in win)[W // 2]
            if not (MIN_SIZE <= med <= MAX_SIZE):
                continue
            out.write(json.dumps(dict(
                title=title, churn_score=round(score, 4), window_start=i,
                n_total_revs=len(revs),
                revs=[dict(revid=r.get("revid"), ts=r.get("ts"),
                           size=r["size"], comment=r.get("comment", ""),
                           text=r["text"]) for r in win]),
                ensure_ascii=False) + "\n")
            kept += 1
            if scanned % 2000 == 0:
                print(f"  scanned {scanned} articles, kept {kept}", flush=True)
                out.flush()
            if max_windows and kept >= max_windows:
                print(f"  early stop at {kept} windows ({scanned} articles scanned)",
                      flush=True)
                break

    # finalise: sort by title for a canonical on-disk order
    rows = [json.loads(l) for l in open(stream_path, encoding="utf8")]
    rows.sort(key=lambda r: r["title"])
    attribution = [(r["title"], rv.get("revid"),
                    f"https://en.wikivoyage.org/w/index.php?oldid={rv.get('revid')}")
                   for r in rows for rv in r["revs"]]
    with open(os.path.join(out_dir, "windows.jsonl"), "w", encoding="utf8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.remove(stream_path)
    with open(os.path.join(out_dir, "attribution.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "revid", "url"])
        w.writerows(sorted(set(attribution)))
    meta = dict(source_url=DUMP_URL, project="en.wikivoyage", license="CC BY-SA 4.0",
                revisions_per_window=W, min_revisions=MIN_REVS, min_churn_bytes=MIN_CHURN,
                median_size_range=[MIN_SIZE, MAX_SIZE],
                articles_scanned=scanned, articles_kept=len(rows),
                revisions_kept=len(rows) * W)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
