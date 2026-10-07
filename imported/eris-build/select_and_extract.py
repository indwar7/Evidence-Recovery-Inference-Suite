"""Select 360 qualifying books deterministically and extract a 45-paragraph
excerpt from each. Produces:
  - selected_books.csv (raw_slot_id, gutenberg_id, title, author, downloads, subjects)
  - book_excerpts.csv  (raw_slot_id, paragraph_index, text)
"""
import csv
import glob
import hashlib
import re

import numpy as np

N_SELECT = 360
EXCERPT_LEN = 45


def stable_seed(tag: str) -> int:
    digest = hashlib.sha256(tag.encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def strip_boilerplate(text: str) -> str:
    m_start = re.search(
        r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text, re.IGNORECASE | re.DOTALL,
    )
    m_end = re.search(
        r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        text, re.IGNORECASE,
    )
    start = m_start.end() if m_start else 0
    end = m_end.start() if m_end else len(text)
    return text[start:end]


def paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    body = strip_boilerplate(text)
    parts = re.split(r"\n\s*\n+", body)
    out = []
    for p in parts:
        p = " ".join(p.split())
        if 40 <= len(p) <= 700 and not p.isupper() and p.count(" ") >= 5:
            out.append(p)
    return out


def main():
    manifest_rows = list(csv.DictReader(open("fetched_manifest.csv", encoding="utf-8")))
    # bring in real subjects metadata from the SPGC catalog for genre bucketing
    subjects_by_id = {}
    with open("spgc_metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subjects_by_id[row["id"]] = row["subjects"]

    qualifying = []
    for row in manifest_rows:
        gid = row["id"][2:]
        path = f"raw_books/{gid}.txt"
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except FileNotFoundError:
            continue
        paras = paragraphs(text)
        if len(paras) >= 75:
            qualifying.append((row, paras))

    print(f"qualifying books: {len(qualifying)}")
    assert len(qualifying) >= N_SELECT, "not enough qualifying books"

    rng = np.random.default_rng(stable_seed("skeinsplit:book-select"))
    idx = rng.permutation(len(qualifying))[:N_SELECT]
    idx = sorted(idx.tolist())  # stable order by original discovery order, then shuffle slots
    chosen = [qualifying[i] for i in idx]

    slot_rng = np.random.default_rng(stable_seed("skeinsplit:slot-order"))
    slot_order = slot_rng.permutation(N_SELECT)

    selected_meta = []
    excerpt_rows = []
    for slot_pos, chosen_idx in enumerate(slot_order):
        row, paras = chosen[chosen_idx]
        gid = row["id"][2:]
        slot_id = f"BOOK{slot_pos + 1:04d}"
        n = len(paras)
        max_start = n - EXCERPT_LEN
        lo = int(0.12 * n)
        hi = max(lo, max_start)
        off_rng = np.random.default_rng(stable_seed(f"skeinsplit:offset:{gid}"))
        start = int(off_rng.integers(lo, hi + 1)) if hi > lo else 0
        excerpt = paras[start:start + EXCERPT_LEN]
        assert len(excerpt) == EXCERPT_LEN, (gid, len(excerpt))

        selected_meta.append({
            "raw_slot_id": slot_id,
            "gutenberg_id": gid,
            "title": row["title"],
            "author": row["author"],
            "downloads": row["downloads"],
            "subjects": subjects_by_id.get(row["id"], ""),
        })
        for i, p in enumerate(excerpt):
            excerpt_rows.append((slot_id, i, p))

    with open("selected_books.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_slot_id", "gutenberg_id", "title",
                                           "author", "downloads", "subjects"])
        w.writeheader()
        for r in selected_meta:
            w.writerow(r)

    with open("book_excerpts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_slot_id", "paragraph_index", "text"])
        for row in excerpt_rows:
            w.writerow(row)

    print(f"wrote selected_books.csv ({len(selected_meta)} books)")
    print(f"wrote book_excerpts.csv ({len(excerpt_rows)} paragraphs)")


if __name__ == "__main__":
    main()
