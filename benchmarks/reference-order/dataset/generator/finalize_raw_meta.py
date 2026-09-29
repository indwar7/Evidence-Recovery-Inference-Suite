"""Generate meta.json and build_meta.json from the finished raw fetch.

Run once build_raw.py has finished (or been stopped) and
wiki_citation_units.jsonl is in its final state for this build.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    units_path = RAW / "wiki_citation_units.jsonl"
    lines = units_path.read_text(encoding="utf-8").splitlines()
    units = [json.loads(l) for l in lines if l.strip()]
    n_lines = len(units)
    size = units_path.stat().st_size
    checksum = md5(units_path)

    titles_path = RAW / "_fetched_titles.json"
    titles = json.loads(titles_path.read_text()) if titles_path.exists() else []

    n_refs_dist = Counter(u["n_refs"] for u in units)
    articles = {u["page"] for u in units}

    meta = {
        "dataset_name": "wikipedia-paragraph-citation-order",
        "source": {
            "publisher": "Wikimedia Foundation / Wikipedia contributors",
            "collection": "English Wikipedia (Good articles / Featured articles)",
            "url": "https://en.wikipedia.org/w/api.php",
            "retrieved_on": "2026-09-25",
        },
        "license": {
            "name": "CC BY-SA 4.0",
            "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
            "publisher_statement": "Creative Commons Attribution-Share Alike 4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
        "files": {
            "wiki_citation_units.jsonl": {
                "n_rows": n_lines,
                "columns": ["page", "text", "n_refs", "refs"],
                "size_bytes": size,
                "md5": checksum,
            }
        },
        "build": {
            "n_articles_attempted": len(titles),
            "n_articles_with_units": len(articles),
            "n_units_extracted": n_lines,
            "n_refs_distribution": {str(k): v for k, v in sorted(n_refs_dist.items())},
        },
        "modifications": [
            "Every <ref>...</ref> marker replaced with the token [CITE].",
            "Wiki markup (templates, wikilinks, bold/italic, HTML) stripped to plain prose.",
            "Each retained citation's {{cite ...}} template parsed into structured fields (title, author, venue, year).",
            "Only paragraphs where every citation is a distinct, fully-inline cite template are kept.",
        ],
    }
    (RAW / "meta.json").write_text(json.dumps(meta, indent=2))

    build_meta = {
        "snapshot_date": "2026-09-25",
        "source": "https://en.wikipedia.org/w/api.php",
        "license": "CC BY-SA 4.0",
        "n_articles_attempted": len(titles),
        "n_articles_with_units": len(articles),
        "n_units_extracted": n_lines,
        "n_refs_distribution": {str(k): v for k, v in sorted(n_refs_dist.items())},
    }
    (RAW / "build_meta.json").write_text(json.dumps(build_meta, indent=2))

    print(json.dumps(meta, indent=2))
    print(json.dumps(build_meta, indent=2))


if __name__ == "__main__":
    main()
