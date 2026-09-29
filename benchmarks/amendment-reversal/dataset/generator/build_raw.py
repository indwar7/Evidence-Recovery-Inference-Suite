#!/usr/bin/env python3
"""Fetch before/after regulation text pairs from the eCFR versioner API.

Source: Electronic Code of Federal Regulations, Office of the Federal Register
        (National Archives and Records Administration) and the U.S. Government
        Publishing Office.
        https://www.ecfr.gov/api/versioner/v1/titles.json

What is fetched
---------------
The eCFR versioner API records, for every CFR section, each date on which that
section was amended (`amendment_date`), whether the amendment was substantive,
and whether the section was removed. For a section with two consecutive
substantive amendment dates d1 < d2, the API can return the full section text
as it read on each date. That gives a genuine (before, after) pair whose
boundary is the archive's own recorded amendment event -- not a boundary this
script invents.

Only the pair is taken from the archive. No annotation is created, no model is
called, and no text is rewritten: `text_before` and `text_after` are the
agency's own published wording on the two dates the archive recorded.

Deliberate fetch choices (see dataset-description.md "Fetch method")
--------------------------------------------------------------------
- ONE worker, never parallel. The API throttles aggressively under
  concurrency and a single patient worker finishes sooner.
- SLEEP_SECONDS between every request.
- The /full/ endpoint REQUIRES response compression -- without an
  Accept-Encoding header that permits gzip it returns HTTP 406. This is not
  documented prominently and cost a full debugging cycle to find.
- Broad `except Exception` in the retry loop. ConnectionResetError and
  TimeoutError are not HTTPError/URLError subclasses and would otherwise kill
  a long run outright.
- Each section pair is appended to the output file as soon as it is parsed, so
  an interrupted run loses at most the pair in flight.

Usage
-----
    python build_raw.py --out ../raw
    python build_raw.py --out ../raw --titles 21,40,29,42,49 --since 2023-01-01
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

API = "https://www.ecfr.gov/api/versioner/v1"
SOURCE_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"
UA = "eris-dataset-builder/1.0 (research benchmark construction)"

SLEEP_SECONDS = 1.2
MAX_RETRIES = 5
TIMEOUT = 90

# Titles sampled for breadth across regulatory subject matter. Chosen because
# each is large and actively amended; no title is included or excluded on the
# basis of what its amendments say.
DEFAULT_TITLES = [21, 40, 29, 42, 49, 45, 7, 14, 30, 33]


def _fetch(url, expect_json=True):
    """GET with gzip enabled, bounded retries, broad exception catching."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA,
                              "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                elif resp.headers.get("Content-Encoding") == "deflate":
                    import zlib
                    raw = zlib.decompress(raw)
                return json.loads(raw) if expect_json else raw.decode(
                    "utf8", errors="replace")
        except urllib.error.HTTPError as e:
            # 404 means this section did not exist on that date -- not an error
            # worth retrying, just an unusable pair.
            if e.code in (404, 400):
                return None
            last = e
        except Exception as e:          # ConnectionReset, Timeout, JSON, ...
            last = e
        time.sleep(SLEEP_SECONDS * (attempt + 1))
    sys.stderr.write(f"  give up after {MAX_RETRIES}: {url} ({last})\n")
    return None


def strip_xml(x):
    """XML -> plain text. Structural markup only; no wording is altered."""
    if not x:
        return ""
    x = re.sub(r"<\?xml[^>]*\?>", " ", x)
    x = re.sub(r"<!--.*?-->", " ", x, flags=re.S)
    x = re.sub(r"<[^>]+>", " ", x)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#8217;", "'"), ("&#8220;", '"'),
                 ("&#8221;", '"'), ("&nbsp;", " "), ("&#167;", "§")):
        x = x.replace(a, b)
    x = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), x)
    return re.sub(r"\s+", " ", x).strip()


# The eCFR inserts a forward-reference banner into the current text of a
# section whose amendment has been published but not yet incorporated. It
# names the Federal Register citation and the amendment date, so it would hand
# a solver the answer's date directly. Removed from both sides.
AMEND_BANNER = re.compile(
    r"Link to an amendment published at [^.]{0,120}\.\s*", re.I)
EFF_NOTE = re.compile(
    r"(Effective date note:|Editorial Note:)[^.]{0,200}\.\s*", re.I)


def clean_section_text(t):
    t = AMEND_BANNER.sub(" ", t)
    t = EFF_NOTE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def section_text(title, date, part, identifier):
    url = (f"{API}/full/{date}/title-{title}.xml"
           f"?part={part}&section={identifier}")
    xml = _fetch(url, expect_json=False)
    if xml is None:
        return None
    if "requires response compression" in xml[:300]:
        raise RuntimeError("compression header rejected by API")
    return clean_section_text(strip_xml(xml))


def versions_for_title(title, since):
    url = f"{API}/versions/title-{title}.json?issue_date%5Bgte%5D={since}"
    d = _fetch(url)
    return (d or {}).get("content_versions", [])


def build(out_dir, titles, since, max_pairs_per_title, limit_total):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "amendments.jsonl")
    written = 0
    seen_pair = set()
    per_title = {}

    with open(out_path, "w", encoding="utf8") as fh:
        for title in titles:
            if limit_total and written >= limit_total:
                break
            sys.stderr.write(f"title {title}: listing versions\n")
            cv = versions_for_title(title, since)
            time.sleep(SLEEP_SECONDS)

            # Group substantive, non-removal amendments by section, then walk
            # consecutive amendment dates to form (before, after) pairs.
            by_sec = defaultdict(set)
            meta = {}
            for v in cv:
                if not v.get("substantive") or v.get("removed"):
                    continue
                ident = v.get("identifier")
                part = v.get("part")
                ad = v.get("amendment_date")
                if not (ident and part and ad):
                    continue
                by_sec[ident].add(ad)
                meta[ident] = {"part": part, "name": v.get("name", ""),
                               "subpart": v.get("subpart")}

            made = 0
            for ident in sorted(by_sec):
                if max_pairs_per_title and made >= max_pairs_per_title:
                    break
                if limit_total and written >= limit_total:
                    break
                dates = sorted(by_sec[ident])
                if len(dates) < 2:
                    continue
                part = meta[ident]["part"]
                for d1, d2 in zip(dates, dates[1:]):
                    if max_pairs_per_title and made >= max_pairs_per_title:
                        break
                    key = (title, ident, d1, d2)
                    if key in seen_pair:
                        continue
                    seen_pair.add(key)

                    before = section_text(title, d1, part, ident)
                    time.sleep(SLEEP_SECONDS)
                    if not before:
                        continue
                    after = section_text(title, d2, part, ident)
                    time.sleep(SLEEP_SECONDS)
                    if not after:
                        continue
                    if before == after:
                        continue                # no observable change
                    if len(before) < 200 or len(after) < 200:
                        continue                # too little text to reason over
                    if len(before) > 20000 or len(after) > 20000:
                        continue                # outlier-length sections

                    rec = {
                        "record_id": f"t{title}-{ident}-{d1}-{d2}",
                        "title": title,
                        "part": part,
                        "section": ident,
                        "section_name": meta[ident]["name"],
                        "subpart": meta[ident]["subpart"],
                        "date_before": d1,
                        "date_after": d2,
                        "text_before": before,
                        "text_after": after,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()              # incremental: survive interruption
                    written += 1
                    made += 1
                    if written % 25 == 0:
                        sys.stderr.write(f"  wrote {written}\n")
            per_title[title] = made
            sys.stderr.write(f"title {title}: {made} pairs\n")

    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    meta = {
        "generated_by": "build_raw.py",
        "source": "Electronic Code of Federal Regulations (eCFR)",
        "source_url": SOURCE_URL,
        "api_base": API,
        "publisher": ("Office of the Federal Register, National Archives and "
                      "Records Administration; U.S. Government Publishing "
                      "Office"),
        "licence": "U.S. Government Work (17 U.S.C. 105) - public domain",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "titles_requested": titles,
        "issue_date_gte": since,
        "pairs_per_title": per_title,
        "n_pairs": written,
        "amendments_sha256": sha,
        "filters": {
            "substantive_only": True,
            "removed_excluded": True,
            "identical_pairs_excluded": True,
            "min_chars_per_side": 200,
            "max_chars_per_side": 20000,
            "amendment_banner_stripped": True,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "ATTRIBUTION.txt"), "w",
              encoding="utf8") as f:
        f.write(
            "Electronic Code of Federal Regulations (eCFR)\n"
            "Office of the Federal Register, National Archives and Records\n"
            "Administration; U.S. Government Publishing Office.\n"
            f"API: {API}\n"
            f"Documentation: {SOURCE_URL}\n"
            f"Retrieved: {meta['retrieved_utc']}\n\n"
            "LICENCE\n"
            "United States Government Work. Under 17 U.S.C. 105, works of the\n"
            "U.S. Government are not subject to copyright protection in the\n"
            "United States and are in the public domain.\n\n"
            "The eCFR is an editorial compilation and is NOT the official\n"
            "legal edition of the Code of Federal Regulations. This dataset\n"
            "is a research benchmark, is not official government material,\n"
            "and must not be relied on for any legal or compliance purpose.\n"
            "The Office of the Federal Register, NARA and GPO do not endorse\n"
            "this dataset or any work derived from it.\n")
    sys.stderr.write(f"\nwrote {written} pairs -> {out_path}\nsha256 {sha}\n")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="../raw")
    p.add_argument("--titles", default=",".join(map(str, DEFAULT_TITLES)))
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--max-per-title", type=int, default=400)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    build(a.out, [int(t) for t in a.titles.split(",") if t.strip()],
          a.since, a.max_per_title, a.limit)


if __name__ == "__main__":
    main()
