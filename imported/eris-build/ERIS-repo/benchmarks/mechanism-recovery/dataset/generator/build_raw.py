#!/usr/bin/env python3
"""Fetch drug-label pharmacology text and FDA's own mechanism-of-action classes.

Source: openFDA Drug Label API (Structured Product Labeling submitted to FDA).
        https://open.fda.gov/apis/drug/label/download/
        (queried live via https://api.fda.gov/drug/label.json)
        Licence: CC0 1.0 Universal (https://open.fda.gov/license/)

What is fetched
---------------
For every label that carries BOTH a Clinical Pharmacology section and at least
one `openfda.pharm_class_moa` value:

    clinical_pharmacology   the label's own pharmacology prose
    pharm_class_moa         FDA's assigned mechanism-of-action class(es),
                            drawn from the NDC pharmacologic class vocabulary

The MoA classes are FDA's structured assignment, published in the label's
openfda block. They are not annotations created for this dataset, not inferred
from the text by a heuristic, and not model output.

WHY THE TEXT IS MASKED (measured, not assumed)
----------------------------------------------
Before masking, 57% of labels state their own mechanism verbatim in the
pharmacology prose -- a label classed "Cyclooxygenase Inhibitors [MoA]" often
contains the phrase "cyclooxygenase inhibitor". A keyword matcher would score
highly while learning nothing about pharmacology.

So a mask vocabulary is built from EVERY MoA class name in the corpus (not
just the row's own class -- otherwise which words are masked would itself
reveal the answer) and every occurrence is replaced with [MASK]. Measured on
1,401 sampled labels: 1,860 of 2,082 label-class pairs stopped being
keyword-recoverable, and the surviving text still runs to a median 7,308
characters of genuine pharmacology.

The vocabulary is written to the raw directory so the masking is auditable and
reproducible, and so prepare.py masks with exactly the vocabulary this fetch
saw.

Deliberate fetch choices
------------------------
- ONE worker, never parallel. openFDA rate-limits by IP; a single patient
  worker finishes sooner than several contended ones.
- SLEEP_SECONDS between every request.
- Broad `except Exception` in the retry loop. ConnectionResetError and
  TimeoutError are not HTTPError/URLError subclasses and would otherwise end a
  long run outright.
- Records are appended and flushed as each page is parsed, so an interrupted
  run loses at most the page in flight.
- Paging uses `skip`, which openFDA caps at 25,000; the fetch stops cleanly
  there rather than erroring.

Usage:
    python build_raw.py --out ../raw
    python build_raw.py --out ../raw --max-records 12000
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

API = "https://api.fda.gov/drug/label.json"
SOURCE_URL = "https://api.fda.gov/drug/label.json"
LICENSE_URL = "https://open.fda.gov/license/"
UA = "eris-dataset-builder/1.0 (research benchmark construction)"

SLEEP_SECONDS = 0.35
MAX_RETRIES = 5
TIMEOUT = 90
PAGE = 100
SKIP_CAP = 25000          # openFDA refuses skip beyond this

SEARCH = ("_exists_:openfda.pharm_class_moa+AND+_exists_:clinical_pharmacology")

MIN_TEXT_CHARS = 800      # below this there is too little pharmacology to read
MAX_TEXT_CHARS = 60000    # a handful of labels are enormous; cap the tail


def _fetch(url):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:          # openFDA returns 404 for "no more results"
                return None
            if e.code == 400:          # skip past the cap
                return None
            last = e
        except Exception as e:         # ConnectionReset, Timeout, JSON, ...
            last = e
        time.sleep(SLEEP_SECONDS * (attempt + 1) * 3)
    sys.stderr.write(f"  give up after {MAX_RETRIES}: {url} ({last})\n")
    return None


def join_section(v):
    if not v:
        return ""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    return re.sub(r"\s+", " ", str(v)).strip()


# Words that appear in MoA class names but carry no mechanism information --
# masking them would gut the prose without hiding anything.
STOP_IN_CLASS = {
    "inhibitors", "agonists", "antagonists", "inducers", "receptor",
    "receptors", "hormone", "agents", "other", "activity", "activation",
    "acid", "alpha", "beta", "type", "class", "cell", "cells", "factor",
    "blockers", "modulators", "stimulants", "substrates", "binding",
}


def class_terms(moa_term):
    """Mechanism words inside one MoA class name, e.g.
    'Cyclooxygenase Inhibitors [MoA]' -> {'cyclooxygenase'}."""
    t = re.sub(r"\s*\[\w+\]\s*$", "", moa_term)
    out = set()
    for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", t):
        wl = w.lower()
        if wl not in STOP_IN_CLASS:
            out.add(wl)
    return out


def build(out_dir, max_records):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "labels.jsonl")

    # ---- pass 1: collect records --------------------------------------- #
    records = []
    skip = 0
    seen_ids = set()
    while skip < SKIP_CAP and (not max_records or len(records) < max_records):
        url = f"{API}?limit={PAGE}&skip={skip}&search={SEARCH}"
        d = _fetch(url)
        if not d or not d.get("results"):
            break
        for r in d["results"]:
            ofa = r.get("openfda", {}) or {}
            moa = ofa.get("pharm_class_moa") or []
            text = join_section(r.get("clinical_pharmacology"))
            if not moa or not text:
                continue
            if not (MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS):
                continue
            rid = r.get("id") or r.get("set_id")
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            records.append({
                "record_id": rid,
                "set_id": r.get("set_id"),
                "effective_time": r.get("effective_time"),
                "generic_name": (ofa.get("generic_name") or [None])[0],
                "substance_name": ofa.get("substance_name") or [],
                "route": ofa.get("route") or [],
                "manufacturer_name": (ofa.get("manufacturer_name") or [None])[0],
                "pharm_class_moa": sorted(set(moa)),
                "pharm_class_epc": sorted(set(ofa.get("pharm_class_epc") or [])),
                "clinical_pharmacology": text,
            })
        skip += PAGE
        if len(records) % 500 < PAGE:
            sys.stderr.write(f"  collected {len(records)} (skip={skip})\n")
        time.sleep(SLEEP_SECONDS)

    sys.stderr.write(f"collected {len(records)} usable labels\n")
    if not records:
        raise SystemExit("no records fetched")

    # ---- pass 2: mask vocabulary from EVERY class seen ------------------ #
    # Built from the whole corpus, never per row: if only a row's own class
    # words were masked, WHICH words are masked would reveal the answer.
    vocab = set()
    for r in records:
        for t in r["pharm_class_moa"]:
            vocab |= class_terms(t)
    vocab = sorted(vocab)
    sys.stderr.write(f"mask vocabulary: {len(vocab)} terms\n")

    pat = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in
                          sorted(vocab, key=len, reverse=True)) + r")\b",
        re.I)

    n_masks = []
    for r in records:
        masked, n = pat.subn("[MASK]", r["clinical_pharmacology"])
        r["pharmacology_masked"] = re.sub(r"\s+", " ", masked).strip()
        r["n_masked"] = n
        n_masks.append(n)
        del r["clinical_pharmacology"]      # unmasked text is never shipped

    with open(out_path, "w", encoding="utf8") as fh:
        for r in sorted(records, key=lambda x: x["record_id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "mask_vocabulary.json"), "w",
              encoding="utf8") as fh:
        json.dump({
            "n_terms": len(vocab),
            "built_from": "every pharm_class_moa value in this corpus",
            "note": ("Built corpus-wide, never per row: masking only a row's "
                     "own class words would make the masking pattern itself "
                     "reveal the answer."),
            "stopwords_excluded": sorted(STOP_IN_CLASS),
            "terms": vocab,
        }, fh, indent=2)

    cls = Counter(t for r in records for t in r["pharm_class_moa"])
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    meta = {
        "generated_by": "build_raw.py",
        "source": "openFDA Drug Label API (Structured Product Labeling)",
        "source_url": SOURCE_URL,
        "publisher": "U.S. Food and Drug Administration",
        "licence": "CC0 1.0 Universal",
        "licence_url": LICENSE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "api_search": SEARCH,
        "n_labels": len(records),
        "n_distinct_moa_classes": len(cls),
        "mask_vocabulary_terms": len(vocab),
        "median_masks_per_label": sorted(n_masks)[len(n_masks) // 2],
        "labels_sha256": sha,
        "filters": {
            "requires_pharm_class_moa": True,
            "requires_clinical_pharmacology": True,
            "min_text_chars": MIN_TEXT_CHARS,
            "max_text_chars": MAX_TEXT_CHARS,
            "deduplicated_by": "label id",
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)

    with open(os.path.join(out_dir, "ATTRIBUTION.txt"), "w",
              encoding="utf8") as fh:
        fh.write(
            "openFDA Drug Label API\n"
            "U.S. Food and Drug Administration\n"
            f"Endpoint: {SOURCE_URL}\n"
            f"Retrieved: {meta['retrieved_utc']}\n\n"
            "LICENCE\n"
            "Creative Commons CC0 1.0 Universal.\n"
            f"{LICENSE_URL}\n"
            "openFDA states: \"the content, data, documentation, code, and\n"
            "related materials on openFDA is public domain and made available\n"
            "with a Creative Commons CC0 1.0 Universal dedication.\"\n\n"
            "DISCLAIMER\n"
            "openFDA states that its data should not be relied upon to make\n"
            "decisions regarding medical care, and that results are\n"
            "unvalidated. This dataset is a research benchmark built from that\n"
            "data, with the pharmacology text deliberately masked. It is not\n"
            "medical information and must not be used for any clinical,\n"
            "prescribing or regulatory purpose. The FDA does not endorse this\n"
            "dataset or any work derived from it.\n")

    sys.stderr.write(f"\nwrote {len(records)} labels -> {out_path}\n"
                     f"distinct MoA classes: {len(cls)}\nsha256 {sha}\n")
    return len(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="../raw")
    p.add_argument("--max-records", type=int, default=0)
    a = p.parse_args()
    build(a.out, a.max_records)


if __name__ == "__main__":
    main()
