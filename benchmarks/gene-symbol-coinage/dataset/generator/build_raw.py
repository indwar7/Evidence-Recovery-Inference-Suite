#!/usr/bin/env python3
"""Fetch the raw archive for "Coining Gene Symbols from Gene Names".

SOURCE
------
HUGO Gene Nomenclature Committee (HGNC), complete gene set, dated quarterly
archive snapshot. Landing page: https://www.genenames.org/download/archive/

    snapshot : hgnc_complete_set_2026-07-07.txt   (tab-separated, UTF-8)
    licence  : CC0 1.0 -- https://www.genenames.org/about/license/

WHAT THIS SCRIPT DOES
---------------------
Downloads ONE file, byte for byte, and records its provenance. It performs no
annotation, no filtering, no rewriting and calls no model. Every value in the
raw archive is exactly what the HGNC published in that snapshot. All task
construction (filtering to protein-coding genes, the date-based split)
happens later, in prepare.py, from this file alone.

A dated archive snapshot is used rather than the rolling "current" file so a
re-run fetches the same bytes; the sha256 of the shipped copy is pinned below
and checked after download.

FETCH DISCIPLINE
----------------
  - one worker, one request; nothing is fetched in parallel
  - broad `except Exception` in the retry loop (ConnectionResetError and
    TimeoutError are not HTTPError subclasses)
  - bounded retries with linear backoff, 1.2 s minimum between attempts
  - the download is streamed to a temporary file and renamed only once
    complete, so an interrupted run never leaves a truncated raw file

Usage:
    python build_raw.py [--out ../raw]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = "hgnc_complete_set_2026-07-07.txt"
SNAPSHOT_URL = ("https://storage.googleapis.com/public-download-files/hgnc/"
                "archive/archive/quarterly/tsv/" + SNAPSHOT)
LANDING_URL = "https://www.genenames.org/download/archive/"
LICENCE_URL = "https://www.genenames.org/about/license/"
EXPECTED_SHA256 = ("e73e9259177884b5994fc81ed733c1b3"
                   "d4df34c84290bc9dddc86e960d5d6419")

USER_AGENT = "gene-symbol-coinage-build/1.0 (research dataset build)"
MAX_RETRIES = 5
SLEEP_S = 1.2

ATTRIBUTION = """\
Coining Gene Symbols from Gene Names -- raw archive attribution
===============================================================

Source      : HUGO Gene Nomenclature Committee (HGNC), complete gene set
Publisher   : HGNC, University of Cambridge
Landing page: https://www.genenames.org/download/archive/
Snapshot    : {snapshot}
Snapshot URL: {url}

Licence     : CC0 1.0 Universal (Public Domain Dedication)
Licence page: https://www.genenames.org/about/license/

The HGNC states on its licence page: "all data is released under the Creative
Commons Public Domain (CC0) License. This means that any form of reuse of the
content is permitted."

CC0 does not require attribution. It is given here as good practice, and
because the HGNC asks that reuse cite the resource:

  Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in
  2023. Nucleic Acids Research 51(D1):D1003-D1009.

The HGNC does not endorse this dataset or any work derived from it. This is a
research benchmark, not a nomenclature authority: approved symbols change, and
the current approved symbol for any gene must be taken from genenames.org.
"""


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, \
                    tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001 -- deliberate, see docstring
            last = e
            print(f"attempt {attempt}/{MAX_RETRIES} failed: {e!r}",
                  file=sys.stderr)
            time.sleep(SLEEP_S * attempt)
    raise SystemExit(f"could not fetch {url}: {last!r}")


def sha256_of(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(here.parent / "raw"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dest = out / SNAPSHOT
    if not dest.exists():
        print(f"fetching {SNAPSHOT_URL}")
        fetch(SNAPSHOT_URL, dest)
    digest = sha256_of(dest)
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"sha256 mismatch for {SNAPSHOT}: got {digest}, "
            f"expected {EXPECTED_SHA256}")

    with dest.open(encoding="utf8", newline="") as f:
        rd = csv.reader(f, delimiter="\t", quotechar='"')
        header = next(rd)
        n_rows = 0
        n_pc = 0
        i_type, i_status = header.index("locus_type"), header.index("status")
        for row in rd:
            if len(row) != len(header):
                raise SystemExit(f"row {n_rows + 2} has {len(row)} fields, "
                                 f"header has {len(header)}")
            n_rows += 1
            if (row[i_status] == "Approved"
                    and row[i_type] == "gene with protein product"):
                n_pc += 1

    meta = {
        "generated_by": "dataset/generator/build_raw.py",
        "source": "HUGO Gene Nomenclature Committee (HGNC) complete gene set",
        "publisher": "HGNC, University of Cambridge",
        "source_url": LANDING_URL,
        "snapshot_url": SNAPSHOT_URL,
        "licence": "CC0 1.0",
        "licence_url": LICENCE_URL,
        "retrieved_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "files": {
            SNAPSHOT: {
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "n_rows": n_rows,
                "n_columns": len(header),
                "n_approved_protein_coding": n_pc,
            }
        },
        "filters": "none -- the snapshot is stored byte for byte",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                   encoding="utf8")
    (out / "ATTRIBUTION.txt").write_text(
        ATTRIBUTION.format(snapshot=SNAPSHOT, url=SNAPSHOT_URL),
        encoding="utf8")
    print(f"{SNAPSHOT}: {n_rows:,} rows, {n_pc:,} approved protein-coding, "
          f"sha256 {digest}")


if __name__ == "__main__":
    main()
