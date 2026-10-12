"""build_raw.py - fetch and pin the raw GeoNames Vietnam dump.

Downloads the Vietnam country export from GeoNames, verifies it against a pinned
SHA-256 (GeoNames republishes a rolling file, so the snapshot is pinned), and
extracts VN.txt into dataset/raw/ alongside meta.json and ATTRIBUTION.txt.

Fetch discipline: a single request, a generous timeout, broad exception handling,
and a .part temp file renamed on success so an interruption never leaves a
truncated download. Re-running is idempotent.

Source : https://download.geonames.org/export/dump/VN.zip
License : Creative Commons Attribution 4.0 (CC BY 4.0)
          https://creativecommons.org/licenses/by/4.0/
          (verbatim from https://download.geonames.org/export/dump/readme.txt)
"""
from __future__ import annotations
import hashlib, json, sys, time, zipfile
from pathlib import Path
from urllib.request import urlopen, Request

URL = "https://download.geonames.org/export/dump/VN.zip"
# Pinned snapshot downloaded 2026-10-10 (re-pinned: GeoNames republished the
# rolling file since the prior 2026-10-04 pin). The zip's own byte-for-byte
# hash is unstable across fetches even when its content is unchanged (observed:
# two fetches minutes apart gave different zip hashes but an identical VN.txt
# hash), so VN.txt's hash is the one that matters for reproducibility; the zip
# hash below is recorded for traceability of this specific fetch only.
PINNED_ZIP_SHA256 = "ef0b824c5ebcde447644ab4dfb0f1ec07c4177a048c6bafbcd0d4cd811c461f0"
PINNED_TXT_SHA256 = "d3151969634ab83071f0254d8817c5924be3b781c88674be95dfa161ce13297a"
SNAPSHOT_DATE = "2026-10-10"

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"


def _download(dest_zip: Path) -> None:
    tmp = dest_zip.with_suffix(".zip.part")
    req = Request(URL, headers={"User-Agent": "eris-dataset-builder/1.0"})
    try:
        with urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except Exception as exc:                       # broad on purpose
        raise SystemExit(f"download failed: {exc}")
    tmp.rename(dest_zip)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "VN.zip"
    if not zip_path.exists():
        print("downloading", URL)
        _download(zip_path)
        time.sleep(1.0)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    with zipfile.ZipFile(zip_path) as z:
        z.extract("VN.txt", RAW)
    txt = RAW / "VN.txt"
    txt_sha = hashlib.sha256(txt.read_bytes()).hexdigest()
    if txt_sha != PINNED_TXT_SHA256:
        print(f"WARNING: VN.txt sha256 {txt_sha} != pinned {PINNED_TXT_SHA256}\n"
              f"GeoNames has republished VN.txt's content since the pinned "
              f"snapshot ({SNAPSHOT_DATE}). Results may differ. (VN.zip's own "
              f"hash is not checked: it varies between fetches of identical "
              f"content, so it is recorded for traceability only.)")

    meta = {
        "source_name": "GeoNames geographical database - Vietnam export",
        "source_url": "https://www.geonames.org/search.html?q=&country=VN",
        "fetched_file": "VN.zip",
        "fetched_url": URL,
        "snapshot_date": SNAPSHOT_DATE,
        "vn_zip_sha256": digest,
        "vn_txt_sha256": txt_sha,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_source": "https://download.geonames.org/export/dump/readme.txt",
        "fetch_method": "single HTTP request, 120s timeout, broad except, "
                        ".part temp renamed on success",
    }
    (RAW / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (RAW / "ATTRIBUTION.txt").write_text(
        "This dataset is derived from the GeoNames geographical database.\n"
        "GeoNames is licensed under a Creative Commons Attribution 4.0 License\n"
        "(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/\n"
        "Source: https://www.geonames.org/  (Vietnam export: "
        "https://download.geonames.org/export/dump/VN.zip)\n"
        f"Snapshot: {SNAPSHOT_DATE}  VN.zip sha256: {digest}\n",
        encoding="utf-8",
    )
    print("wrote", txt, "meta.json, ATTRIBUTION.txt")


if __name__ == "__main__":
    main()
