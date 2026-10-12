"""Fetches a bounded slice of CourtListener's public bulk data and joins it
into the raw corpus for the citation-graph-reconstruction challenge.

Source: Free Law Project / CourtListener bulk data
  https://www.courtlistener.com/help/api/bulk-data/
License: Public Domain Mark (Free Law Project's own statement, quoted
  verbatim: "Our bulk data files are free of known copyright
  restrictions."). Underlying judicial opinion text is independently
  uncopyrightable U.S. government-edict material (Banks v. Manchester,
  128 U.S. 244 (1888); Georgia v. Public.Resource.Org, 590 U.S. 255
  (2020)).

Snapshot date: 2022-09-30 (the earliest snapshot where all needed tables
-- citation-map, opinion-clusters, opinions, dockets, courts -- are
simultaneously populated; citation-map's 2022-08-03/08-31 snapshots are
empty placeholders, verified by direct inspection before choosing this
date).

MEASURED strategy (see DESIGN.md for the full record of what was tried
and rejected):

1. Download the full citation-map table (~205MB compressed) via
   parallel HTTP range requests. Stream it once to collect the set of
   every distinct opinion id that appears as either endpoint of any
   citation edge (measured: 28,900,122 edges, 3,524,995 distinct ids).

2. Download a BOUNDED PREFIX of the opinions table (the full table is
   ~30GB compressed and not ordered by any join key useful for a
   pre-filter -- see the rejected court/date pre-filter strategy below)
   via parallel HTTP range requests, stream it once, and keep only rows
   whose id is in the citation-map's opinion-id set AND whose
   plain_text is substantial (>500 chars). This is a citation-graph-
   MEMBERSHIP filter, not a court/date filter -- measured far denser
   (~7.6% match rate in a 1GB test read) than a naive court-based
   pre-filter applied before looking at the citation graph at all
   (measured ~0.003% match rate for the same test).

3. Re-stream citation-map a second time, keeping only edges where BOTH
   endpoints survived step 2's filter -- this is the actual internal
   citation-edge set `prepare.py` builds bags from.

4. Download the full opinion-clusters table (~1.2GB compressed, small
   enough for a full parallel download rather than a bounded prefix),
   filtered to only the clusters the kept opinions belong to -- this
   supplies date_filed, case_name (for redaction), and docket_id.

5. Download a bounded prefix of the dockets table, filtered to the
   dockets the kept clusters reference, to resolve court_id (needed for
   the held-out-court train/test split).

REJECTED: an earlier version pre-filtered dockets->court_id to the 13
federal circuit courts of appeals BEFORE looking at opinions at all.
Measured on a 1GB opinions test read: 1 match out of 8,698 well-formed
rows (0.003%) against 12,189 target clusters -- opinions.csv is not
ordered by any key correlated with a narrow court pre-filter, so this
approach would have required reading the large majority of the 30GB
table to find enough matches. The citation-graph-membership filter above
replaced it.

MEMORY NOTE: an earlier version of the join step (superseded by
join_corpus.py's current streaming design) loaded the full 28.9M-row
citation-map into a Python list of dicts, which pushed the process into
8GB+ of swap and stalled for several minutes before being killed. The
current design never holds more than the id SET (not full rows) for the
large tables in memory at once -- see join_corpus.py.

Uses PARALLEL byte-range HTTP requests (S3 supports Accept-Ranges: bytes)
rather than a single serial stream: measured ~3.4x faster aggregate
throughput on this network (2.9 MB/s with 4 parallel 50MB chunks vs.
0.86 MB/s serial). Concatenating in-order parallel chunks is byte-
identical to a serial download of the same byte range, so bz2 correctly
decompresses the concatenated file.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

csv.field_size_limit(10**9)

BASE = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data"
SNAPSHOT = "2022-09-30"

# Bounded prefix sizes, in bytes, for the two large tables that are not
# downloaded in full. See module docstring for why these bounds exist.
OPINIONS_PREFIX_BYTES = 2_000_000_000     # opinions table is ~30GB total
DOCKETS_PREFIX_BYTES = 1_200_000_000      # dockets table is ~3.4GB total


def parallel_range_download(url: str, out_path: Path, total_bytes: int,
                             chunk_size: int = 100_000_000, concurrency: int = 8) -> None:
    """Downloads [0, total_bytes) of `url` as N parallel byte-range
    requests, concatenated in order into `out_path`."""
    import concurrent.futures

    n_chunks = (total_bytes + chunk_size - 1) // chunk_size
    chunk_dir = out_path.parent / f"{out_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_bytes) - 1
        plan.append((i, start, end, chunk_dir / f"chunk_{i:03d}.bin"))

    def fetch(i, start, end, chunk_path):
        if chunk_path.exists() and chunk_path.stat().st_size == end - start + 1:
            return
        subprocess.run(
            ["curl", "-sSL", "--range", f"{start}-{end}", url, "-o", str(chunk_path)],
            check=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(fetch, *p) for p in plan]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    with open(out_path, "wb") as out:
        for i, start, end, chunk_path in plan:
            out.write(chunk_path.read_bytes())
    for _, _, _, chunk_path in plan:
        chunk_path.unlink(missing_ok=True)
    chunk_dir.rmdir()


def decompress_bz2(bz2_path: Path, csv_path: Path) -> None:
    """Decompresses a (possibly truncated-prefix) bz2 file. A truncated
    prefix decompresses cleanly up to its last complete bz2 block; the
    trailing partial block is simply dropped (bzip2's own CLI and the
    stdlib module both handle this without raising on a clean prefix)."""
    subprocess.run(["bash", "-c", f"bzcat '{bz2_path}' 2>/dev/null > '{csv_path}'"])


def stream_ids(csv_path: Path, id_cols: tuple) -> set:
    ids = set()
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            for c in id_cols:
                ids.add(row[c])
    return ids


def stream_filtered(csv_path: Path, row_filter, wanted_cols: list, out_path: Path) -> int:
    n = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=wanted_cols)
        writer.writeheader()
        for row in reader:
            if row_filter(row):
                writer.writerow({k: row[k] for k in wanted_cols})
                n += 1
    return n


def build(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Step 1: courts (full table)...", file=sys.stderr)
    courts_bz2 = raw_dir / "courts_dl.csv.bz2"
    subprocess.run(["curl", "-sSL", f"{BASE}/courts-{SNAPSHOT}.csv.bz2",
                     "-o", str(courts_bz2)], check=True)
    decompress_bz2(courts_bz2, raw_dir / "courts.csv")

    print("Step 2: citation-map (full table, parallel download)...", file=sys.stderr)
    cm_bz2 = raw_dir / "citation_map_dl.csv.bz2"
    parallel_range_download(f"{BASE}/citation-map-{SNAPSHOT}.csv.bz2", cm_bz2,
                             total_bytes=205_000_000, chunk_size=50_000_000, concurrency=5)
    cm_csv = raw_dir / "citation_map_full.csv"
    decompress_bz2(cm_bz2, cm_csv)
    citation_opinion_ids = stream_ids(cm_csv, ("citing_opinion_id", "cited_opinion_id"))
    print(f"  {len(citation_opinion_ids)} distinct opinion ids in the citation graph",
          file=sys.stderr)

    print(f"Step 3: opinions (first {OPINIONS_PREFIX_BYTES/1e9:.1f} GB, parallel)...",
          file=sys.stderr)
    op_bz2 = raw_dir / "opinions_dl.csv.bz2"
    parallel_range_download(f"{BASE}/opinions-{SNAPSHOT}.csv.bz2", op_bz2,
                             total_bytes=OPINIONS_PREFIX_BYTES,
                             chunk_size=100_000_000, concurrency=8)
    op_csv = raw_dir / "opinions_full.csv"
    decompress_bz2(op_bz2, op_csv)
    n_opinions = stream_filtered(
        op_csv,
        lambda r: r.get("id") in citation_opinion_ids and len(r.get("plain_text", "")) > 500,
        ["id", "cluster_id", "type", "plain_text"],
        raw_dir / "opinions.csv",
    )
    print(f"  {n_opinions} opinions matched", file=sys.stderr)

    matched_opinion_ids = stream_ids(raw_dir / "opinions.csv", ("id",))
    n_internal = stream_filtered(
        cm_csv,
        lambda r: r["citing_opinion_id"] in matched_opinion_ids
        and r["cited_opinion_id"] in matched_opinion_ids,
        ["citing_opinion_id", "cited_opinion_id"],
        raw_dir / "citation_edges.csv",
    )
    print(f"  {n_internal} internal citation edges", file=sys.stderr)

    needed_clusters = set()
    with open(raw_dir / "opinions.csv", newline="") as f:
        for row in csv.DictReader(f):
            needed_clusters.add(row["cluster_id"])

    print("Step 4: opinion-clusters (full table, parallel)...", file=sys.stderr)
    oc_bz2 = raw_dir / "opinion_clusters_dl.csv.bz2"
    parallel_range_download(f"{BASE}/opinion-clusters-{SNAPSHOT}.csv.bz2", oc_bz2,
                             total_bytes=1_250_000_000, chunk_size=100_000_000, concurrency=8)
    oc_csv = raw_dir / "opinion_clusters_full.csv"
    decompress_bz2(oc_bz2, oc_csv)
    n_clusters = stream_filtered(
        oc_csv,
        lambda r: r.get("id") in needed_clusters,
        ["id", "date_filed", "case_name", "case_name_short", "docket_id",
         "precedential_status", "blocked"],
        raw_dir / "opinion_clusters.csv",
    )
    print(f"  {n_clusters} clusters matched", file=sys.stderr)

    needed_dockets = set()
    with open(raw_dir / "opinion_clusters.csv", newline="") as f:
        for row in csv.DictReader(f):
            needed_dockets.add(row["docket_id"])

    print(f"Step 5: dockets (first {DOCKETS_PREFIX_BYTES/1e9:.1f} GB, parallel)...",
          file=sys.stderr)
    dk_bz2 = raw_dir / "dockets_dl.csv.bz2"
    parallel_range_download(f"{BASE}/dockets-{SNAPSHOT}.csv.bz2", dk_bz2,
                             total_bytes=DOCKETS_PREFIX_BYTES,
                             chunk_size=100_000_000, concurrency=8)
    dk_csv = raw_dir / "dockets_full.csv"
    decompress_bz2(dk_bz2, dk_csv)
    n_dockets = stream_filtered(
        dk_csv,
        lambda r: r.get("id") in needed_dockets,
        ["id", "court_id"],
        raw_dir / "docket_court_raw.csv",
    )
    # rename id -> docket_id for the shipped schema
    with open(raw_dir / "docket_court_raw.csv", newline="") as fin, \
         open(raw_dir / "docket_court.csv", "w", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout)
        w.writerow(["docket_id", "court_id"])
        for row in r:
            w.writerow([row["id"], row["court_id"]])
    print(f"  {n_dockets}/{len(needed_dockets)} dockets resolved (prefix-bounded read)",
          file=sys.stderr)

    meta = {
        "snapshot_date": SNAPSHOT,
        "source": "https://www.courtlistener.com/help/api/bulk-data/",
        "license": "Public Domain Mark",
        "n_opinions": n_opinions,
        "n_internal_citation_edges": n_internal,
        "n_clusters": n_clusters,
        "n_dockets_resolved": n_dockets,
        "n_dockets_needed": len(needed_dockets),
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(raw_dir / "build_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    # cleanup large intermediates, keep only the final joined files + raw courts/citation-map
    for p in [courts_bz2, op_bz2, oc_bz2, dk_bz2, op_csv, oc_csv, dk_csv,
              raw_dir / "docket_court_raw.csv"]:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    build(Path(__file__).parents[1] / "raw")
