"""Joins the already-downloaded raw table fragments (courts, citation-map,
opinions prefix) into the final raw corpus. Memory-bounded: never holds
the full opinions or citation-map table in memory at once -- streams
each file once, keeping only the id sets and the specific fields needed
downstream.

(v2: the first version of this script loaded the full 28.9M-row
citation-map into a Python list of dicts AND the full opinions text into
another list before filtering -- this pushed the process into 8GB+ of
swap and stalled. Fixed by computing only the id sets needed for
filtering in a first streaming pass, then writing matched rows directly
to disk in a second streaming pass, never materializing the full tables.)
"""
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(10**9)

RAW = Path(__file__).resolve().parent.parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def build_citation_opinion_ids(citation_map_path: str) -> tuple:
    """Single streaming pass: returns (distinct_opinion_ids, n_edges)."""
    ids = set()
    n = 0
    with open(citation_map_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ids.add(row["citing_opinion_id"])
            ids.add(row["cited_opinion_id"])
            n += 1
    return ids, n


def stream_matched_opinions(opinions_csv_path: str, citation_opinion_ids: set,
                             out_path: Path) -> set:
    """Single streaming pass over the opinions prefix: writes matched rows
    directly to out_path, never holding the full match set in memory
    beyond the ids needed for the edge-filtering pass. Returns the set of
    matched opinion ids."""
    matched_ids = set()
    with open(opinions_csv_path, newline="", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "cluster_id", "type", "plain_text"])
        writer.writeheader()
        for row in reader:
            if row.get("id") in citation_opinion_ids and len(row.get("plain_text", "")) > 500:
                writer.writerow({
                    "id": row["id"], "cluster_id": row["cluster_id"],
                    "type": row["type"], "plain_text": row["plain_text"],
                })
                matched_ids.add(row["id"])
    return matched_ids


def stream_internal_edges(citation_map_path: str, opinion_ids: set, out_path: Path) -> int:
    """Single streaming pass over citation-map: writes only edges where
    BOTH endpoints are in opinion_ids. Returns count written."""
    n = 0
    with open(citation_map_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["citing_opinion_id", "cited_opinion_id"])
        writer.writeheader()
        for row in reader:
            if row["citing_opinion_id"] in opinion_ids and row["cited_opinion_id"] in opinion_ids:
                writer.writerow({
                    "citing_opinion_id": row["citing_opinion_id"],
                    "cited_opinion_id": row["cited_opinion_id"],
                })
                n += 1
    return n


if __name__ == "__main__":
    citation_map_path = "/tmp/citation_map_full.csv"
    opinions_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/op_combined.csv"

    print("Pass 1/3: streaming citation-map for distinct opinion ids...", file=sys.stderr)
    citation_opinion_ids, n_edges_total = build_citation_opinion_ids(citation_map_path)
    print(f"  {n_edges_total} total edges, {len(citation_opinion_ids)} distinct opinion ids",
          file=sys.stderr)

    print("Pass 2/3: streaming opinions, writing matches directly to disk...", file=sys.stderr)
    matched_ids = stream_matched_opinions(opinions_path, citation_opinion_ids,
                                           RAW / "opinions.csv")
    print(f"  {len(matched_ids)} opinions matched and written", file=sys.stderr)

    print("Pass 3/3: streaming citation-map again for internal edges only...", file=sys.stderr)
    n_internal = stream_internal_edges(citation_map_path, matched_ids,
                                        RAW / "citation_edges.csv")
    print(f"  {n_internal} internal citation edges written", file=sys.stderr)

    with open(RAW / "matched_opinion_ids.json", "w") as f:
        json.dump(sorted(matched_ids), f)

    print(json.dumps({
        "n_opinions_matched": len(matched_ids),
        "n_internal_edges": n_internal,
        "n_citation_edges_total": n_edges_total,
    }, indent=2))
