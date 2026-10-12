"""
Fetches a stratified sample of OpenAlex works (title, abstract, primary_topic)
across a fixed, seeded list of OpenAlex Topic IDs, reconstructs each abstract
from its inverted-index representation, and writes dataset/raw/works.jsonl.

OpenAlex data is released under CC0 (https://openalex.org); abstracts are
redistributed by this script as word->position inverted indexes reconstructed
to plain text, consistent with OpenAlex's own documented reconstruction method.

Usage: python build_raw.py --out ../raw/works.jsonl --n-topics 220 --per-topic 10
"""
import argparse
import json
import os
import random
import time
import urllib.request

API = "https://api.openalex.org"
HEADERS = {"User-Agent": "eris-topic-exemplar-match (mailto:aseem12321@gmail.com)"}
SEED = 20261008


def fetch(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def reconstruct_abstract(inv):
    if not inv:
        return ""
    maxpos = max(p for ps in inv.values() for p in ps)
    arr = [""] * (maxpos + 1)
    for word, positions in inv.items():
        for p in positions:
            arr[p] = word
    return " ".join(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../raw/works.jsonl")
    ap.add_argument("--n-topics", type=int, default=220)
    ap.add_argument("--per-topic", type=int, default=10)
    ap.add_argument("--topic-map-out", default="../../build/topic_map.json",
                    help="build-only; embedded into prepare_groups.py, never shipped in raw/")
    args = ap.parse_args()

    rng = random.Random(SEED)

    topic_pool = []
    cursor = "*"
    while cursor:
        url = (
            f"{API}/topics?per-page=200&select=id,display_name,works_count"
            f"&sort=works_count:desc&cursor={cursor}"
        )
        resp = fetch(url)
        for t in resp["results"]:
            if t["works_count"] >= 500:
                topic_pool.append(t["id"].split("/")[-1])
        cursor = resp["meta"].get("next_cursor")
        time.sleep(0.3)

    rng.shuffle(topic_pool)
    topic_ids = topic_pool[: args.n_topics]
    print(f"discovered {len(topic_pool)} eligible topics, using {len(topic_ids)}")

    out_f = open(args.out, "w")
    work_topic = {}
    n_written = 0
    for i, tid in enumerate(topic_ids):
        url = (
            f"{API}/works?filter=primary_topic.id:{tid},has_abstract:true,"
            f"publication_year:2015-2023,type:article"
            f"&select=id,title,abstract_inverted_index,primary_topic,publication_year"
            f"&per-page={args.per_topic}&sample={args.per_topic}&seed={SEED}"
        )
        try:
            d = fetch(url)
        except Exception as e:
            print(f"[{i}] skip {tid}: {e}")
            continue
        for r in d.get("results", []):
            inv = r.get("abstract_inverted_index")
            topic = r.get("primary_topic")
            if not inv or not topic:
                continue
            abstract = reconstruct_abstract(inv)
            if len(abstract.split()) < 40:
                continue
            work_id = r["id"].split("/")[-1]
            row = {
                "work_id": work_id,
                "title": r["title"],
                "abstract": abstract,
                "publication_year": r.get("publication_year"),
            }
            out_f.write(json.dumps(row) + "\n")
            work_topic[work_id] = topic["id"].split("/")[-1]
            n_written += 1
        time.sleep(0.8)

    out_f.close()
    # The topic assignment is written to a build-only file (not raw/), keyed by
    # work_id and re-indexed to opaque integers; build_prepare_groups.py embeds
    # it into prepare_groups.py so nothing in the data mount pairs an abstract
    # with its topic.
    topics = sorted(set(work_topic.values()))
    random.Random(SEED).shuffle(topics)
    t2i = {t: i for i, t in enumerate(topics)}
    map_path = args.topic_map_out
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    with open(map_path, "w") as f:
        json.dump({w: t2i[t] for w, t in work_topic.items()}, f)
    print(f"wrote {n_written} rows to {args.out} and {len(work_topic)} entries to {map_path}")


if __name__ == "__main__":
    main()
