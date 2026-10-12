"""
Precompute, for every topic with >=2 works in dataset/raw/works.jsonl, its
topics ranked by mean-embedding similarity (most similar first), using a
pretrained sentence encoder over each topic's own abstracts. This is a
one-time, offline DATA-BUILD step run before `prepare.py`; its output,
dataset/raw/hard_neighbors.json, is shipped as a raw file alongside
works.jsonl and is read by prepare.py to select distractor topics that are
genuinely confusable with a query's true topic, rather than merely sharing
a coarse domain label. It does not run at grading time and is not available
to a solver beyond what prepare.py derives from it into the public split.

Writes dataset/raw/hard_neighbors.json:
  {topic_id: [most_similar_topic_id, second_most_similar, ...]}
  (self excluded; only topics with >=2 deduplicated works are included)

Reads the topic of each work from topic_map.json beside works.jsonl.

Usage: python build_hard_distractors.py --raw ../raw/works.jsonl --topic-map ../../build/topic_map.json --out ../../build/hard_neighbors.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

MIN_ITEMS_PER_TOPIC = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../raw/works.jsonl")
    ap.add_argument("--out", default="../../build/hard_neighbors.json")
    ap.add_argument("--topic-map", default="../../build/topic_map.json")
    args = ap.parse_args()

    rows = []
    seen = set()
    with open(args.raw) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["work_id"] in seen:
                continue
            seen.add(r["work_id"])
            rows.append(r)

    abstract_counts = defaultdict(int)
    for r in rows:
        abstract_counts[r["abstract"]] += 1
    rows = [r for r in rows if abstract_counts[r["abstract"]] == 1]

    with open(args.topic_map) as f:
        topic_map = json.load(f)
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[str(topic_map[r["work_id"]])].append(r)
    viable = {k: v for k, v in by_topic.items() if len(v) >= MIN_ITEMS_PER_TOPIC}
    print("viable topics:", len(viable))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_ids = sorted(viable.keys())
    centroids = {}
    for tid in topic_ids:
        items = sorted(viable[tid], key=lambda r: r["work_id"])
        texts = [r["abstract"] for r in items[:5]]
        embs = model.encode(texts, show_progress_bar=False)
        centroids[tid] = embs.mean(axis=0)

    mat = np.stack([centroids[t] for t in topic_ids])
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sim = mat_norm @ mat_norm.T

    neighbors = {}
    for i, tid in enumerate(topic_ids):
        order = np.argsort(-sim[i])
        nn = [topic_ids[j] for j in order if topic_ids[j] != tid]
        neighbors[tid] = nn

    with open(args.out, "w") as f:
        json.dump(neighbors, f)
    print("wrote", args.out, "for", len(neighbors), "topics")


if __name__ == "__main__":
    main()
