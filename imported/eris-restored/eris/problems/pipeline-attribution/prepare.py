"""
prepare(raw, public, private) -> None

Builds the public/private split for the Digitization Pipeline Attribution
challenge from dataset/raw/snippets.jsonl + batches_meta.csv.

Design (see DESIGN.md for the full rationale):
- Split is BATCH-DISJOINT: test batches never appear in train. This forces a
  solver to learn a GENERAL OCR-pipeline-fingerprint notion from labeled train
  batches, rather than memorize a small fixed vocabulary of known batch IDs —
  and it blunts the "just globally cluster the whole corpus" shortcut, since
  test batches carry no train-time label a solver could look up.
- train.csv is FLAT and fully labeled (snippet_id, text, batch_id) — a normal
  supervised training set.
- test.csv is BAG-STRUCTURED and unlabeled: (bag_id, snippet_id, text). Each
  bag draws a random subset of 3-6 DISTINCT test batches and 3-10 snippets per
  chosen batch (both randomized per bag — a fixed bag shape would itself be a
  structural tell). Snippet order within a bag is shuffled so position carries
  no signal. Snippets are sampled WITHOUT replacement within one bag but WITH
  replacement across different bags (reusing a snippet in two different bags
  leaks nothing new — it's the same row's content, independently graded).
- Ground truth (private/answers.csv) stores the real batch_id per (bag_id,
  snippet_id) — never exposed to solvers. Grading only ever needs pairwise
  same-batch-or-not within a bag, never the batch_id's name — see grade.py.
"""
import csv
import json
import random
import shutil
from pathlib import Path

SEED = 20260812
N_GROUPS_PER_BAG = (3, 5)
N_SNIPPETS_PER_GROUP = (3, 8)
N_TEST_BAGS = 200
TEST_BATCH_FRACTION = 0.45


def _load_snippets(raw: Path):
    rows = [json.loads(line) for line in (raw / "snippets.jsonl").open()]
    by_batch = {}
    for r in rows:
        by_batch.setdefault(r["batch_id"], []).append(r)
    return by_batch


def _split_batches(by_batch, rng):
    batch_ids = sorted(by_batch.keys())
    rng.shuffle(batch_ids)
    n_test = max(1, round(len(batch_ids) * TEST_BATCH_FRACTION))
    test_batches = sorted(batch_ids[:n_test])
    train_batches = sorted(batch_ids[n_test:])
    return train_batches, test_batches


def _build_train(by_batch, train_batches):
    rows = []
    for b in train_batches:
        for r in by_batch[b]:
            rows.append({"snippet_id": r["snippet_id"], "text": r["text"], "batch_id": r["batch_id"]})
    return rows


def _build_bags(by_batch, test_batches, rng):
    bags = []  # list of (bag_id, [(snippet_id, text, batch_id), ...])
    for i in range(N_TEST_BAGS):
        bag_id = f"bag_{i:04d}"
        n_groups = rng.randint(*N_GROUPS_PER_BAG)
        chosen_batches = rng.sample(test_batches, min(n_groups, len(test_batches)))
        items = []
        for b in chosen_batches:
            pool = by_batch[b]
            n_snip = rng.randint(*N_SNIPPETS_PER_GROUP)
            n_snip = min(n_snip, len(pool))
            chosen = rng.sample(pool, n_snip)
            for r in chosen:
                items.append((r["snippet_id"], r["text"], r["batch_id"]))
        rng.shuffle(items)
        bags.append((bag_id, items))
    return bags


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    by_batch = _load_snippets(raw)
    train_batches, test_batches = _split_batches(by_batch, rng)

    train_rows = _build_train(by_batch, train_batches)
    bags = _build_bags(by_batch, test_batches, rng)

    with (public / "train.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snippet_id", "text", "batch_id"])
        w.writeheader()
        w.writerows(train_rows)

    # row_id is the genuinely unique per-row key: neither bag_id (repeats
    # within a bag) nor snippet_id (the same snippet is deliberately reused
    # across DIFFERENT bags) is unique alone. The platform's grading harness
    # keys submissions on the answers file's FIRST column, so this must be
    # both present and actually unique (see REJECTION-REGISTER.md 1.8).
    def row_id(bag_id, snippet_id):
        return f"{bag_id}::{snippet_id}"

    with (public / "test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "bag_id", "snippet_id", "text"])
        w.writeheader()
        for bag_id, items in bags:
            for snippet_id, text, _ in items:
                w.writerow({"row_id": row_id(bag_id, snippet_id), "bag_id": bag_id, "snippet_id": snippet_id, "text": text})

    with (private / "answers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "bag_id", "snippet_id", "batch_id"])
        w.writeheader()
        for bag_id, items in bags:
            for snippet_id, _, batch_id in items:
                w.writerow({"row_id": row_id(bag_id, snippet_id), "bag_id": bag_id, "snippet_id": snippet_id, "batch_id": batch_id})

    with (public / "sample_submission.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "bag_id", "snippet_id", "group_label"])
        w.writeheader()
        for bag_id, items in bags:
            for snippet_id, _, _ in items:
                w.writerow({"row_id": row_id(bag_id, snippet_id), "bag_id": bag_id, "snippet_id": snippet_id, "group_label": 0})

    stats = {
        "n_train_batches": len(train_batches),
        "n_test_batches": len(test_batches),
        "train_batches": train_batches,
        "test_batches": test_batches,
        "n_train_snippets": len(train_rows),
        "n_test_bags": len(bags),
        "n_test_snippets": sum(len(items) for _, items in bags),
        "bag_size_min": min(len(items) for _, items in bags),
        "bag_size_max": max(len(items) for _, items in bags),
    }
    with (public / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    validator_src = Path(__file__).parent / "dataset" / "validate_submission.py"
    if validator_src.exists():
        shutil.copyfile(validator_src, public / "validate_submission.py")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    prepare(Path("dataset/raw"), Path("dataset/public"), Path("dataset/private"))
