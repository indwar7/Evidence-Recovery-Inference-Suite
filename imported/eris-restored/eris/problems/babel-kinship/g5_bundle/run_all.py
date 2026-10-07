"""
Babel Kinship -- self-contained build + baseline script for the g5 box.
Downloads real MASSIVE data itself. Writes everything under ./bk_work/ so it
never touches any other running job's files or GPU memory beyond what
sentence-transformers allocates for its own model.
"""
import csv, json, random, tarfile, urllib.request
from pathlib import Path
from collections import defaultdict

ROOT = Path("./bk_work")
RAW = ROOT / "raw"
WORK = ROOT / "out"
RAW.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
TARBALL = RAW / "massive.tar.gz"

if not (RAW / "data" / "en-US.jsonl").exists():
    print("downloading MASSIVE...")
    urllib.request.urlretrieve(MASSIVE_URL, TARBALL)
    with tarfile.open(TARBALL) as t:
        t.extractall(RAW)
    # archive extracts to raw/1.1/... -- flatten
    inner = RAW / "1.1"
    if inner.exists():
        for p in inner.iterdir():
            p.rename(RAW / p.name)
        inner.rmdir()
    print("done.")
else:
    print("MASSIVE already present, skipping download.")

# ---- DEFAULT config (adjustable -- see TODO(human) note from earlier) ----
LANGS = ["en-US","hi-IN","ta-IN","mn-MN","ar-SA","zh-CN","ja-JP","ko-KR",
         "th-TH","ru-RU","he-IL","el-GR","vi-VN","sw-KE","fi-FI","ka-GE",
         "km-KH","my-MM","am-ET","hy-AM"]
N_TRAIN_BAGS, N_TEST_BAGS = 300, 100
PAIR_IDS_PER_BAG, SINGLE_IDS_PER_BAG = 6, 6
random.seed(20260909)
# ---------------------------------------------------------------------------

print(f"loading {len(LANGS)} locales...")
data = {}
for lang in LANGS:
    d = {}
    with open(RAW / "data" / f"{lang}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            d[r["id"]] = r["utt"]
    data[lang] = d
all_ids = sorted(data["en-US"].keys(), key=int)
print(f"ids available: {len(all_ids)}")

def build_bags(ids_pool, n_bags, pair_ids_per_bag, single_ids_per_bag, seed):
    rng = random.Random(seed)
    ids_pool = list(ids_pool)
    rows = []
    for b in range(n_bags):
        bag_id = f"b{b:05d}"
        rng.shuffle(ids_pool)
        need = pair_ids_per_bag + single_ids_per_bag
        chosen = ids_pool[:need]
        pair_pool_ids = chosen[:pair_ids_per_bag]
        single_pool_ids = chosen[pair_ids_per_bag:need]
        items = []
        for uid in pair_pool_ids:
            for lg in rng.sample(LANGS, 2):
                if uid in data[lg]:
                    items.append((uid, lg, data[lg][uid]))
        for uid in single_pool_ids:
            lg = rng.choice(LANGS)
            if uid in data[lg]:
                items.append((uid, lg, data[lg][uid]))
        rng.shuffle(items)
        local = [f"{bag_id}_n{i:02d}" for i in range(len(items))]
        uid_to_local = defaultdict(list)
        for loc, (uid, lg, utt) in zip(local, items):
            uid_to_local[uid].append(loc)
        for loc, (uid, lg, utt) in zip(local, items):
            partners = [x for x in uid_to_local[uid] if x != loc]
            rows.append({"bag_id": bag_id, "item_id": loc, "text": utt,
                         "partner_id": partners[0] if partners else "ALONE"})
    return rows

train_rows = build_bags(all_ids[:8000], N_TRAIN_BAGS, PAIR_IDS_PER_BAG, SINGLE_IDS_PER_BAG, seed=1)
test_rows  = build_bags(all_ids[8000:], N_TEST_BAGS,  PAIR_IDS_PER_BAG, SINGLE_IDS_PER_BAG, seed=2)

def write_csv(rows, path, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

write_csv(train_rows, WORK / "train.csv", ["bag_id","item_id","text","partner_id"])
write_csv(test_rows,  WORK / "test_public.csv", ["bag_id","item_id","text"])
write_csv(test_rows,  WORK / "test_answers.csv", ["bag_id","item_id","partner_id"])
print(f"train rows: {len(train_rows)}  test rows: {len(test_rows)}")

# ---- baselines, including a REAL GPU multilingual model this time ----
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

test_pub = load(WORK / "test_public.csv")
answers = {r["item_id"]: r["partner_id"] for r in load(WORK / "test_answers.csv")}
bags = defaultdict(list)
for r in test_pub:
    bags[r["bag_id"]].append(r)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}  (nvidia-smi should show this process if cuda)")

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # small, matches local CPU proxy result (0.587)
    "sentence-transformers/LaBSE",  # bigger, real GPU-scale multilingual model
]

def score(model, thresh):
    correct = total = 0
    for bag_id, items in bags.items():
        texts = [it["text"] for it in items]
        ids = [it["item_id"] for it in items]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                            device=device, batch_size=64)
        sim = emb @ emb.T
        np.fill_diagonal(sim, -1)
        for i, iid in enumerate(ids):
            j = int(sim[i].argmax())
            pred = ids[j] if sim[i, j] >= thresh else "ALONE"
            total += 1
            if pred == answers[iid]:
                correct += 1
    return correct / total

for name in MODELS:
    print(f"\nloading {name} on {device} ...")
    model = SentenceTransformer(name, device=device)
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        acc = score(model, th)
        print(f"  {name.split('/')[-1]:35s} thresh={th:<4} acc={acc:.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

print("\ndone. Results are in ./bk_work/out/")
