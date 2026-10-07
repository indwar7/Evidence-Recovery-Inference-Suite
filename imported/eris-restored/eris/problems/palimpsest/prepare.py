"""Palimpsest -- build public/ and private/ splits from the raw revision windows.

Deterministic: fixed seed, no wall-clock, no network. Running it twice produces
byte-identical output.

An ITEM is a bag of snapshots drawn from 2-4 different Wikivoyage articles.
Within an article the snapshots are consecutive revisions, so the ground-truth
structure is a forest of chains: one chain per article, its earliest surviving
revision having parent ROOT. Snapshots are shuffled and relabelled `n00, n01,
...` in the shuffled order, so the identifier carries no order information.
Timestamps, revision ids, edit comments, usernames and article titles are never
released.

Anti-shortcut measures applied here (all documented in DESIGN.md):
  * `lastedit=` template parameters are stripped -- they are edit metadata that
    would otherwise order revisions at 0.64 pairwise accuracy.
  * The article title is masked out of the text, so chains cannot be grouped by
    reading the page name off the prose.
  * The number of articles per item varies and is never disclosed.
"""
from pathlib import Path
import json, random, re, csv

SEED = 20260805
TRAIN_FRACTION = 0.70
ARTICLES_PER_ITEM = (2, 4)        # distinct ARTICLES per item
REVS_PER_ARTICLE = (3, 5)         # revisions per lineage. Chain orientation is
                                  # ONE global decision, so long chains average
                                  # per-pair noise away: measured agent chain-
                                  # orientation accuracy was 0.95 at length 10,
                                  # 0.86 at length 6. Short chains remove that
                                  # amplifier.
SPLIT_PROB = 0.92                 # chance an article contributes SEVERAL lineages
                                  # from adjacent periods of its own history,
                                  # separated by a dropped revision. The two
                                  # are near-identical lexically, so grouping
                                  # them apart is genuinely ambiguous and a
                                  # path-cover decoder cannot merge its way out.
MAX_ITEMS = 900                   # cap keeps the public set ~200 MB and the
                                  # solver's quadratic pair featurisation cheap

TITLE_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")
# `lastedit=YYYY-MM-DD` inside {{listing}} templates records when an editor last
# verified the listing -- edit metadata, not content, and a measured 0.64-accuracy
# ordering shortcut. Editors misspell the parameter (`lastedited`, `lastedot`,
# `lastedir`, `last edit`), so the pattern matches the whole family; citation
# access dates are stripped for the same reason.
LASTEDIT = re.compile(
    r"\|\s*(?:last\s*ed[a-z]*|access\s*-?\s*date)\s*=\s*[^|}\n]*", re.I)
# Editors misspell it in many ways (`lastdit`, `lasted it`, `lastedot`, ...), so
# rather than chase spellings, drop ANY template parameter whose value is a bare
# ISO date. That is exhaustive by construction and cannot be defeated by a typo.
ISO_PARAM = re.compile(
    r"\|\s*[A-Za-z_][A-Za-z0-9_ .-]{0,24}\s*=\s*(?:19|20)\d{2}-\d{2}-\d{2}[^|}\n]*")


def scrub(text, title):
    """Strip edit metadata and the article title from a revision's wikitext."""
    text = LASTEDIT.sub("", text or "")
    text = ISO_PARAM.sub("", text)
    for p in sorted({p for p in TITLE_TOKEN.findall(title or "") if len(p) > 3},
                    key=len, reverse=True):
        text = re.sub(re.escape(p), "___", text, flags=re.I)
    return text


def load_windows(raw: Path):
    out = []
    with open(raw / "windows.jsonl", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_items(windows, rng):
    """Group article windows into items; return list of item dicts."""
    pool = list(windows)
    rng.shuffle(pool)
    items, idx = [], 0
    while idx < len(pool) and len(items) < MAX_ITEMS:
        m = rng.randint(*ARTICLES_PER_ITEM)
        if idx + m > len(pool):
            break
        group, idx = pool[idx:idx + m], idx + m
        snaps, truth = [], {}
        n_lineages = 0
        for art in group:
            revs = art["revs"]
            texts = [scrub(r["text"], art["title"]) for r in revs]

            # Exact-duplicate texts (reverts restoring a byte-identical page)
            # make the parent undecidable BY ANY METHOD -- and make the answer
            # key arbitrary. Well-posedness filter: only take contiguous runs
            # whose scrubbed texts are all distinct.
            def runs_ok(kk):
                return [s for s in range(len(revs) - kk + 1)
                        if len(set(texts[s:s + kk])) == kk]

            # An article contributes either one lineage, or TWO taken from
            # adjacent stretches of its own history with one revision dropped
            # between them. The dropped revision is the second lineage's true
            # parent, so its head is genuinely a ROOT -- the item stays
            # well-posed. But the two lineages are the same article days apart,
            # so they are nearly identical lexically: deciding whether the bag
            # holds one long lineage or two short ones is a real judgement, and
            # a decoder cannot merge its way out of it.
            n_split = (rng.choice([2, 2, 3]) if rng.random() < SPLIT_PROB else 1)
            ks = [rng.randint(*REVS_PER_ARTICLE) for _ in range(n_split)]
            spans = None
            while n_split > 1 and spans is None:
                need = sum(ks) + (n_split - 1)          # one dropped rev per gap
                cands = [s for s in range(len(revs) - need + 1)
                         if len(set(texts[s:s + need])) == need]
                if cands:
                    s0 = cands[rng.randrange(len(cands))]
                    spans, off = [], s0
                    for kk in ks:
                        spans.append((off, kk))
                        off += kk + 1                    # skip the dropped rev
                else:
                    n_split -= 1
                    ks = ks[:n_split]
            k1 = ks[0]
            if spans is None:                            # fall back to one lineage
                k = min(k1, len(revs))
                starts = runs_ok(k)
                while not starts and k > 3:
                    k -= 1
                    starts = runs_ok(k)
                if not starts:
                    continue                             # article unusable
                spans = [(starts[rng.randrange(len(starts))], k)]

            for start, k in spans:
                local = []
                for r, txt in zip(revs[start:start + k], texts[start:start + k]):
                    local.append(dict(tmp=f"t{len(snaps) + len(local)}", text=txt,
                                      title=art["title"], ts=r["ts"],
                                      revid=r.get("revid")))
                for j, s in enumerate(local):
                    truth[s["tmp"]] = "ROOT" if j == 0 else local[j - 1]["tmp"]
                snaps += local
                n_lineages += 1

        if len({s["title"] for s in snaps}) < 2 or n_lineages < 3:
            continue                          # item degenerated; drop it

        order = list(range(len(snaps)))
        rng.shuffle(order)
        relabel = {snaps[old]["tmp"]: f"n{new:02d}" for new, old in enumerate(order)}
        final = []
        for old_i in order:
            s = dict(snaps[old_i])
            s["sid"] = relabel[s["tmp"]]
            final.append(s)
        final.sort(key=lambda s: s["sid"])
        new_truth = {relabel[a]: (b if b == "ROOT" else relabel[b])
                     for a, b in truth.items()}
        items.append(dict(item_id=f"it{len(items):05d}", snapshots=final,
                          truth=new_truth,
                          provenance=[dict(title=g["title"]) for g in group]))
    return items


def _write_items(items, out_dir: Path, with_labels: bool):
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        payload = {"item_id": it["item_id"],
                   "snapshots": {s["sid"]: s["text"] for s in it["snapshots"]}}
        with open(snap_dir / f"{it['item_id']}.json", "w", encoding="utf8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    windows = load_windows(raw)
    windows.sort(key=lambda w: w["title"])          # deterministic input order
    items = build_items(windows, rng)

    n_train = int(len(items) * TRAIN_FRACTION)
    train, test = items[:n_train], items[n_train:]

    _write_items(train, public, with_labels=True)
    _write_items(test, public, with_labels=False)

    with open(public / "train.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "snapshot_id", "parent_id"])
        for it in train:
            for s in it["snapshots"]:
                w.writerow([it["item_id"], s["sid"], it["truth"][s["sid"]]])

    with open(public / "test.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "snapshot_id"])
        for it in test:
            for s in it["snapshots"]:
                w.writerow([it["item_id"], s["sid"]])

    with open(public / "sample_submission.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "snapshot_id", "parent_id"])
        for it in test:
            for s in it["snapshots"]:
                w.writerow([it["item_id"], s["sid"], "ROOT"])

    with open(private / "answers.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "snapshot_id", "parent_id"])
        for it in test:
            for s in it["snapshots"]:
                w.writerow([it["item_id"], s["sid"], it["truth"][s["sid"]]])

    meta = dict(seed=SEED, n_items=len(items), n_train_items=len(train),
                n_test_items=len(test),
                n_train_snapshots=sum(len(i["snapshots"]) for i in train),
                n_test_snapshots=sum(len(i["snapshots"]) for i in test),
                articles_per_item=list(ARTICLES_PER_ITEM),
                revs_per_article=list(REVS_PER_ARTICLE))
    with open(public / "dataset_stats.json", "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    prepare(Path(a[0] if a else "dataset/raw"),
            Path(a[1] if len(a) > 1 else "dataset/public"),
            Path(a[2] if len(a) > 2 else "dataset/private"))
